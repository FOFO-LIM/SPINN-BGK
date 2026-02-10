"""
GPU-accelerated numerical solver for 3D+3D Boltzmann-BGK equation using JAX.

Equation:
    df/dt + v . grad_x f = nu * (M[f] - f)

where M[f] is the local Maxwellian constructed from the moments of f:
    M[f] = rho / (2*pi*T)^(3/2) * exp(-|v - u|^2 / (2*T))

Numerical method:
    - Strang splitting: half-collision -> full-transport -> half-collision
    - Collision: exact exponential BGK relaxation
        f_new = M + (f - M) * exp(-nu * dt/2)
    - Transport: FFT spectral method with 3 sequential 1D sweeps (x, y, z)
        f_hat(k, t+dt) = f_hat(k, t) * exp(-i * k * v * dt)

Initial condition:
    rho(0,x,y,z) = 1 + 0.5 * sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z)
    u(0,x,y,z) = 0
    T(0,x,y,z) = 1
    f(0,x,y,z,vx,vy,vz) = Maxwellian(rho, u, T)

Domain:
    x, y, z in [-0.5, 0.5] (periodic)
    vx, vy, vz in [-6, 6]
    t in [0, T_final]

Parameters:
    Kn = 0.01  =>  nu = 1/Kn = 100
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np
import time
import os
import json
import argparse
from datetime import datetime

# Optional tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def get_gpu_memory_gib(device_idx=0, peak=False):
    """Get GPU memory usage in GiB for a single device."""
    try:
        device = jax.local_devices()[device_idx]
        stats = device.memory_stats()
        if stats:
            key = 'peak_bytes_in_use' if peak else 'bytes_in_use'
            return stats.get(key, 0) / (1024 ** 3)
    except Exception:
        pass
    return 0.0


class BGKSolver3D:
    """
    GPU-accelerated solver for 3D+3D Boltzmann-BGK equation using JAX.

    Array convention: f.shape = (Nx, Ny, Nz, Nvx, Nvy, Nvz)
        - spatial axes first (0,1,2), velocity axes last (3,4,5)

    Default: (16, 16, 16, 24, 24, 24) = ~57M points, ~0.43 GB float64
    """

    def __init__(self, Nx=16, Nv=24, Nt=5000, X=0.5, V=6.0, T_final=5.0, Kn=0.01):
        """
        Initialize the 3D BGK solver.

        Parameters
        ----------
        Nx : int
            Number of spatial grid points in each direction.
        Nv : int
            Number of velocity grid points in each direction.
        Nt : int
            Number of time steps.
        X : float
            Spatial domain half-width: x,y,z in [-X, X].
        V : float
            Velocity domain half-width: vx,vy,vz in [-V, V].
        T_final : float
            Final simulation time.
        Kn : float
            Knudsen number. Collision frequency nu = 1/Kn.
        """
        self.Nx = Nx
        self.Nv = Nv
        self.Nt = Nt
        self.X = X
        self.V = V
        self.T_final = T_final
        self.Kn = Kn
        self.nu = 1.0 / Kn

        self.dt = T_final / Nt

        self._setup_grids()
        self._precompute_phase_shifts()
        self._compile_functions()

        # Print summary
        total_points = Nx**3 * Nv**3
        mem_gb = total_points * 8 / (1024**3)  # float64
        print(f"BGKSolver3D initialized:")
        print(f"  Spatial: {Nx}^3 = {Nx**3} points, x in [-{X}, {X}]")
        print(f"  Velocity: {Nv}^3 = {Nv**3} points, v in [-{V}, {V}]")
        print(f"  Total: {total_points:,} points ({mem_gb:.2f} GB float64)")
        print(f"  Time: Nt={Nt}, dt={self.dt:.6e}, T_final={T_final}")
        print(f"  Kn={Kn}, nu={self.nu}")
        print(f"  JAX devices: {jax.devices()}")

    def _setup_grids(self):
        """Set up spatial, velocity, and wavenumber grids."""
        Nx, Nv, X, V = self.Nx, self.Nv, self.X, self.V

        # Spatial grid (periodic, exclude right endpoint)
        self.dx = 2 * X / Nx
        self.x = jnp.linspace(-X, X - self.dx, Nx)
        self.y = jnp.linspace(-X, X - self.dx, Nx)
        self.z = jnp.linspace(-X, X - self.dx, Nx)

        # Velocity grid (include endpoints for trapezoidal rule)
        self.dv = 2 * V / (Nv - 1)
        self.vx = jnp.linspace(-V, V, Nv)
        self.vy = jnp.linspace(-V, V, Nv)
        self.vz = jnp.linspace(-V, V, Nv)

        # Wavenumbers for FFT
        self.kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, self.dx)
        self.ky = 2 * jnp.pi * jnp.fft.fftfreq(Nx, self.dx)
        self.kz = 2 * jnp.pi * jnp.fft.fftfreq(Nx, self.dx)

        # Trapezoidal weights for velocity integration
        wvx = jnp.ones(Nv) * self.dv
        wvx = wvx.at[0].set(self.dv / 2)
        wvx = wvx.at[-1].set(self.dv / 2)
        self.wvx = wvx

        wvy = jnp.ones(Nv) * self.dv
        wvy = wvy.at[0].set(self.dv / 2)
        wvy = wvy.at[-1].set(self.dv / 2)
        self.wvy = wvy

        wvz = jnp.ones(Nv) * self.dv
        wvz = wvz.at[0].set(self.dv / 2)
        wvz = wvz.at[-1].set(self.dv / 2)
        self.wvz = wvz

        # 3D velocity weight tensor: (Nvx, Nvy, Nvz)
        self.w3d = wvx[:, None, None] * wvy[None, :, None] * wvz[None, None, :]

    def _precompute_phase_shifts(self):
        """
        Precompute phase shift arrays for FFT-based transport.

        For x-direction: phase_x[kx, vx] = exp(-i * kx * vx * dt)
        Similarly for y and z.
        """
        dt = self.dt

        # phase_x: shape (Nx, Nvx)
        self.phase_x = jnp.exp(-1j * jnp.outer(self.kx, self.vx) * dt)
        # phase_y: shape (Ny, Nvy)
        self.phase_y = jnp.exp(-1j * jnp.outer(self.ky, self.vy) * dt)
        # phase_z: shape (Nz, Nvz)
        self.phase_z = jnp.exp(-1j * jnp.outer(self.kz, self.vz) * dt)

    def _compile_functions(self):
        """JIT compile the main computational functions."""
        self._strang_step_jit = jit(self._strang_splitting_step)
        self._compute_moments_jit = jit(self._compute_moments)
        self._compute_conservation_jit = jit(self._compute_conservation)
        self._compute_fneq_rms_jit = jit(self._compute_fneq_rms)

    def _maxwellian_from_moments(self, rho, ux, uy, uz, T):
        """
        Compute 3D Maxwellian distribution from macroscopic moments.

        M = rho / (2*pi*T)^(3/2) * exp(-((vx-ux)^2 + (vy-uy)^2 + (vz-uz)^2) / (2*T))

        Parameters
        ----------
        rho : (Nx, Ny, Nz)
        ux, uy, uz : (Nx, Ny, Nz)
        T : (Nx, Ny, Nz)

        Returns
        -------
        M : (Nx, Ny, Nz, Nvx, Nvy, Nvz)
        """
        # Expand spatial dims for broadcasting: (Nx,Ny,Nz,1,1,1)
        rho_e = rho[:, :, :, None, None, None]
        ux_e = ux[:, :, :, None, None, None]
        uy_e = uy[:, :, :, None, None, None]
        uz_e = uz[:, :, :, None, None, None]
        T_e = T[:, :, :, None, None, None]

        # Velocity grids: (1,1,1,Nvx,1,1), etc.
        vx_e = self.vx[None, None, None, :, None, None]
        vy_e = self.vy[None, None, None, None, :, None]
        vz_e = self.vz[None, None, None, None, None, :]

        v_sq = (vx_e - ux_e)**2 + (vy_e - uy_e)**2 + (vz_e - uz_e)**2

        prefactor = rho_e / (2 * jnp.pi * T_e)**(3.0 / 2.0)
        M = prefactor * jnp.exp(-v_sq / (2 * T_e))

        return M

    def _compute_moments(self, f):
        """
        Compute macroscopic moments from f using trapezoidal integration.

        Parameters
        ----------
        f : (Nx, Ny, Nz, Nvx, Nvy, Nvz)

        Returns
        -------
        rho : (Nx, Ny, Nz) -- density
        ux, uy, uz : (Nx, Ny, Nz) -- bulk velocities
        T : (Nx, Ny, Nz) -- temperature
        """
        # Weight tensor: (1,1,1,Nvx,Nvy,Nvz) for broadcasting
        w = self.w3d[None, None, None, :, :, :]

        # Density: rho = int f dv
        rho = jnp.sum(f * w, axis=(3, 4, 5))

        rho_safe = jnp.maximum(rho, 1e-16)

        # Momentum: rho*u = int v*f dv
        vx_e = self.vx[None, None, None, :, None, None]
        vy_e = self.vy[None, None, None, None, :, None]
        vz_e = self.vz[None, None, None, None, None, :]

        mom_x = jnp.sum(f * vx_e * w, axis=(3, 4, 5))
        mom_y = jnp.sum(f * vy_e * w, axis=(3, 4, 5))
        mom_z = jnp.sum(f * vz_e * w, axis=(3, 4, 5))

        ux = mom_x / rho_safe
        uy = mom_y / rho_safe
        uz = mom_z / rho_safe

        # Energy: (3/2)*rho*T = int 0.5*|v-u|^2 * f dv
        # => T = (1/(3*rho)) * int (|v-u|^2) * f dv
        # Alternatively: T = (E_kin/rho - |u|^2) / 3  where E_kin = int |v|^2 f dv / rho
        # More numerically stable: compute int |v|^2 f dv first
        energy = jnp.sum(f * (vx_e**2 + vy_e**2 + vz_e**2) * w, axis=(3, 4, 5))
        T = (energy / rho_safe - (ux**2 + uy**2 + uz**2)) / 3.0

        return rho, ux, uy, uz, T

    def _collision_half_step(self, f):
        """
        Exact exponential BGK collision for half time step.

        f_new = M + (f - M) * exp(-nu * dt/2)

        Parameters
        ----------
        f : (Nx, Ny, Nz, Nvx, Nvy, Nvz)

        Returns
        -------
        f_new : (Nx, Ny, Nz, Nvx, Nvy, Nvz)
        """
        rho, ux, uy, uz, T = self._compute_moments(f)
        M = self._maxwellian_from_moments(rho, ux, uy, uz, T)
        decay = jnp.exp(-self.nu * self.dt / 2)
        f_new = M + (f - M) * decay
        return f_new

    def _transport_step(self, f):
        """
        FFT-based spectral transport for full time step.

        Three sequential 1D sweeps (x, y, z):
            f_hat_x(kx) *= exp(-i * kx * vx * dt)
            f_hat_y(ky) *= exp(-i * ky * vy * dt)
            f_hat_z(kz) *= exp(-i * kz * vz * dt)

        Parameters
        ----------
        f : (Nx, Ny, Nz, Nvx, Nvy, Nvz)

        Returns
        -------
        f_new : (Nx, Ny, Nz, Nvx, Nvy, Nvz)
        """
        # X-sweep: FFT along axis 0, multiply by phase_x[kx, vx]
        # phase_x shape: (Nx, Nvx) -> (Nx, 1, 1, Nvx, 1, 1)
        f_hat = jnp.fft.fft(f, axis=0)
        phase = self.phase_x[:, None, None, :, None, None]
        f = jnp.real(jnp.fft.ifft(f_hat * phase, axis=0))

        # Y-sweep: FFT along axis 1, multiply by phase_y[ky, vy]
        # phase_y shape: (Ny, Nvy) -> (1, Ny, 1, 1, Nvy, 1)
        f_hat = jnp.fft.fft(f, axis=1)
        phase = self.phase_y[None, :, None, None, :, None]
        f = jnp.real(jnp.fft.ifft(f_hat * phase, axis=1))

        # Z-sweep: FFT along axis 2, multiply by phase_z[kz, vz]
        # phase_z shape: (Nz, Nvz) -> (1, 1, Nz, 1, 1, Nvz)
        f_hat = jnp.fft.fft(f, axis=2)
        phase = self.phase_z[None, None, :, None, None, :]
        f = jnp.real(jnp.fft.ifft(f_hat * phase, axis=2))

        return f

    def _strang_splitting_step(self, f):
        """
        One Strang splitting step:
            1. Half collision
            2. Full transport
            3. Half collision

        Parameters
        ----------
        f : (Nx, Ny, Nz, Nvx, Nvy, Nvz)

        Returns
        -------
        f_new : (Nx, Ny, Nz, Nvx, Nvy, Nvz)
        """
        f = self._collision_half_step(f)
        f = self._transport_step(f)
        f = self._collision_half_step(f)
        return f

    def _compute_fneq_rms(self, f):
        """
        Compute RMS of non-equilibrium part: sqrt(mean((f - M[f])^2)).

        Parameters
        ----------
        f : (Nx, Ny, Nz, Nvx, Nvy, Nvz)

        Returns
        -------
        fneq_rms : scalar
        """
        rho, ux, uy, uz, T = self._compute_moments(f)
        M = self._maxwellian_from_moments(rho, ux, uy, uz, T)
        fneq = f - M
        return jnp.sqrt(jnp.mean(fneq**2))

    def _compute_conservation(self, f):
        """
        Compute conserved quantities integrated over all of phase space.

        Returns
        -------
        mass : scalar (integral of f over x and v)
        momentum : (3,) (integral of v*f)
        energy : scalar (integral of 0.5*|v|^2*f)
        """
        w = self.w3d[None, None, None, :, :, :]

        vx_e = self.vx[None, None, None, :, None, None]
        vy_e = self.vy[None, None, None, None, :, None]
        vz_e = self.vz[None, None, None, None, None, :]

        # Integrate over velocity
        rho = jnp.sum(f * w, axis=(3, 4, 5))           # (Nx, Ny, Nz)
        mom_x = jnp.sum(f * vx_e * w, axis=(3, 4, 5))
        mom_y = jnp.sum(f * vy_e * w, axis=(3, 4, 5))
        mom_z = jnp.sum(f * vz_e * w, axis=(3, 4, 5))
        energy_density = jnp.sum(
            f * 0.5 * (vx_e**2 + vy_e**2 + vz_e**2) * w, axis=(3, 4, 5)
        )

        # Integrate over space (periodic grid, uniform spacing => sum * dx^3)
        dx3 = self.dx**3
        mass = jnp.sum(rho) * dx3
        momentum_x = jnp.sum(mom_x) * dx3
        momentum_y = jnp.sum(mom_y) * dx3
        momentum_z = jnp.sum(mom_z) * dx3
        energy = jnp.sum(energy_density) * dx3

        return mass, jnp.array([momentum_x, momentum_y, momentum_z]), energy

    def initial_condition(self):
        """
        Compute initial distribution function.

        f_0 = Maxwellian(rho_0, 0, 1)
        rho_0 = 1 + 0.5 * sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z)

        Returns
        -------
        f0 : (Nx, Ny, Nz, Nvx, Nvy, Nvz)
        """
        # 3D density field
        xx = self.x[:, None, None]
        yy = self.y[None, :, None]
        zz = self.z[None, None, :]

        rho0 = 1.0 + 0.5 * jnp.sin(2 * jnp.pi * xx) * \
                           jnp.sin(2 * jnp.pi * yy) * \
                           jnp.sin(2 * jnp.pi * zz)

        ux0 = jnp.zeros_like(rho0)
        uy0 = jnp.zeros_like(rho0)
        uz0 = jnp.zeros_like(rho0)
        T0 = jnp.ones_like(rho0)

        return self._maxwellian_from_moments(rho0, ux0, uy0, uz0, T0)

    def solve(self, save_every=100, verbose=True, save_f=False):
        """
        Solve the Boltzmann-BGK equation.

        Parameters
        ----------
        save_every : int
            Save moments every N time steps.
        verbose : bool
            Print progress information.
        save_f : bool
            Whether to save full distribution snapshots.

        Returns
        -------
        results : dict
            Solution data and diagnostics.
        """
        if verbose:
            print(f"\n{'='*70}")
            print("3D+3D Boltzmann-BGK Equation Solver (JAX/GPU)")
            print('='*70)
            print(f"Grid: Nx={self.Nx}^3, Nv={self.Nv}^3, Nt={self.Nt}")
            print(f"Spatial domain: [-{self.X}, {self.X}]^3, dx={self.dx:.6e}")
            print(f"Velocity domain: [-{self.V}, {self.V}]^3, dv={self.dv:.6e}")
            print(f"Time: t in [0, {self.T_final}], dt={self.dt:.6e}")
            print(f"Kn={self.Kn}, nu={self.nu}")
            print(f"Method: Strang splitting (half-collision + transport + half-collision)")
            print('='*70)

        # Initialize
        f = self.initial_condition()

        # Warm-up JIT
        if verbose:
            print("JIT compiling... ", end="", flush=True)
        _ = self._strang_step_jit(f)
        _ = self._compute_moments_jit(f)
        _ = self._compute_conservation_jit(f)
        _ = self._compute_fneq_rms_jit(f)
        jax.block_until_ready(_)
        if verbose:
            print("done")

        # Initial diagnostics
        mass0, mom0, energy0 = self._compute_conservation_jit(f)
        jax.block_until_ready(mass0)

        # Storage
        times = [0.0]
        rho_history = []
        ux_history = []
        uy_history = []
        uz_history = []
        T_history = []
        fneq_rms_history = []
        conservation_history = []

        # Record initial state
        rho, ux, uy, uz, T = self._compute_moments_jit(f)
        fneq_rms = self._compute_fneq_rms_jit(f)
        jax.block_until_ready(fneq_rms)

        rho_history.append(np.array(rho))
        ux_history.append(np.array(ux))
        uy_history.append(np.array(uy))
        uz_history.append(np.array(uz))
        T_history.append(np.array(T))
        fneq_rms_history.append(float(fneq_rms))
        conservation_history.append(
            (float(mass0), float(mom0[0]), float(mom0[1]), float(mom0[2]), float(energy0))
        )

        if save_f:
            f_snapshots = [np.array(f)]
            f_snapshot_indices = [0]

        if verbose:
            print(f"Initial: rho=[{float(rho.min()):.6f}, {float(rho.max()):.6f}], "
                  f"T=[{float(T.min()):.6f}, {float(T.max()):.6f}], "
                  f"fneq_rms={float(fneq_rms):.6e}")

        # Time stepping
        start_time = time.time()

        if verbose and HAS_TQDM:
            iterator = tqdm(range(self.Nt), desc="Time stepping", unit="step")
        elif verbose:
            print("Time stepping...", flush=True)
            iterator = range(self.Nt)
        else:
            iterator = range(self.Nt)

        for n in iterator:
            f = self._strang_step_jit(f)

            if (n + 1) % save_every == 0:
                jax.block_until_ready(f)
                t_now = (n + 1) * self.dt
                times.append(t_now)

                rho, ux, uy, uz, T = self._compute_moments_jit(f)
                fneq_rms = self._compute_fneq_rms_jit(f)
                mass, mom, energy = self._compute_conservation_jit(f)
                jax.block_until_ready(energy)

                rho_history.append(np.array(rho))
                ux_history.append(np.array(ux))
                uy_history.append(np.array(uy))
                uz_history.append(np.array(uz))
                T_history.append(np.array(T))
                fneq_rms_history.append(float(fneq_rms))
                conservation_history.append(
                    (float(mass), float(mom[0]), float(mom[1]), float(mom[2]), float(energy))
                )

                if save_f:
                    f_snapshots.append(np.array(f))
                    f_snapshot_indices.append(n + 1)

                if verbose and not HAS_TQDM:
                    mass_err = abs(float(mass) - float(mass0)) / abs(float(mass0))
                    print(f"  Step {n+1}/{self.Nt}, t={t_now:.4f}, "
                          f"fneq_rms={float(fneq_rms):.6e}, mass_err={mass_err:.2e}")

        jax.block_until_ready(f)
        elapsed_time = time.time() - start_time

        # Final diagnostics
        rho_f, ux_f, uy_f, uz_f, T_f = self._compute_moments_jit(f)
        mass_f, mom_f, energy_f = self._compute_conservation_jit(f)
        fneq_rms_f = self._compute_fneq_rms_jit(f)
        jax.block_until_ready(fneq_rms_f)

        if verbose:
            print(f"\nCompleted in {elapsed_time:.2f} seconds")
            print(f"Throughput: {self.Nt / elapsed_time:.1f} steps/sec")
            print(f"\nFinal moments:")
            print(f"  rho: [{float(rho_f.min()):.6f}, {float(rho_f.max()):.6f}], "
                  f"mean={float(rho_f.mean()):.6f}")
            print(f"  ux:  [{float(ux_f.min()):.6e}, {float(ux_f.max()):.6e}]")
            print(f"  uy:  [{float(uy_f.min()):.6e}, {float(uy_f.max()):.6e}]")
            print(f"  uz:  [{float(uz_f.min()):.6e}, {float(uz_f.max()):.6e}]")
            print(f"  T:   [{float(T_f.min()):.6f}, {float(T_f.max()):.6f}], "
                  f"mean={float(T_f.mean()):.6f}")
            print(f"  fneq_rms: {float(fneq_rms_f):.6e}")
            print(f"\nConservation errors (relative):")
            mass0_f = float(mass0)
            energy0_f = float(energy0)
            print(f"  Mass:     {abs(float(mass_f) - mass0_f) / abs(mass0_f):.2e}")
            mom0_np = np.array(mom0)
            mom_f_np = np.array(mom_f)
            for i, label in enumerate(['x', 'y', 'z']):
                if abs(mom0_np[i]) > 1e-10:
                    print(f"  Mom_{label}:   "
                          f"{abs(mom_f_np[i] - mom0_np[i]) / abs(mom0_np[i]):.2e} (relative)")
                else:
                    print(f"  Mom_{label}:   {abs(mom_f_np[i] - mom0_np[i]):.2e} (absolute)")
            print(f"  Energy:   {abs(float(energy_f) - energy0_f) / abs(energy0_f):.2e}")

        results = {
            'times': np.array(times),
            'rho_history': np.stack(rho_history, axis=0),
            'ux_history': np.stack(ux_history, axis=0),
            'uy_history': np.stack(uy_history, axis=0),
            'uz_history': np.stack(uz_history, axis=0),
            'T_history': np.stack(T_history, axis=0),
            'fneq_rms_history': np.array(fneq_rms_history),
            'conservation_history': np.array(conservation_history),
            'x': np.array(self.x),
            'y': np.array(self.y),
            'z': np.array(self.z),
            'vx': np.array(self.vx),
            'vy': np.array(self.vy),
            'vz': np.array(self.vz),
            'elapsed_time': elapsed_time,
            'params': {
                'Nx': self.Nx,
                'Nv': self.Nv,
                'Nt': self.Nt,
                'X': self.X,
                'V': self.V,
                'T_final': self.T_final,
                'Kn': self.Kn,
                'nu': self.nu,
                'dt': float(self.dt),
                'dx': float(self.dx),
                'dv': float(self.dv),
            }
        }

        if save_f:
            results['f_snapshots'] = f_snapshots
            results['f_snapshot_indices'] = f_snapshot_indices

        return results


def main():
    parser = argparse.ArgumentParser(
        description='3D+3D Boltzmann-BGK numerical solver (JAX/GPU)'
    )
    parser.add_argument('--nx', type=int, default=16,
                        help='Spatial grid points per direction (default: 16)')
    parser.add_argument('--nv', type=int, default=24,
                        help='Velocity grid points per direction (default: 24)')
    parser.add_argument('--nt', type=int, default=5000,
                        help='Number of time steps (default: 5000)')
    parser.add_argument('--T', type=float, default=5.0,
                        help='Final time (default: 5.0)')
    parser.add_argument('--Kn', type=float, default=0.01,
                        help='Knudsen number (default: 0.01)')
    parser.add_argument('--save_every', type=int, default=100,
                        help='Save moments every N steps (default: 100)')
    parser.add_argument('--save_f', action='store_true',
                        help='Save full distribution snapshots')
    parser.add_argument('--X', type=float, default=0.5,
                        help='Spatial domain half-width (default: 0.5)')
    parser.add_argument('--V', type=float, default=6.0,
                        help='Velocity domain half-width (default: 6.0)')
    args = parser.parse_args()

    start_datetime = datetime.now()
    timestamp = start_datetime.strftime("%Y%m%d_%H%M%S")

    # Create solver
    solver = BGKSolver3D(
        Nx=args.nx,
        Nv=args.nv,
        Nt=args.nt,
        X=args.X,
        V=args.V,
        T_final=args.T,
        Kn=args.Kn,
    )

    # Solve
    results = solver.solve(
        save_every=args.save_every,
        verbose=True,
        save_f=args.save_f,
    )

    # Create output directory
    out_dir = "data/bgk_3d"
    os.makedirs(out_dir, exist_ok=True)

    base_name = (f"bgk3d_Nx{args.nx}_Nv{args.nv}_Nt{args.nt}"
                 f"_Kn{args.Kn}_T{args.T}_{timestamp}")

    # Save moments
    moments_file = os.path.join(out_dir, f"{base_name}_moments.npz")
    np.savez(
        moments_file,
        times=results['times'],
        rho_history=results['rho_history'],
        ux_history=results['ux_history'],
        uy_history=results['uy_history'],
        uz_history=results['uz_history'],
        T_history=results['T_history'],
        fneq_rms_history=results['fneq_rms_history'],
        conservation_history=results['conservation_history'],
        x=results['x'],
        y=results['y'],
        z=results['z'],
        vx=results['vx'],
        vy=results['vy'],
        vz=results['vz'],
    )
    print(f"\nMoments saved to {moments_file}")

    # Save config
    gpu_memory_peak = get_gpu_memory_gib(device_idx=0, peak=True)

    device = jax.devices()[0]
    device_str = str(device).lower()
    device_type = "GPU" if ('cuda' in device_str or 'gpu' in device_str) else "CPU"

    end_datetime = datetime.now()

    config = {
        "run_date": start_datetime.strftime("%Y-%m-%d"),
        "run_start_time": start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "run_end_time": end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_time_sec": results['elapsed_time'],

        "equation": "3D Boltzmann-BGK",
        "equation_form": "df/dt + v.grad_x f = nu*(M[f] - f)",
        "collision_model": "BGK (exact exponential relaxation)",

        "device_type": device_type,
        "gpu_memory_peak_gib": gpu_memory_peak,

        "Nx": args.nx,
        "Nv": args.nv,
        "Nt": args.nt,
        "X": args.X,
        "V": args.V,
        "T_final": args.T,
        "Kn": args.Kn,
        "nu": 1.0 / args.Kn,
        "dt": float(solver.dt),
        "dx": float(solver.dx),
        "dv": float(solver.dv),
        "total_points": args.nx**3 * args.nv**3,

        "method": "Strang splitting",
        "transport_solver": "FFT spectral (3 sequential 1D sweeps)",
        "collision_solver": "Exact exponential BGK",

        "initial_condition": {
            "rho": "1 + 0.5*sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)",
            "u": "0",
            "T": "1"
        },

        "final_rho_range": [
            float(results['rho_history'][-1].min()),
            float(results['rho_history'][-1].max()),
        ],
        "final_T_range": [
            float(results['T_history'][-1].min()),
            float(results['T_history'][-1].max()),
        ],
        "final_fneq_rms": float(results['fneq_rms_history'][-1]),

        "mass_error_relative": float(
            abs(results['conservation_history'][-1, 0] -
                results['conservation_history'][0, 0]) /
            abs(results['conservation_history'][0, 0])
        ),
        "energy_error_relative": float(
            abs(results['conservation_history'][-1, 4] -
                results['conservation_history'][0, 4]) /
            abs(results['conservation_history'][0, 4])
        ),

        "moments_file": f"{base_name}_moments.npz",
    }

    config_file = os.path.join(out_dir, f"{base_name}_config.json")
    with open(config_file, 'w') as fp:
        json.dump(config, fp, indent=2)
    print(f"Config saved to {config_file}")

    # Save full f snapshots if requested
    if args.save_f and 'f_snapshots' in results:
        for i, (f_snap, idx) in enumerate(
            zip(results['f_snapshots'], results['f_snapshot_indices'])
        ):
            f_file = os.path.join(out_dir, f"{base_name}_f_{idx}.npy")
            np.save(f_file, f_snap)
        print(f"Saved {len(results['f_snapshots'])} distribution snapshots")

    print(f"\nAll outputs in {out_dir}/")
    return results


if __name__ == "__main__":
    main()
