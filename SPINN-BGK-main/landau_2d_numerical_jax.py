# 2026-02-09: Added runtime estimation (warm steps + extrapolation); Added chunked collision (lax.scan); Initial creation — 2D Boltzmann-Landau solver
"""
GPU-accelerated numerical solver for 2D Boltzmann-Landau equation using JAX.

Equation:
    ∂f/∂t + vx ∂f/∂x + vy ∂f/∂y = Q_L(f,f)

where Q_L is the 2D Landau collision operator (tensor formulation with spectral
derivatives) from landau_collision_operator.py.

Numerical method:
    - Strang splitting: advection (dt/2) -> collision (dt) -> advection (dt/2)
    - Advection: spectral method (exact in Fourier space, 2D spatial FFT)
    - Collision: LandauOperator2D_Spectral + RK4 time integration
    - All operations vectorized for GPU acceleration

Initial condition:
    ρ(0,x,y) = 1 + 0.5 sin(2πx) sin(2πy)
    u(0,x,y) = (0, 0)
    T(0,x,y) = 1
    f(0,x,y,vx,vy) = Maxwellian(ρ, u, T)

Domain:
    x ∈ [-X, X], y ∈ [-X, X] with periodic BC
    vx ∈ [-V, V], vy ∈ [-V, V] (truncated)
    t ∈ [0, T_final]

Data shape:
    f has shape (N_x, N_x, N_v, N_v) — (x, y, vx, vy)
"""

import sys
import os

# Add parent directory so we can import landau_collision_operator
sys.path.insert(0, os.path.expanduser("~"))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

from landau_collision_operator import LandauOperator2D_Spectral

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
            if peak:
                return stats.get('peak_bytes_in_use', 0) / (1024 ** 3)
            else:
                return stats.get('bytes_in_use', 0) / (1024 ** 3)
    except:
        pass
    return 0.0


class LandauSolver2D_JAX:
    """
    GPU-accelerated solver for 2D Boltzmann-Landau equation using JAX.

    f has shape (N_x, N_x, N_v, N_v) — (x, y, vx, vy).
    """

    def __init__(self, N_x, N_v, N_t, X=0.5, V=6.0, T_final=0.1, lambda_D=10.0,
                 collision_batch_size=4096):
        """
        Initialize the solver.

        Parameters
        ----------
        N_x : int - Number of spatial grid points per dimension (x and y)
        N_v : int - Number of velocity grid points per dimension (vx and vy)
        N_t : int - Number of time steps
        X : float - Spatial domain is [-X, X] in each direction
        V : float - Velocity domain is [-V, V] in each direction
        T_final : float - Final time
        lambda_D : float - Debye length for Coulomb cutoff
        collision_batch_size : int - spatial points per chunk in collision step
            (controls memory vs parallelism tradeoff; must divide N_x*N_x)
        """
        self.N_x = N_x
        self.N_v = N_v
        self.N_t = N_t
        self.X = X
        self.V = V
        self.T_final = T_final
        self.lambda_D = lambda_D
        self.cutoff = 1.0 / lambda_D
        self.collision_batch_size = collision_batch_size

        # Spatial grid (periodic, exclude right endpoint)
        self.dx = 2 * X / N_x
        self.x = jnp.linspace(-X, X - self.dx, N_x)
        # y grid is the same as x grid
        self.y = self.x

        # Velocity grid
        self.dv = 2 * V / (N_v - 1)
        self.v = jnp.linspace(-V, V, N_v)

        # Time step
        self.dt = T_final / N_t

        # Wavenumbers for spatial derivatives (spectral method)
        self.kx = 2 * jnp.pi * jnp.fft.fftfreq(N_x, self.dx)

        # 2D Landau collision operator
        self.collision_op = LandauOperator2D_Spectral(Nv=N_v, V=V, lambda_D=lambda_D)

        # Precompute phase shifts for advection
        self._precompute_advection_phase()

        # JIT compile functions
        self._compile_functions()

        # Validate collision batch size
        n_spatial = N_x * N_x
        assert n_spatial % collision_batch_size == 0, \
            f"N_x*N_x ({n_spatial}) must be divisible by collision_batch_size ({collision_batch_size})"
        self.n_collision_chunks = n_spatial // collision_batch_size

        # Print info
        v_max = float(jnp.max(jnp.abs(self.v)))
        cfl_advection = v_max * self.dt / self.dx
        mem_f_gib = N_x * N_x * N_v * N_v * 8 / (1024**3)
        print(f"CFL number (advection): {cfl_advection:.4f}")
        print(f"f array size: {N_x}x{N_x}x{N_v}x{N_v} = {N_x*N_x*N_v*N_v:.2e} elements ({mem_f_gib:.2f} GiB)")
        print(f"Collision: {self.n_collision_chunks} chunks of {collision_batch_size} spatial points")
        print(f"JAX devices: {jax.devices()}")

    def _precompute_advection_phase(self):
        """
        Precompute separable phase shift arrays for 2D spectral advection.

        Phase = exp(-i*(kx*vx + ky*vy)*dt) = exp(-i*kx*vx*dt) * exp(-i*ky*vy*dt)

        phase_x has shape (N_x, 1, N_v, 1) — broadcasts over y, vy
        phase_y has shape (1, N_x, 1, N_v) — broadcasts over x, vx
        """
        kx = self.kx  # (N_x,)
        v = self.v    # (N_v,)

        # Half-step phases (for Strang splitting)
        phase_x_half = jnp.exp(-1j * jnp.outer(kx, v) * self.dt / 2)  # (N_x, N_v)
        phase_y_half = jnp.exp(-1j * jnp.outer(kx, v) * self.dt / 2)  # (N_x, N_v)

        # Reshape for broadcasting: (N_x, 1, N_v, 1) and (1, N_x, 1, N_v)
        self.phase_x_half = phase_x_half[:, jnp.newaxis, :, jnp.newaxis]
        self.phase_y_half = phase_y_half[jnp.newaxis, :, jnp.newaxis, :]

    def _compile_functions(self):
        """JIT compile the main computational functions."""
        self._strang_step_jit = jit(self._strang_splitting_step)
        self._compute_moments_jit = jit(self._compute_moments)
        self._compute_conservation_jit = jit(self._compute_conservation)
        self._compute_optimal_tau_jit = jit(self._compute_optimal_tau)

    def _advection_step(self, f, phase_x, phase_y):
        """
        Solve advection equation ∂f/∂t + vx ∂f/∂x + vy ∂f/∂y = 0 using spectral method.

        Parameters
        ----------
        f : ndarray of shape (N_x, N_x, N_v, N_v)
        phase_x : precomputed phase shift, shape (N_x, 1, N_v, 1)
        phase_y : precomputed phase shift, shape (1, N_x, 1, N_v)

        Returns
        -------
        f_new : ndarray of shape (N_x, N_x, N_v, N_v)
        """
        # 2D FFT in spatial dimensions (x, y)
        f_hat = jnp.fft.fft2(f, axes=(0, 1))

        # Apply separable phase shift
        f_hat = f_hat * phase_x * phase_y

        # Inverse FFT
        f_new = jnp.real(jnp.fft.ifft2(f_hat, axes=(0, 1)))

        return f_new

    def _collision_step_rk4(self, f):
        """
        Solve collision equation ∂f/∂t = Q_L(f,f) using RK4.

        Uses chunked processing via lax.scan to limit GPU memory:
        spatial points are split into chunks, each chunk is independently
        advanced through a full RK4 step using vmap.

        Parameters
        ----------
        f : ndarray of shape (N_x, N_x, N_v, N_v)

        Returns
        -------
        f_new : ndarray of shape (N_x, N_x, N_v, N_v)
        """
        N_x = self.N_x
        N_v = self.N_v
        dt = self.dt
        bs = self.collision_batch_size

        # Reshape to (n_chunks, batch_size, N_v, N_v)
        f_chunked = f.reshape(self.n_collision_chunks, bs, N_v, N_v)

        # vmap collision operator over batch dimension within a chunk
        Q_vmap = vmap(self.collision_op.collision_operator)

        def rk4_one_chunk(carry, f_chunk):
            """RK4 for a single spatial chunk. f_chunk: (bs, N_v, N_v)."""
            k1 = Q_vmap(f_chunk)
            k2 = Q_vmap(f_chunk + 0.5 * dt * k1)
            k3 = Q_vmap(f_chunk + 0.5 * dt * k2)
            k4 = Q_vmap(f_chunk + dt * k3)
            f_new = f_chunk + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            f_new = jnp.maximum(f_new, 0.0)
            return carry, f_new

        _, f_new_chunked = lax.scan(rk4_one_chunk, None, f_chunked)

        return f_new_chunked.reshape(N_x, N_x, N_v, N_v)

    def _strang_splitting_step(self, f):
        """
        Perform one Strang splitting step:
            1. Advection for dt/2
            2. Collision for dt
            3. Advection for dt/2
        """
        f = self._advection_step(f, self.phase_x_half, self.phase_y_half)
        f = self._collision_step_rk4(f)
        f = self._advection_step(f, self.phase_x_half, self.phase_y_half)
        return f

    def _maxwellian(self, rho, ux, uy, T):
        """
        Compute 2D Maxwellian distribution.

        M(x,y,vx,vy) = ρ(x,y) / (2πT(x,y)) * exp(-((vx-ux)² + (vy-uy)²) / (2T(x,y)))

        Parameters
        ----------
        rho : ndarray of shape (N_x, N_x)
        ux : ndarray of shape (N_x, N_x)
        uy : ndarray of shape (N_x, N_x)
        T : ndarray of shape (N_x, N_x)

        Returns
        -------
        f : ndarray of shape (N_x, N_x, N_v, N_v)
        """
        # Expand dims for broadcasting: spatial (N_x, N_x, 1, 1), velocity (1, 1, N_v) or (1, 1, 1, N_v)
        rho_4d = rho[:, :, jnp.newaxis, jnp.newaxis]
        ux_4d = ux[:, :, jnp.newaxis, jnp.newaxis]
        uy_4d = uy[:, :, jnp.newaxis, jnp.newaxis]
        T_4d = T[:, :, jnp.newaxis, jnp.newaxis]

        vx = self.v[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]  # (1, 1, N_v, 1)
        vy = self.v[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]  # (1, 1, 1, N_v)

        return rho_4d / (2 * jnp.pi * T_4d) * jnp.exp(
            -((vx - ux_4d)**2 + (vy - uy_4d)**2) / (2 * T_4d)
        )

    def initial_condition(self):
        """
        Set initial condition:
            ρ(0,x,y) = 1 + 0.5 sin(2πx) sin(2πy)
            u(0,x,y) = (0, 0)
            T(0,x,y) = 1
        """
        # 2D meshgrid for spatial
        xx, yy = jnp.meshgrid(self.x, self.x, indexing='ij')  # (N_x, N_x)

        rho0 = 1 + 0.5 * jnp.sin(2 * jnp.pi * xx) * jnp.sin(2 * jnp.pi * yy)
        ux0 = jnp.zeros((self.N_x, self.N_x))
        uy0 = jnp.zeros((self.N_x, self.N_x))
        T0 = jnp.ones((self.N_x, self.N_x))

        return self._maxwellian(rho0, ux0, uy0, T0)

    def _compute_moments(self, f):
        """
        Compute macroscopic moments from distribution function.

        Parameters
        ----------
        f : ndarray of shape (N_x, N_x, N_v, N_v)

        Returns
        -------
        rho : ndarray of shape (N_x, N_x) — density
        ux : ndarray of shape (N_x, N_x) — bulk velocity x
        uy : ndarray of shape (N_x, N_x) — bulk velocity y
        T : ndarray of shape (N_x, N_x) — temperature
        """
        dv = self.dv

        # Trapezoidal weights for velocity integration
        wv = jnp.ones(self.N_v) * dv
        wv = wv.at[0].set(dv / 2)
        wv = wv.at[-1].set(dv / 2)

        # Outer product for 2D velocity weight: (N_v, N_v)
        wv2d = wv[:, jnp.newaxis] * wv[jnp.newaxis, :]

        # Velocity grids for integration
        vx = self.v[:, jnp.newaxis]   # (N_v, 1)
        vy = self.v[jnp.newaxis, :]   # (1, N_v)

        # ρ = ∫∫ f dvx dvy
        rho = jnp.sum(f * wv2d[jnp.newaxis, jnp.newaxis, :, :], axis=(2, 3))

        # momentum_x = ∫∫ vx * f dvx dvy
        momentum_x = jnp.sum(f * (vx * wv2d)[jnp.newaxis, jnp.newaxis, :, :], axis=(2, 3))

        # momentum_y = ∫∫ vy * f dvx dvy
        momentum_y = jnp.sum(f * (vy * wv2d)[jnp.newaxis, jnp.newaxis, :, :], axis=(2, 3))

        # energy = ∫∫ (vx² + vy²) * f dvx dvy
        v_sq = vx**2 + vy**2  # (N_v, N_v)
        energy = jnp.sum(f * (v_sq * wv2d)[jnp.newaxis, jnp.newaxis, :, :], axis=(2, 3))

        rho_safe = rho + 1e-16
        ux = momentum_x / rho_safe
        uy = momentum_y / rho_safe

        # T = (E_kin / ρ - |u|²) / d, d=2
        T = (energy / rho_safe - ux**2 - uy**2) / 2.0

        return rho, ux, uy, T

    def _compute_conservation(self, f):
        """
        Compute conserved quantities integrated over (x, y, vx, vy).

        Returns
        -------
        total_mass, total_momentum_x, total_momentum_y, total_energy : floats
        """
        rho, ux, uy, T = self._compute_moments(f)

        # Trapezoidal weights for x integration (periodic grid — uniform weights)
        dx = self.dx
        wx = jnp.ones(self.N_x) * dx
        # 2D spatial weight
        wx2d = wx[:, jnp.newaxis] * wx[jnp.newaxis, :]

        total_mass = jnp.sum(rho * wx2d)
        total_momentum_x = jnp.sum(rho * ux * wx2d)
        total_momentum_y = jnp.sum(rho * uy * wx2d)
        total_energy = jnp.sum(rho * (T + (ux**2 + uy**2) / 2) * wx2d)

        return total_mass, total_momentum_x, total_momentum_y, total_energy

    def _compute_optimal_tau(self, f):
        """
        Compute the analytically optimal BGK relaxation time τ.

        τ_opt = <f_neq, f_neq> / <Q_L, f_neq>

        where f_neq = f_eq - f, integrated over (x, y, vx, vy).

        Parameters
        ----------
        f : ndarray of shape (N_x, N_x, N_v, N_v)

        Returns
        -------
        tau : float
        """
        # Moments and equilibrium
        rho, ux, uy, T = self._compute_moments(f)
        f_eq = self._maxwellian(rho, ux, uy, T)
        f_neq = f_eq - f

        # Collision operator (chunked vmap over spatial batch)
        N_x = self.N_x
        N_v = self.N_v
        bs = self.collision_batch_size

        f_chunked = f.reshape(self.n_collision_chunks, bs, N_v, N_v)
        Q_vmap = vmap(self.collision_op.collision_operator)

        def eval_chunk(carry, f_chunk):
            return carry, Q_vmap(f_chunk)

        _, Q_chunked = lax.scan(eval_chunk, None, f_chunked)
        Q_L = Q_chunked.reshape(N_x, N_x, N_v, N_v)

        # Trapezoidal weights for velocity
        dv = self.dv
        wv = jnp.ones(N_v) * dv
        wv = wv.at[0].set(dv / 2)
        wv = wv.at[-1].set(dv / 2)
        wv2d = wv[:, jnp.newaxis] * wv[jnp.newaxis, :]

        # Trapezoidal weights for space (periodic — uniform)
        dx = self.dx
        wx = jnp.ones(N_x) * dx
        wx2d = wx[:, jnp.newaxis] * wx[jnp.newaxis, :]

        # <f_neq, f_neq> = ∫∫∫∫ f_neq² dvx dvy dx dy
        vel_integral_num = jnp.sum(f_neq**2 * wv2d[jnp.newaxis, jnp.newaxis, :, :], axis=(2, 3))
        numerator = jnp.sum(vel_integral_num * wx2d)

        # <Q_L, f_neq> = ∫∫∫∫ Q_L · f_neq dvx dvy dx dy
        vel_integral_den = jnp.sum(Q_L * f_neq * wv2d[jnp.newaxis, jnp.newaxis, :, :], axis=(2, 3))
        denominator = jnp.sum(vel_integral_den * wx2d)

        tau = jnp.where(jnp.abs(denominator) > 1e-30,
                        numerator / denominator,
                        jnp.inf)

        return tau

    def solve(self, save_every=None, verbose=True, x_stride=1, v_stride=1,
              save_tau=True, N_tau_measure=None):
        """
        Solve the 2D Boltzmann-Landau equation.

        Parameters
        ----------
        save_every : int or None - save solution every N steps
        verbose : bool - print progress
        x_stride : int - stride for subsampling f in x/y when saving
        v_stride : int - stride for subsampling f in vx/vy when saving
        save_tau : bool - compute and save optimal tau history
        N_tau_measure : int or None - compute tau every N steps (default: save_every)

        Returns
        -------
        results : dict containing solution and diagnostics
        """
        if N_tau_measure is None:
            N_tau_measure = save_every
        if verbose:
            print(f"\n{'='*60}")
            print("2D Boltzmann-Landau Equation Solver (JAX/GPU)")
            print('='*60)
            print(f"Grid: N_x={self.N_x}, N_v={self.N_v}, N_t={self.N_t}")
            print(f"Domain: x,y ∈ [-{self.X}, {self.X}], vx,vy ∈ [-{self.V}, {self.V}]")
            print(f"Time: t ∈ [0, {self.T_final}], dt = {self.dt:.6e}")
            print(f"Debye length: λ_D = {self.lambda_D}, cutoff = {self.cutoff:.4f}")
            print('='*60)

        # Initialize
        f = self.initial_condition()

        # Warm-up JIT compilation
        if verbose:
            print("JIT compiling... ", end="", flush=True)
        t_jit0 = time.time()
        _ = self._strang_step_jit(f)
        _ = self._compute_moments_jit(f)
        _ = self._compute_conservation_jit(f)
        if save_tau:
            _ = self._compute_optimal_tau_jit(f)
        jax.block_until_ready(_)
        t_jit = time.time() - t_jit0
        if verbose:
            print(f"done ({t_jit:.1f}s)")

        # Time estimation: run 3 warm steps and extrapolate
        if verbose:
            print("Estimating runtime... ", end="", flush=True)
        n_warmup = min(3, self.N_t)
        f_tmp = f
        t_est0 = time.time()
        for _ in range(n_warmup):
            f_tmp = self._strang_step_jit(f_tmp)
        jax.block_until_ready(f_tmp)
        t_per_step = (time.time() - t_est0) / n_warmup
        est_total = t_per_step * self.N_t
        if verbose:
            print(f"{t_per_step:.4f}s/step")
            if est_total < 60:
                print(f"Estimated total time: {est_total:.1f}s")
            elif est_total < 3600:
                print(f"Estimated total time: {est_total/60:.1f} min")
            else:
                print(f"Estimated total time: {est_total/3600:.2f} hr")
        del f_tmp

        # Initial conservation quantities
        mass0, momx0, momy0, energy0 = self._compute_conservation_jit(f)

        # Storage
        times = [0.0]
        rho_history = []
        ux_history = []
        uy_history = []
        T_history = []
        conservation_history = [(float(mass0), float(momx0), float(momy0), float(energy0))]

        rho, ux, uy, T = self._compute_moments_jit(f)
        rho_history.append(np.array(rho))
        ux_history.append(np.array(ux))
        uy_history.append(np.array(uy))
        T_history.append(np.array(T))

        if save_every is not None:
            f_history = [np.array(f[::x_stride, ::x_stride, ::v_stride, ::v_stride])]

        # Tau tracking
        if save_tau and N_tau_measure is not None:
            tau0 = self._compute_optimal_tau_jit(f)
            tau_history = [float(tau0)]
            tau_times = [0.0]
        else:
            tau_history = []
            tau_times = []

        # Time stepping
        start_time = time.time()

        if verbose and HAS_TQDM:
            iterator = tqdm(range(self.N_t), desc="Time stepping", unit="step")
        elif verbose:
            print("Time stepping...", flush=True)
            iterator = range(self.N_t)
        else:
            iterator = range(self.N_t)

        for n in iterator:
            f = self._strang_step_jit(f)

            if save_every is not None and (n + 1) % save_every == 0:
                jax.block_until_ready(f)
                times.append((n + 1) * self.dt)
                rho, ux, uy, T = self._compute_moments_jit(f)
                rho_history.append(np.array(rho))
                ux_history.append(np.array(ux))
                uy_history.append(np.array(uy))
                T_history.append(np.array(T))
                cons = self._compute_conservation_jit(f)
                conservation_history.append((float(cons[0]), float(cons[1]),
                                             float(cons[2]), float(cons[3])))
                f_history.append(np.array(f[::x_stride, ::x_stride, ::v_stride, ::v_stride]))

            if save_tau and N_tau_measure is not None and (n + 1) % N_tau_measure == 0:
                jax.block_until_ready(f)
                tau_val = self._compute_optimal_tau_jit(f)
                tau_history.append(float(tau_val))
                tau_times.append((n + 1) * self.dt)

        jax.block_until_ready(f)
        elapsed_time = time.time() - start_time

        # Final state
        rho_final, ux_final, uy_final, T_final = self._compute_moments_jit(f)
        mass_final, momx_final, momy_final, energy_final = self._compute_conservation_jit(f)

        rho_final = np.array(rho_final)
        ux_final = np.array(ux_final)
        uy_final = np.array(uy_final)
        T_final_arr = np.array(T_final)
        f_final = np.array(f)

        if verbose:
            print(f"\nCompleted in {elapsed_time:.2f} seconds")
            print(f"Throughput: {self.N_t / elapsed_time:.1f} steps/sec")
            print(f"\nFinal moments:")
            print(f"  ρ: min={rho_final.min():.6f}, max={rho_final.max():.6f}, mean={rho_final.mean():.6f}")
            print(f"  ux: min={ux_final.min():.6f}, max={ux_final.max():.6f}, mean={ux_final.mean():.6f}")
            print(f"  uy: min={uy_final.min():.6f}, max={uy_final.max():.6f}, mean={uy_final.mean():.6f}")
            print(f"  T: min={T_final_arr.min():.6f}, max={T_final_arr.max():.6f}, mean={T_final_arr.mean():.6f}")

            mass0_f = float(mass0)
            mass_f = float(mass_final)
            energy0_f = float(energy0)
            energy_f = float(energy_final)
            momx0_f = float(momx0)
            momx_f = float(momx_final)
            momy0_f = float(momy0)
            momy_f = float(momy_final)

            print(f"\nConservation errors:")
            print(f"  Mass:       {abs(mass_f - mass0_f) / abs(mass0_f):.2e} (relative)")
            if abs(momx0_f) > 1e-10:
                print(f"  Momentum_x: {abs(momx_f - momx0_f) / abs(momx0_f):.2e} (relative)")
            else:
                print(f"  Momentum_x: {abs(momx_f - momx0_f):.2e} (absolute, initial ≈ 0)")
            if abs(momy0_f) > 1e-10:
                print(f"  Momentum_y: {abs(momy_f - momy0_f) / abs(momy0_f):.2e} (relative)")
            else:
                print(f"  Momentum_y: {abs(momy_f - momy0_f):.2e} (absolute, initial ≈ 0)")
            print(f"  Energy:     {abs(energy_f - energy0_f) / abs(energy0_f):.2e} (relative)")

            if save_tau and len(tau_history) > 0:
                tau_arr = np.array(tau_history)
                print(f"\nOptimal τ (BGK relaxation time):")
                print(f"  Range: [{tau_arr.min():.6e}, {tau_arr.max():.6e}]")
                print(f"  Final: {tau_arr[-1]:.6e}")

        results = {
            'f': f_final,
            'x': np.array(self.x),
            'y': np.array(self.y),
            'v': np.array(self.v),
            'rho': rho_final,
            'ux': ux_final,
            'uy': uy_final,
            'T': T_final_arr,
            'times': np.array(times),
            'rho_history': np.array(rho_history),
            'ux_history': np.array(ux_history),
            'uy_history': np.array(uy_history),
            'T_history': np.array(T_history),
            'conservation_history': np.array(conservation_history),
            'elapsed_time': elapsed_time,
            'params': {
                'N_x': self.N_x,
                'N_v': self.N_v,
                'N_t': self.N_t,
                'X': self.X,
                'V': self.V,
                'T_final': self.T_final,
                'lambda_D': self.lambda_D,
            }
        }

        if save_every is not None:
            results['f_history'] = np.array(f_history)

        if save_tau and len(tau_history) > 0:
            results['tau_history'] = np.array(tau_history)
            results['tau_times'] = np.array(tau_times)

        return results


def main(N_x: int = 128, N_v: int = 64, N_t: int = 1024,
         N_x_save: int = 32, N_v_save: int = 32, N_t_save: int = 64,
         X: float = 0.5, V: float = 6.0, T_final: float = 0.1,
         lambda_D: float = 10.0, plot: bool = True,
         save_tau: bool = True, N_tau_measure: int = None,
         collision_batch_size: int = 4096):
    """
    Run the 2D Boltzmann-Landau solver (JAX/GPU version).

    Parameters
    ----------
    N_x : int - Number of spatial grid points per dimension
    N_v : int - Number of velocity grid points per dimension
    N_t : int - Number of time steps
    N_x_save : int - Number of spatial points to save per dimension
    N_v_save : int - Number of velocity points to save per dimension
    N_t_save : int - Number of time snapshots to save
    X : float - Spatial domain half-width [-X, X]
    V : float - Velocity domain half-width [-V, V]
    T_final : float - Final simulation time
    lambda_D : float - Debye length for Coulomb cutoff
    plot : bool - Generate plots
    save_tau : bool - Compute and save optimal BGK tau history
    N_tau_measure : int - Compute tau every N steps (default: save_every)
    collision_batch_size : int - spatial points per chunk in collision step
    """
    # Validate save grid divides compute grid
    assert N_x % N_x_save == 0, f"N_x ({N_x}) must be divisible by N_x_save ({N_x_save})"
    assert N_v % N_v_save == 0, f"N_v ({N_v}) must be divisible by N_v_save ({N_v_save})"
    assert N_t % N_t_save == 0, f"N_t ({N_t}) must be divisible by N_t_save ({N_t_save})"

    x_stride = N_x // N_x_save
    v_stride = N_v // N_v_save
    save_every = N_t // N_t_save

    # Record start time
    start_datetime = datetime.now()
    start_time_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # Get device info
    device = jax.devices()[0]
    device_str = str(device).lower()
    device_type = "GPU" if ('cuda' in device_str or 'gpu' in device_str) else "CPU"

    print(f"Starting JAX/GPU 2D simulation with N_x={N_x}, N_v={N_v}, N_t={N_t}")
    print(f"Save grid: N_x_save={N_x_save}, N_v_save={N_v_save}, N_t_save={N_t_save} "
          f"(strides: x={x_stride}, v={v_stride}, save_every={save_every})")
    print(f"Start time: {start_time_str}")
    print(f"Device: {device_type}")

    # Create solver
    solver = LandauSolver2D_JAX(
        N_x=N_x, N_v=N_v, N_t=N_t,
        X=X, V=V, T_final=T_final, lambda_D=lambda_D,
        collision_batch_size=collision_batch_size
    )

    # Solve
    results = solver.solve(save_every=save_every, x_stride=x_stride, v_stride=v_stride,
                           save_tau=save_tau, N_tau_measure=N_tau_measure)

    # Record end time
    end_datetime = datetime.now()
    end_time_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # GPU memory
    gpu_memory_peak_gib = get_gpu_memory_gib(device_idx=0, peak=True)

    # Create output directories
    os.makedirs("data/landau_2d", exist_ok=True)
    os.makedirs("figures/landau_2d", exist_ok=True)

    # Generate filename with timestamp
    timestamp = start_datetime.strftime("%Y%m%d_%H%M%S")
    base_name = f"landau2d_Nx{N_x}_Nv{N_v}_Nt{N_t}_{timestamp}"

    # ========== Save metadata to JSON ==========
    config = {
        "created_by": "landau_2d_numerical_jax.py",

        # Run info
        "run_date": start_datetime.strftime("%Y-%m-%d"),
        "run_start_time": start_time_str,
        "run_end_time": end_time_str,
        "elapsed_time_sec": results['elapsed_time'],

        # Equation info
        "equation": "2D Boltzmann-Landau",
        "equation_form": "∂f/∂t + vx ∂f/∂x + vy ∂f/∂y = Q_L(f,f)",
        "collision_operator": "LandauOperator2D_Spectral (tensor kernel with spectral derivatives)",
        "collision_kernel": "Coulomb with Debye cutoff",
        "kernel_formula": "Φ(|z|) = 1 / max(|z|, 1/λ_D)",

        # Hardware info
        "device_type": device_type,
        "gpu_memory_peak_gib": gpu_memory_peak_gib,

        # Simulation parameters (compute grid)
        "N_x": N_x,
        "N_v": N_v,
        "N_t": N_t,
        "lambda_D": lambda_D,
        "cutoff": 1.0 / lambda_D,

        # Save grid (subsampled)
        "N_x_save": N_x_save,
        "N_v_save": N_v_save,
        "N_t_save": N_t_save,
        "x_stride": x_stride,
        "v_stride": v_stride,
        "save_every": save_every,

        # Domain
        "X": X,
        "V": V,
        "T_final": T_final,
        "dx": float(solver.dx),
        "dv": float(solver.dv),
        "dt": float(solver.dt),

        # Initial condition
        "initial_condition": {
            "rho": "1 + 0.5 * sin(2πx) * sin(2πy)",
            "ux": "0",
            "uy": "0",
            "T": "1"
        },

        # Numerical method
        "method": "Strang operator splitting",
        "advection_solver": "Spectral (2D FFT in x,y)",
        "collision_solver": "RK4 with LandauOperator2D_Spectral (chunked vmap via lax.scan)",
        "collision_batch_size": collision_batch_size,
        "n_collision_chunks": solver.n_collision_chunks,

        # Final state summary
        "final_rho_min": float(results['rho'].min()),
        "final_rho_max": float(results['rho'].max()),
        "final_rho_mean": float(results['rho'].mean()),
        "final_T_min": float(results['T'].min()),
        "final_T_max": float(results['T'].max()),
        "final_T_mean": float(results['T'].mean()),

        # Conservation
        "mass_error_relative": float(abs(results['conservation_history'][-1, 0] - results['conservation_history'][0, 0])
                                     / abs(results['conservation_history'][0, 0])),
        "energy_error_relative": float(abs(results['conservation_history'][-1, 3] - results['conservation_history'][0, 3])
                                       / abs(results['conservation_history'][0, 3])),

        # Optimal tau
        "save_tau": save_tau,
        "N_tau_measure": N_tau_measure if N_tau_measure is not None else save_every,

        # File references
        "data_file": f"{base_name}_f.npy",
        "grid_file": f"{base_name}_grid.npz",
    }

    if save_tau and 'tau_history' in results:
        tau_arr = results['tau_history']
        config["tau_initial"] = float(tau_arr[0])
        config["tau_final"] = float(tau_arr[-1])
        config["tau_min"] = float(tau_arr.min())
        config["tau_max"] = float(tau_arr.max())
        config["tau_num_samples"] = len(tau_arr)

    config_file = f"data/landau_2d/{base_name}_config.json"
    with open(config_file, 'w') as fp:
        json.dump(config, fp, indent=2)
    print(f"\nConfig saved to {config_file}")

    # ========== Save distribution function f(x,y,vx,vy,t) to NPY ==========
    if 'f_history' in results:
        # f_history shape: (N_t_save+1, N_x_save, N_x_save, N_v_save, N_v_save)
        f_data = results['f_history']
    else:
        f_data = results['f'][::x_stride, ::x_stride, ::v_stride, ::v_stride][np.newaxis, :, :, :, :]

    f_file = f"data/landau_2d/{base_name}_f.npy"
    np.save(f_file, f_data)
    print(f"Distribution f(x,y,vx,vy,t) saved to {f_file}")
    print(f"  Shape: {f_data.shape}")
    print(f"  Size: {f_data.nbytes / (1024**3):.2f} GiB")

    # ========== Save grid info to NPZ ==========
    x_save = results['x'][::x_stride]
    y_save = results['y'][::x_stride]
    v_save = results['v'][::v_stride]
    grid_file = f"data/landau_2d/{base_name}_grid.npz"
    grid_data = dict(
        x=x_save,
        y=y_save,
        v=v_save,
        times=results['times'],
        rho_history=np.array([r[::x_stride, ::x_stride] for r in results['rho_history']]),
        ux_history=np.array([u[::x_stride, ::x_stride] for u in results['ux_history']]),
        uy_history=np.array([u[::x_stride, ::x_stride] for u in results['uy_history']]),
        T_history=np.array([t[::x_stride, ::x_stride] for t in results['T_history']]),
        conservation_history=results['conservation_history'],
    )
    if 'tau_history' in results:
        grid_data['tau_history'] = results['tau_history']
        grid_data['tau_times'] = results['tau_times']
    np.savez(grid_file, **grid_data)
    print(f"Grid and moments saved to {grid_file}")

    # ========== Plots ==========
    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        x_np = np.array(solver.x)
        y_np = np.array(solver.y)

        # (0,0) Initial density heatmap
        ax = axes[0, 0]
        rho_init = results['rho_history'][0]
        im = ax.pcolormesh(x_np, y_np, rho_init.T, shading='auto', cmap='viridis')
        plt.colorbar(im, ax=ax, label='ρ')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Initial Density ρ(x,y)')
        ax.set_aspect('equal')

        # (0,1) Final density heatmap
        ax = axes[0, 1]
        im = ax.pcolormesh(x_np, y_np, results['rho'].T, shading='auto', cmap='viridis')
        plt.colorbar(im, ax=ax, label='ρ')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Final Density ρ(x,y) at t={solver.T_final}')
        ax.set_aspect('equal')

        # (0,2) Final temperature heatmap
        ax = axes[0, 2]
        im = ax.pcolormesh(x_np, y_np, results['T'].T, shading='auto', cmap='inferno')
        plt.colorbar(im, ax=ax, label='T')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Final Temperature T(x,y) at t={solver.T_final}')
        ax.set_aspect('equal')

        # (1,0) Distribution slice at x=0, y=0
        ax = axes[1, 0]
        mid = N_x // 2
        v_np = np.array(solver.v)
        f0 = np.array(solver.initial_condition())
        f0_slice = f0[mid, mid, :, :]
        f_final_slice = results['f'][mid, mid, :, :]
        # Plot 1D slice along vx at vy=0
        mid_v = N_v // 2
        ax.plot(v_np, f0_slice[:, mid_v], 'b--', label='Initial', alpha=0.7)
        ax.plot(v_np, f_final_slice[:, mid_v], 'r-', label='Final')
        ax.set_xlabel('vx')
        ax.set_ylabel('f')
        ax.set_title('Distribution at (x,y)=(0,0), vy=0')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (1,1) Final distribution f(vx,vy) heatmap at (x,y)=(0,0)
        ax = axes[1, 1]
        im = ax.pcolormesh(v_np, v_np, f_final_slice.T, shading='auto', cmap='hot')
        plt.colorbar(im, ax=ax, label='f(vx,vy)')
        ax.set_xlabel('vx')
        ax.set_ylabel('vy')
        ax.set_title('Final f(vx,vy) at (x,y)=(0,0)')
        ax.set_aspect('equal')

        # (1,2) Conservation errors
        ax = axes[1, 2]
        cons = results['conservation_history']
        mass_err = np.abs(cons[:, 0] - cons[0, 0]) / np.abs(cons[0, 0])
        energy_err = np.abs(cons[:, 3] - cons[0, 3]) / np.abs(cons[0, 3])
        ax.semilogy(results['times'], mass_err + 1e-16, 'b-', label='Mass')
        ax.semilogy(results['times'], energy_err + 1e-16, 'r-', label='Energy')
        ax.set_xlabel('t')
        ax.set_ylabel('Relative Error')
        ax.set_title('Conservation Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_file = f"figures/landau_2d/{base_name}.png"
        plt.savefig(fig_file, dpi=150)
        print(f"Figure saved to {fig_file}")

        # Save figure provenance JSON
        fig_json_file = f"figures/landau_2d/{base_name}.json"
        fig_meta = {
            "created_by": "landau_2d_numerical_jax.py",
            "data_files": [f_file, grid_file],
            "config_file": config_file,
        }
        with open(fig_json_file, 'w') as fp:
            json.dump(fig_meta, fp, indent=2)
        print(f"Figure metadata saved to {fig_json_file}")

        plt.show()

    return results


if __name__ == "__main__":
    try:
        import fire
        fire.Fire({'main': main})
    except ImportError:
        import argparse
        parser = argparse.ArgumentParser(description='2D Boltzmann-Landau solver (JAX/GPU)')
        parser.add_argument('--N_x', type=int, default=128, help='Spatial grid points per dimension')
        parser.add_argument('--N_v', type=int, default=64, help='Velocity grid points per dimension')
        parser.add_argument('--N_t', type=int, default=1024, help='Number of time steps')
        parser.add_argument('--N_x_save', type=int, default=32, help='Saved spatial points per dimension')
        parser.add_argument('--N_v_save', type=int, default=32, help='Saved velocity points per dimension')
        parser.add_argument('--N_t_save', type=int, default=64, help='Saved time snapshots')
        parser.add_argument('--X', type=float, default=0.5, help='Spatial domain half-width')
        parser.add_argument('--V', type=float, default=6.0, help='Velocity domain half-width')
        parser.add_argument('--T_final', type=float, default=0.1, help='Final time')
        parser.add_argument('--lambda_D', type=float, default=10.0, help='Debye length')
        parser.add_argument('--no_plot', action='store_true', help='Disable plotting')
        parser.add_argument('--no_save_tau', action='store_true', help='Disable optimal tau computation')
        parser.add_argument('--N_tau_measure', type=int, default=None, help='Compute tau every N steps')
        parser.add_argument('--collision_batch_size', type=int, default=4096, help='Spatial points per collision chunk')
        args = parser.parse_args()

        main(N_x=args.N_x, N_v=args.N_v, N_t=args.N_t,
             N_x_save=args.N_x_save, N_v_save=args.N_v_save, N_t_save=args.N_t_save,
             X=args.X, V=args.V, T_final=args.T_final,
             lambda_D=args.lambda_D, plot=not args.no_plot,
             save_tau=not args.no_save_tau, N_tau_measure=args.N_tau_measure,
             collision_batch_size=args.collision_batch_size)
