# 2026-02-07: Added analytical optimal tau computation (τ_opt = <f_neq,f_neq>/<Q_L,f_neq>); 2026-02-07: Added save grid downsampling (N_x_save, N_v_save, N_t_save); separate N_x/N_v params
"""
GPU-accelerated numerical solver for 1D Boltzmann-Landau equation using JAX.

Equation:
    ∂f/∂t + v ∂f/∂x = Q_L(f,f)

where Q_L is the 1D Landau collision operator:
    Q_L(f,f) = ∂/∂v [ A[f] ∂f/∂v - B[f] f ]

with:
    A[f](v) = (Φ * f)(v) = ∫ Φ(|v-v'|) f(v') dv'
    B[f](v) = (Φ * ∂_v f)(v) = ∫ Φ(|v-v'|) ∂f(v')/∂v' dv'

Coulomb kernel with Debye cutoff:
    Φ(|u|) = 1 / max(|u|, 1/λ_D)

Numerical method:
    - Strang splitting: advection (dt/2) -> collision (dt) -> advection (dt/2)
    - Advection: spectral method (exact in Fourier space)
    - Collision: FFT-based convolution + RK4 time integration
    - All operations vectorized for GPU acceleration

Initial condition:
    ρ(0,x) = 1 + 0.5 sin(2πx)
    u(0,x) = 0
    T(0,x) = 1
    f(0,x,v) = Maxwellian(ρ, u, T)

Domain:
    x ∈ [-X, X] with periodic BC
    v ∈ [-V, V] (truncated)
    t ∈ [0, T_final]
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
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
            if peak:
                return stats.get('peak_bytes_in_use', 0) / (1024 ** 3)
            else:
                return stats.get('bytes_in_use', 0) / (1024 ** 3)
    except:
        pass
    return 0.0


class LandauSolver1D_JAX:
    """
    GPU-accelerated solver for 1D Boltzmann-Landau equation using JAX.
    All operations are vectorized and JIT-compiled for maximum performance.
    """

    def __init__(self, N_x, N_v, N_t, X=0.5, V=6.0, T_final=0.1, lambda_D=10.0):
        """
        Initialize the solver.

        Parameters:
        -----------
        N_x : int - Number of spatial grid points
        N_v : int - Number of velocity grid points
        N_t : int - Number of time steps
        X : float - Spatial domain is [-X, X]
        V : float - Velocity domain is [-V, V]
        T_final : float - Final time
        lambda_D : float - Debye length for cutoff (Φ = 1/max(|u|, 1/λ_D))
        """
        self.N_x = N_x
        self.N_v = N_v
        self.N_t = N_t
        self.X = X
        self.V = V
        self.T_final = T_final
        self.lambda_D = lambda_D
        self.cutoff = 1.0 / lambda_D

        # Spatial grid (periodic, exclude right endpoint)
        self.dx = 2 * X / N_x
        self.x = jnp.linspace(-X, X - self.dx, N_x)

        # Velocity grid
        self.dv = 2 * V / (N_v - 1)
        self.v = jnp.linspace(-V, V, N_v)

        # Time step
        self.dt = T_final / N_t

        # Wavenumbers for spatial derivatives (spectral method)
        self.kx = 2 * jnp.pi * jnp.fft.fftfreq(N_x, self.dx)

        # Precompute Coulomb kernel FFT for convolutions
        self._precompute_kernel_fft()

        # Precompute phase shift for advection
        self._precompute_advection_phase()

        # JIT compile the step function
        self._compile_functions()

        # Print info
        cfl_advection = float(jnp.max(jnp.abs(self.v)) * self.dt / self.dx)
        print(f"CFL number (advection): {cfl_advection:.4f}")
        print(f"JAX devices: {jax.devices()}")

    def _precompute_kernel_fft(self):
        """
        Precompute FFT of the Coulomb kernel with cutoff for convolution.
        """
        # For linear convolution, pad to size 2*N_v - 1
        self.N_conv = 2 * self.N_v - 1

        # Velocity difference grid for kernel (centered at 0)
        u = jnp.arange(-(self.N_v - 1), self.N_v) * self.dv

        # Coulomb kernel with cutoff: Φ(|u|) = 1/max(|u|, 1/λ_D)
        self.Phi = 1.0 / jnp.maximum(jnp.abs(u), self.cutoff)

        # FFT of kernel (for use in convolution)
        self.Phi_fft = jnp.fft.fft(self.Phi)

    def _precompute_advection_phase(self):
        """
        Precompute phase shift matrices for advection step.
        """
        # Phase shift: exp(-i k v dt) for half step and full step
        self.phase_half = jnp.exp(-1j * jnp.outer(self.kx, self.v) * self.dt / 2)
        self.phase_full = jnp.exp(-1j * jnp.outer(self.kx, self.v) * self.dt)

    def _compile_functions(self):
        """
        JIT compile the main computational functions.
        """
        # Compile the Strang splitting step
        self._strang_step_jit = jit(self._strang_splitting_step)

        # Compile moment computation
        self._compute_moments_jit = jit(self._compute_moments)

        # Compile conservation computation
        self._compute_conservation_jit = jit(self._compute_conservation)

        # Compile optimal tau computation
        self._compute_optimal_tau_jit = jit(self._compute_optimal_tau)

    def _compute_dv(self, f):
        """
        Compute ∂f/∂v using central differences (vectorized).

        Parameters:
        -----------
        f : ndarray of shape (N_x, N_v)

        Returns:
        --------
        df_dv : ndarray of shape (N_x, N_v)
        """
        # Central differences for interior points
        df_dv_center = (f[:, 2:] - f[:, :-2]) / (2 * self.dv)

        # One-sided differences at boundaries
        df_dv_left = (f[:, 1:2] - f[:, 0:1]) / self.dv
        df_dv_right = (f[:, -1:] - f[:, -2:-1]) / self.dv

        # Concatenate
        df_dv = jnp.concatenate([df_dv_left, df_dv_center, df_dv_right], axis=1)

        return df_dv

    def _convolution_fft(self, f_row):
        """
        Compute convolution Φ * f for a single spatial point using FFT.

        Parameters:
        -----------
        f_row : ndarray of shape (N_v,)

        Returns:
        --------
        conv : ndarray of shape (N_v,)
        """
        # Zero-pad for linear convolution
        f_padded = jnp.zeros(self.N_conv)
        f_padded = f_padded.at[:self.N_v].set(f_row)

        # Convolution via FFT
        conv_full = jnp.real(jnp.fft.ifft(self.Phi_fft * jnp.fft.fft(f_padded)))

        # Extract valid part
        start = self.N_v - 1
        conv = conv_full[start:start + self.N_v] * self.dv

        return conv

    def _compute_collision_coefficients(self, f):
        """
        Compute A[f] and B[f] using FFT-based convolution (vectorized over x).

        A[f](v) = ∫ Φ(|v-v'|) f(v') dv'
        B[f](v) = ∫ Φ(|v-v'|) ∂f(v')/∂v' dv'

        Parameters:
        -----------
        f : ndarray of shape (N_x, N_v)

        Returns:
        --------
        A, B : ndarrays of shape (N_x, N_v)
        """
        # Compute ∂f/∂v
        df_dv = self._compute_dv(f)

        # Vectorized convolution over all spatial points using vmap
        A = vmap(self._convolution_fft)(f)
        B = vmap(self._convolution_fft)(df_dv)

        return A, B

    def _collision_operator(self, f):
        """
        Compute the Landau collision operator Q_L(f,f).

        Q_L = ∂/∂v [ A[f] ∂f/∂v - B[f] f ]

        Parameters:
        -----------
        f : ndarray of shape (N_x, N_v)

        Returns:
        --------
        Q : ndarray of shape (N_x, N_v)
        """
        A, B = self._compute_collision_coefficients(f)

        # Compute ∂f/∂v
        df_dv = self._compute_dv(f)

        # Flux: J = A ∂f/∂v - B f
        J = A * df_dv - B * f

        # Q = ∂J/∂v
        Q = self._compute_dv(J)

        # 2026-02-06: Sign correction - negate Q to satisfy H-theorem (entropy dissipation)
        # The original code had wrong sign, causing entropy to increase instead of decrease
        return -Q

    def _advection_step(self, f, phase):
        """
        Solve advection equation ∂f/∂t + v ∂f/∂x = 0 using spectral method.

        Parameters:
        -----------
        f : ndarray of shape (N_x, N_v)
        phase : precomputed phase shift matrix

        Returns:
        --------
        f_new : ndarray of shape (N_x, N_v)
        """
        # FFT in x direction
        f_hat = jnp.fft.fft(f, axis=0)

        # Apply phase shift and inverse FFT
        f_new = jnp.real(jnp.fft.ifft(f_hat * phase, axis=0))

        return f_new

    def _collision_step_rk4(self, f):
        """
        Solve collision equation ∂f/∂t = Q_L(f,f) using RK4.

        Parameters:
        -----------
        f : ndarray of shape (N_x, N_v)

        Returns:
        --------
        f_new : ndarray of shape (N_x, N_v)
        """
        dt = self.dt

        k1 = self._collision_operator(f)
        k2 = self._collision_operator(f + 0.5 * dt * k1)
        k3 = self._collision_operator(f + 0.5 * dt * k2)
        k4 = self._collision_operator(f + dt * k3)

        f_new = f + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Ensure non-negativity
        f_new = jnp.maximum(f_new, 0.0)

        return f_new

    def _strang_splitting_step(self, f):
        """
        Perform one Strang splitting step:
            1. Advection for dt/2
            2. Collision for dt
            3. Advection for dt/2
        """
        f = self._advection_step(f, self.phase_half)
        f = self._collision_step_rk4(f)
        f = self._advection_step(f, self.phase_half)
        return f

    def _maxwellian(self, rho, u, T):
        """
        Compute Maxwellian distribution.

        M(x,v) = ρ(x) / √(2πT(x)) · exp(-(v - u(x))² / (2T(x)))
        """
        rho = rho[:, jnp.newaxis]
        u = u[:, jnp.newaxis]
        T = T[:, jnp.newaxis]
        v = self.v[jnp.newaxis, :]

        return rho / jnp.sqrt(2 * jnp.pi * T) * jnp.exp(-(v - u)**2 / (2 * T))

    def _compute_moments(self, f):
        """
        Compute macroscopic moments from distribution function.

        Returns: rho, u, T
        """
        # Trapezoidal rule weights
        w = jnp.ones(self.N_v) * self.dv
        w = w.at[0].set(self.dv / 2)
        w = w.at[-1].set(self.dv / 2)

        rho = jnp.sum(f * w, axis=1)
        momentum = jnp.sum(f * self.v * w, axis=1)
        energy = jnp.sum(f * self.v**2 * w, axis=1)

        u = momentum / (rho + 1e-16)
        T = (energy / (rho + 1e-16)) - u**2

        return rho, u, T

    def _compute_conservation(self, f):
        """
        Compute conserved quantities (integrated over x and v).
        """
        rho, u, T = self._compute_moments(f)

        # Trapezoidal rule for x integration
        wx = jnp.ones(self.N_x) * self.dx
        wx = wx.at[0].set(self.dx / 2)
        wx = wx.at[-1].set(self.dx / 2)

        total_mass = jnp.sum(rho * wx)
        total_momentum = jnp.sum(rho * u * wx)
        total_energy = jnp.sum(rho * (T + u**2) * wx) / 2

        return total_mass, total_momentum, total_energy

    def _compute_optimal_tau(self, f):
        """
        Compute the analytically optimal BGK relaxation time τ.

        Minimizes ||Q_L - (1/τ)(f_eq - f)||² in L2(x,v), yielding:
            τ_opt = <f_neq, f_neq> / <Q_L, f_neq>
        where f_neq = f_eq - f.

        Parameters:
        -----------
        f : ndarray of shape (N_x, N_v)

        Returns:
        --------
        tau : float - optimal relaxation time
        """
        # Compute local moments and equilibrium
        rho, u, T = self._compute_moments(f)
        f_eq = self._maxwellian(rho, u, T)

        # Non-equilibrium part
        f_neq = f_eq - f

        # Landau collision operator
        Q_L = self._collision_operator(f)

        # Trapezoidal weights for v integration
        wv = jnp.ones(self.N_v) * self.dv
        wv = wv.at[0].set(self.dv / 2)
        wv = wv.at[-1].set(self.dv / 2)

        # Trapezoidal weights for x integration
        wx = jnp.ones(self.N_x) * self.dx
        wx = wx.at[0].set(self.dx / 2)
        wx = wx.at[-1].set(self.dx / 2)

        # <f_neq, f_neq> = ∫∫ f_neq² dv dx
        integrand_num = jnp.sum(f_neq**2 * wv[jnp.newaxis, :], axis=1)
        numerator = jnp.sum(integrand_num * wx)

        # <Q_L, f_neq> = ∫∫ Q_L · f_neq dv dx
        integrand_den = jnp.sum(Q_L * f_neq * wv[jnp.newaxis, :], axis=1)
        denominator = jnp.sum(integrand_den * wx)

        # tau = numerator / denominator, with edge case protection
        tau = jnp.where(jnp.abs(denominator) > 1e-30,
                        numerator / denominator,
                        jnp.inf)

        return tau

    def initial_condition(self):
        """
        Set initial condition:
            ρ(0,x) = 1 + 0.5 sin(2πx)
            u(0,x) = 0
            T(0,x) = 1
        """
        rho0 = 1 + 0.5 * jnp.sin(2 * jnp.pi * self.x)
        u0 = jnp.zeros(self.N_x)
        T0 = jnp.ones(self.N_x)

        return self._maxwellian(rho0, u0, T0)

    def solve(self, save_every=None, verbose=True, x_stride=1, v_stride=1,
              save_tau=True, N_tau_measure=None):
        """
        Solve the Boltzmann-Landau equation.

        Parameters:
        -----------
        save_every : int or None - save solution every N steps
        verbose : bool - print progress
        x_stride : int - stride for subsampling f in x when saving
        v_stride : int - stride for subsampling f in v when saving
        save_tau : bool - compute and save optimal tau history
        N_tau_measure : int or None - compute tau every N steps (default: save_every)

        Returns:
        --------
        results : dict containing solution and diagnostics
        """
        if N_tau_measure is None:
            N_tau_measure = save_every
        if verbose:
            print(f"\n{'='*60}")
            print("1D Boltzmann-Landau Equation Solver (JAX/GPU)")
            print('='*60)
            print(f"Grid: N_x={self.N_x}, N_v={self.N_v}, N_t={self.N_t}")
            print(f"Domain: x ∈ [-{self.X}, {self.X}], v ∈ [-{self.V}, {self.V}]")
            print(f"Time: t ∈ [0, {self.T_final}], dt = {self.dt:.6e}")
            print(f"Debye length: λ_D = {self.lambda_D}, cutoff = {self.cutoff:.4f}")
            print('='*60)

        # Initialize
        f = self.initial_condition()

        # Warm-up JIT compilation
        if verbose:
            print("JIT compiling... ", end="", flush=True)
        _ = self._strang_step_jit(f)
        _ = self._compute_moments_jit(f)
        _ = self._compute_conservation_jit(f)
        if save_tau:
            _ = self._compute_optimal_tau_jit(f)
        # Block until compilation is done
        jax.block_until_ready(_)
        if verbose:
            print("done")

        # Initial conservation quantities
        mass0, mom0, energy0 = self._compute_conservation_jit(f)

        # Storage (on CPU for efficiency)
        times = [0.0]
        rho_history = []
        u_history = []
        T_history = []
        conservation_history = [(float(mass0), float(mom0), float(energy0))]

        rho, u, T = self._compute_moments_jit(f)
        rho_history.append(np.array(rho))
        u_history.append(np.array(u))
        T_history.append(np.array(T))

        if save_every is not None:
            f_history = [np.array(f[::x_stride, ::v_stride])]

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

        # Create progress bar iterator
        if verbose and HAS_TQDM:
            iterator = tqdm(range(self.N_t), desc="Time stepping", unit="step")
        elif verbose:
            print("Time stepping...", flush=True)
            iterator = range(self.N_t)
        else:
            iterator = range(self.N_t)

        # Run time stepping with progress bar
        for n in iterator:
            f = self._strang_step_jit(f)

            if save_every is not None and (n + 1) % save_every == 0:
                # Block and save
                jax.block_until_ready(f)
                times.append((n + 1) * self.dt)
                rho, u, T = self._compute_moments_jit(f)
                rho_history.append(np.array(rho))
                u_history.append(np.array(u))
                T_history.append(np.array(T))
                cons = self._compute_conservation_jit(f)
                conservation_history.append((float(cons[0]), float(cons[1]), float(cons[2])))
                f_history.append(np.array(f[::x_stride, ::v_stride]))

            if save_tau and N_tau_measure is not None and (n + 1) % N_tau_measure == 0:
                jax.block_until_ready(f)
                tau_val = self._compute_optimal_tau_jit(f)
                tau_history.append(float(tau_val))
                tau_times.append((n + 1) * self.dt)

        # Ensure final result is ready
        jax.block_until_ready(f)

        elapsed_time = time.time() - start_time

        # Final state
        rho_final, u_final, T_final = self._compute_moments_jit(f)
        mass_final, mom_final, energy_final = self._compute_conservation_jit(f)

        # Convert to numpy for output
        rho_final = np.array(rho_final)
        u_final = np.array(u_final)
        T_final = np.array(T_final)
        f_final = np.array(f)

        if verbose:
            print(f"\nCompleted in {elapsed_time:.2f} seconds")
            print(f"Throughput: {self.N_t / elapsed_time:.1f} steps/sec")
            print(f"\nFinal moments:")
            print(f"  ρ: min={rho_final.min():.6f}, max={rho_final.max():.6f}, mean={rho_final.mean():.6f}")
            print(f"  u: min={u_final.min():.6f}, max={u_final.max():.6f}, mean={u_final.mean():.6f}")
            print(f"  T: min={T_final.min():.6f}, max={T_final.max():.6f}, mean={T_final.mean():.6f}")
            mass0_f, mom0_f, energy0_f = float(mass0), float(mom0), float(energy0)
            mass_f, mom_f, energy_f = float(mass_final), float(mom_final), float(energy_final)
            print(f"\nConservation errors:")
            print(f"  Mass:     {abs(mass_f - mass0_f) / abs(mass0_f):.2e} (relative)")
            if abs(mom0_f) > 1e-10:
                print(f"  Momentum: {abs(mom_f - mom0_f) / abs(mom0_f):.2e} (relative)")
            else:
                print(f"  Momentum: {abs(mom_f - mom0_f):.2e} (absolute, initial ≈ 0)")
            print(f"  Energy:   {abs(energy_f - energy0_f) / abs(energy0_f):.2e} (relative)")

            if save_tau and len(tau_history) > 0:
                tau_arr = np.array(tau_history)
                print(f"\nOptimal τ (BGK relaxation time):")
                print(f"  Range: [{tau_arr.min():.6e}, {tau_arr.max():.6e}]")
                print(f"  Final: {tau_arr[-1]:.6e}")

        results = {
            'f': f_final,
            'x': np.array(self.x),
            'v': np.array(self.v),
            'rho': rho_final,
            'u': u_final,
            'T': T_final,
            'times': np.array(times),
            'rho_history': np.array(rho_history),
            'u_history': np.array(u_history),
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

    def solve_parallel(self, save_every=None, verbose=True, num_devices=4, x_stride=1, v_stride=1,
                        save_tau=True, N_tau_measure=None):
        """
        Solve the Boltzmann-Landau equation using multiple GPUs.

        Uses domain decomposition: spatial domain is split across GPUs.
        - Collision step: computed locally on each GPU (independent per x)
        - Advection step: uses all_gather for global FFT, then scatter back

        Parameters:
        -----------
        save_every : int or None - save solution every N steps
        verbose : bool - print progress
        num_devices : int - number of GPUs to use
        save_tau : bool - compute and save optimal tau history
        N_tau_measure : int or None - compute tau every N steps (default: save_every)

        Returns:
        --------
        results : dict containing solution and diagnostics
        """
        if N_tau_measure is None:
            N_tau_measure = save_every
        from jax import pmap
        from jax.lax import all_gather, pmean

        # Ensure N_x is divisible by num_devices
        assert self.N_x % num_devices == 0, f"N_x ({self.N_x}) must be divisible by num_devices ({num_devices})"
        chunk_size = self.N_x // num_devices

        if verbose:
            print(f"\n{'='*60}")
            print(f"1D Boltzmann-Landau Equation Solver (JAX/Multi-GPU)")
            print('='*60)
            print(f"Grid: N_x={self.N_x}, N_v={self.N_v}, N_t={self.N_t}")
            print(f"Domain: x ∈ [-{self.X}, {self.X}], v ∈ [-{self.V}, {self.V}]")
            print(f"Time: t ∈ [0, {self.T_final}], dt = {self.dt:.6e}")
            print(f"Debye length: λ_D = {self.lambda_D}, cutoff = {self.cutoff:.4f}")
            print(f"Devices: {num_devices} GPUs, {chunk_size} x-points per GPU")
            print('='*60)

        # Initialize full domain
        f_full = self.initial_condition()

        # Define convolution for parallel execution (inline to avoid closure issues)
        N_v = self.N_v
        N_conv = self.N_conv
        dv = self.dv
        Phi_fft = self.Phi_fft
        dt = self.dt

        def convolution_fft_inline(f_row):
            """Compute convolution Φ * f for a single spatial point."""
            f_padded = jnp.zeros(N_conv)
            f_padded = f_padded.at[:N_v].set(f_row)
            conv_full = jnp.real(jnp.fft.ifft(Phi_fft * jnp.fft.fft(f_padded)))
            start = N_v - 1
            return conv_full[start:start + N_v] * dv

        def collision_step_chunk(f_chunk):
            """Collision step for a spatial chunk. f_chunk: (chunk_size, N_v)"""
            def collision_one_step(f):
                # Compute velocity derivative
                df_dv = jnp.zeros_like(f)
                df_dv = df_dv.at[:, 1:-1].set((f[:, 2:] - f[:, :-2]) / (2 * dv))
                df_dv = df_dv.at[:, 0].set((f[:, 1] - f[:, 0]) / dv)
                df_dv = df_dv.at[:, -1].set((f[:, -1] - f[:, -2]) / dv)

                # Compute collision coefficients via vmap
                A = vmap(convolution_fft_inline)(f)
                B = vmap(convolution_fft_inline)(df_dv)

                # Flux and divergence
                J = A * df_dv - B * f
                Q = jnp.zeros_like(f)
                Q = Q.at[:, 1:-1].set((J[:, 2:] - J[:, :-2]) / (2 * dv))
                Q = Q.at[:, 0].set((J[:, 1] - J[:, 0]) / dv)
                Q = Q.at[:, -1].set((J[:, -1] - J[:, -2]) / dv)
                return Q

            # RK4 time integration
            k1 = collision_one_step(f_chunk)
            k2 = collision_one_step(f_chunk + 0.5 * dt * k1)
            k3 = collision_one_step(f_chunk + 0.5 * dt * k2)
            k4 = collision_one_step(f_chunk + dt * k3)
            f_new = f_chunk + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            return jnp.maximum(f_new, 0.0)

        # Precompute phase for advection
        phase_half = self.phase_half
        N_x = self.N_x

        def advection_step_full(f_full):
            """Advection step on full domain."""
            f_hat = jnp.fft.fft(f_full, axis=0)
            f_new = jnp.real(jnp.fft.ifft(f_hat * phase_half, axis=0))
            return f_new

        # Define one Strang splitting step with multi-GPU
        def strang_step_parallel(f_chunk):
            """
            One Strang splitting step distributed across GPUs.
            Inside pmap, f_chunk has shape (chunk_size, N_v) per device.
            """
            # Gather from all devices: result is (num_devices, chunk_size, N_v)
            f_gathered = jax.lax.all_gather(f_chunk, axis_name='devices')
            f_full = f_gathered.reshape(N_x, N_v)

            # Advection half step (on full domain)
            f_full = advection_step_full(f_full)

            # Split back into chunks and get this device's chunk
            f_split = f_full.reshape(num_devices, chunk_size, N_v)
            device_idx = jax.lax.axis_index('devices')
            f_chunk = f_split[device_idx]

            # Collision full step (local on this device)
            f_chunk = collision_step_chunk(f_chunk)

            # Gather for second advection
            f_gathered = jax.lax.all_gather(f_chunk, axis_name='devices')
            f_full = f_gathered.reshape(N_x, N_v)

            # Advection half step
            f_full = advection_step_full(f_full)

            # Split and return this device's chunk
            f_split = f_full.reshape(num_devices, chunk_size, N_v)
            f_chunk = f_split[device_idx]
            return f_chunk

        # pmap the step function
        strang_step_pmap = pmap(strang_step_parallel, axis_name='devices')

        # JIT compile
        if verbose:
            print("JIT compiling parallel code... ", end="", flush=True)

        # Reshape initial condition into chunks
        f_chunks = f_full.reshape(num_devices, chunk_size, self.N_v)

        # Warm-up compilation
        _ = strang_step_pmap(f_chunks)
        if save_tau:
            _ = self._compute_optimal_tau_jit(f_full)
        jax.block_until_ready(_)
        if verbose:
            print("done")

        # Initial conservation
        mass0 = float(jnp.trapezoid(jnp.trapezoid(f_full, self.v, axis=1), self.x))

        # Storage
        times = [0.0]
        rho_history = []
        rho, _, _ = self._compute_moments(f_full)
        rho_history.append(np.array(rho))

        if save_every is not None:
            f_history = [np.array(f_full[::x_stride, ::v_stride])]

        # Tau tracking
        if save_tau and N_tau_measure is not None:
            tau0 = self._compute_optimal_tau_jit(f_full)
            tau_history = [float(tau0)]
            tau_times = [0.0]
        else:
            tau_history = []
            tau_times = []

        # Time stepping
        start_time = time.time()

        if verbose and HAS_TQDM:
            iterator = tqdm(range(self.N_t), desc="Time stepping (parallel)", unit="step")
        elif verbose:
            print("Time stepping...", flush=True)
            iterator = range(self.N_t)
        else:
            iterator = range(self.N_t)

        for n in iterator:
            f_chunks = strang_step_pmap(f_chunks)

            need_save = save_every is not None and (n + 1) % save_every == 0
            need_tau = save_tau and N_tau_measure is not None and (n + 1) % N_tau_measure == 0

            if need_save or need_tau:
                jax.block_until_ready(f_chunks)
                f_full = f_chunks.reshape(self.N_x, self.N_v)

            if need_save:
                times.append((n + 1) * self.dt)
                rho, u, T = self._compute_moments(f_full)
                rho_history.append(np.array(rho))
                f_history.append(np.array(f_full[::x_stride, ::v_stride]))

            if need_tau:
                tau_val = self._compute_optimal_tau_jit(f_full)
                tau_history.append(float(tau_val))
                tau_times.append((n + 1) * self.dt)

        jax.block_until_ready(f_chunks)
        elapsed_time = time.time() - start_time

        # Final state
        f_full = f_chunks.reshape(self.N_x, self.N_v)
        rho_final, u_final, T_final = self._compute_moments(f_full)
        mass_final = float(jnp.trapezoid(jnp.trapezoid(f_full, self.v, axis=1), self.x))

        rho_final = np.array(rho_final)
        u_final = np.array(u_final)
        T_final = np.array(T_final)
        f_final = np.array(f_full)

        if verbose:
            print(f"\nCompleted in {elapsed_time:.2f} seconds")
            print(f"Throughput: {self.N_t / elapsed_time:.1f} steps/sec")
            print(f"\nFinal moments:")
            print(f"  ρ: min={rho_final.min():.6f}, max={rho_final.max():.6f}, mean={rho_final.mean():.6f}")
            print(f"  u: min={u_final.min():.6f}, max={u_final.max():.6f}, mean={u_final.mean():.6f}")
            print(f"  T: min={T_final.min():.6f}, max={T_final.max():.6f}, mean={T_final.mean():.6f}")
            print(f"\nConservation errors:")
            print(f"  Mass: {abs(mass_final - mass0) / abs(mass0):.2e} (relative)")

            if save_tau and len(tau_history) > 0:
                tau_arr = np.array(tau_history)
                print(f"\nOptimal τ (BGK relaxation time):")
                print(f"  Range: [{tau_arr.min():.6e}, {tau_arr.max():.6e}]")
                print(f"  Final: {tau_arr[-1]:.6e}")

        # Build results
        conservation_history = np.array([[mass0, 0, 0], [mass_final, 0, 0]])

        results = {
            'f': f_final,
            'x': np.array(self.x),
            'v': np.array(self.v),
            'rho': rho_final,
            'u': u_final,
            'T': T_final,
            'times': np.array(times),
            'rho_history': np.array(rho_history),
            'u_history': np.array(rho_history),  # placeholder
            'T_history': np.array(rho_history),  # placeholder
            'conservation_history': conservation_history,
            'elapsed_time': elapsed_time,
            'params': {
                'N_x': self.N_x,
                'N_v': self.N_v,
                'N_t': self.N_t,
                'X': self.X,
                'V': self.V,
                'T_final': self.T_final,
                'lambda_D': self.lambda_D,
                'num_devices': num_devices,
            }
        }

        if save_every is not None:
            results['f_history'] = np.array(f_history)

        if save_tau and len(tau_history) > 0:
            results['tau_history'] = np.array(tau_history)
            results['tau_times'] = np.array(tau_times)

        return results


def main(N_x: int = 2**16, N_v: int = 2**8, N_t: int = 2**20,
         N_x_save: int = 2**10, N_v_save: int = 2**6, N_t_save: int = 2**13,
         X: float = 0.5, V: float = 6.0, T_final: float = 5.0,
         lambda_D: float = 10.0, plot: bool = True,
         parallel: bool = False, num_gpus: int = None,
         save_tau: bool = True, N_tau_measure: int = None):
    """
    Run the 1D Boltzmann-Landau solver (JAX/GPU version).

    Parameters:
    -----------
    N_x : int - Number of spatial grid points
    N_v : int - Number of velocity grid points
    N_t : int - Number of time steps
    N_x_save : int - Number of spatial points to save (must divide N_x)
    N_v_save : int - Number of velocity points to save (must divide N_v)
    N_t_save : int - Number of time snapshots to save (must divide N_t)
    X : float - Spatial domain half-width [-X, X]
    V : float - Velocity domain half-width [-V, V]
    T_final : float - Final simulation time
    lambda_D : float - Debye length for Coulomb cutoff
    plot : bool - Generate plots
    parallel : bool - Use multi-GPU parallelism
    num_gpus : int - Number of GPUs to use (default: all available)
    save_tau : bool - Compute and save optimal BGK tau history
    N_tau_measure : int - Compute tau every N steps (default: save_every)
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
    available_devices = jax.local_device_count()
    if num_gpus is None:
        num_gpus = available_devices if parallel else 1
    num_gpus = min(num_gpus, available_devices)

    device = jax.devices()[0]
    device_str = str(device).lower()
    if 'cuda' in device_str or 'gpu' in device_str:
        device_type = "GPU"
    else:
        device_type = "CPU"

    print(f"Starting JAX/GPU simulation with N_x={N_x}, N_v={N_v}, N_t={N_t}")
    print(f"Save grid: N_x_save={N_x_save}, N_v_save={N_v_save}, N_t_save={N_t_save} (strides: x={x_stride}, v={v_stride}, save_every={save_every})")
    print(f"Start time: {start_time_str}")
    print(f"Parallel: {parallel}, Using {num_gpus} {device_type}(s)")

    # Create solver
    solver = LandauSolver1D_JAX(
        N_x=N_x,
        N_v=N_v,
        N_t=N_t,
        X=X,
        V=V,
        T_final=T_final,
        lambda_D=lambda_D
    )

    # Solve (with or without parallelism)
    if parallel and num_gpus > 1:
        results = solver.solve_parallel(save_every=save_every, num_devices=num_gpus,
                                        x_stride=x_stride, v_stride=v_stride,
                                        save_tau=save_tau, N_tau_measure=N_tau_measure)
    else:
        results = solver.solve(save_every=save_every, x_stride=x_stride, v_stride=v_stride,
                               save_tau=save_tau, N_tau_measure=N_tau_measure)

    # Record end time
    end_datetime = datetime.now()
    end_time_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # Get GPU memory usage
    gpu_memory_peak_gib = get_gpu_memory_gib(device_idx=0, peak=True)

    # Create output directories
    os.makedirs("data/landau_1d", exist_ok=True)
    os.makedirs("figures/landau_1d", exist_ok=True)

    # Generate filename with timestamp
    timestamp = start_datetime.strftime("%Y%m%d_%H%M%S")
    base_name = f"landau_Nx{N_x}_Nv{N_v}_Nt{N_t}_{timestamp}"

    # ========== Save metadata to JSON ==========
    config = {
        "created_by": "landau_1d_numerical_jax.py",

        # Run info
        "run_date": start_datetime.strftime("%Y-%m-%d"),
        "run_start_time": start_time_str,
        "run_end_time": end_time_str,
        "elapsed_time_sec": results['elapsed_time'],

        # Equation info
        "equation": "1D Boltzmann-Landau",
        "equation_form": "∂f/∂t + v ∂f/∂x = Q_L(f,f), Q_L = ∂/∂v[A ∂f/∂v - B f]",
        "collision_kernel": "Coulomb with Debye cutoff",
        "kernel_formula": "Φ(|u|) = 1 / max(|u|, 1/λ_D)",

        # Hardware info
        "num_gpus": num_gpus,
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
            "rho": "1 + 0.5 * sin(2πx)",
            "u": "0",
            "T": "1"
        },

        # Numerical method
        "method": "Strang operator splitting",
        "advection_solver": "Spectral (FFT)",
        "collision_solver": "RK4 with FFT-based convolution",

        # Final state summary
        "final_rho_min": float(results['rho'].min()),
        "final_rho_max": float(results['rho'].max()),
        "final_rho_mean": float(results['rho'].mean()),
        "final_T_min": float(results['T'].min()),
        "final_T_max": float(results['T'].max()),
        "final_T_mean": float(results['T'].mean()),

        # Conservation
        "mass_error_relative": float(abs(results['conservation_history'][-1, 0] - results['conservation_history'][0, 0]) / abs(results['conservation_history'][0, 0])),
        "energy_error_relative": float(abs(results['conservation_history'][-1, 2] - results['conservation_history'][0, 2]) / abs(results['conservation_history'][0, 2])),

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

    config_file = f"data/landau_1d/{base_name}_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_file}")

    # ========== Save distribution function f(x,v,t) to NPY ==========
    # f_history is already subsampled in x and v via strides
    if 'f_history' in results:
        # f_history shape: (N_t_save+1, N_x_save, N_v_save)
        f_data = results['f_history']
    else:
        # Just final state, subsampled, shape: (1, N_x_save, N_v_save)
        f_data = results['f'][::x_stride, ::v_stride][np.newaxis, :, :]

    f_file = f"data/landau_1d/{base_name}_f.npy"
    np.save(f_file, f_data)
    print(f"Distribution f(x,v,t) saved to {f_file}")
    print(f"  Shape: {f_data.shape} (N_t_save+1, N_x_save, N_v_save)")
    print(f"  Size: {f_data.nbytes / (1024**3):.2f} GiB")

    # ========== Save grid info to NPZ ==========
    # Save subsampled grids and moment histories
    x_save = results['x'][::x_stride]
    v_save = results['v'][::v_stride]
    grid_file = f"data/landau_1d/{base_name}_grid.npz"
    grid_data = dict(
        x=x_save,
        v=v_save,
        times=results['times'],
        rho_history=results['rho_history'][:, ::x_stride],
        u_history=results['u_history'][:, ::x_stride],
        T_history=results['T_history'][:, ::x_stride],
        conservation_history=results['conservation_history'],
    )
    if 'tau_history' in results:
        grid_data['tau_history'] = results['tau_history']
        grid_data['tau_times'] = results['tau_times']
    np.savez(grid_file, **grid_data)
    print(f"Grid and moments saved to {grid_file}")

    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))

        # (0,0) Density evolution
        ax = axes[0, 0]
        for i, t in enumerate(results['times']):
            alpha = 0.3 + 0.7 * i / max(1, len(results['times']) - 1)
            label = f't={t:.3f}' if i == 0 or i == len(results['times'])-1 else None
            ax.plot(solver.x, results['rho_history'][i], alpha=alpha, label=label)
        ax.set_xlabel('x')
        ax.set_ylabel('ρ')
        ax.set_title('Density Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (0,1) Temperature evolution
        ax = axes[0, 1]
        for i, t in enumerate(results['times']):
            alpha = 0.3 + 0.7 * i / max(1, len(results['times']) - 1)
            label = f't={t:.3f}' if i == 0 or i == len(results['times'])-1 else None
            ax.plot(solver.x, results['T_history'][i], alpha=alpha, label=label)
        ax.set_xlabel('x')
        ax.set_ylabel('T')
        ax.set_title('Temperature Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (0,2) Bulk velocity evolution
        ax = axes[0, 2]
        for i, t in enumerate(results['times']):
            alpha = 0.3 + 0.7 * i / max(1, len(results['times']) - 1)
            ax.plot(solver.x, results['u_history'][i], alpha=alpha)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title('Bulk Velocity Evolution')
        ax.grid(True, alpha=0.3)

        # (1,0) Final distribution at x=0
        ax = axes[1, 0]
        mid_x = N_x // 2
        f0 = np.array(solver.initial_condition())
        ax.plot(solver.v, f0[mid_x, :], 'b--', label='Initial', alpha=0.7)
        ax.plot(solver.v, results['f'][mid_x, :], 'r-', label='Final')
        ax.set_xlabel('v')
        ax.set_ylabel('f')
        ax.set_title(f'Distribution at x=0')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (1,1) Distribution function heatmap (final)
        ax = axes[1, 1]
        im = ax.pcolormesh(np.array(solver.x), np.array(solver.v),
                          results['f'].T, shading='auto', cmap='hot')
        plt.colorbar(im, ax=ax, label='f(x,v)')
        ax.set_xlabel('x')
        ax.set_ylabel('v')
        ax.set_title('Final Distribution f(x,v)')

        # (1,2) Conservation errors
        ax = axes[1, 2]
        cons = np.array(results['conservation_history'])
        mass_err = np.abs(cons[:, 0] - cons[0, 0]) / np.abs(cons[0, 0])
        energy_err = np.abs(cons[:, 2] - cons[0, 2]) / np.abs(cons[0, 2])
        ax.semilogy(results['times'], mass_err + 1e-16, 'b-', label='Mass')
        ax.semilogy(results['times'], energy_err + 1e-16, 'r-', label='Energy')
        ax.set_xlabel('t')
        ax.set_ylabel('Relative Error')
        ax.set_title('Conservation Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_file = f"figures/landau_1d/{base_name}.png"
        plt.savefig(fig_file, dpi=150)
        print(f"Figure saved to {fig_file}")
        plt.show()

    return results


def benchmark(N_values=[32, 64, 128, 256], T_final=0.01):
    """
    Benchmark the solver for different grid sizes.
    """
    print("\n" + "="*60)
    print("BENCHMARK: 1D Boltzmann-Landau Solver (JAX/GPU)")
    print("="*60)
    print(f"{'N_i':<8} {'N_t':<8} {'Time (s)':<12} {'Steps/sec':<12} {'Throughput':<15}")
    print("-"*60)

    results = []
    for N in N_values:
        solver = LandauSolver1D_JAX(N_x=N, N_v=N, N_t=N, T_final=T_final)
        res = solver.solve(save_every=None, verbose=False)
        throughput = N * N * N / res['elapsed_time']  # total grid points per second
        print(f"{N:<8} {N:<8} {res['elapsed_time']:<12.4f} {N/res['elapsed_time']:<12.1f} {throughput:<15.2e}")
        results.append({
            'N': N,
            'time': res['elapsed_time'],
            'steps_per_sec': N / res['elapsed_time'],
            'throughput': throughput
        })

    print("="*60)
    return results


if __name__ == "__main__":
    try:
        import fire
        fire.Fire({'main': main, 'benchmark': benchmark})
    except ImportError:
        import argparse
        parser = argparse.ArgumentParser(description='1D Boltzmann-Landau solver (JAX/GPU)')
        parser.add_argument('--N_x', type=int, default=2**16, help='Number of spatial grid points')
        parser.add_argument('--N_v', type=int, default=2**8, help='Number of velocity grid points')
        parser.add_argument('--N_t', type=int, default=2**20, help='Number of time steps')
        parser.add_argument('--N_x_save', type=int, default=2**10, help='Saved spatial grid points')
        parser.add_argument('--N_v_save', type=int, default=2**6, help='Saved velocity grid points')
        parser.add_argument('--N_t_save', type=int, default=2**13, help='Saved time snapshots')
        parser.add_argument('--X', type=float, default=0.5, help='Spatial domain half-width')
        parser.add_argument('--V', type=float, default=6.0, help='Velocity domain half-width')
        parser.add_argument('--T_final', type=float, default=5.0, help='Final time')
        parser.add_argument('--lambda_D', type=float, default=10.0, help='Debye length')
        parser.add_argument('--no_plot', action='store_true', help='Disable plotting')
        parser.add_argument('--parallel', action='store_true', help='Use multi-GPU parallelism')
        parser.add_argument('--num_gpus', type=int, default=None, help='Number of GPUs to use')
        parser.add_argument('--no_save_tau', action='store_true', help='Disable optimal tau computation')
        parser.add_argument('--N_tau_measure', type=int, default=None, help='Compute tau every N steps (default: save_every)')
        parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
        args = parser.parse_args()

        if args.benchmark:
            benchmark()
        else:
            main(N_x=args.N_x, N_v=args.N_v, N_t=args.N_t,
                 N_x_save=args.N_x_save, N_v_save=args.N_v_save, N_t_save=args.N_t_save,
                 X=args.X, V=args.V, T_final=args.T_final,
                 lambda_D=args.lambda_D, plot=not args.no_plot,
                 parallel=args.parallel, num_gpus=args.num_gpus,
                 save_tau=not args.no_save_tau, N_tau_measure=args.N_tau_measure)
