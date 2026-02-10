#!/usr/bin/env python3
"""
Compute optimal μ for Fokker-Planck operator approximation of Landau collision.

This script finds the optimal μ that minimizes ||Q(f,f) - μ·P(f)||² where:
- Q(f,f) is the Landau collision operator (computed from time derivative)
- P(f) is the Fokker-Planck operator: P(f) = ∇_v · (M[f] ∇_v(f/M[f]))

Analytical solution for L2 norm:
    μ_opt = ⟨Q·P⟩ / ⟨P²⟩ = B / A

This is analogous to the BGK analysis where τ_opt = A/B for:
    ||Q(f,f) - (M-f)/τ||²

Run on GPUs 0 and 1.
"""

import os
# MUST set before importing JAX
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
from jax.sharding import PositionalSharding
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from functools import partial
from tqdm import tqdm
import time

print(f"JAX devices: {jax.devices()}")
print(f"Number of devices: {jax.local_device_count()}")


# =============================================================================
# Core computation functions (JIT-compiled)
# =============================================================================

@jit
def compute_maxwellian(rho, u, T, v):
    """Compute Maxwellian distribution M[f](x, v)."""
    rho = rho[:, jnp.newaxis]
    u = u[:, jnp.newaxis]
    T = T[:, jnp.newaxis]
    T = jnp.maximum(T, 1e-10)
    v_grid = v[jnp.newaxis, :]
    return rho / jnp.sqrt(2 * jnp.pi * T) * jnp.exp(-(v_grid - u)**2 / (2 * T))


@jit
def compute_moments(f, v, dv):
    """Compute macroscopic moments (ρ, u, T) from distribution function."""
    N_v = len(v)
    w = jnp.ones(N_v) * dv
    w = w.at[0].set(dv / 2)
    w = w.at[-1].set(dv / 2)

    rho = jnp.sum(f * w, axis=1)
    momentum = jnp.sum(f * v * w, axis=1)
    energy = jnp.sum(f * v**2 * w, axis=1)

    rho = jnp.maximum(rho, 1e-16)
    u = momentum / rho
    T = (energy / rho) - u**2
    T = jnp.maximum(T, 1e-10)

    return rho, u, T


@jit
def compute_spatial_derivative_periodic(f, dx):
    """Compute ∂f/∂x using central differences with periodic BC."""
    df_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * dx)
    return df_dx


@jit
def compute_velocity_derivative(f, dv):
    """Compute ∂f/∂v using central differences."""
    # Interior points: central difference
    df_dv = jnp.zeros_like(f)
    df_dv = df_dv.at[:, 1:-1].set((f[:, 2:] - f[:, :-2]) / (2 * dv))
    # Boundaries: one-sided
    df_dv = df_dv.at[:, 0].set((f[:, 1] - f[:, 0]) / dv)
    df_dv = df_dv.at[:, -1].set((f[:, -1] - f[:, -2]) / dv)
    return df_dv


@jit
def compute_fokker_planck_operator(f, v, dv):
    """
    Compute the normalized Fokker-Planck operator P(f).

    P(f) = ∇_v · (M[f] ∇_v(f / M[f]))

    This can be rewritten as:
    P(f) = ∇_v · (∇_v f - f · ∇_v ln(M))
         = ∇²_v f - ∇_v · (f · ∇_v ln(M))

    For 1D:
    P(f) = ∂/∂v [M · ∂/∂v(f/M)]
    """
    # Compute moments and Maxwellian
    rho, u, T = compute_moments(f, v, dv)
    M = compute_maxwellian(rho, u, T, v)
    M = jnp.maximum(M, 1e-30)  # Avoid division by zero

    # Compute f/M
    f_over_M = f / M

    # Compute ∂/∂v (f/M)
    d_f_over_M_dv = compute_velocity_derivative(f_over_M, dv)

    # Compute M · ∂/∂v(f/M)
    flux = M * d_f_over_M_dv

    # Compute ∂/∂v [M · ∂/∂v(f/M)] = P(f)
    P = compute_velocity_derivative(flux, dv)

    return P


@partial(jit, static_argnums=())
def compute_coefficients_for_mu(f, f_prev, f_next, v, dx, dt):
    """
    Compute A, B, C coefficients for μ optimization.

    For minimizing ||Q - μP||²:
        Error(μ) = ⟨Q²⟩ - 2μ⟨QP⟩ + μ²⟨P²⟩
                 = C - 2μB + μ²A

    Optimal μ = B/A (when A > 0)

    Returns: A, B, C where
        A = ⟨P²⟩
        B = ⟨Q·P⟩
        C = ⟨Q²⟩
    """
    dv = v[1] - v[0]

    # Compute Q from time derivative: Q = ∂f/∂t + v·∂f/∂x
    df_dt = (f_next - f_prev) / (2 * dt)
    df_dx = compute_spatial_derivative_periodic(f, dx)
    advection = v[jnp.newaxis, :] * df_dx
    Q = df_dt + advection  # Collision term Q(f,f) = ∂f/∂t + v·∂f/∂x

    # Compute FP operator P(f)
    P = compute_fokker_planck_operator(f, v, dv)

    # Compute coefficients
    A = jnp.mean(P**2)  # ⟨P²⟩
    B = jnp.mean(Q * P)  # ⟨Q·P⟩
    C = jnp.mean(Q**2)   # ⟨Q²⟩

    return A, B, C


@partial(jit, static_argnums=())
def compute_errors_for_all_mu(f, f_prev, f_next, v, dx, dt, mu_values):
    """Compute L2 errors for all μ values at a single time point."""
    dv = v[1] - v[0]

    # Compute Q from time derivative
    df_dt = (f_next - f_prev) / (2 * dt)
    df_dx = compute_spatial_derivative_periodic(f, dx)
    advection = v[jnp.newaxis, :] * df_dx
    Q = df_dt + advection

    # Compute FP operator P(f)
    P = compute_fokker_planck_operator(f, v, dv)

    def error_for_mu(mu):
        fp_term = mu * P
        diff = fp_term - Q
        return jnp.mean(diff**2)

    errors = vmap(error_for_mu)(mu_values)
    return errors


# =============================================================================
# Optimization functions
# =============================================================================

def optimize_mu_all_snapshots(f_history, x, v, times, mu_values, N_size=1, N_bin=1):
    """
    Find optimal μ for all time snapshots.

    Returns both:
    1. Analytical μ_opt = B/A
    2. Numerical μ_opt from grid search
    """
    N_snapshots = f_history.shape[0]
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    dx = x[1] - x[0]
    dv = v[1] - v[0]

    print(f"\nProcessing {N_snapshots} snapshots...")
    print(f"N_size={N_size}, N_bin={N_bin}")
    print(f"Time step between snapshots: dt = {dt:.6e}")
    print(f"Spatial step: dx = {dx:.6e}")
    print(f"Velocity step: dv = {dv:.6e}")

    window_total = N_size * N_bin
    if window_total > N_snapshots - 2:
        print(f"Warning: window_total={window_total} > available snapshots-2={N_snapshots-2}")
        print(f"Adjusting to per-snapshot analysis...")
        window_total = 1
        N_size = 1
        N_bin = 1

    valid_snapshots = N_snapshots - 2
    N_output = valid_snapshots // window_total

    print(f"Valid snapshots for analysis: {valid_snapshots}")
    print(f"Number of output points: {N_output}")

    # JIT compile
    print("\nJIT compiling...")
    v_jax = jnp.array(v)
    mu_jax = jnp.array(mu_values)

    f_test = jnp.array(f_history[1])
    f_prev_test = jnp.array(f_history[0])
    f_next_test = jnp.array(f_history[2])

    # Warm-up JIT
    _ = compute_errors_for_all_mu(f_test, f_prev_test, f_next_test, v_jax, dx, dt, mu_jax)
    _ = compute_coefficients_for_mu(f_test, f_prev_test, f_next_test, v_jax, dx, dt)
    jax.block_until_ready(_)
    print("JIT compilation done.")

    # Storage for results
    optimal_mus_numerical = []
    optimal_mus_analytical = []
    A_values = []
    B_values = []
    C_values = []
    min_errors = []
    all_errors_list = []
    time_centers = []

    print("\nComputing optimal μ for each time window...")

    for out_idx in tqdm(range(N_output), desc="Processing"):
        window_start = 1 + out_idx * window_total
        window_end = window_start + window_total

        # Accumulate over window
        window_errors = np.zeros(len(mu_values))
        window_A = 0.0
        window_B = 0.0
        window_C = 0.0
        count = 0

        for idx in range(window_start, min(window_end, N_snapshots - 1)):
            f = jnp.array(f_history[idx])
            f_prev = jnp.array(f_history[idx - 1])
            f_next = jnp.array(f_history[idx + 1])

            # Numerical errors
            errors = compute_errors_for_all_mu(f, f_prev, f_next, v_jax, dx, dt, mu_jax)
            window_errors += np.array(errors)

            # Analytical coefficients
            A, B, C = compute_coefficients_for_mu(f, f_prev, f_next, v_jax, dx, dt)
            window_A += float(A)
            window_B += float(B)
            window_C += float(C)
            count += 1

        # Average
        window_errors /= count
        window_A /= count
        window_B /= count
        window_C /= count

        # Numerical optimal
        min_idx = np.argmin(window_errors)
        mu_opt_numerical = mu_values[min_idx]
        min_error = window_errors[min_idx]

        # Analytical optimal: μ_opt = B/A
        if window_A > 1e-30:
            mu_opt_analytical = window_B / window_A
        else:
            mu_opt_analytical = np.nan

        # Store results
        optimal_mus_numerical.append(mu_opt_numerical)
        optimal_mus_analytical.append(mu_opt_analytical)
        A_values.append(window_A)
        B_values.append(window_B)
        C_values.append(window_C)
        min_errors.append(min_error)
        all_errors_list.append(window_errors)

        t_center = (times[window_start] + times[min(window_end - 1, N_snapshots - 1)]) / 2
        time_centers.append(t_center)

    return {
        'mu_opt_numerical': np.array(optimal_mus_numerical),
        'mu_opt_analytical': np.array(optimal_mus_analytical),
        'A': np.array(A_values),
        'B': np.array(B_values),
        'C': np.array(C_values),
        'time_centers': np.array(time_centers),
        'min_errors': np.array(min_errors),
        'all_errors': np.array(all_errors_list),
        'mu_values': mu_values,
    }


# =============================================================================
# 1D Landau Solver (JAX-accelerated)
# =============================================================================

class LandauSolver1D_JAX:
    """
    JAX-accelerated 1D Boltzmann-Landau solver using Strang splitting.
    """

    def __init__(self, N_x, N_v, N_t, X=0.5, V=6.0, T_final=0.1, lambda_D=10.0):
        self.N_x = N_x
        self.N_v = N_v
        self.N_t = N_t
        self.X = X
        self.V = V
        self.T_final = T_final
        self.lambda_D = lambda_D
        self.cutoff = 1.0 / lambda_D

        # Grids
        self.dx = 2 * X / N_x
        self.x = jnp.linspace(-X, X - self.dx, N_x)
        self.dv = 2 * V / (N_v - 1)
        self.v = jnp.linspace(-V, V, N_v)
        self.dt = T_final / N_t

        # Wavenumbers
        self.kx = 2 * jnp.pi * jnp.fft.fftfreq(N_x, self.dx)

        # Precompute kernel
        self._precompute_kernel()

        # JIT compile methods
        self._jit_compile()

        print(f"CFL number (advection): {float(jnp.max(jnp.abs(self.v))) * self.dt / self.dx:.4f}")

    def _precompute_kernel(self):
        """Precompute FFT of Coulomb kernel."""
        self.N_conv = 2 * self.N_v - 1
        u = jnp.arange(-(self.N_v - 1), self.N_v) * self.dv
        self.Phi = 1.0 / jnp.maximum(jnp.abs(u), self.cutoff)
        self.Phi_fft = jnp.fft.fft(self.Phi)

    def _jit_compile(self):
        """JIT compile the step functions."""
        # Capture constants
        N_v = self.N_v
        N_conv = self.N_conv
        dv = self.dv

        @jit
        def advection_step(f, dt, kx, v):
            f_hat = jnp.fft.fft(f, axis=0)
            phase = jnp.exp(-1j * jnp.outer(kx, v) * dt)
            return jnp.real(jnp.fft.ifft(f_hat * phase, axis=0))

        @jit
        def collision_coefficients(f, Phi_fft):
            df_dv = jnp.zeros_like(f)
            df_dv = df_dv.at[:, 1:-1].set((f[:, 2:] - f[:, :-2]) / (2 * dv))
            df_dv = df_dv.at[:, 0].set((f[:, 1] - f[:, 0]) / dv)
            df_dv = df_dv.at[:, -1].set((f[:, -1] - f[:, -2]) / dv)

            def conv_single(f_row, df_row):
                f_padded = jnp.zeros(N_conv)
                f_padded = f_padded.at[:N_v].set(f_row)
                df_padded = jnp.zeros(N_conv)
                df_padded = df_padded.at[:N_v].set(df_row)

                A_full = jnp.real(jnp.fft.ifft(Phi_fft * jnp.fft.fft(f_padded)))
                B_full = jnp.real(jnp.fft.ifft(Phi_fft * jnp.fft.fft(df_padded)))

                start = N_v - 1
                A = A_full[start:start + N_v] * dv
                B = B_full[start:start + N_v] * dv
                return A, B

            A, B = vmap(conv_single)(f, df_dv)
            return A, B

        @jit
        def collision_operator(f, Phi_fft):
            A, B = collision_coefficients(f, Phi_fft)

            df_dv = jnp.zeros_like(f)
            df_dv = df_dv.at[:, 1:-1].set((f[:, 2:] - f[:, :-2]) / (2 * dv))
            df_dv = df_dv.at[:, 0].set((f[:, 1] - f[:, 0]) / dv)
            df_dv = df_dv.at[:, -1].set((f[:, -1] - f[:, -2]) / dv)

            J = A * df_dv - B * f

            Q = jnp.zeros_like(f)
            Q = Q.at[:, 1:-1].set((J[:, 2:] - J[:, :-2]) / (2 * dv))
            Q = Q.at[:, 0].set((J[:, 1] - J[:, 0]) / dv)
            Q = Q.at[:, -1].set((J[:, -1] - J[:, -2]) / dv)

            return Q

        @jit
        def collision_step_rk4(f, dt, Phi_fft):
            k1 = collision_operator(f, Phi_fft)
            k2 = collision_operator(f + 0.5 * dt * k1, Phi_fft)
            k3 = collision_operator(f + 0.5 * dt * k2, Phi_fft)
            k4 = collision_operator(f + dt * k3, Phi_fft)
            f_new = f + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            return jnp.maximum(f_new, 0.0)

        self._advection_step = advection_step
        self._collision_step = collision_step_rk4

    def strang_step(self, f):
        """Single Strang splitting step."""
        f = self._advection_step(f, self.dt / 2, self.kx, self.v)
        f = self._collision_step(f, self.dt, self.Phi_fft)
        f = self._advection_step(f, self.dt / 2, self.kx, self.v)
        return f

    def initial_condition(self):
        """Set initial condition: ρ = 1 + 0.5*sin(2πx), u = 0, T = 1."""
        rho0 = 1 + 0.5 * jnp.sin(2 * jnp.pi * self.x)
        u0 = jnp.zeros(self.N_x)
        T0 = jnp.ones(self.N_x)
        return compute_maxwellian(rho0, u0, T0, self.v)

    def solve(self, save_every=1, verbose=True):
        """Solve the Boltzmann-Landau equation."""
        if verbose:
            print(f"\n{'='*60}")
            print("1D Boltzmann-Landau Equation Solver (JAX)")
            print('='*60)
            print(f"Grid: N_x={self.N_x}, N_v={self.N_v}, N_t={self.N_t}")
            print(f"Domain: x ∈ [-{self.X}, {self.X}], v ∈ [-{self.V}, {self.V}]")
            print(f"Time: t ∈ [0, {self.T_final}], dt = {self.dt:.6e}")
            print(f"Saving every {save_every} steps")
            print('='*60)

        f = self.initial_condition()

        # Storage
        n_saves = self.N_t // save_every + 1
        f_history = [np.array(f)]
        times = [0.0]

        # Time stepping
        start_time = time.time()

        for n in tqdm(range(self.N_t), desc="Time stepping", disable=not verbose):
            f = self.strang_step(f)

            if (n + 1) % save_every == 0:
                f_history.append(np.array(f))
                times.append((n + 1) * float(self.dt))

        elapsed = time.time() - start_time

        if verbose:
            print(f"\nCompleted in {elapsed:.2f} seconds")

        return {
            'f_history': np.array(f_history),
            'x': np.array(self.x),
            'v': np.array(self.v),
            'times': np.array(times),
            'elapsed_time': elapsed,
            'params': {
                'N_x': self.N_x, 'N_v': self.N_v, 'N_t': self.N_t,
                'X': self.X, 'V': self.V, 'T_final': self.T_final,
                'lambda_D': self.lambda_D, 'save_every': save_every,
            }
        }


# =============================================================================
# Plotting functions
# =============================================================================

def plot_mu_opt_analytical(results, save_path):
    """
    Create analytical μ_opt plot similar to tau_opt_analytical.png

    Top panel: Analytical μ_opt = B/A (green = positive, red = negative)
    Bottom panel: |Analytical μ_opt| vs Numerical μ_opt
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    times = results['time_centers']
    mu_analytical = results['mu_opt_analytical']
    mu_numerical = results['mu_opt_numerical']
    mu_values = results['mu_values']

    # Set bounds
    mu_min = mu_values.min()
    mu_max = mu_values.max()

    # Top panel: Analytical μ_opt = B/A with sign
    ax = axes[0]

    # Separate positive and negative
    positive_mask = mu_analytical > 0
    negative_mask = mu_analytical < 0

    # Plot with color coding
    if np.any(positive_mask):
        ax.scatter(times[positive_mask], np.log10(np.abs(mu_analytical[positive_mask])),
                   c='green', s=30, label='Positive', alpha=0.7)
    if np.any(negative_mask):
        ax.scatter(times[negative_mask], np.log10(np.abs(mu_analytical[negative_mask])),
                   c='red', s=30, label='Negative', alpha=0.7)

    ax.axhline(np.log10(mu_min), color='gray', linestyle='--', alpha=0.5, label=f'μ_min = {mu_min}')
    ax.axhline(np.log10(mu_max), color='gray', linestyle='--', alpha=0.5, label=f'μ_max = {mu_max}')
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('log₁₀|μ_opt| (analytical = B/A)', fontsize=12)
    ax.set_title('Analytical μ_opt = B/A (green = positive, red = negative)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Bottom panel: |Analytical| vs Numerical
    ax = axes[1]

    ax.scatter(times, np.abs(mu_analytical), c='blue', s=20, alpha=0.6,
               label='|Analytical μ_opt| = |B/A|')
    ax.plot(times, mu_numerical, 'r-', linewidth=1.5, label='Numerical μ_opt')

    ax.axhline(mu_min, color='gray', linestyle='--', alpha=0.5, label=f'μ_min = {mu_min}')
    ax.axhline(mu_max, color='gray', linestyle='--', alpha=0.5, label=f'μ_max = {mu_max}')
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('μ_opt', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Analytical |B/A| vs Numerical μ_opt', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")

    return fig


def plot_mu_optimization_summary(results, save_path):
    """Plot comprehensive μ optimization results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    times = results['time_centers']
    mu_numerical = results['mu_opt_numerical']
    mu_analytical = results['mu_opt_analytical']
    min_errors = results['min_errors']
    all_errors = results['all_errors']
    mu_values = results['mu_values']

    # (0,0) μ evolution
    ax = axes[0, 0]
    ax.plot(times, mu_numerical, 'b.-', markersize=4, linewidth=1, label='Numerical')
    ax.plot(times, np.abs(mu_analytical), 'r--', alpha=0.7, label='|Analytical|')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Optimal μ')
    ax.set_title('Optimal μ for FP Approximation of Landau')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(mu_numerical.mean(), color='g', linestyle=':',
               label=f'Mean = {mu_numerical.mean():.3f}')

    # (0,1) Histogram
    ax = axes[0, 1]
    ax.hist(mu_numerical, bins=min(30, len(np.unique(mu_numerical))),
            edgecolor='black', alpha=0.7)
    ax.axvline(mu_numerical.mean(), color='r', linestyle='--',
               label=f'Mean = {mu_numerical.mean():.3f}')
    ax.set_xlabel('Optimal μ')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Optimal μ Values')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Error evolution
    ax = axes[1, 0]
    ax.semilogy(times, min_errors, 'g.-', markersize=4, linewidth=1)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Minimum L2 Error')
    ax.set_title('FP Approximation Error with Optimal μ')
    ax.grid(True, alpha=0.3)

    # (1,1) Error landscape
    ax = axes[1, 1]
    if len(all_errors) > 10:
        im = ax.pcolormesh(mu_values, times, np.log10(all_errors + 1e-20),
                          shading='auto', cmap='viridis')
        plt.colorbar(im, ax=ax, label='log₁₀(L2 Error)')
        ax.set_xlabel('μ')
        ax.set_ylabel('Time t')
        ax.set_title('Error Landscape (log scale)')
        ax.plot(mu_numerical, times, 'r.', markersize=2, alpha=0.5)
    else:
        avg_error_profile = all_errors.mean(axis=0)
        ax.plot(mu_values, avg_error_profile, 'b-', linewidth=2)
        ax.set_xlabel('μ')
        ax.set_ylabel('Average L2 Error')
        ax.set_title('Error Profile Averaged Over All Time Windows')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Main function
# =============================================================================

def main():
    """
    Main function: Run 1D Landau simulation and compute optimal μ for FP operator.

    Parameters optimized for 2x RTX A6000 GPUs (48GB each):
    - N_x = 8192 (spatial resolution)
    - N_v = 512 (velocity resolution)
    - N_t = 16384 (time steps)
    - save_every = 16 (yields 1024 snapshots)
    - Memory: ~34 GB for f_history + working memory
    - Estimated time: 8-16 hours
    """
    # =====================================================================
    # SIMULATION PARAMETERS
    # =====================================================================
    N_x = 8192        # Spatial grid points
    N_v = 512         # Velocity grid points
    N_t = 16384       # Time steps
    save_every = 16   # Save every N steps -> 1024 snapshots

    X = 0.5           # Spatial domain: [-X, X]
    V = 6.0           # Velocity domain: [-V, V]
    T_final = 0.1     # Final time
    lambda_D = 10.0   # Debye length

    # μ search parameters
    mu_min = 0.001
    mu_max = 100.0
    n_mu = 200        # Number of μ values to test
    mu_values = np.logspace(np.log10(mu_min), np.log10(mu_max), n_mu)

    # Window parameters for time averaging
    N_size = 1        # Snapshots per window
    N_bin = 1         # Windows per output point

    # =====================================================================
    # PRINT CONFIGURATION
    # =====================================================================
    print("="*70)
    print("Optimal μ (mu) Finder for FP Approximation of Landau Collision")
    print("="*70)
    print(f"\nSimulation Parameters:")
    print(f"  N_x = {N_x}, N_v = {N_v}, N_t = {N_t}")
    print(f"  save_every = {save_every} -> {N_t // save_every + 1} snapshots")
    print(f"  Domain: x ∈ [-{X}, {X}], v ∈ [-{V}, {V}]")
    print(f"  T_final = {T_final}, λ_D = {lambda_D}")

    print(f"\nμ Search Parameters:")
    print(f"  μ range: [{mu_min}, {mu_max}] ({n_mu} values, log-spaced)")
    print(f"  N_size = {N_size}, N_bin = {N_bin}")

    # Memory estimate
    n_snapshots = N_t // save_every + 1
    mem_gb = n_snapshots * N_x * N_v * 8 / 1e9
    print(f"\nMemory Estimate:")
    print(f"  f_history: {n_snapshots} × {N_x} × {N_v} × 8 bytes = {mem_gb:.2f} GB")

    print(f"\nUsing {jax.local_device_count()} GPU(s): {[str(d) for d in jax.devices()]}")
    print("="*70)

    # =====================================================================
    # RUN SIMULATION
    # =====================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "="*70)
    print("Step 1: Running 1D Landau Simulation")
    print("="*70)

    solver = LandauSolver1D_JAX(
        N_x=N_x, N_v=N_v, N_t=N_t,
        X=X, V=V, T_final=T_final, lambda_D=lambda_D
    )

    sim_results = solver.solve(save_every=save_every, verbose=True)

    f_history = sim_results['f_history']
    x = sim_results['x']
    v = sim_results['v']
    times = sim_results['times']

    print(f"\nSimulation completed!")
    print(f"  f_history shape: {f_history.shape}")
    print(f"  Time range: [{times[0]:.6f}, {times[-1]:.6f}]")

    # =====================================================================
    # COMPUTE OPTIMAL μ
    # =====================================================================
    print("\n" + "="*70)
    print("Step 2: Computing Optimal μ for FP Operator")
    print("="*70)

    opt_start = time.time()

    opt_results = optimize_mu_all_snapshots(
        f_history, x, v, times, mu_values,
        N_size=N_size, N_bin=N_bin
    )

    opt_elapsed = time.time() - opt_start
    print(f"\nOptimization completed in {opt_elapsed:.2f} seconds")

    # =====================================================================
    # RESULTS SUMMARY
    # =====================================================================
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)

    mu_numerical = opt_results['mu_opt_numerical']
    mu_analytical = opt_results['mu_opt_analytical']

    print(f"Number of output points: {len(mu_numerical)}")
    print(f"\nNumerical μ_opt:")
    print(f"  Range: [{mu_numerical.min():.4f}, {mu_numerical.max():.4f}]")
    print(f"  Mean ± Std: {mu_numerical.mean():.4f} ± {mu_numerical.std():.4f}")

    valid_analytical = mu_analytical[~np.isnan(mu_analytical)]
    if len(valid_analytical) > 0:
        print(f"\nAnalytical μ_opt (B/A):")
        print(f"  Range: [{valid_analytical.min():.4f}, {valid_analytical.max():.4f}]")
        print(f"  Mean ± Std: {valid_analytical.mean():.4f} ± {valid_analytical.std():.4f}")
        print(f"  Positive fraction: {np.mean(valid_analytical > 0) * 100:.1f}%")

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)

    os.makedirs("data/landau_1d", exist_ok=True)
    os.makedirs("figures/landau_1d", exist_ok=True)

    base_name = f"mu_opt_fp_Nx{N_x}_Nv{N_v}_Nt{N_t}_{timestamp}"

    # Save optimization results
    opt_file = f"data/landau_1d/{base_name}_results.npz"
    np.savez(opt_file,
             mu_opt_numerical=opt_results['mu_opt_numerical'],
             mu_opt_analytical=opt_results['mu_opt_analytical'],
             A=opt_results['A'],
             B=opt_results['B'],
             C=opt_results['C'],
             time_centers=opt_results['time_centers'],
             min_errors=opt_results['min_errors'],
             all_errors=opt_results['all_errors'],
             mu_values=mu_values,
             N_size=N_size,
             N_bin=N_bin)
    print(f"Results saved to: {opt_file}")

    # Save config
    config = {
        "timestamp": timestamp,
        "simulation": {
            "N_x": N_x, "N_v": N_v, "N_t": N_t,
            "save_every": save_every,
            "X": X, "V": V, "T_final": T_final,
            "lambda_D": lambda_D,
        },
        "optimization": {
            "mu_min": mu_min, "mu_max": mu_max, "n_mu": n_mu,
            "N_size": N_size, "N_bin": N_bin,
        },
        "results": {
            "mu_numerical_mean": float(mu_numerical.mean()),
            "mu_numerical_std": float(mu_numerical.std()),
            "mu_numerical_min": float(mu_numerical.min()),
            "mu_numerical_max": float(mu_numerical.max()),
        },
        "timing": {
            "simulation_time_sec": sim_results['elapsed_time'],
            "optimization_time_sec": opt_elapsed,
        }
    }

    config_file = f"data/landau_1d/{base_name}_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_file}")

    # Save f_history for future analysis
    f_file = f"data/landau_1d/{base_name}_f.npy"
    np.save(f_file, f_history)
    print(f"f_history saved to: {f_file}")

    grid_file = f"data/landau_1d/{base_name}_grid.npz"
    np.savez(grid_file, x=x, v=v, times=times)
    print(f"Grid saved to: {grid_file}")

    # =====================================================================
    # GENERATE PLOTS
    # =====================================================================
    print("\n" + "="*70)
    print("Generating Plots")
    print("="*70)

    # Analytical plot (similar to tau_opt_analytical.png)
    fig1 = plot_mu_opt_analytical(opt_results, f"figures/landau_1d/{base_name}_analytical.png")
    plt.close(fig1)

    # Summary plot
    fig2 = plot_mu_optimization_summary(opt_results, f"figures/landau_1d/{base_name}_summary.png")
    plt.close(fig2)

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

    return opt_results, sim_results


if __name__ == "__main__":
    results = main()
