#!/usr/bin/env python3
"""
Compute optimal μ for Fokker-Planck operator approximation of Landau collision.
Multi-GPU version using pmap to split spatial domain across 2 GPUs.

Run on GPUs 0 and 1.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, lax
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from functools import partial
from tqdm import tqdm
import time

print(f"JAX devices: {jax.devices()}")
print(f"Number of devices: {jax.local_device_count()}")
N_DEVICES = jax.local_device_count()

# =============================================================================
# Core computation functions
# =============================================================================

@jit
def compute_maxwellian(rho, u, T, v):
    """Compute Maxwellian distribution."""
    rho = rho[:, jnp.newaxis]
    u = u[:, jnp.newaxis]
    T = jnp.maximum(T[:, jnp.newaxis], 1e-10)
    v_grid = v[jnp.newaxis, :]
    return rho / jnp.sqrt(2 * jnp.pi * T) * jnp.exp(-(v_grid - u)**2 / (2 * T))


@jit
def compute_moments(f, v, dv):
    """Compute moments (ρ, u, T) from distribution function."""
    N_v = len(v)
    w = jnp.ones(N_v) * dv
    w = w.at[0].set(dv / 2)
    w = w.at[-1].set(dv / 2)

    rho = jnp.sum(f * w, axis=1)
    momentum = jnp.sum(f * v * w, axis=1)
    energy = jnp.sum(f * v**2 * w, axis=1)

    rho = jnp.maximum(rho, 1e-16)
    u = momentum / rho
    T = jnp.maximum((energy / rho) - u**2, 1e-10)

    return rho, u, T


@jit
def compute_velocity_derivative(f, dv):
    """Compute ∂f/∂v using central differences."""
    df_dv = jnp.zeros_like(f)
    df_dv = df_dv.at[:, 1:-1].set((f[:, 2:] - f[:, :-2]) / (2 * dv))
    df_dv = df_dv.at[:, 0].set((f[:, 1] - f[:, 0]) / dv)
    df_dv = df_dv.at[:, -1].set((f[:, -1] - f[:, -2]) / dv)
    return df_dv


@jit
def compute_fokker_planck_operator(f, v, dv):
    """Compute P(f) = ∇_v · (M ∇_v(f/M))."""
    rho, u, T = compute_moments(f, v, dv)
    M = compute_maxwellian(rho, u, T, v)
    M = jnp.maximum(M, 1e-30)

    f_over_M = f / M
    d_f_over_M_dv = compute_velocity_derivative(f_over_M, dv)
    flux = M * d_f_over_M_dv
    P = compute_velocity_derivative(flux, dv)

    return P


# =============================================================================
# Multi-GPU Landau Solver using pmap
# =============================================================================

class LandauSolver1D_MultiGPU:
    """
    Multi-GPU 1D Boltzmann-Landau solver.
    Splits spatial domain across GPUs using pmap.
    """

    def __init__(self, N_x, N_v, N_t, X=0.5, V=6.0, T_final=0.1, lambda_D=10.0):
        assert N_x % N_DEVICES == 0, f"N_x must be divisible by {N_DEVICES}"

        self.N_x = N_x
        self.N_v = N_v
        self.N_t = N_t
        self.N_x_per_device = N_x // N_DEVICES
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

        # Wavenumbers for full domain (used in advection)
        self.kx = 2 * jnp.pi * jnp.fft.fftfreq(N_x, self.dx)

        # Precompute collision kernel
        self._precompute_kernel()

        # Build pmapped functions
        self._build_pmap_functions()

        print(f"CFL number (advection): {float(jnp.max(jnp.abs(self.v))) * self.dt / self.dx:.4f}")
        print(f"Spatial points per GPU: {self.N_x_per_device}")

    def _precompute_kernel(self):
        """Precompute FFT of Coulomb kernel."""
        self.N_conv = 2 * self.N_v - 1
        u = jnp.arange(-(self.N_v - 1), self.N_v) * self.dv
        self.Phi = 1.0 / jnp.maximum(jnp.abs(u), self.cutoff)
        self.Phi_fft = jnp.fft.fft(self.Phi)

    def _build_pmap_functions(self):
        """Build pmapped functions for multi-GPU execution."""
        N_v = self.N_v
        N_conv = self.N_conv
        dv = self.dv
        Phi_fft = self.Phi_fft

        # Collision operator for a chunk of spatial points
        @jit
        def collision_operator_chunk(f_chunk):
            """Compute collision operator for a chunk of f (N_x_chunk, N_v)."""
            # Velocity derivative
            df_dv = jnp.zeros_like(f_chunk)
            df_dv = df_dv.at[:, 1:-1].set((f_chunk[:, 2:] - f_chunk[:, :-2]) / (2 * dv))
            df_dv = df_dv.at[:, 0].set((f_chunk[:, 1] - f_chunk[:, 0]) / dv)
            df_dv = df_dv.at[:, -1].set((f_chunk[:, -1] - f_chunk[:, -2]) / dv)

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

            A, B = vmap(conv_single)(f_chunk, df_dv)

            # Flux J = A * df_dv - B * f
            J = A * df_dv - B * f_chunk

            # Q = dJ/dv
            Q = jnp.zeros_like(f_chunk)
            Q = Q.at[:, 1:-1].set((J[:, 2:] - J[:, :-2]) / (2 * dv))
            Q = Q.at[:, 0].set((J[:, 1] - J[:, 0]) / dv)
            Q = Q.at[:, -1].set((J[:, -1] - J[:, -2]) / dv)

            return Q

        @jit
        def collision_step_rk4_chunk(f_chunk, dt):
            """RK4 collision step for a chunk."""
            k1 = collision_operator_chunk(f_chunk)
            k2 = collision_operator_chunk(f_chunk + 0.5 * dt * k1)
            k3 = collision_operator_chunk(f_chunk + 0.5 * dt * k2)
            k4 = collision_operator_chunk(f_chunk + dt * k3)
            f_new = f_chunk + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            return jnp.maximum(f_new, 0.0)

        # pmap the collision step
        self._collision_step_pmap = pmap(collision_step_rk4_chunk, in_axes=(0, None))

        # Advection uses full domain FFT (runs on device 0, then redistributes)
        @jit
        def advection_step_full(f_full, dt, kx, v):
            f_hat = jnp.fft.fft(f_full, axis=0)
            phase = jnp.exp(-1j * jnp.outer(kx, v) * dt)
            return jnp.real(jnp.fft.ifft(f_hat * phase, axis=0))

        self._advection_step = advection_step_full

    def _reshape_for_pmap(self, f):
        """Reshape f from (N_x, N_v) to (N_devices, N_x/N_devices, N_v)."""
        return f.reshape(N_DEVICES, self.N_x_per_device, self.N_v)

    def _reshape_from_pmap(self, f_sharded):
        """Reshape from (N_devices, N_x/N_devices, N_v) to (N_x, N_v)."""
        return f_sharded.reshape(self.N_x, self.N_v)

    def strang_step(self, f):
        """Single Strang splitting step using multi-GPU."""
        # Advection dt/2 (full domain on device 0)
        f = self._advection_step(f, self.dt / 2, self.kx, self.v)

        # Collision dt (distributed across GPUs)
        f_sharded = self._reshape_for_pmap(f)
        f_sharded = self._collision_step_pmap(f_sharded, self.dt)
        f = self._reshape_from_pmap(f_sharded)

        # Advection dt/2
        f = self._advection_step(f, self.dt / 2, self.kx, self.v)

        return f

    def initial_condition(self):
        """Initial condition: ρ = 1 + 0.5*sin(2πx), u = 0, T = 1."""
        rho0 = 1 + 0.5 * jnp.sin(2 * jnp.pi * self.x)
        u0 = jnp.zeros(self.N_x)
        T0 = jnp.ones(self.N_x)
        return compute_maxwellian(rho0, u0, T0, self.v)

    def solve(self, save_every=1, verbose=True):
        """Solve the equation."""
        if verbose:
            print(f"\n{'='*60}")
            print("1D Boltzmann-Landau Solver (Multi-GPU)")
            print('='*60)
            print(f"Grid: N_x={self.N_x}, N_v={self.N_v}, N_t={self.N_t}")
            print(f"GPUs: {N_DEVICES}, N_x per GPU: {self.N_x_per_device}")
            print(f"Domain: x ∈ [-{self.X}, {self.X}], v ∈ [-{self.V}, {self.V}]")
            print(f"Time: t ∈ [0, {self.T_final}], dt = {self.dt:.6e}")
            print('='*60)

        f = self.initial_condition()

        # Warm-up JIT
        if verbose:
            print("Warming up JIT compilation...")
        _ = self.strang_step(f)
        jax.block_until_ready(_)
        if verbose:
            print("JIT warm-up done.")

        # Storage
        f_history = [np.array(f)]
        times = [0.0]

        start_time = time.time()

        for n in tqdm(range(self.N_t), desc="Time stepping", disable=not verbose):
            f = self.strang_step(f)

            if (n + 1) % save_every == 0:
                f_history.append(np.array(f))
                times.append((n + 1) * float(self.dt))

        elapsed = time.time() - start_time

        if verbose:
            print(f"\nCompleted in {elapsed:.2f} seconds")
            print(f"Throughput: {self.N_t / elapsed:.1f} steps/sec")

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
# μ optimization functions
# =============================================================================

@jit
def compute_spatial_derivative_periodic(f, dx):
    """Compute ∂f/∂x with periodic BC."""
    return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * dx)


@partial(jit, static_argnums=())
def compute_coefficients_for_mu(f, f_prev, f_next, v, dx, dt):
    """Compute A, B, C coefficients for μ optimization."""
    dv = v[1] - v[0]

    # Q from time derivative
    df_dt = (f_next - f_prev) / (2 * dt)
    df_dx = compute_spatial_derivative_periodic(f, dx)
    Q = df_dt + v[jnp.newaxis, :] * df_dx

    # FP operator
    P = compute_fokker_planck_operator(f, v, dv)

    A = jnp.mean(P**2)
    B = jnp.mean(Q * P)
    C = jnp.mean(Q**2)

    return A, B, C


@partial(jit, static_argnums=())
def compute_errors_for_all_mu(f, f_prev, f_next, v, dx, dt, mu_values):
    """Compute L2 errors for all μ values."""
    dv = v[1] - v[0]

    df_dt = (f_next - f_prev) / (2 * dt)
    df_dx = compute_spatial_derivative_periodic(f, dx)
    Q = df_dt + v[jnp.newaxis, :] * df_dx
    P = compute_fokker_planck_operator(f, v, dv)

    def error_for_mu(mu):
        diff = mu * P - Q
        return jnp.mean(diff**2)

    return vmap(error_for_mu)(mu_values)


def optimize_mu_all_snapshots(f_history, x, v, times, mu_values):
    """Find optimal μ for all time snapshots."""
    N_snapshots = f_history.shape[0]
    dt = times[1] - times[0]
    dx = x[1] - x[0]

    print(f"\nProcessing {N_snapshots} snapshots...")
    print(f"dt = {dt:.6e}, dx = {dx:.6e}")

    # JIT warm-up
    print("JIT compiling optimization functions...")
    v_jax = jnp.array(v)
    mu_jax = jnp.array(mu_values)
    f_test = jnp.array(f_history[1])
    _ = compute_errors_for_all_mu(f_test, jnp.array(f_history[0]),
                                   jnp.array(f_history[2]), v_jax, dx, dt, mu_jax)
    jax.block_until_ready(_)
    print("Done.")

    # Results storage
    mu_numerical = []
    mu_analytical = []
    A_vals, B_vals, C_vals = [], [], []
    min_errors = []
    all_errors_list = []
    time_centers = []

    print("Computing optimal μ...")
    for idx in tqdm(range(1, N_snapshots - 1)):
        f = jnp.array(f_history[idx])
        f_prev = jnp.array(f_history[idx - 1])
        f_next = jnp.array(f_history[idx + 1])

        # Numerical
        errors = compute_errors_for_all_mu(f, f_prev, f_next, v_jax, dx, dt, mu_jax)
        errors_np = np.array(errors)
        min_idx = np.argmin(errors_np)

        # Analytical
        A, B, C = compute_coefficients_for_mu(f, f_prev, f_next, v_jax, dx, dt)
        A, B, C = float(A), float(B), float(C)

        mu_numerical.append(mu_values[min_idx])
        mu_analytical.append(B / A if A > 1e-30 else np.nan)
        A_vals.append(A)
        B_vals.append(B)
        C_vals.append(C)
        min_errors.append(errors_np[min_idx])
        all_errors_list.append(errors_np)
        time_centers.append(times[idx])

    return {
        'mu_opt_numerical': np.array(mu_numerical),
        'mu_opt_analytical': np.array(mu_analytical),
        'A': np.array(A_vals),
        'B': np.array(B_vals),
        'C': np.array(C_vals),
        'time_centers': np.array(time_centers),
        'min_errors': np.array(min_errors),
        'all_errors': np.array(all_errors_list),
        'mu_values': mu_values,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_mu_opt_analytical(results, save_path):
    """Create analytical μ_opt plot similar to tau_opt_analytical.png."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    times = results['time_centers']
    mu_analytical = results['mu_opt_analytical']
    mu_numerical = results['mu_opt_numerical']
    mu_values = results['mu_values']
    mu_min, mu_max = mu_values.min(), mu_values.max()

    # Top: Analytical with sign
    ax = axes[0]
    pos = mu_analytical > 0
    neg = mu_analytical < 0
    if np.any(pos):
        ax.scatter(times[pos], np.log10(np.abs(mu_analytical[pos])),
                   c='green', s=30, alpha=0.7, label='Positive')
    if np.any(neg):
        ax.scatter(times[neg], np.log10(np.abs(mu_analytical[neg])),
                   c='red', s=30, alpha=0.7, label='Negative')
    ax.axhline(np.log10(mu_min), color='gray', linestyle='--', alpha=0.5)
    ax.axhline(np.log10(mu_max), color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time t')
    ax.set_ylabel('log₁₀|μ_opt| (analytical = B/A)')
    ax.set_title('Analytical μ_opt = B/A (green = positive, red = negative)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom: |Analytical| vs Numerical
    ax = axes[1]
    ax.scatter(times, np.abs(mu_analytical), c='blue', s=20, alpha=0.6,
               label='|Analytical μ_opt| = |B/A|')
    ax.plot(times, mu_numerical, 'r-', linewidth=1.5, label='Numerical μ_opt')
    ax.axhline(mu_min, color='gray', linestyle='--', alpha=0.5, label=f'μ_min = {mu_min}')
    ax.axhline(mu_max, color='gray', linestyle='--', alpha=0.5, label=f'μ_max = {mu_max}')
    ax.set_xlabel('Time t')
    ax.set_ylabel('μ_opt')
    ax.set_yscale('log')
    ax.set_title('Analytical |B/A| vs Numerical μ_opt')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    return fig


def plot_summary(results, save_path):
    """Plot summary."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    times = results['time_centers']
    mu_num = results['mu_opt_numerical']
    mu_ana = results['mu_opt_analytical']
    min_errors = results['min_errors']
    all_errors = results['all_errors']
    mu_values = results['mu_values']

    # (0,0) μ evolution
    ax = axes[0, 0]
    ax.plot(times, mu_num, 'b.-', ms=2, lw=1, label='Numerical')
    ax.plot(times, np.abs(mu_ana), 'r--', alpha=0.5, label='|Analytical|')
    ax.axhline(mu_num.mean(), color='g', ls=':', label=f'Mean={mu_num.mean():.3f}')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Optimal μ')
    ax.set_title('Optimal μ for FP Approximation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Histogram
    ax = axes[0, 1]
    ax.hist(mu_num, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(mu_num.mean(), color='r', ls='--', label=f'Mean={mu_num.mean():.3f}')
    ax.set_xlabel('Optimal μ')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Optimal μ')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Error
    ax = axes[1, 0]
    ax.semilogy(times, min_errors, 'g.-', ms=2, lw=1)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Min L2 Error')
    ax.set_title('FP Approximation Error')
    ax.grid(True, alpha=0.3)

    # (1,1) Error landscape
    ax = axes[1, 1]
    if len(all_errors) > 10:
        im = ax.pcolormesh(mu_values, times, np.log10(all_errors + 1e-20),
                          shading='auto', cmap='viridis')
        plt.colorbar(im, ax=ax, label='log₁₀(Error)')
        ax.plot(mu_num, times, 'r.', ms=1, alpha=0.5)
    ax.set_xlabel('μ')
    ax.set_ylabel('Time t')
    ax.set_title('Error Landscape')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    """Main function."""
    # Parameters
    N_x = 8192        # Must be divisible by N_DEVICES
    N_v = 512
    N_t = 16384
    save_every = 16   # 1025 snapshots

    X = 0.5
    V = 6.0
    T_final = 0.1
    lambda_D = 10.0

    mu_min = 0.001
    mu_max = 100.0
    n_mu = 200
    mu_values = np.logspace(np.log10(mu_min), np.log10(mu_max), n_mu)

    # Print config
    print("="*70)
    print("Optimal μ Finder for FP Approximation (Multi-GPU)")
    print("="*70)
    print(f"\nSimulation: N_x={N_x}, N_v={N_v}, N_t={N_t}")
    print(f"GPUs: {N_DEVICES}, N_x per GPU: {N_x // N_DEVICES}")
    print(f"Memory estimate: {(N_t // save_every + 1) * N_x * N_v * 8 / 1e9:.2f} GB")
    print(f"μ range: [{mu_min}, {mu_max}] ({n_mu} values)")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run simulation
    print("\n" + "="*70)
    print("Step 1: Running Simulation")
    print("="*70)

    solver = LandauSolver1D_MultiGPU(
        N_x=N_x, N_v=N_v, N_t=N_t,
        X=X, V=V, T_final=T_final, lambda_D=lambda_D
    )
    sim_results = solver.solve(save_every=save_every)

    f_history = sim_results['f_history']
    x = sim_results['x']
    v = sim_results['v']
    times = sim_results['times']

    print(f"f_history shape: {f_history.shape}")

    # Optimize μ
    print("\n" + "="*70)
    print("Step 2: Computing Optimal μ")
    print("="*70)

    opt_start = time.time()
    opt_results = optimize_mu_all_snapshots(f_history, x, v, times, mu_values)
    opt_elapsed = time.time() - opt_start
    print(f"Optimization done in {opt_elapsed:.2f}s")

    # Summary
    mu_num = opt_results['mu_opt_numerical']
    mu_ana = opt_results['mu_opt_analytical']

    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    print(f"Numerical μ: {mu_num.mean():.4f} ± {mu_num.std():.4f}")
    print(f"  Range: [{mu_num.min():.4f}, {mu_num.max():.4f}]")
    valid = mu_ana[~np.isnan(mu_ana)]
    if len(valid) > 0:
        print(f"Analytical μ: {valid.mean():.4f} ± {valid.std():.4f}")
        print(f"  Positive: {100*np.mean(valid > 0):.1f}%")

    # Save
    os.makedirs("data/landau_1d", exist_ok=True)
    os.makedirs("figures/landau_1d", exist_ok=True)

    base = f"mu_opt_fp_Nx{N_x}_Nv{N_v}_{timestamp}"

    np.savez(f"data/landau_1d/{base}_results.npz", **opt_results)
    np.save(f"data/landau_1d/{base}_f.npy", f_history)
    np.savez(f"data/landau_1d/{base}_grid.npz", x=x, v=v, times=times)

    config = {
        "timestamp": timestamp,
        "N_x": N_x, "N_v": N_v, "N_t": N_t,
        "N_devices": N_DEVICES,
        "sim_time_sec": sim_results['elapsed_time'],
        "opt_time_sec": opt_elapsed,
        "mu_mean": float(mu_num.mean()),
        "mu_std": float(mu_num.std()),
    }
    with open(f"data/landau_1d/{base}_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved to data/landau_1d/{base}_*")

    # Plots
    print("\nGenerating plots...")
    fig1 = plot_mu_opt_analytical(opt_results, f"figures/landau_1d/{base}_analytical.png")
    plt.close(fig1)
    fig2 = plot_summary(opt_results, f"figures/landau_1d/{base}_summary.png")
    plt.close(fig2)

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

    return opt_results


if __name__ == "__main__":
    main()
