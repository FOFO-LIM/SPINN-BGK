#!/usr/bin/env python3
"""
Optimize τ (tau) for BGK approximation from pre-computed Landau simulation data.

Uses single GPU (GPU 7) for efficiency.
Loads pre-saved distribution function and computes optimal τ.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'  # Use GPU 7

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

print(f"JAX devices: {jax.devices()}")


def run_tau_optimization_from_data(data_file):
    """
    Compute optimal τ from pre-saved Landau simulation data.
    """
    # Load data
    print(f"\nLoading data from {data_file}...")
    f_history = np.load(data_file)
    print(f"Shape: {f_history.shape}")
    print(f"Dtype: {f_history.dtype}")
    print(f"Size: {f_history.nbytes / 1e9:.2f} GB")

    N_snapshots, N_x, N_v = f_history.shape

    # Parameters (matching the simulation)
    X = 0.5
    V = 6.0
    T_final = 0.1
    lambda_D = 10.0

    # Grids
    dx = 2 * X / N_x
    x = jnp.linspace(-X, X - dx, N_x)
    dv = 2 * V / (N_v - 1)
    v = jnp.linspace(-V, V, N_v)

    # Time between snapshots
    # Original: N_t = 131072, save_every = 1296 (approx) -> 101 snapshots
    # dt_snapshot = T_final / (N_snapshots - 1)
    dt_snapshot = T_final / (N_snapshots - 1)
    times = np.linspace(0, T_final, N_snapshots)

    # τ grid: log-uniform from 0.1 to 1000
    tau_values = jnp.logspace(jnp.log10(0.1), jnp.log10(1000), 100)
    N_tau = len(tau_values)

    print(f"\n{'='*70}")
    print("τ Optimization from Pre-computed Data (GPU 7)")
    print('='*70)
    print(f"Grid: N_x={N_x}, N_v={N_v}")
    print(f"Snapshots: {N_snapshots}")
    print(f"Time step between snapshots: {dt_snapshot:.6f}")
    print(f"τ grid: {float(tau_values[0]):.2f} to {float(tau_values[-1]):.2f} ({N_tau} log-uniform)")
    print('='*70)

    # Quadrature weights
    w = jnp.ones(N_v) * dv
    w = w.at[0].set(dv / 2)
    w = w.at[-1].set(dv / 2)

    @jit
    def compute_tau_errors(f, f_prev, f_next, dt_meas):
        """Compute L2 error for all τ values."""
        # Moments
        rho = jnp.sum(f * w, axis=1)
        momentum = jnp.sum(f * v * w, axis=1)
        energy = jnp.sum(f * v**2 * w, axis=1)
        rho = jnp.maximum(rho, 1e-16)
        u = momentum / rho
        T = jnp.maximum((energy / rho) - u**2, 1e-10)

        # Maxwellian equilibrium
        f_eq = (rho[:, None] / jnp.sqrt(2 * jnp.pi * T[:, None]) *
                jnp.exp(-(v[None, :] - u[:, None])**2 / (2 * T[:, None])))

        # Time derivative (central difference)
        df_dt = (f_next - f_prev) / (2 * dt_meas)

        # Spatial derivative (periodic, central difference)
        df_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * dx)

        # Collision term from PDE: df/dt + v * df/dx = Q(f)
        collision_term = df_dt + v[None, :] * df_dx

        # f_neq = f_eq - f
        f_neq = f_eq - f

        # Error for each tau: |f_neq/tau - collision_term|^2
        def error_for_tau(tau):
            bgk = f_neq / tau
            diff = bgk - collision_term
            return jnp.mean(diff**2)

        return vmap(error_for_tau)(tau_values)

    # JIT warmup
    print("\nJIT compiling...", end=" ", flush=True)
    f_test = jnp.array(f_history[0])
    _ = compute_tau_errors(f_test, f_test, f_test, dt_snapshot)
    jax.block_until_ready(_)
    print("done")

    # Main loop - compute τ for each snapshot
    print(f"\nComputing optimal τ for {N_snapshots - 2} snapshots...")

    all_tau_errors = []
    measurement_times = []

    start_time = time.time()

    for i in range(1, N_snapshots - 1):
        if i % 10 == 0:
            print(f"  Processing snapshot {i}/{N_snapshots - 2}...")

        f_prev = jnp.array(f_history[i - 1])
        f_curr = jnp.array(f_history[i])
        f_next = jnp.array(f_history[i + 1])

        errors = compute_tau_errors(f_curr, f_prev, f_next, dt_snapshot)
        jax.block_until_ready(errors)

        all_tau_errors.append(np.array(errors))
        measurement_times.append(times[i])

    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.2f} seconds")

    # Convert
    all_tau_errors = np.array(all_tau_errors)
    measurement_times = np.array(measurement_times)

    # Find optimal τ for each snapshot
    optimal_indices = np.argmin(all_tau_errors, axis=1)
    optimal_taus = np.array(tau_values)[optimal_indices]
    min_errors = np.array([all_tau_errors[i, optimal_indices[i]] for i in range(len(optimal_indices))])

    print(f"\n{'='*60}")
    print("Results")
    print('='*60)
    print(f"Snapshots analyzed: {len(optimal_taus)}")
    print(f"τ range: {optimal_taus.min():.2f} to {optimal_taus.max():.2f}")
    print(f"τ mean: {optimal_taus.mean():.2f} ± {optimal_taus.std():.2f}")
    print(f"τ median: {np.median(optimal_taus):.2f}")

    return {
        'optimal_taus': optimal_taus,
        'measurement_times': measurement_times,
        'all_tau_errors': all_tau_errors,
        'tau_values': np.array(tau_values),
        'min_errors': min_errors,
        'elapsed_time': elapsed,
        'data_file': data_file,
    }


def main():
    data_file = "data/landau_1d/landau_large_Nx65536_Nv1024_Nt131072_20260125_214408_f.npy"

    results = run_tau_optimization_from_data(data_file)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f"tau_from_data_{timestamp}"
    data_dir = "data/landau_1d"
    fig_dir = "figures/landau_1d"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    np.savez(f"{data_dir}/{output_base}.npz", **results)
    print(f"\nSaved to {data_dir}/{output_base}.npz")

    # Plot
    optimal_taus = results['optimal_taus']
    measurement_times = results['measurement_times']
    tau_values = results['tau_values']
    all_tau_errors = results['all_tau_errors']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Optimal τ vs time
    ax = axes[0, 0]
    ax.plot(measurement_times, optimal_taus, 'b.-', markersize=4)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Optimal τ')
    ax.set_yscale('log')
    ax.set_title('Optimal τ vs Time')
    ax.axhline(optimal_taus.mean(), color='r', linestyle='--',
               label=f'Mean = {optimal_taus.mean():.2f}')
    ax.axhline(np.median(optimal_taus), color='g', linestyle=':',
               label=f'Median = {np.median(optimal_taus):.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[0, 1]
    ax.hist(optimal_taus, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(optimal_taus.mean(), color='r', linestyle='--', label=f'Mean={optimal_taus.mean():.2f}')
    ax.axvline(np.median(optimal_taus), color='g', linestyle=':', label=f'Median={np.median(optimal_taus):.2f}')
    ax.set_xlabel('Optimal τ')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Optimal τ')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error vs τ for selected times
    ax = axes[1, 0]
    indices = [0, len(measurement_times)//4, len(measurement_times)//2, 3*len(measurement_times)//4, -1]
    for idx in indices:
        t = measurement_times[idx]
        ax.plot(tau_values, all_tau_errors[idx], label=f't={t:.4f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('τ')
    ax.set_ylabel('L2 Error')
    ax.set_title('Error vs τ at Different Times')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error landscape
    ax = axes[1, 1]
    im = ax.pcolormesh(tau_values, measurement_times, np.log10(all_tau_errors + 1e-20),
                       shading='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, label='log₁₀(Error)')
    ax.set_xscale('log')
    ax.set_xlabel('τ')
    ax.set_ylabel('Time t')
    ax.set_title('Error Landscape')
    ax.plot(optimal_taus, measurement_times, 'r.', markersize=3, label='Optimal τ')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{output_base}.png", dpi=150)
    print(f"Figure: {fig_dir}/{output_base}.png")

    # Timeseries plot
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(measurement_times, optimal_taus, 'b-o', markersize=3)
    ax2.axhline(optimal_taus.mean(), color='r', linestyle='--',
               label=f'Mean τ = {optimal_taus.mean():.2f}')
    ax2.axhline(np.median(optimal_taus), color='g', linestyle=':',
               label=f'Median τ = {np.median(optimal_taus):.2f}')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Optimal τ')
    ax2.set_yscale('log')
    ax2.set_title('Optimal BGK τ vs Time (from pre-computed data)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.savefig(f"{fig_dir}/{output_base}_timeseries.png", dpi=150)
    print(f"Timeseries: {fig_dir}/{output_base}_timeseries.png")

    plt.close('all')

    return results


if __name__ == "__main__":
    main()
