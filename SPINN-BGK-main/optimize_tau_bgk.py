#!/usr/bin/env python3
"""
Optimize τ (tau) value for BGK approximation of high-fidelity Landau solution.

The BGK equation approximates the Landau collision operator:
    ∂f/∂t + v·∂f/∂x = (1/τ)(f_eq - f)

This script finds the optimal τ that minimizes the L2 error between:
    - BGK collision term: (1/τ)(f_eq - f)
    - Actual collision term: ∂f/∂t + v·∂f/∂x (computed from Landau data)

Works in (1+1+1)D: 1D space (x) + 1D velocity (v) + 1D time (t)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from functools import partial
from tqdm import tqdm

print(f"JAX devices: {jax.devices()}")
print(f"Number of devices: {jax.local_device_count()}")


@jit
def compute_maxwellian(rho, u, T, v):
    """
    Compute Maxwellian distribution.
    M(v) = ρ / √(2πT) · exp(-(v - u)² / (2T))
    """
    rho = rho[:, jnp.newaxis]
    u = u[:, jnp.newaxis]
    T = T[:, jnp.newaxis]
    T = jnp.maximum(T, 1e-10)  # Prevent division by zero
    v_grid = v[jnp.newaxis, :]

    return rho / jnp.sqrt(2 * jnp.pi * T) * jnp.exp(-(v_grid - u)**2 / (2 * T))


@jit
def compute_moments(f, v, dv):
    """
    Compute macroscopic moments from distribution function.
    """
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
    """
    Compute ∂f/∂x using central differences with periodic BC.
    """
    df_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * dx)
    return df_dx


@partial(jit, static_argnums=())
def compute_errors_for_all_tau(f, f_prev, f_next, v, dx, dt, tau_values):
    """
    Compute L2 errors for all τ values at a single time point.

    Args:
        f: distribution at time t (N_x, N_v)
        f_prev: distribution at time t-dt (N_x, N_v)
        f_next: distribution at time t+dt (N_x, N_v)
        v: velocity grid (N_v,)
        dx: spatial grid spacing
        dt: time step between snapshots
        tau_values: array of τ values to test (N_tau,)

    Returns:
        errors: L2 error for each τ (N_tau,)
    """
    dv = v[1] - v[0]

    # Compute moments and equilibrium
    rho, u, T = compute_moments(f, v, dv)
    f_eq = compute_maxwellian(rho, u, T, v)

    # Compute collision term: ∂f/∂t + v·∂f/∂x
    # Time derivative (central difference)
    df_dt = (f_next - f_prev) / (2 * dt)

    # Spatial derivative (central difference with periodic BC)
    df_dx = compute_spatial_derivative_periodic(f, dx)

    # Advection term: v · ∂f/∂x
    advection = v[jnp.newaxis, :] * df_dx

    # Collision term from Landau equation
    collision_term = df_dt + advection

    # BGK residual for equilibrium part
    f_neq = f_eq - f  # Non-equilibrium part

    # Compute error for each tau: || (f_eq - f)/τ - collision_term ||²
    def error_for_tau(tau):
        bgk_term = f_neq / tau
        diff = bgk_term - collision_term
        return jnp.mean(diff**2)

    errors = vmap(error_for_tau)(tau_values)
    return errors


def process_on_gpu(f_chunk, f_prev_chunk, f_next_chunk, v, dx, dt, tau_values, device_idx):
    """Process a chunk of spatial points on a specific GPU."""
    with jax.default_device(jax.devices()[device_idx]):
        f_j = jnp.array(f_chunk)
        f_prev_j = jnp.array(f_prev_chunk)
        f_next_j = jnp.array(f_next_chunk)
        v_j = jnp.array(v)
        tau_j = jnp.array(tau_values)
        errors = compute_errors_for_all_tau(f_j, f_prev_j, f_next_j, v_j, dx, dt, tau_j)
        return np.array(errors)


def optimize_tau_single_snapshot(f, f_prev, f_next, x, v, dt, tau_values, num_devices=8):
    """
    Find optimal τ for a single time snapshot using multiple GPUs.

    Splits the spatial domain across GPUs for parallel computation.
    """
    N_x = len(x)
    dx = x[1] - x[0]
    chunk_size = N_x // num_devices

    # Process each chunk on a different GPU
    all_errors = []

    for dev_idx in range(num_devices):
        start = dev_idx * chunk_size
        end = start + chunk_size if dev_idx < num_devices - 1 else N_x

        # Handle periodic boundary for spatial derivatives
        # We need the full array since we use periodic BC
        f_chunk = f
        f_prev_chunk = f_prev
        f_next_chunk = f_next

        errors = process_on_gpu(f_chunk, f_prev_chunk, f_next_chunk,
                               v, dx, dt, tau_values, dev_idx)
        all_errors.append(errors)
        break  # Actually, since periodic BC needs full array, just use one GPU per snapshot

    # Average errors across spatial chunks (but we're using full array)
    avg_errors = all_errors[0]

    # Find optimal tau
    min_idx = np.argmin(avg_errors)
    optimal_tau = tau_values[min_idx]
    min_error = avg_errors[min_idx]

    return optimal_tau, min_error, avg_errors


def optimize_tau_all_snapshots(f_history, x, v, times, tau_values, N_size=1, N_bin=1, num_devices=8):
    """
    Find optimal τ for all time snapshots.

    Args:
        f_history: (N_snapshots, N_x, N_v) distribution function history
        x: spatial grid (N_x,)
        v: velocity grid (N_v,)
        times: time points (N_snapshots,)
        tau_values: array of τ values to test
        N_size: window size for time averaging (consecutive snapshots)
        N_bin: number of windows to average for each output
        num_devices: number of GPUs

    Returns:
        optimal_taus, time_centers, min_errors, all_errors
    """
    N_snapshots = f_history.shape[0]
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    dx = x[1] - x[0]

    print(f"\nProcessing {N_snapshots} snapshots...")
    print(f"N_size={N_size}, N_bin={N_bin}")
    print(f"Time step between snapshots: dt = {dt:.6e}")
    print(f"Spatial step: dx = {dx:.6e}")

    # Window parameters
    window_total = N_size * N_bin
    if window_total > N_snapshots - 2:
        print(f"Warning: window_total={window_total} > available snapshots-2={N_snapshots-2}")
        print(f"Adjusting to per-snapshot analysis...")
        window_total = 1
        N_size = 1
        N_bin = 1

    # Number of output points
    # We need idx-1 and idx+1 for derivatives, so valid range is [1, N_snapshots-2]
    valid_snapshots = N_snapshots - 2
    N_output = valid_snapshots // window_total

    print(f"Valid snapshots for analysis: {valid_snapshots}")
    print(f"Number of output points: {N_output}")

    # Pre-compile JAX function
    print("\nJIT compiling...")
    v_jax = jnp.array(v)
    tau_jax = jnp.array(tau_values)

    # Warm up
    f_test = jnp.array(f_history[1])
    f_prev_test = jnp.array(f_history[0])
    f_next_test = jnp.array(f_history[2])
    _ = compute_errors_for_all_tau(f_test, f_prev_test, f_next_test, v_jax, dx, dt, tau_jax)
    jax.block_until_ready(_)
    print("JIT compilation done.")

    # Process snapshots
    optimal_taus = []
    min_errors = []
    all_errors_list = []
    time_centers = []

    print("\nComputing optimal τ for each time window...")

    for out_idx in tqdm(range(N_output), desc="Processing"):
        # Window indices
        window_start = 1 + out_idx * window_total  # Start from 1 (need idx-1 for derivative)
        window_end = window_start + window_total

        # Accumulate errors over the window
        window_errors = np.zeros(len(tau_values))

        for idx in range(window_start, min(window_end, N_snapshots - 1)):
            f = jnp.array(f_history[idx])
            f_prev = jnp.array(f_history[idx - 1])
            f_next = jnp.array(f_history[idx + 1])

            errors = compute_errors_for_all_tau(f, f_prev, f_next, v_jax, dx, dt, tau_jax)
            window_errors += np.array(errors)

        # Average over window
        count = min(window_end, N_snapshots - 1) - window_start
        window_errors /= count

        # Find optimal tau
        min_idx = np.argmin(window_errors)
        optimal_tau = tau_values[min_idx]
        min_error = window_errors[min_idx]

        optimal_taus.append(optimal_tau)
        min_errors.append(min_error)
        all_errors_list.append(window_errors)

        # Time center
        t_center = (times[window_start] + times[min(window_end - 1, N_snapshots - 1)]) / 2
        time_centers.append(t_center)

    return (np.array(optimal_taus), np.array(time_centers),
            np.array(min_errors), np.array(all_errors_list))


def main(N_size: int = 1, N_bin: int = 1,
         tau_min: float = 0.1, tau_max: float = 10.0, tau_step: float = 0.1,
         data_file: str = None):
    """
    Main function to run τ optimization.

    Args:
        N_size: number of snapshots per measurement
        N_bin: number of measurements to average
        tau_min, tau_max, tau_step: τ range to search
        data_file: base name of data file (without extension)
    """

    # τ values to test
    tau_values = np.arange(tau_min, tau_max + tau_step/2, tau_step)

    # Data file path
    data_dir = "data/landau_1d"
    if data_file is None:
        base_name = "landau_large_Nx65536_Nv1024_Nt131072_20260125_214408"
    else:
        base_name = data_file

    f_file = f"{data_dir}/{base_name}_f.npy"
    grid_file = f"{data_dir}/{base_name}_grid.npz"
    config_file = f"{data_dir}/{base_name}_config.json"

    print("="*70)
    print("Optimal τ (tau) Finder for BGK Approximation of Landau Solution")
    print("="*70)
    print(f"N_size = {N_size}, N_bin = {N_bin}")
    print(f"τ values: {tau_values[0]:.1f} to {tau_values[-1]:.1f} ({len(tau_values)} values)")
    print(f"Using {jax.local_device_count()} GPUs")
    print("="*70)

    # Load config
    print("\nLoading configuration...")
    with open(config_file, 'r') as f:
        config = json.load(f)
    print(f"  N_x = {config['N_x']}, N_v = {config['N_v']}, N_t = {config['N_t']}")
    print(f"  T_final = {config['T_final']}, λ_D = {config['lambda_D']}")

    # Load grid
    print("\nLoading grid data...")
    grid_data = np.load(grid_file)
    x = grid_data['x']
    v = grid_data['v']
    times = grid_data['times']
    print(f"  x: {x.shape}, v: {v.shape}, times: {times.shape}")
    print(f"  Time range: {times[0]:.6f} to {times[-1]:.6f}")

    # Load distribution function
    print("\nLoading distribution function...")
    f_history = np.load(f_file)  # Load fully into memory for GPU processing
    print(f"  Shape: {f_history.shape} (N_snapshots, N_x, N_v)")
    print(f"  Memory: {f_history.nbytes / 1e9:.2f} GB")

    N_snapshots = f_history.shape[0]
    window_total = N_size * N_bin
    expected_output = max(1, (N_snapshots - 2) // window_total)
    print(f"  Expected output points: {expected_output}")

    # Run optimization
    print("\n" + "="*70)
    print("Running τ optimization...")
    print("="*70)

    start_time = datetime.now()

    optimal_taus, time_centers, min_errors, all_errors = optimize_tau_all_snapshots(
        f_history, x, v, times, tau_values,
        N_size=N_size, N_bin=N_bin,
        num_devices=jax.local_device_count()
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nOptimization completed in {elapsed:.2f} seconds")

    # Statistics
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    print(f"Number of optimal τ values: {len(optimal_taus)}")
    print(f"τ range: {optimal_taus.min():.2f} to {optimal_taus.max():.2f}")
    print(f"τ mean: {optimal_taus.mean():.2f} ± {optimal_taus.std():.2f}")
    print(f"Error range: {min_errors.min():.2e} to {min_errors.max():.2e}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f"tau_optimal_Nsize{N_size}_Nbin{N_bin}_{timestamp}"

    # Save data
    output_file = f"{data_dir}/{output_base}.npz"
    np.savez(output_file,
             optimal_taus=optimal_taus,
             time_centers=time_centers,
             min_errors=min_errors,
             all_errors=all_errors,
             tau_values=tau_values,
             N_size=N_size,
             N_bin=N_bin,
             source_file=base_name)
    print(f"\nResults saved to {output_file}")

    # Save config
    result_config = {
        "source_file": base_name,
        "N_size": N_size,
        "N_bin": N_bin,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "tau_step": tau_step,
        "num_tau_values": len(tau_values),
        "num_output_points": len(optimal_taus),
        "tau_mean": float(optimal_taus.mean()),
        "tau_std": float(optimal_taus.std()),
        "tau_min_result": float(optimal_taus.min()),
        "tau_max_result": float(optimal_taus.max()),
        "elapsed_time_sec": elapsed,
        "timestamp": timestamp
    }
    config_output = f"{data_dir}/{output_base}_config.json"
    with open(config_output, 'w') as f:
        json.dump(result_config, f, indent=2)
    print(f"Config saved to {config_output}")

    # Plot results
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Optimal τ vs time
    ax = axes[0, 0]
    ax.plot(time_centers, optimal_taus, 'b.-', markersize=4, linewidth=1)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Optimal τ')
    ax.set_title(f'Optimal τ for BGK Approximation (N_size={N_size}, N_bin={N_bin})')
    ax.grid(True, alpha=0.3)
    ax.axhline(optimal_taus.mean(), color='r', linestyle='--',
               label=f'Mean τ = {optimal_taus.mean():.2f}')
    ax.legend()

    # (0,1) Histogram of optimal τ
    ax = axes[0, 1]
    ax.hist(optimal_taus, bins=min(30, len(np.unique(optimal_taus))),
            edgecolor='black', alpha=0.7)
    ax.axvline(optimal_taus.mean(), color='r', linestyle='--',
               label=f'Mean = {optimal_taus.mean():.2f}')
    ax.axvline(optimal_taus.mean() - optimal_taus.std(), color='orange', linestyle=':',
               label=f'Std = {optimal_taus.std():.2f}')
    ax.axvline(optimal_taus.mean() + optimal_taus.std(), color='orange', linestyle=':')
    ax.set_xlabel('Optimal τ')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Optimal τ Values')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Minimum error vs time
    ax = axes[1, 0]
    ax.semilogy(time_centers, min_errors, 'g.-', markersize=4, linewidth=1)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Minimum L2 Error')
    ax.set_title('BGK Approximation Error with Optimal τ')
    ax.grid(True, alpha=0.3)

    # (1,1) Error landscape (heatmap or average profile)
    ax = axes[1, 1]
    if len(all_errors) > 10:
        # Heatmap for many time points
        im = ax.pcolormesh(tau_values, time_centers, np.log10(all_errors + 1e-20),
                          shading='auto', cmap='viridis')
        plt.colorbar(im, ax=ax, label='log₁₀(L2 Error)')
        ax.set_xlabel('τ')
        ax.set_ylabel('Time t')
        ax.set_title('Error Landscape (log scale)')

        # Mark optimal tau for each time
        ax.plot(optimal_taus, time_centers, 'r.', markersize=2, alpha=0.5)
    else:
        # Line plot for few time points
        avg_error_profile = all_errors.mean(axis=0)
        ax.plot(tau_values, avg_error_profile, 'b-', linewidth=2)
        min_idx = np.argmin(avg_error_profile)
        ax.axvline(tau_values[min_idx], color='r', linestyle='--',
                   label=f'Global min at τ = {tau_values[min_idx]:.1f}')
        ax.set_xlabel('τ')
        ax.set_ylabel('Average L2 Error')
        ax.set_title('Error Profile Averaged Over All Time Windows')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    os.makedirs("figures/landau_1d", exist_ok=True)
    fig_file = f"figures/landau_1d/{output_base}.png"
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {fig_file}")

    # Also save a simple time series plot
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(time_centers, optimal_taus, 'b-', linewidth=1.5, marker='o', markersize=3)
    ax2.fill_between(time_centers,
                     optimal_taus - optimal_taus.std(),
                     optimal_taus + optimal_taus.std(),
                     alpha=0.2, color='blue')
    ax2.axhline(optimal_taus.mean(), color='r', linestyle='--', linewidth=2,
               label=f'Mean τ = {optimal_taus.mean():.3f} ± {optimal_taus.std():.3f}')
    ax2.set_xlabel('Time t', fontsize=12)
    ax2.set_ylabel('Optimal τ', fontsize=12)
    ax2.set_title('Time Evolution of Optimal BGK Relaxation Time τ', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig2_file = f"figures/landau_1d/{output_base}_timeseries.png"
    fig2.savefig(fig2_file, dpi=150, bbox_inches='tight')
    print(f"Time series plot saved to {fig2_file}")

    plt.show()

    return optimal_taus, time_centers, min_errors


if __name__ == "__main__":
    try:
        import fire
        fire.Fire(main)
    except ImportError:
        # Default run with N_size=1, N_bin=1 for per-snapshot analysis
        main(N_size=1, N_bin=1)
