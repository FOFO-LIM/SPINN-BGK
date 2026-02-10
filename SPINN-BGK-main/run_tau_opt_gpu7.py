#!/usr/bin/env python3
"""
Wrapper script to run tau optimization on GPU 7 only.
"""
import os
# MUST set before importing JAX
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import jax
import jax.numpy as jnp
from jax import jit, vmap
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
    """Compute Maxwellian distribution."""
    rho = rho[:, jnp.newaxis]
    u = u[:, jnp.newaxis]
    T = T[:, jnp.newaxis]
    T = jnp.maximum(T, 1e-10)
    v_grid = v[jnp.newaxis, :]
    return rho / jnp.sqrt(2 * jnp.pi * T) * jnp.exp(-(v_grid - u)**2 / (2 * T))


@jit
def compute_moments(f, v, dv):
    """Compute macroscopic moments from distribution function."""
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


@partial(jit, static_argnums=())
def compute_errors_for_all_tau(f, f_prev, f_next, v, dx, dt, tau_values):
    """Compute L2 errors for all τ values at a single time point."""
    dv = v[1] - v[0]

    rho, u, T = compute_moments(f, v, dv)
    f_eq = compute_maxwellian(rho, u, T, v)

    df_dt = (f_next - f_prev) / (2 * dt)
    df_dx = compute_spatial_derivative_periodic(f, dx)
    advection = v[jnp.newaxis, :] * df_dx
    collision_term = df_dt + advection

    f_neq = f_eq - f

    def error_for_tau(tau):
        bgk_term = f_neq / tau
        diff = bgk_term - collision_term
        return jnp.mean(diff**2)

    errors = vmap(error_for_tau)(tau_values)
    return errors


def optimize_tau_all_snapshots(f_history, x, v, times, tau_values, N_size=1, N_bin=1):
    """Find optimal τ for all time snapshots."""
    N_snapshots = f_history.shape[0]
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    dx = x[1] - x[0]

    print(f"\nProcessing {N_snapshots} snapshots...")
    print(f"N_size={N_size}, N_bin={N_bin}")
    print(f"Time step between snapshots: dt = {dt:.6e}")
    print(f"Spatial step: dx = {dx:.6e}")

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

    print("\nJIT compiling...")
    v_jax = jnp.array(v)
    tau_jax = jnp.array(tau_values)

    f_test = jnp.array(f_history[1])
    f_prev_test = jnp.array(f_history[0])
    f_next_test = jnp.array(f_history[2])
    _ = compute_errors_for_all_tau(f_test, f_prev_test, f_next_test, v_jax, dx, dt, tau_jax)
    jax.block_until_ready(_)
    print("JIT compilation done.")

    optimal_taus = []
    min_errors = []
    all_errors_list = []
    time_centers = []

    print("\nComputing optimal τ for each time window...")

    for out_idx in tqdm(range(N_output), desc="Processing"):
        window_start = 1 + out_idx * window_total
        window_end = window_start + window_total

        window_errors = np.zeros(len(tau_values))

        for idx in range(window_start, min(window_end, N_snapshots - 1)):
            f = jnp.array(f_history[idx])
            f_prev = jnp.array(f_history[idx - 1])
            f_next = jnp.array(f_history[idx + 1])

            errors = compute_errors_for_all_tau(f, f_prev, f_next, v_jax, dx, dt, tau_jax)
            window_errors += np.array(errors)

        count = min(window_end, N_snapshots - 1) - window_start
        window_errors /= count

        min_idx = np.argmin(window_errors)
        optimal_tau = tau_values[min_idx]
        min_error = window_errors[min_idx]

        optimal_taus.append(optimal_tau)
        min_errors.append(min_error)
        all_errors_list.append(window_errors)

        t_center = (times[window_start] + times[min(window_end - 1, N_snapshots - 1)]) / 2
        time_centers.append(t_center)

    return (np.array(optimal_taus), np.array(time_centers),
            np.array(min_errors), np.array(all_errors_list))


def main():
    """Main function - runs with N_size=32, N_bin=32 on GPU 7."""
    N_size = 32
    N_bin = 32
    tau_min = 0.1
    tau_max = 10.0
    tau_step = 0.1

    tau_values = np.arange(tau_min, tau_max + tau_step/2, tau_step)

    data_dir = "data/landau_1d"
    base_name = "landau_large_Nx65536_Nv1024_Nt131072_20260125_214408"

    f_file = f"{data_dir}/{base_name}_f.npy"
    grid_file = f"{data_dir}/{base_name}_grid.npz"
    config_file = f"{data_dir}/{base_name}_config.json"

    print("="*70)
    print("Optimal τ (tau) Finder for BGK Approximation of Landau Solution")
    print("="*70)
    print(f"N_size = {N_size}, N_bin = {N_bin}")
    print(f"τ values: {tau_values[0]:.1f} to {tau_values[-1]:.1f} ({len(tau_values)} values)")
    print(f"Using {jax.local_device_count()} GPU(s)")
    print("="*70)

    print("\nLoading configuration...")
    with open(config_file, 'r') as f:
        config = json.load(f)
    print(f"  N_x = {config['N_x']}, N_v = {config['N_v']}, N_t = {config['N_t']}")
    print(f"  T_final = {config['T_final']}, λ_D = {config['lambda_D']}")

    print("\nLoading grid data...")
    grid_data = np.load(grid_file)
    x = grid_data['x']
    v = grid_data['v']
    times = grid_data['times']
    print(f"  x: {x.shape}, v: {v.shape}, times: {times.shape}")
    print(f"  Time range: {times[0]:.6f} to {times[-1]:.6f}")

    print("\nLoading distribution function...")
    f_history = np.load(f_file)
    print(f"  Shape: {f_history.shape} (N_snapshots, N_x, N_v)")
    print(f"  Memory: {f_history.nbytes / 1e9:.2f} GB")

    N_snapshots = f_history.shape[0]
    window_total = N_size * N_bin
    expected_output = max(1, (N_snapshots - 2) // window_total)
    print(f"  Expected output points: {expected_output}")

    print("\n" + "="*70)
    print("Running τ optimization...")
    print("="*70)

    start_time = datetime.now()

    optimal_taus, time_centers, min_errors, all_errors = optimize_tau_all_snapshots(
        f_history, x, v, times, tau_values,
        N_size=N_size, N_bin=N_bin
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nOptimization completed in {elapsed:.2f} seconds")

    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    print(f"Number of optimal τ values: {len(optimal_taus)}")
    print(f"τ range: {optimal_taus.min():.2f} to {optimal_taus.max():.2f}")
    print(f"τ mean: {optimal_taus.mean():.2f} ± {optimal_taus.std():.2f}")
    print(f"Error range: {min_errors.min():.2e} to {min_errors.max():.2e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f"tau_optimal_Nsize{N_size}_Nbin{N_bin}_{timestamp}"

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

    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(time_centers, optimal_taus, 'b.-', markersize=4, linewidth=1)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Optimal τ')
    ax.set_title(f'Optimal τ for BGK Approximation (N_size={N_size}, N_bin={N_bin})')
    ax.grid(True, alpha=0.3)
    ax.axhline(optimal_taus.mean(), color='r', linestyle='--',
               label=f'Mean τ = {optimal_taus.mean():.2f}')
    ax.legend()

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

    ax = axes[1, 0]
    ax.semilogy(time_centers, min_errors, 'g.-', markersize=4, linewidth=1)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Minimum L2 Error')
    ax.set_title('BGK Approximation Error with Optimal τ')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if len(all_errors) > 10:
        im = ax.pcolormesh(tau_values, time_centers, np.log10(all_errors + 1e-20),
                          shading='auto', cmap='viridis')
        plt.colorbar(im, ax=ax, label='log₁₀(L2 Error)')
        ax.set_xlabel('τ')
        ax.set_ylabel('Time t')
        ax.set_title('Error Landscape (log scale)')
        ax.plot(optimal_taus, time_centers, 'r.', markersize=2, alpha=0.5)
    else:
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

    os.makedirs("figures/landau_1d", exist_ok=True)
    fig_file = f"figures/landau_1d/{output_base}.png"
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {fig_file}")

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

    print("\nDone!")

    return optimal_taus, time_centers, min_errors


if __name__ == "__main__":
    main()
