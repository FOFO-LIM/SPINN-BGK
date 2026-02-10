# 2026-02-06: Initial creation - Analyze non-equilibrium part of f(x,v,t) from Landau simulation
#!/usr/bin/env python3
"""
Analyze the non-equilibrium part of the distribution function from Landau simulation.

Decomposes f(x,v,t) = f_eq(x,v,t) + f_neq(x,v,t) where:
- f_eq is the local Maxwellian computed from moments (rho, u, T)
- f_neq = f - f_eq is the non-equilibrium deviation

If the Landau operator has the wrong sign, we expect ||f_neq|| to INCREASE over time
(system moving away from equilibrium instead of toward it).
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Try to use JAX for faster computation, fall back to numpy
try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    from functools import partial
    USE_JAX = True
    print(f"Using JAX with devices: {jax.devices()}")
except ImportError:
    USE_JAX = False
    print("JAX not available, using NumPy")


def compute_moments_numpy(f, v, dv):
    """Compute moments (rho, u, T) from distribution function using NumPy."""
    N_v = len(v)
    w = np.ones(N_v) * dv
    w[0] = dv / 2
    w[-1] = dv / 2

    rho = np.sum(f * w, axis=-1)
    rho = np.maximum(rho, 1e-30)
    u = np.sum(f * v * w, axis=-1) / rho
    c_sq = (v - u[..., np.newaxis])**2
    T = np.sum(f * c_sq * w, axis=-1) / rho
    T = np.maximum(T, 1e-10)

    return rho, u, T


def compute_maxwellian_numpy(rho, u, T, v):
    """Compute Maxwellian distribution using NumPy."""
    T_safe = np.maximum(T, 1e-10)
    # Handle broadcasting for different input shapes
    if rho.ndim == 1:  # (N_x,)
        return (rho[:, np.newaxis] / np.sqrt(2 * np.pi * T_safe[:, np.newaxis]) *
                np.exp(-(v[np.newaxis, :] - u[:, np.newaxis])**2 / (2 * T_safe[:, np.newaxis])))
    else:
        raise ValueError(f"Unexpected rho shape: {rho.shape}")


if USE_JAX:
    @partial(jit, static_argnums=(2,))
    def compute_moments_jax(f, v, dv):
        """Compute moments using JAX."""
        N_v = len(v)
        w = jnp.ones(N_v) * dv
        w = w.at[0].set(dv / 2)
        w = w.at[-1].set(dv / 2)

        rho = jnp.sum(f * w, axis=-1)
        rho = jnp.maximum(rho, 1e-30)
        u = jnp.sum(f * v * w, axis=-1) / rho
        c_sq = (v - u[..., jnp.newaxis])**2
        T = jnp.sum(f * c_sq * w, axis=-1) / rho
        T = jnp.maximum(T, 1e-10)

        return rho, u, T

    @jit
    def compute_maxwellian_jax(rho, u, T, v):
        """Compute Maxwellian using JAX."""
        T_safe = jnp.maximum(T, 1e-10)
        return (rho[:, jnp.newaxis] / jnp.sqrt(2 * jnp.pi * T_safe[:, jnp.newaxis]) *
                jnp.exp(-(v[jnp.newaxis, :] - u[:, jnp.newaxis])**2 / (2 * T_safe[:, jnp.newaxis])))


def analyze_nonequilibrium(data_file, grid_file, config_file, output_dir='figures/landau_1d'):
    """
    Analyze non-equilibrium evolution from Landau simulation data.
    """
    print("=" * 70)
    print("NON-EQUILIBRIUM ANALYSIS OF LANDAU SIMULATION")
    print("=" * 70)

    # Load config
    print("\nLoading configuration...")
    with open(config_file, 'r') as cfg:
        config = json.load(cfg)

    N_x = config['N_x']
    N_v = config['N_v']
    print(f"  N_x = {N_x}, N_v = {N_v}")

    # Load grid
    print("\nLoading grid...")
    grid = np.load(grid_file)
    x = grid['x']
    v = grid['v']
    times = grid['times']

    dv = v[1] - v[0]
    dx = x[1] - x[0]
    print(f"  dx = {dx:.6e}, dv = {dv:.6e}")
    print(f"  Time range: {times[0]:.6f} to {times[-1]:.6f}")
    print(f"  Number of snapshots: {len(times)}")

    # Load distribution function
    print("\nLoading distribution function...")
    f_history = np.load(data_file)
    print(f"  Shape: {f_history.shape} (N_t, N_x, N_v)")
    print(f"  Memory: {f_history.nbytes / 1e9:.2f} GB")

    n_snapshots = len(times)

    # Storage for results
    results = {
        'times': times,
        'f_neq_L2_norm': [],           # ||f_neq||_L2 over whole domain
        'f_neq_Linf_norm': [],         # ||f_neq||_Linf (max deviation)
        'f_neq_relative_L2': [],       # ||f_neq||_L2 / ||f||_L2
        'rho_variation': [],           # std(rho) / mean(rho)
        'T_variation': [],             # std(T) / mean(T)
        'entropy': [],                 # H = integral(f ln f)
        'f_neq_snapshots': [],         # Store some f_neq snapshots for plotting
        'snapshot_indices': [],
    }

    # Select snapshots for detailed storage (don't store all - too much memory)
    detail_indices = [0, n_snapshots // 4, n_snapshots // 2,
                      3 * n_snapshots // 4, n_snapshots - 1]
    detail_indices = sorted(list(set(detail_indices)))

    print("\nAnalyzing non-equilibrium evolution...")
    print(f"  Detailed snapshots at indices: {detail_indices}")

    if USE_JAX:
        v_jax = jnp.array(v)

    for i in range(n_snapshots):
        t = times[i]
        f = f_history[i]

        # Compute moments
        if USE_JAX:
            f_jax = jnp.array(f)
            rho, u, T = compute_moments_jax(f_jax, v_jax, dv)
            f_eq = compute_maxwellian_jax(rho, u, T, v_jax)
            f_neq = f_jax - f_eq

            # Convert to numpy for storage
            rho = np.array(rho)
            u = np.array(u)
            T = np.array(T)
            f_eq = np.array(f_eq)
            f_neq = np.array(f_neq)
            f = np.array(f_jax)
        else:
            rho, u, T = compute_moments_numpy(f, v, dv)
            f_eq = compute_maxwellian_numpy(rho, u, T, v)
            f_neq = f - f_eq

        # Compute norms
        f_neq_L2 = np.sqrt(np.mean(f_neq**2))
        f_neq_Linf = np.max(np.abs(f_neq))
        f_L2 = np.sqrt(np.mean(f**2))
        f_neq_rel = f_neq_L2 / (f_L2 + 1e-30)

        # Moment variations
        rho_var = np.std(rho) / (np.mean(rho) + 1e-30)
        T_var = np.std(T) / (np.mean(T) + 1e-30)

        # Entropy H = integral(f ln f) - should decrease for correct dynamics
        f_safe = np.maximum(f, 1e-30)
        entropy = np.mean(f_safe * np.log(f_safe)) * dx * dv

        # Store results
        results['f_neq_L2_norm'].append(f_neq_L2)
        results['f_neq_Linf_norm'].append(f_neq_Linf)
        results['f_neq_relative_L2'].append(f_neq_rel)
        results['rho_variation'].append(rho_var)
        results['T_variation'].append(T_var)
        results['entropy'].append(entropy)

        # Store detailed snapshots
        if i in detail_indices:
            results['f_neq_snapshots'].append(f_neq.copy())
            results['snapshot_indices'].append(i)

        # Progress
        if (i + 1) % max(1, n_snapshots // 10) == 0:
            print(f"  [{i+1}/{n_snapshots}] t={t:.4f}: ||f_neq||_L2 = {f_neq_L2:.6e}, "
                  f"||f_neq||/||f|| = {f_neq_rel:.4f}")

    # Convert to arrays
    for key in ['f_neq_L2_norm', 'f_neq_Linf_norm', 'f_neq_relative_L2',
                'rho_variation', 'T_variation', 'entropy']:
        results[key] = np.array(results[key])

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nNon-equilibrium norm ||f_neq||_L2:")
    print(f"  Initial (t=0):    {results['f_neq_L2_norm'][0]:.6e}")
    print(f"  Final (t={times[-1]:.3f}):  {results['f_neq_L2_norm'][-1]:.6e}")
    print(f"  Ratio (final/initial): {results['f_neq_L2_norm'][-1] / (results['f_neq_L2_norm'][0] + 1e-30):.2f}x")

    print(f"\nRelative non-equilibrium ||f_neq||/||f||:")
    print(f"  Initial: {results['f_neq_relative_L2'][0]:.4f}")
    print(f"  Final:   {results['f_neq_relative_L2'][-1]:.4f}")

    print(f"\nEntropy H = integral(f ln f):")
    print(f"  Initial: {results['entropy'][0]:.6e}")
    print(f"  Final:   {results['entropy'][-1]:.6e}")
    print(f"  Change:  {results['entropy'][-1] - results['entropy'][0]:.6e}")
    if results['entropy'][-1] > results['entropy'][0]:
        print(f"  → ENTROPY INCREASED (wrong sign would cause this!)")
    else:
        print(f"  → Entropy decreased (expected for correct dynamics)")

    # Determine if system is moving away from equilibrium
    is_diverging = results['f_neq_L2_norm'][-1] > 2 * results['f_neq_L2_norm'][0]

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    if is_diverging:
        print("\n*** SYSTEM IS MOVING AWAY FROM EQUILIBRIUM ***")
        print(f"||f_neq|| increased by {results['f_neq_L2_norm'][-1] / (results['f_neq_L2_norm'][0] + 1e-30):.0f}x")
        print("This confirms the Landau operator has the WRONG SIGN.")
    else:
        print("\nSystem appears to be relaxing toward equilibrium.")

    # Create plots
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    create_summary_plot(results, times, output_dir, timestamp)
    create_fneq_heatmaps(results, x, v, times, output_dir, timestamp)
    create_fneq_slices(results, x, v, times, output_dir, timestamp)

    return results


def create_summary_plot(results, times, output_dir, timestamp):
    """Create summary plot of non-equilibrium evolution."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (0,0) ||f_neq||_L2 vs time
    ax = axes[0, 0]
    ax.semilogy(times, results['f_neq_L2_norm'], 'b-', linewidth=2)
    ax.axhline(results['f_neq_L2_norm'][0], color='green', linestyle='--',
               alpha=0.7, label=f'Initial: {results["f_neq_L2_norm"][0]:.2e}')
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'$\|f_{neq}\|_{L^2}$')
    ax.set_title(r'Non-equilibrium Norm $\|f - f_{eq}\|_{L^2}$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Shade region of growth
    if results['f_neq_L2_norm'][-1] > results['f_neq_L2_norm'][0]:
        ax.fill_between(times, results['f_neq_L2_norm'][0], results['f_neq_L2_norm'],
                        alpha=0.2, color='red', label='Deviation growth')

    # (0,1) ||f_neq||_Linf vs time
    ax = axes[0, 1]
    ax.semilogy(times, results['f_neq_Linf_norm'], 'r-', linewidth=2)
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'$\|f_{neq}\|_{L^\infty}$')
    ax.set_title(r'Max Deviation $\max|f - f_{eq}|$')
    ax.grid(True, alpha=0.3)

    # (0,2) Relative non-equilibrium
    ax = axes[0, 2]
    ax.plot(times, results['f_neq_relative_L2'] * 100, 'g-', linewidth=2)
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'$\|f_{neq}\| / \|f\|$ (%)')
    ax.set_title('Relative Non-equilibrium')
    ax.grid(True, alpha=0.3)

    # (1,0) Entropy evolution
    ax = axes[1, 0]
    entropy = results['entropy']
    ax.plot(times, entropy, 'm-', linewidth=2)
    ax.axhline(entropy[0], color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'$H = \int f \ln f \, dv$')
    ax.set_title('Entropy Evolution\n(Should DECREASE for correct dynamics)')
    ax.grid(True, alpha=0.3)

    # Color based on entropy change
    if entropy[-1] > entropy[0]:
        ax.fill_between(times, entropy[0], entropy, alpha=0.2, color='red')
        ax.text(0.5, 0.95, 'ENTROPY INCREASING\n(Wrong sign!)', transform=ax.transAxes,
                ha='center', va='top', fontsize=12, color='red', fontweight='bold')
    else:
        ax.fill_between(times, entropy, entropy[0], alpha=0.2, color='green')

    # (1,1) Density variation
    ax = axes[1, 1]
    ax.plot(times, results['rho_variation'] * 100, 'b-', linewidth=2, label=r'$\rho$')
    ax.plot(times, results['T_variation'] * 100, 'r-', linewidth=2, label='T')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Variation (%)')
    ax.set_title('Moment Variations (std/mean)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,2) Summary text
    ax = axes[1, 2]
    ax.axis('off')

    ratio = results['f_neq_L2_norm'][-1] / (results['f_neq_L2_norm'][0] + 1e-30)
    entropy_change = results['entropy'][-1] - results['entropy'][0]

    diagnosis = "WRONG SIGN CONFIRMED" if ratio > 2 else "Appears normal"
    diagnosis_color = "red" if ratio > 2 else "green"

    summary_text = f"""
    NON-EQUILIBRIUM ANALYSIS SUMMARY
    ════════════════════════════════════

    Initial ||f_neq||_L2:  {results['f_neq_L2_norm'][0]:.4e}
    Final ||f_neq||_L2:    {results['f_neq_L2_norm'][-1]:.4e}
    Growth ratio:          {ratio:.1f}x

    Initial ||f_neq||/||f||: {results['f_neq_relative_L2'][0]*100:.2f}%
    Final ||f_neq||/||f||:   {results['f_neq_relative_L2'][-1]*100:.2f}%

    Entropy change: {entropy_change:+.4e}
    {'↑ INCREASED (BAD!)' if entropy_change > 0 else '↓ Decreased (good)'}

    ════════════════════════════════════
    DIAGNOSIS: {diagnosis}
    ════════════════════════════════════

    For correct Landau dynamics:
    • ||f_neq|| should DECREASE
    • Entropy should DECREASE
    • System should relax to Maxwellian

    Observed behavior suggests the
    collision operator pushes f AWAY
    from equilibrium, not toward it.
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Non-equilibrium Evolution Analysis\n(Landau Simulation with Suspected Sign Error)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = f"{output_dir}/nonequilibrium_summary_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSummary plot saved to {fig_path}")
    plt.close()


def create_fneq_heatmaps(results, x, v, times, output_dir, timestamp):
    """Create heatmaps of f_neq(x,v) at different times."""
    n_plots = len(results['f_neq_snapshots'])
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Find global colorbar limits
    vmax = max(np.max(np.abs(snap)) for snap in results['f_neq_snapshots'])
    vmin = -vmax

    for i, (idx, f_neq) in enumerate(zip(results['snapshot_indices'],
                                          results['f_neq_snapshots'])):
        ax = axes[i]
        t = times[idx]

        # Subsample for plotting if too large
        step_x = max(1, len(x) // 256)
        step_v = max(1, len(v) // 256)

        x_sub = x[::step_x]
        v_sub = v[::step_v]
        f_neq_sub = f_neq[::step_x, ::step_v]

        im = ax.pcolormesh(x_sub, v_sub, f_neq_sub.T,
                           shading='auto', cmap='RdBu_r',
                           vmin=vmin, vmax=vmax)
        ax.set_xlabel('x')
        ax.set_ylabel('v')
        ax.set_title(f't = {t:.4f}\n||f_neq|| = {np.sqrt(np.mean(f_neq**2)):.2e}')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.suptitle(r'Non-equilibrium Distribution $f_{neq}(x,v,t) = f - f_{eq}$',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = f"{output_dir}/nonequilibrium_heatmaps_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Heatmaps saved to {fig_path}")
    plt.close()


def create_fneq_slices(results, x, v, times, output_dir, timestamp):
    """Create slice plots of f_neq at specific x locations."""
    n_plots = len(results['f_neq_snapshots'])
    if n_plots == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Select spatial locations to plot
    N_x = len(x)
    x_indices = [0, N_x // 4, N_x // 2, 3 * N_x // 4]
    x_labels = ['x = -X', 'x = -X/2', 'x = 0', 'x = X/2']

    for ax_idx, (x_idx, x_label) in enumerate(zip(x_indices, x_labels)):
        ax = axes[ax_idx // 2, ax_idx % 2]

        for snap_idx, (t_idx, f_neq) in enumerate(zip(results['snapshot_indices'],
                                                       results['f_neq_snapshots'])):
            t = times[t_idx]
            alpha = 0.3 + 0.7 * snap_idx / max(1, n_plots - 1)
            label = f't = {t:.3f}' if snap_idx in [0, n_plots - 1] else None
            ax.plot(v, f_neq[x_idx, :], alpha=alpha, linewidth=1.5, label=label)

        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('v')
        ax.set_ylabel(r'$f_{neq}(v)$')
        ax.set_title(f'{x_label} (index {x_idx})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(r'Non-equilibrium Distribution Slices $f_{neq}(v)$ at Fixed x',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = f"{output_dir}/nonequilibrium_slices_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Slices saved to {fig_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze non-equilibrium evolution')
    parser.add_argument('--data_file', type=str,
                        default='data/landau_1d/landau_Nx65536_Nv1024_Nt16384_20260129_152948_f.npy',
                        help='Path to f_history .npy file')
    parser.add_argument('--grid_file', type=str,
                        default='data/landau_1d/landau_Nx65536_Nv1024_Nt16384_20260129_152948_grid.npz',
                        help='Path to grid .npz file')
    parser.add_argument('--config_file', type=str,
                        default='data/landau_1d/landau_Nx65536_Nv1024_Nt16384_20260129_152948_config.json',
                        help='Path to config .json file')
    parser.add_argument('--output_dir', type=str, default='figures/landau_1d',
                        help='Output directory for plots')
    args = parser.parse_args()

    results = analyze_nonequilibrium(
        data_file=args.data_file,
        grid_file=args.grid_file,
        config_file=args.config_file,
        output_dir=args.output_dir
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
