#!/usr/bin/env python3
"""
Run Landau simulation and optimize both ES-BGK and Shakhov parameters.

Usage:
    CUDA_VISIBLE_DEVICES=7 python run_simulation_shakhov.py
"""

import numpy as np
import os
import json
from datetime import datetime

# Set up JAX before importing
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from landau_1d_numerical_jax import LandauSolver1D_JAX
from optimize_esbgk import (
    optimize_esbgk, compute_moments as compute_moments_esbgk,
    plot_esbgk_optimization
)
from optimize_shakhov import (
    optimize_shakhov, compute_moments, compute_heat_flux,
    plot_shakhov_optimization
)
import matplotlib.pyplot as plt


def main():
    # Parameters
    N_x = 2**16  # 65536
    N_v = 2**10  # 1024
    N_t = 2**13  # 8192

    X = 0.5
    V = 6.0
    T_final = 0.1
    lambda_D = 10.0

    # Save every N steps (adjust based on memory)
    save_every = N_t // 64  # Save 64 snapshots

    print("=" * 70)
    print("Landau Simulation + ES-BGK + Shakhov Optimization")
    print("=" * 70)
    print(f"Grid: N_x={N_x}, N_v={N_v}, N_t={N_t}")
    print(f"Save every: {save_every} steps ({N_t // save_every} snapshots)")
    print(f"Domain: x in [-{X}, {X}], v in [-{V}, {V}], t in [0, {T_final}]")
    print(f"Debye length: lambda_D = {lambda_D}")
    print("=" * 70)

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

    # Run simulation
    print("\n[1/4] Running Landau simulation...")
    results = solver.solve(save_every=save_every, verbose=True)

    # Save simulation data
    os.makedirs("data/landau_1d", exist_ok=True)
    os.makedirs("figures/landau_1d", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"landau_Nx{N_x}_Nv{N_v}_Nt{N_t}_{timestamp}"

    # Save f_history
    f_file = f"data/landau_1d/{base_name}_f.npy"
    np.save(f_file, results['f_history'])
    print(f"\nSaved f_history to {f_file}")
    print(f"  Shape: {results['f_history'].shape}")

    # Save grid
    grid_file = f"data/landau_1d/{base_name}_grid.npz"
    np.savez(grid_file,
             x=results['x'],
             v=results['v'],
             times=results['times'],
             rho_history=results['rho_history'],
             u_history=results['u_history'],
             T_history=results['T_history'])
    print(f"Saved grid to {grid_file}")

    # =============================================
    # ES-BGK and Shakhov Optimization
    # =============================================
    print("\n[2/4] Computing collision operator...")

    f_history = results['f_history']
    v = results['v']
    dv = solver.dv
    times = results['times']

    n_optimize = min(10, len(times) - 1)
    indices = np.linspace(1, len(times) - 1, n_optimize, dtype=int)

    esbgk_results = []
    shakhov_results = []

    print("\n[3/4] Optimizing ES-BGK and Shakhov at each snapshot...")

    for idx in indices:
        print(f"\n  Snapshot {idx}/{len(times)-1}, t = {times[idx]:.4f}")
        print("  " + "-" * 50)

        f = f_history[idx]

        # Approximate Q_landau from time derivative
        if idx > 0:
            dt_snapshot = times[idx] - times[idx-1]
            Q_approx = (f_history[idx] - f_history[idx-1]) / dt_snapshot
        else:
            Q_approx = np.zeros_like(f)

        # Compute moments
        rho, u, T = compute_moments(f, v, dv)

        # Compute heat flux for Shakhov
        q = compute_heat_flux(f, v, dv, u)

        # ---- ES-BGK Optimization ----
        esbgk_result = optimize_esbgk(f, Q_approx, rho, u, T, v, norm='L2', n_grid=30)
        esbgk_result['time'] = times[idx]
        esbgk_result['snapshot_idx'] = idx
        esbgk_results.append(esbgk_result)

        print(f"    ES-BGK:  tau={esbgk_result['tau_opt']:.4f}, "
              f"alpha={esbgk_result['alpha_opt']:.4f}, "
              f"improvement={esbgk_result['improvement']:.2f}%")

        # ---- Shakhov Optimization ----
        shakhov_result = optimize_shakhov(f, Q_approx, rho, u, T, v, dv, norm='L2', n_grid=30)
        shakhov_result['time'] = times[idx]
        shakhov_result['snapshot_idx'] = idx
        shakhov_results.append(shakhov_result)

        print(f"    Shakhov: tau={shakhov_result['tau_opt']:.4f}, "
              f"Pr={shakhov_result['Pr_opt']:.4f}, "
              f"improvement={shakhov_result['improvement']:.2f}%")

        # Heat flux info
        print(f"    Heat flux: mean|q|={np.mean(np.abs(q)):.4e}, max|q|={np.max(np.abs(q)):.4e}")

    # =============================================
    # Save optimization results
    # =============================================
    print("\n[4/4] Saving results...")

    # ES-BGK results
    esbgk_file = f"data/landau_1d/{base_name}_esbgk_opt.npz"
    np.savez(esbgk_file,
             times=np.array([r['time'] for r in esbgk_results]),
             tau_esbgk=np.array([r['tau_opt'] for r in esbgk_results]),
             alpha_esbgk=np.array([r['alpha_opt'] for r in esbgk_results]),
             tau_bgk=np.array([r['tau_bgk'] for r in esbgk_results]),
             error_esbgk=np.array([r['error_esbgk'] for r in esbgk_results]),
             error_bgk=np.array([r['error_bgk'] for r in esbgk_results]),
             improvement=np.array([r['improvement'] for r in esbgk_results]))
    print(f"Saved ES-BGK results to {esbgk_file}")

    # Shakhov results
    shakhov_file = f"data/landau_1d/{base_name}_shakhov_opt.npz"
    np.savez(shakhov_file,
             times=np.array([r['time'] for r in shakhov_results]),
             tau_shakhov=np.array([r['tau_opt'] for r in shakhov_results]),
             Pr_shakhov=np.array([r['Pr_opt'] for r in shakhov_results]),
             tau_bgk=np.array([r['tau_bgk'] for r in shakhov_results]),
             error_shakhov=np.array([r['error_shakhov'] for r in shakhov_results]),
             error_bgk=np.array([r['error_bgk'] for r in shakhov_results]),
             improvement=np.array([r['improvement'] for r in shakhov_results]),
             heat_flux_mean=np.array([np.mean(np.abs(r['heat_flux'])) for r in shakhov_results]),
             heat_flux_max=np.array([np.max(np.abs(r['heat_flux'])) for r in shakhov_results]))
    print(f"Saved Shakhov results to {shakhov_file}")

    # =============================================
    # Plot ES-BGK results
    # =============================================
    fig = plot_esbgk_optimization(
        esbgk_results[-1],
        f"figures/landau_1d/{base_name}_esbgk_landscape.png"
    )
    plt.close(fig)

    # =============================================
    # Plot Shakhov results
    # =============================================
    fig = plot_shakhov_optimization(
        shakhov_results[-1],
        f"figures/landau_1d/{base_name}_shakhov_landscape.png"
    )
    plt.close(fig)

    # =============================================
    # Combined comparison plot
    # =============================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    opt_times = [r['time'] for r in esbgk_results]

    # tau comparison (all 3 models)
    ax = axes[0, 0]
    ax.semilogy(opt_times, [r['tau_bgk'] for r in esbgk_results], 'b-o', label='BGK', linewidth=2)
    ax.semilogy(opt_times, [r['tau_opt'] for r in esbgk_results], 'g-s', label='ES-BGK', linewidth=2)
    ax.semilogy(opt_times, [r['tau_opt'] for r in shakhov_results], 'orange', marker='^',
                linestyle='-', label='Shakhov', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Optimal tau')
    ax.set_title('Optimal Relaxation Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ES-BGK alpha
    ax = axes[0, 1]
    ax.plot(opt_times, [r['alpha_opt'] for r in esbgk_results], 'g-s', linewidth=2)
    ax.axhline(1.0, color='b', linestyle='--', label='alpha=1 (BGK)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Optimal alpha')
    ax.set_title('ES-BGK Temperature Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Shakhov Pr
    ax = axes[0, 2]
    ax.plot(opt_times, [r['Pr_opt'] for r in shakhov_results], 'orange', marker='^',
            linestyle='-', linewidth=2)
    ax.axhline(1.0, color='b', linestyle='--', label='Pr=1 (BGK)')
    ax.axhline(2/3, color='green', linestyle=':', linewidth=2, label='Pr=2/3 (physical)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Optimal Pr')
    ax.set_title('Shakhov Prandtl Number')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error comparison (all 3 models)
    ax = axes[1, 0]
    ax.semilogy(opt_times, [r['error_bgk'] for r in esbgk_results], 'b-o', label='BGK', linewidth=2)
    ax.semilogy(opt_times, [r['error_esbgk'] for r in esbgk_results], 'g-s', label='ES-BGK', linewidth=2)
    ax.semilogy(opt_times, [r['error_shakhov'] for r in shakhov_results], 'orange', marker='^',
                linestyle='-', label='Shakhov', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Error (L2)')
    ax.set_title('Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Improvement comparison
    ax = axes[1, 1]
    width = 0.35
    x = np.arange(len(opt_times))
    ax.bar(x - width/2, [r['improvement'] for r in esbgk_results], width,
           label='ES-BGK', color='green', alpha=0.7)
    ax.bar(x + width/2, [r['improvement'] for r in shakhov_results], width,
           label='Shakhov', color='orange', alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Snapshot')
    ax.set_ylabel('Improvement over BGK (%)')
    ax.set_title('Model Improvements')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Heat flux vs Pr correlation
    ax = axes[1, 2]
    heat_flux_mean = [np.mean(np.abs(r['heat_flux'])) for r in shakhov_results]
    sc = ax.scatter(heat_flux_mean, [r['Pr_opt'] for r in shakhov_results],
                    c=opt_times, cmap='viridis', s=100, edgecolor='black')
    ax.axhline(2/3, color='green', linestyle=':', linewidth=2, label='Pr=2/3')
    ax.axhline(1.0, color='blue', linestyle='--', label='Pr=1')
    ax.set_xlabel('Mean |q| (heat flux)')
    ax.set_ylabel('Optimal Pr')
    ax.set_title('Shakhov: Pr vs Heat Flux')
    plt.colorbar(sc, ax=ax, label='Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_file = f"figures/landau_1d/{base_name}_comparison.png"
    plt.savefig(comparison_file, dpi=150)
    print(f"Saved comparison plot to {comparison_file}")
    plt.close(fig)

    # =============================================
    # Summary
    # =============================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Simulation time: {results['elapsed_time']:.2f} seconds")
    print()
    print("BGK:")
    print(f"  Average tau: {np.mean([r['tau_bgk'] for r in esbgk_results]):.4f}")
    print()
    print("ES-BGK:")
    print(f"  Average tau: {np.mean([r['tau_opt'] for r in esbgk_results]):.4f}")
    print(f"  Average alpha: {np.mean([r['alpha_opt'] for r in esbgk_results]):.4f}")
    print(f"  Average improvement: {np.mean([r['improvement'] for r in esbgk_results]):.2f}%")
    print()
    print("Shakhov:")
    print(f"  Average tau: {np.mean([r['tau_opt'] for r in shakhov_results]):.4f}")
    print(f"  Average Pr: {np.mean([r['Pr_opt'] for r in shakhov_results]):.4f}")
    print(f"  Average improvement: {np.mean([r['improvement'] for r in shakhov_results]):.2f}%")
    print("=" * 70)

    # Save config
    config = {
        'N_x': N_x,
        'N_v': N_v,
        'N_t': N_t,
        'X': X,
        'V': V,
        'T_final': T_final,
        'lambda_D': lambda_D,
        'save_every': save_every,
        'elapsed_time': results['elapsed_time'],
        'esbgk': {
            'avg_tau': float(np.mean([r['tau_opt'] for r in esbgk_results])),
            'avg_alpha': float(np.mean([r['alpha_opt'] for r in esbgk_results])),
            'avg_improvement': float(np.mean([r['improvement'] for r in esbgk_results])),
        },
        'shakhov': {
            'avg_tau': float(np.mean([r['tau_opt'] for r in shakhov_results])),
            'avg_Pr': float(np.mean([r['Pr_opt'] for r in shakhov_results])),
            'avg_improvement': float(np.mean([r['improvement'] for r in shakhov_results])),
        },
    }

    config_file = f"data/landau_1d/{base_name}_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_file}")


if __name__ == "__main__":
    main()
