#!/usr/bin/env python3
"""
Run Landau simulation and then optimize collision operator parameters.

Usage:
    CUDA_VISIBLE_DEVICES=7 python run_simulation_and_optimize.py --operator bgk
    CUDA_VISIBLE_DEVICES=7 python run_simulation_and_optimize.py --operator esbgk
    CUDA_VISIBLE_DEVICES=7 python run_simulation_and_optimize.py --operator shakhov
    CUDA_VISIBLE_DEVICES=7 python run_simulation_and_optimize.py --operator fokker-planck
"""

import numpy as np
import os
import json
import argparse
from datetime import datetime

# Set up JAX before importing
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from landau_1d_numerical_jax import LandauSolver1D_JAX
from optimize_esbgk import (
    optimize_esbgk, optimize_bgk_tau,
    compute_moments as compute_moments_esbgk,
    plot_esbgk_optimization
)
from optimize_shakhov import (
    optimize_shakhov, compute_moments, compute_heat_flux,
    plot_shakhov_optimization
)
from optimize_fokker_planck import (
    optimize_fokker_planck, plot_fokker_planck_optimization
)
import matplotlib.pyplot as plt


OPERATOR_CHOICES = ['bgk', 'esbgk', 'shakhov', 'fokker-planck']

OPERATOR_LABELS = {
    'bgk': 'BGK',
    'esbgk': 'ES-BGK',
    'shakhov': 'Shakhov',
    'fokker-planck': 'Fokker-Planck',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Landau simulation and optimize collision operator parameters.'
    )
    parser.add_argument(
        '--operator', type=str, default='esbgk',
        choices=OPERATOR_CHOICES,
        help='Collision operator to optimize (default: esbgk)'
    )
    return parser.parse_args()


def optimize_at_snapshot(operator, f, Q_approx, v, dv):
    """Run optimization for the chosen operator at a single snapshot."""
    rho, u, T = compute_moments(f, v, dv)

    if operator == 'bgk':
        result = optimize_bgk_tau(f, Q_approx, rho, u, T, v, norm='L2')
        # Normalize keys for uniform output
        return {
            'tau_opt': result['tau_opt'],
            'error_model': result['error'],
            'error_bgk': result['error'],
            'tau_bgk': result['tau_opt'],
            'improvement': 0.0,
            'norm': result['norm'],
        }

    elif operator == 'esbgk':
        result = optimize_esbgk(f, Q_approx, rho, u, T, v, norm='L2', n_grid=30)
        return {
            'tau_opt': result['tau_opt'],
            'alpha_opt': result['alpha_opt'],
            'error_model': result['error_esbgk'],
            'error_esbgk': result['error_esbgk'],
            'error_bgk': result['error_bgk'],
            'tau_bgk': result['tau_bgk'],
            'improvement': result['improvement'],
            'error_grid': result['error_grid'],
            'tau_grid': result['tau_grid'],
            'alpha_grid': result['alpha_grid'],
            'norm': result['norm'],
        }

    elif operator == 'shakhov':
        result = optimize_shakhov(f, Q_approx, rho, u, T, v, dv, norm='L2', n_grid=30)
        return {
            'tau_opt': result['tau_opt'],
            'Pr_opt': result['Pr_opt'],
            'error_model': result['error_shakhov'],
            'error_shakhov': result['error_shakhov'],
            'error_bgk': result['error_bgk'],
            'tau_bgk': result['tau_bgk'],
            'improvement': result['improvement'],
            'heat_flux': result['heat_flux'],
            'error_grid': result['error_grid'],
            'tau_grid': result['tau_grid'],
            'Pr_grid': result['Pr_grid'],
            'norm': result['norm'],
        }

    elif operator == 'fokker-planck':
        result = optimize_fokker_planck(f, Q_approx, rho, u, T, v, dv, norm='L2', n_grid=30)
        return {
            'mu_opt': result['mu_opt'],
            'error_model': result['error_fp'],
            'error_bgk': result['error_bgk'],
            'tau_bgk': result['tau_bgk'],
            'improvement': result['improvement'],
            'norm': result['norm'],
        }


def print_snapshot_result(operator, result):
    """Print optimization result for a single snapshot."""
    if operator == 'bgk':
        print(f"    BGK: tau={result['tau_opt']:.4f}")
    elif operator == 'esbgk':
        print(f"    ES-BGK: tau={result['tau_opt']:.4f}, "
              f"alpha={result['alpha_opt']:.4f}, "
              f"improvement={result['improvement']:.2f}%")
    elif operator == 'shakhov':
        print(f"    Shakhov: tau={result['tau_opt']:.4f}, "
              f"Pr={result['Pr_opt']:.4f}, "
              f"improvement={result['improvement']:.2f}%")
        q = result['heat_flux']
        print(f"    Heat flux: mean|q|={np.mean(np.abs(q)):.4e}, "
              f"max|q|={np.max(np.abs(q)):.4e}")
    elif operator == 'fokker-planck':
        print(f"    Fokker-Planck: mu={result['mu_opt']:.4f}, "
              f"improvement={result['improvement']:.2f}%")


def save_optimization_results(operator, optimization_results, base_name):
    """Save optimization results to .npz file."""
    times = np.array([r['time'] for r in optimization_results])
    error_model = np.array([r['error_model'] for r in optimization_results])
    error_bgk = np.array([r['error_bgk'] for r in optimization_results])
    improvement = np.array([r['improvement'] for r in optimization_results])

    if operator == 'bgk':
        opt_file = f"data/landau_1d/{base_name}_bgk_opt.npz"
        np.savez(opt_file,
                 times=times,
                 tau_bgk=np.array([r['tau_opt'] for r in optimization_results]),
                 error_bgk=error_bgk)

    elif operator == 'esbgk':
        opt_file = f"data/landau_1d/{base_name}_esbgk_opt.npz"
        np.savez(opt_file,
                 times=times,
                 tau_esbgk=np.array([r['tau_opt'] for r in optimization_results]),
                 alpha_esbgk=np.array([r['alpha_opt'] for r in optimization_results]),
                 tau_bgk=np.array([r['tau_bgk'] for r in optimization_results]),
                 error_esbgk=error_model,
                 error_bgk=error_bgk,
                 improvement=improvement)

    elif operator == 'shakhov':
        opt_file = f"data/landau_1d/{base_name}_shakhov_opt.npz"
        np.savez(opt_file,
                 times=times,
                 tau_shakhov=np.array([r['tau_opt'] for r in optimization_results]),
                 Pr_shakhov=np.array([r['Pr_opt'] for r in optimization_results]),
                 tau_bgk=np.array([r['tau_bgk'] for r in optimization_results]),
                 error_shakhov=error_model,
                 error_bgk=error_bgk,
                 improvement=improvement,
                 heat_flux_mean=np.array([np.mean(np.abs(r['heat_flux']))
                                          for r in optimization_results]),
                 heat_flux_max=np.array([np.max(np.abs(r['heat_flux']))
                                         for r in optimization_results]))

    elif operator == 'fokker-planck':
        opt_file = f"data/landau_1d/{base_name}_fp_opt.npz"
        np.savez(opt_file,
                 times=times,
                 mu_fp=np.array([r['mu_opt'] for r in optimization_results]),
                 tau_bgk=np.array([r['tau_bgk'] for r in optimization_results]),
                 error_fp=error_model,
                 error_bgk=error_bgk,
                 improvement=improvement)

    print(f"Saved optimization results to {opt_file}")
    return opt_file


def plot_trajectory(operator, optimization_results, base_name):
    """Generate trajectory plots for the chosen operator."""
    opt_times = [r['time'] for r in optimization_results]
    label = OPERATOR_LABELS[operator]

    if operator == 'bgk':
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # tau trajectory
        ax = axes[0]
        ax.semilogy(opt_times, [r['tau_opt'] for r in optimization_results],
                     'b-o', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Optimal tau')
        ax.set_title('BGK Optimal Relaxation Time')
        ax.grid(True, alpha=0.3)

        # Error trajectory
        ax = axes[1]
        ax.semilogy(opt_times, [r['error_bgk'] for r in optimization_results],
                     'b-o', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Error (L2)')
        ax.set_title('BGK Error')
        ax.grid(True, alpha=0.3)

    elif operator == 'esbgk':
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax = axes[0, 0]
        ax.semilogy(opt_times, [r['tau_bgk'] for r in optimization_results],
                     'b-o', label='BGK')
        ax.semilogy(opt_times, [r['tau_opt'] for r in optimization_results],
                     'g-o', label='ES-BGK')
        ax.set_xlabel('Time')
        ax.set_ylabel('Optimal tau')
        ax.set_title('Optimal Relaxation Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(opt_times, [r['alpha_opt'] for r in optimization_results], 'g-o')
        ax.axhline(1.0, color='b', linestyle='--', label='alpha=1 (BGK)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Optimal alpha')
        ax.set_title('ES-BGK Temperature Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.semilogy(opt_times, [r['error_bgk'] for r in optimization_results],
                     'b-o', label='BGK')
        ax.semilogy(opt_times, [r['error_model'] for r in optimization_results],
                     'g-o', label='ES-BGK')
        ax.set_xlabel('Time')
        ax.set_ylabel('Error (L2)')
        ax.set_title('Error Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        improvements = [r['improvement'] for r in optimization_results]
        ax.plot(opt_times, improvements, 'g-o', linewidth=2)
        ax.fill_between(opt_times, 0, improvements, alpha=0.3, color='green')
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('ES-BGK Improvement over BGK')
        ax.grid(True, alpha=0.3)

    elif operator == 'shakhov':
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        ax = axes[0, 0]
        ax.semilogy(opt_times, [r['tau_bgk'] for r in optimization_results],
                     'b-o', label='BGK')
        ax.semilogy(opt_times, [r['tau_opt'] for r in optimization_results],
                     'orange', marker='^', linestyle='-', label='Shakhov')
        ax.set_xlabel('Time')
        ax.set_ylabel('Optimal tau')
        ax.set_title('Optimal Relaxation Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(opt_times, [r['Pr_opt'] for r in optimization_results],
                'orange', marker='^', linestyle='-', linewidth=2)
        ax.axhline(1.0, color='b', linestyle='--', label='Pr=1 (BGK)')
        ax.axhline(2/3, color='green', linestyle=':', linewidth=2,
                   label='Pr=2/3 (physical)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Optimal Pr')
        ax.set_title('Shakhov Prandtl Number')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        heat_flux_mean = [np.mean(np.abs(r['heat_flux'])) for r in optimization_results]
        ax.semilogy(opt_times, heat_flux_mean, 'orange', marker='^',
                     linestyle='-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean |q|')
        ax.set_title('Heat Flux Magnitude')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.semilogy(opt_times, [r['error_bgk'] for r in optimization_results],
                     'b-o', label='BGK')
        ax.semilogy(opt_times, [r['error_model'] for r in optimization_results],
                     'orange', marker='^', linestyle='-', label='Shakhov')
        ax.set_xlabel('Time')
        ax.set_ylabel('Error (L2)')
        ax.set_title('Error Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        improvements = [r['improvement'] for r in optimization_results]
        ax.plot(opt_times, improvements, 'orange', marker='^',
                linestyle='-', linewidth=2)
        ax.fill_between(opt_times, 0, improvements, alpha=0.3, color='orange')
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Shakhov Improvement over BGK')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        sc = ax.scatter(heat_flux_mean,
                        [r['Pr_opt'] for r in optimization_results],
                        c=opt_times, cmap='viridis', s=100, edgecolor='black')
        ax.axhline(2/3, color='green', linestyle=':', linewidth=2, label='Pr=2/3')
        ax.axhline(1.0, color='blue', linestyle='--', label='Pr=1')
        ax.set_xlabel('Mean |q| (heat flux)')
        ax.set_ylabel('Optimal Pr')
        ax.set_title('Shakhov: Pr vs Heat Flux')
        plt.colorbar(sc, ax=ax, label='Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    elif operator == 'fokker-planck':
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax = axes[0, 0]
        ax.semilogy(opt_times, [r['tau_bgk'] for r in optimization_results],
                     'b-o', label='BGK (tau)')
        ax.set_xlabel('Time')
        ax.set_ylabel('BGK Optimal tau')
        ax.set_title('BGK Reference: Relaxation Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.semilogy(opt_times, [r['mu_opt'] for r in optimization_results],
                     'r-s', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Optimal mu')
        ax.set_title('Fokker-Planck Collision Rate')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.semilogy(opt_times, [r['error_bgk'] for r in optimization_results],
                     'b-o', label='BGK')
        ax.semilogy(opt_times, [r['error_model'] for r in optimization_results],
                     'r-s', label='Fokker-Planck')
        ax.set_xlabel('Time')
        ax.set_ylabel('Error (L2)')
        ax.set_title('Error Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        improvements = [r['improvement'] for r in optimization_results]
        ax.plot(opt_times, improvements, 'r-s', linewidth=2)
        ax.fill_between(opt_times, 0, improvements, alpha=0.3, color='red')
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Fokker-Planck Improvement over BGK')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_file = f"figures/landau_1d/{base_name}_{operator}_trajectory.png"
    plt.savefig(fig_file, dpi=150)
    print(f"Saved trajectory plot to {fig_file}")
    plt.close(fig)


def plot_landscape(operator, optimization_results, base_name):
    """Plot error landscape for operators that have grid search data."""
    if operator == 'esbgk':
        fig = plot_esbgk_optimization(
            optimization_results[-1],
            f"figures/landau_1d/{base_name}_esbgk_landscape.png"
        )
        plt.close(fig)
    elif operator == 'shakhov':
        fig = plot_shakhov_optimization(
            optimization_results[-1],
            f"figures/landau_1d/{base_name}_shakhov_landscape.png"
        )
        plt.close(fig)
    elif operator == 'fokker-planck':
        fig = plot_fokker_planck_optimization(
            optimization_results[-1],
            f"figures/landau_1d/{base_name}_fp_comparison.png"
        )
        plt.close(fig)


def print_summary(operator, optimization_results, elapsed_time):
    """Print summary statistics."""
    label = OPERATOR_LABELS[operator]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Simulation time: {elapsed_time:.2f} seconds")
    print(f"Operator: {label}")

    if operator == 'bgk':
        print(f"Average tau: {np.mean([r['tau_opt'] for r in optimization_results]):.4f}")

    elif operator == 'esbgk':
        print(f"Average tau_bgk: "
              f"{np.mean([r['tau_bgk'] for r in optimization_results]):.4f}")
        print(f"Average tau_esbgk: "
              f"{np.mean([r['tau_opt'] for r in optimization_results]):.4f}")
        print(f"Average alpha: "
              f"{np.mean([r['alpha_opt'] for r in optimization_results]):.4f}")
        print(f"Average improvement: "
              f"{np.mean([r['improvement'] for r in optimization_results]):.2f}%")

    elif operator == 'shakhov':
        print(f"Average tau_bgk: "
              f"{np.mean([r['tau_bgk'] for r in optimization_results]):.4f}")
        print(f"Average tau_shakhov: "
              f"{np.mean([r['tau_opt'] for r in optimization_results]):.4f}")
        print(f"Average Pr: "
              f"{np.mean([r['Pr_opt'] for r in optimization_results]):.4f}")
        print(f"Average improvement: "
              f"{np.mean([r['improvement'] for r in optimization_results]):.2f}%")

    elif operator == 'fokker-planck':
        print(f"Average tau_bgk: "
              f"{np.mean([r['tau_bgk'] for r in optimization_results]):.4f}")
        print(f"Average mu: "
              f"{np.mean([r['mu_opt'] for r in optimization_results]):.4f}")
        print(f"Average improvement: "
              f"{np.mean([r['improvement'] for r in optimization_results]):.2f}%")

    print("=" * 70)


def build_config(operator, optimization_results, sim_params):
    """Build config dictionary for saving."""
    config = dict(sim_params)
    config['operator'] = operator

    if operator == 'bgk':
        config['avg_tau'] = float(
            np.mean([r['tau_opt'] for r in optimization_results]))

    elif operator == 'esbgk':
        config['avg_tau_bgk'] = float(
            np.mean([r['tau_bgk'] for r in optimization_results]))
        config['avg_tau_esbgk'] = float(
            np.mean([r['tau_opt'] for r in optimization_results]))
        config['avg_alpha_esbgk'] = float(
            np.mean([r['alpha_opt'] for r in optimization_results]))
        config['avg_improvement'] = float(
            np.mean([r['improvement'] for r in optimization_results]))

    elif operator == 'shakhov':
        config['avg_tau_bgk'] = float(
            np.mean([r['tau_bgk'] for r in optimization_results]))
        config['avg_tau_shakhov'] = float(
            np.mean([r['tau_opt'] for r in optimization_results]))
        config['avg_Pr'] = float(
            np.mean([r['Pr_opt'] for r in optimization_results]))
        config['avg_improvement'] = float(
            np.mean([r['improvement'] for r in optimization_results]))

    elif operator == 'fokker-planck':
        config['avg_tau_bgk'] = float(
            np.mean([r['tau_bgk'] for r in optimization_results]))
        config['avg_mu'] = float(
            np.mean([r['mu_opt'] for r in optimization_results]))
        config['avg_improvement'] = float(
            np.mean([r['improvement'] for r in optimization_results]))

    return config


def main():
    args = parse_args()
    operator = args.operator
    label = OPERATOR_LABELS[operator]

    # Parameters
    N_x = 2**16  # 65536
    N_v = 2**10  # 1024
    N_t = 2**14  # 16384

    X = 0.5
    V = 6.0
    T_final = 0.1
    lambda_D = 10.0

    # Save every N steps (adjust based on memory)
    save_every = N_t // 64  # Save 64 snapshots

    print("=" * 70)
    print(f"Landau Simulation + {label} Optimization")
    print("=" * 70)
    print(f"Grid: N_x={N_x}, N_v={N_v}, N_t={N_t}")
    print(f"Save every: {save_every} steps ({N_t // save_every} snapshots)")
    print(f"Domain: x in [-{X}, {X}], v in [-{V}, {V}], t in [0, {T_final}]")
    print(f"Debye length: lambda_D = {lambda_D}")
    print(f"Operator: {label}")
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
    print(f"\n[1/3] Running Landau simulation...")
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
    # Optimization
    # =============================================
    print(f"\n[2/3] Computing collision operator and optimizing {label}...")

    f_history = results['f_history']
    v = results['v']
    dv = solver.dv
    times = results['times']

    n_optimize = min(10, len(times) - 1)
    indices = np.linspace(1, len(times) - 1, n_optimize, dtype=int)

    optimization_results = []

    for idx in indices:
        print(f"\n  Optimizing at t = {times[idx]:.4f} "
              f"(snapshot {idx}/{len(times)-1})...")

        f = f_history[idx]

        # Approximate Q_landau from time derivative
        if idx > 0:
            dt_snapshot = times[idx] - times[idx-1]
            Q_approx = (f_history[idx] - f_history[idx-1]) / dt_snapshot
        else:
            Q_approx = np.zeros_like(f)

        result = optimize_at_snapshot(operator, f, Q_approx, v, dv)
        result['time'] = times[idx]
        result['snapshot_idx'] = idx
        optimization_results.append(result)

        print_snapshot_result(operator, result)

    # =============================================
    # Save and plot results
    # =============================================
    print(f"\n[3/3] Saving results...")

    save_optimization_results(operator, optimization_results, base_name)
    plot_landscape(operator, optimization_results, base_name)
    plot_trajectory(operator, optimization_results, base_name)

    # Summary
    print_summary(operator, optimization_results, results['elapsed_time'])

    # Save config
    sim_params = {
        'N_x': N_x,
        'N_v': N_v,
        'N_t': N_t,
        'X': X,
        'V': V,
        'T_final': T_final,
        'lambda_D': lambda_D,
        'save_every': save_every,
        'elapsed_time': results['elapsed_time'],
    }
    config = build_config(operator, optimization_results, sim_params)

    config_file = f"data/landau_1d/{base_name}_config.json"
    with open(config_file, 'w') as fp:
        json.dump(config, fp, indent=2)
    print(f"Saved config to {config_file}")


if __name__ == "__main__":
    main()
