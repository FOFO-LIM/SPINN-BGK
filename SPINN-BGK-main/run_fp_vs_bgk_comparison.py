#!/usr/bin/env python3
"""
Compare Fokker-Planck (FP) vs BGK collision operators against Landau.

This script:
1. Runs the 1D Boltzmann-Landau simulation to get Q_Landau(f,f)
2. Computes the Fokker-Planck operator Q_FP(f)
3. Finds optimal mu that minimizes ||Q_Landau(f,f) - mu * Q_FP(f)||_L^2
4. Also computes optimal tau for BGK for comparison
5. Generates comparison plots

Goal: Determine if FP is fundamentally better than BGK in approximating Landau.

Optimized with JAX for GPU-accelerated post-processing.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import time
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Set JAX to use GPUs 0 and 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

print(f"JAX devices: {jax.devices()}")
print(f"Number of devices: {jax.local_device_count()}")

# Import the Landau solver
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from landau_1d_numerical_jax import LandauSolver1D_JAX, get_gpu_memory_gib


# ============================================================================
# JAX-accelerated Operator Implementations
# ============================================================================

@jit
def compute_maxwellian_jax(rho: jnp.ndarray, u: jnp.ndarray, T: jnp.ndarray,
                            v: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Maxwellian distribution f_M(x, v) using JAX.
    """
    T_safe = jnp.maximum(T, 1e-10)
    return (rho[:, None] / jnp.sqrt(2 * jnp.pi * T_safe[:, None]) *
            jnp.exp(-(v[None, :] - u[:, None])**2 / (2 * T_safe[:, None])))


@partial(jit, static_argnums=(2,))
def compute_moments_jax(f: jnp.ndarray, v: jnp.ndarray, dv: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute moments (rho, u, T) from distribution function using JAX.
    """
    N_v = len(v)
    w = jnp.ones(N_v) * dv
    w = w.at[0].set(dv / 2)
    w = w.at[-1].set(dv / 2)

    rho = jnp.sum(f * w, axis=1)
    rho = jnp.maximum(rho, 1e-30)
    u = jnp.sum(f * v[None, :] * w, axis=1) / rho
    c_sq = (v[None, :] - u[:, None])**2
    T = jnp.sum(f * c_sq * w, axis=1) / rho

    return rho, u, T


def create_landau_operator_jax(N_v: int, dv: float, lambda_D: float):
    """
    Create a JIT-compiled Landau operator function.
    """
    cutoff = 1.0 / lambda_D
    N_conv = 2 * N_v - 1

    # Velocity difference grid for kernel
    u_grid = jnp.arange(-(N_v - 1), N_v) * dv

    # Coulomb kernel with cutoff
    Phi = 1.0 / jnp.maximum(jnp.abs(u_grid), cutoff)
    Phi_fft = jnp.fft.fft(Phi)

    def convolution_fft(f_row):
        """Compute convolution Phi * f for a single spatial point."""
        f_padded = jnp.zeros(N_conv)
        f_padded = f_padded.at[:N_v].set(f_row)
        conv_full = jnp.real(jnp.fft.ifft(Phi_fft * jnp.fft.fft(f_padded)))
        start = N_v - 1
        return conv_full[start:start + N_v] * dv

    @jit
    def compute_landau_operator(f: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the 1D Landau collision operator Q_L(f,f).
        """
        N_x = f.shape[0]

        # Compute df/dv using central differences
        df_dv = jnp.zeros_like(f)
        df_dv = df_dv.at[:, 1:-1].set((f[:, 2:] - f[:, :-2]) / (2 * dv))
        df_dv = df_dv.at[:, 0].set((f[:, 1] - f[:, 0]) / dv)
        df_dv = df_dv.at[:, -1].set((f[:, -1] - f[:, -2]) / dv)

        # Compute A[f] and B[f] via convolution (vectorized)
        A = vmap(convolution_fft)(f)
        B = vmap(convolution_fft)(df_dv)

        # Flux: J = A df/dv - B f
        J = A * df_dv - B * f

        # Q = dJ/dv
        Q = jnp.zeros_like(f)
        Q = Q.at[:, 1:-1].set((J[:, 2:] - J[:, :-2]) / (2 * dv))
        Q = Q.at[:, 0].set((J[:, 1] - J[:, 0]) / dv)
        Q = Q.at[:, -1].set((J[:, -1] - J[:, -2]) / dv)

        return Q

    return compute_landau_operator


def create_fokker_planck_operator_jax(N_v: int, dv: float):
    """
    Create a JIT-compiled Fokker-Planck operator function.
    """
    @jit
    def compute_fokker_planck_operator(f: jnp.ndarray, v: jnp.ndarray,
                                        rho: jnp.ndarray, u: jnp.ndarray,
                                        T: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the normalized 1D Fokker-Planck operator P(f).
        P(f) = d/dv [ M[f] d/dv (f / M[f]) ]
        """
        # Local Maxwellian
        M = compute_maxwellian_jax(rho, u, T, v)
        M = jnp.maximum(M, 1e-30)

        # f / M
        f_over_M = f / M

        # d/dv (f / M) using central differences
        grad_f_over_M = jnp.zeros_like(f)
        grad_f_over_M = grad_f_over_M.at[:, 1:-1].set(
            (f_over_M[:, 2:] - f_over_M[:, :-2]) / (2 * dv))
        grad_f_over_M = grad_f_over_M.at[:, 0].set(
            (f_over_M[:, 1] - f_over_M[:, 0]) / dv)
        grad_f_over_M = grad_f_over_M.at[:, -1].set(
            (f_over_M[:, -1] - f_over_M[:, -2]) / dv)

        # M * d/dv (f / M)
        flux = M * grad_f_over_M

        # d/dv [ M * d/dv (f / M) ]
        P = jnp.zeros_like(f)
        P = P.at[:, 1:-1].set((flux[:, 2:] - flux[:, :-2]) / (2 * dv))
        P = P.at[:, 0].set((flux[:, 1] - flux[:, 0]) / dv)
        P = P.at[:, -1].set((flux[:, -1] - flux[:, -2]) / dv)

        return P

    return compute_fokker_planck_operator


# ============================================================================
# Optimization Functions
# ============================================================================

def optimize_fp_mu(Q_landau: np.ndarray, P_fp: np.ndarray,
                   mu_range: Tuple[float, float] = (0.001, 100.0)) -> Dict:
    """
    Find optimal mu that minimizes ||Q_Landau - mu * P_FP||_L2.
    For L2 norm: mu_opt = <Q_Landau, P_FP> / <P_FP, P_FP>
    """
    numerator = np.mean(Q_landau * P_fp)
    denominator = np.mean(P_fp**2)

    if denominator > 1e-20 and numerator > 0:
        mu_opt = np.clip(numerator / denominator, mu_range[0], mu_range[1])
    elif denominator > 1e-20 and numerator < 0:
        mu_opt = mu_range[0]
    else:
        mu_opt = 1.0

    error = np.mean((Q_landau - mu_opt * P_fp)**2)
    return {'mu_opt': mu_opt, 'error_L2': error}


def optimize_bgk_tau(f: np.ndarray, Q_landau: np.ndarray,
                      f_eq: np.ndarray,
                      tau_range: Tuple[float, float] = (0.01, 1000)) -> Dict:
    """
    Find optimal tau for BGK: min ||Q_Landau - (f_eq - f)/tau||_L2.
    For L2: tau_opt = <f_neq, f_neq> / <f_neq, Q_Landau>
    """
    f_neq = f_eq - f

    A = np.mean(f_neq**2)
    B = np.mean(f_neq * Q_landau)

    if B > 1e-20 and A > 0:
        tau_opt = np.clip(A / B, tau_range[0], tau_range[1])
    elif B < 0:
        tau_opt = tau_range[1]
    else:
        tau_opt = 1.0

    Q_bgk = f_neq / tau_opt
    error = np.mean((Q_landau - Q_bgk)**2)
    return {'tau_opt': tau_opt, 'error_L2': error}


# ============================================================================
# Main Simulation and Comparison
# ============================================================================

def run_fp_vs_bgk_comparison(
    N_x: int = 65536,
    N_v: int = 1024,
    N_t: int = 8192,
    X: float = 0.5,
    V: float = 6.0,
    T_final: float = 0.1,
    lambda_D: float = 10.0,
    num_gpus: int = 2,
    save_every: int = None,
    verbose: bool = True
):
    """
    Run simulation and compare FP vs BGK in approximating Landau collision operator.
    """
    start_datetime = datetime.now()

    if save_every is None:
        save_every = max(1, N_t // 64)

    print(f"\n{'='*70}")
    print("Fokker-Planck vs BGK Comparison for Landau Collision Operator")
    print('='*70)
    print(f"N_x = {N_x}")
    print(f"N_v = {N_v}")
    print(f"N_t = {N_t}")
    print(f"T_final = {T_final}")
    print(f"lambda_D = {lambda_D}")
    print(f"Using {num_gpus} GPUs")
    print(f"save_every = {save_every} ({N_t // save_every} snapshots)")
    print('='*70)

    # Create solver
    print("\nInitializing Landau solver...")
    solver = LandauSolver1D_JAX(
        N_x=N_x,
        N_v=N_v,
        N_t=N_t,
        X=X,
        V=V,
        T_final=T_final,
        lambda_D=lambda_D
    )

    # Run Landau simulation
    print("\nRunning Landau simulation...")
    sim_start = time.time()
    if num_gpus > 1 and N_x % num_gpus == 0:
        results = solver.solve_parallel(save_every=save_every, num_devices=num_gpus, verbose=verbose)
    else:
        results = solver.solve(save_every=save_every, verbose=verbose)
    sim_time = time.time() - sim_start

    print(f"\nLandau simulation completed in {sim_time:.2f} seconds ({sim_time/60:.2f} minutes)")

    # Get grid info
    x = results['x']
    v = results['v']
    dv = float(solver.dv)
    times = results['times']
    f_history = results.get('f_history', results['f'][np.newaxis, :, :])

    n_snapshots = len(times)
    print(f"\nProcessing {n_snapshots} snapshots for FP vs BGK comparison...")

    # Create JIT-compiled operators
    print("JIT compiling operators... ", end="", flush=True)
    compute_landau_jax = create_landau_operator_jax(N_v, dv, lambda_D)
    compute_fp_jax = create_fokker_planck_operator_jax(N_v, dv)

    # Warm-up JIT
    f_test = jnp.array(f_history[0])
    v_jax = jnp.array(v)
    _ = compute_landau_jax(f_test)
    rho_test, u_test, T_test = compute_moments_jax(f_test, v_jax, dv)
    _ = compute_fp_jax(f_test, v_jax, rho_test, u_test, T_test)
    _ = compute_maxwellian_jax(rho_test, u_test, T_test, v_jax)
    jax.block_until_ready(_)
    print("done")

    # Storage for results
    fp_results = []
    bgk_results = []

    post_process_start = time.time()

    for i, t in enumerate(times):
        f = jnp.array(f_history[i])

        # Compute moments
        rho, u, T = compute_moments_jax(f, v_jax, dv)

        # Compute Q_Landau using JAX
        Q_landau = compute_landau_jax(f)

        # Compute Fokker-Planck operator P(f)
        P_fp = compute_fp_jax(f, v_jax, rho, u, T)

        # Compute Maxwellian for BGK
        f_eq = compute_maxwellian_jax(rho, u, T, v_jax)

        # Block to ensure computation is done
        jax.block_until_ready(P_fp)

        # Convert to numpy for optimization
        Q_landau_np = np.array(Q_landau)
        P_fp_np = np.array(P_fp)
        f_np = np.array(f)
        f_eq_np = np.array(f_eq)

        # Optimize mu for FP
        fp_opt = optimize_fp_mu(Q_landau_np, P_fp_np)
        Q_norm = np.sqrt(np.mean(Q_landau_np**2))

        fp_results.append({
            'time': t,
            'mu_opt': fp_opt['mu_opt'],
            'error_L2': fp_opt['error_L2'],
            'Q_landau_norm': Q_norm
        })

        # Optimize tau for BGK
        bgk_opt = optimize_bgk_tau(f_np, Q_landau_np, f_eq_np)
        bgk_results.append({
            'time': t,
            'tau_opt': bgk_opt['tau_opt'],
            'error_L2': bgk_opt['error_L2'],
        })

        if verbose and (i + 1) % max(1, n_snapshots // 10) == 0:
            elapsed = time.time() - post_process_start
            eta = elapsed / (i + 1) * (n_snapshots - i - 1)
            print(f"  Snapshot {i+1}/{n_snapshots}: t={t:.4f}, "
                  f"FP mu={fp_opt['mu_opt']:.4f}, BGK tau={bgk_opt['tau_opt']:.4f}, "
                  f"ETA: {eta/60:.1f} min")

    post_process_time = time.time() - post_process_start

    # Convert to arrays
    times_arr = np.array([r['time'] for r in fp_results])
    fp_mu = np.array([r['mu_opt'] for r in fp_results])
    fp_error_L2 = np.array([r['error_L2'] for r in fp_results])
    bgk_tau = np.array([r['tau_opt'] for r in bgk_results])
    bgk_error_L2 = np.array([r['error_L2'] for r in bgk_results])
    Q_norm = np.array([r['Q_landau_norm'] for r in fp_results])

    # Compute relative errors
    fp_rel_error = np.sqrt(fp_error_L2) / (Q_norm + 1e-10)
    bgk_rel_error = np.sqrt(bgk_error_L2) / (Q_norm + 1e-10)

    # Compute improvement of FP over BGK
    improvement = (bgk_error_L2 - fp_error_L2) / (bgk_error_L2 + 1e-10) * 100

    end_datetime = datetime.now()
    total_time = (end_datetime - start_datetime).total_seconds()

    # Summary statistics
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print('='*70)
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"  - Landau simulation: {sim_time:.2f} seconds")
    print(f"  - Post-processing: {post_process_time:.2f} seconds")
    print(f"\nFokker-Planck (optimal mu):")
    print(f"  mu range: [{fp_mu.min():.4f}, {fp_mu.max():.4f}]")
    print(f"  mu mean: {fp_mu.mean():.4f}")
    print(f"  L2 error range: [{fp_error_L2.min():.2e}, {fp_error_L2.max():.2e}]")
    print(f"  Relative error range: [{fp_rel_error.min():.2%}, {fp_rel_error.max():.2%}]")
    print(f"\nBGK (optimal tau):")
    print(f"  tau range: [{bgk_tau.min():.4f}, {bgk_tau.max():.4f}]")
    print(f"  tau mean: {bgk_tau.mean():.4f}")
    print(f"  L2 error range: [{bgk_error_L2.min():.2e}, {bgk_error_L2.max():.2e}]")
    print(f"  Relative error range: [{bgk_rel_error.min():.2%}, {bgk_rel_error.max():.2%}]")
    print(f"\nFP improvement over BGK:")
    print(f"  Improvement range: [{improvement.min():.2f}%, {improvement.max():.2f}%]")
    print(f"  Mean improvement: {improvement.mean():.2f}%")

    # Determine if FP is fundamentally better
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print('='*70)
    if improvement.mean() > 5:
        print(f"Fokker-Planck is SIGNIFICANTLY BETTER than BGK")
        print(f"  Mean improvement: {improvement.mean():.1f}%")
    elif improvement.mean() > 0:
        print(f"Fokker-Planck is SLIGHTLY BETTER than BGK")
        print(f"  Mean improvement: {improvement.mean():.1f}%")
    elif improvement.mean() > -5:
        print(f"Fokker-Planck and BGK are COMPARABLE")
        print(f"  Mean difference: {improvement.mean():.1f}%")
    else:
        print(f"BGK is BETTER than Fokker-Planck")
        print(f"  BGK advantage: {-improvement.mean():.1f}%")
    print('='*70)

    # Create output directories
    os.makedirs("data/landau_1d", exist_ok=True)
    os.makedirs("figures/landau_1d", exist_ok=True)

    timestamp = start_datetime.strftime("%Y%m%d_%H%M%S")
    base_name = f"landau_Nx{N_x}_Nv{N_v}_Nt{N_t}_{timestamp}_fp_vs_bgk"

    # Save results
    comparison_results = {
        'times': times_arr,
        'fp_mu': fp_mu,
        'fp_error_L2': fp_error_L2,
        'fp_rel_error': fp_rel_error,
        'bgk_tau': bgk_tau,
        'bgk_error_L2': bgk_error_L2,
        'bgk_rel_error': bgk_rel_error,
        'improvement': improvement,
        'Q_norm': Q_norm,
    }

    np.savez(f"data/landau_1d/{base_name}_comparison.npz", **comparison_results)

    config = {
        "N_x": N_x,
        "N_v": N_v,
        "N_t": N_t,
        "X": X,
        "V": V,
        "T_final": T_final,
        "lambda_D": lambda_D,
        "save_every": save_every,
        "num_gpus": num_gpus,
        "total_time_sec": total_time,
        "simulation_time_sec": sim_time,
        "post_process_time_sec": post_process_time,
        "n_snapshots": n_snapshots,
        "fp": {
            "mu_min": float(fp_mu.min()),
            "mu_max": float(fp_mu.max()),
            "mu_mean": float(fp_mu.mean()),
            "error_L2_mean": float(fp_error_L2.mean()),
            "rel_error_mean": float(fp_rel_error.mean()),
        },
        "bgk": {
            "tau_min": float(bgk_tau.min()),
            "tau_max": float(bgk_tau.max()),
            "tau_mean": float(bgk_tau.mean()),
            "error_L2_mean": float(bgk_error_L2.mean()),
            "rel_error_mean": float(bgk_rel_error.mean()),
        },
        "improvement_mean_percent": float(improvement.mean()),
        "improvement_max_percent": float(improvement.max()),
        "improvement_min_percent": float(improvement.min()),
        "conclusion": "FP better" if improvement.mean() > 0 else "BGK better",
        "timestamp": timestamp,
    }

    with open(f"data/landau_1d/{base_name}_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nResults saved to data/landau_1d/{base_name}_*")

    # Create comparison plot (similar to the reference)
    create_comparison_plot(comparison_results, config, base_name)

    return comparison_results, config


def create_comparison_plot(results: Dict, config: Dict, base_name: str):
    """
    Create a comparison plot similar to landau_Nx65536_Nv1024_Nt8192_20260129_212125_comparison.png
    """
    times = results['times']
    fp_mu = results['fp_mu']
    bgk_tau = results['bgk_tau']
    fp_error_L2 = results['fp_error_L2']
    bgk_error_L2 = results['bgk_error_L2']
    improvement = results['improvement']
    Q_norm = results['Q_norm']
    fp_rel_error = results['fp_rel_error']
    bgk_rel_error = results['bgk_rel_error']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (0,0) Optimal Relaxation Time / Collision Rate
    ax = axes[0, 0]
    ax.semilogy(times, bgk_tau, 'b-o', label='BGK (τ)', markersize=4, linewidth=1.5)
    ax.semilogy(times, fp_mu, 'r-s', label='FP (μ)', markersize=4, linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Optimal Parameter')
    ax.set_title('Optimal Relaxation Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) FP Collision Rate (mu)
    ax = axes[0, 1]
    ax.plot(times, fp_mu, 'r-s', markersize=4, linewidth=2)
    ax.axhline(y=fp_mu.mean(), color='darkred', linestyle='--', alpha=0.7,
               label=f'Mean: {fp_mu.mean():.4f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Optimal μ')
    ax.set_title('Fokker-Planck Collision Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,2) BGK Relaxation Time (tau)
    ax = axes[0, 2]
    ax.semilogy(times, bgk_tau, 'b-o', markersize=4, linewidth=2)
    ax.axhline(y=bgk_tau.mean(), color='darkblue', linestyle='--', alpha=0.7,
               label=f'Mean: {bgk_tau.mean():.2f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Optimal τ')
    ax.set_title('BGK Relaxation Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Error Comparison (L2)
    ax = axes[1, 0]
    ax.semilogy(times, np.sqrt(bgk_error_L2), 'b-o', label='BGK', markersize=4, linewidth=2)
    ax.semilogy(times, np.sqrt(fp_error_L2), 'r-s', label='Fokker-Planck', markersize=4, linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Error (L2)')
    ax.set_title('Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Model Improvements over BGK
    ax = axes[1, 1]
    n_snap = len(improvement)
    colors = ['green' if imp > 0 else 'crimson' for imp in improvement]
    bars = ax.bar(range(n_snap), improvement, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhline(y=improvement.mean(), color='darkgreen', linestyle='--', linewidth=2,
               label=f'Mean: {improvement.mean():.1f}%')
    ax.set_xlabel('Snapshot')
    ax.set_ylabel('Improvement over BGK (%)')
    ax.set_title('Model Improvements')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # (1,2) FP vs BGK Summary
    ax = axes[1, 2]
    ax.axis('off')

    # Determine conclusion
    if improvement.mean() > 10:
        conclusion = "FP is SIGNIFICANTLY better"
        conclusion_color = "darkgreen"
    elif improvement.mean() > 0:
        conclusion = "FP is slightly better"
        conclusion_color = "green"
    elif improvement.mean() > -10:
        conclusion = "FP and BGK are comparable"
        conclusion_color = "orange"
    else:
        conclusion = "BGK is better"
        conclusion_color = "red"

    summary_text = f"""
    Is Fokker-Planck Better than BGK?
    ═══════════════════════════════════

    Simulation Parameters:
    • N_x = {config['N_x']:,}
    • N_v = {config['N_v']:,}
    • N_t = {config['N_t']:,}
    • T_final = {config['T_final']}

    Fokker-Planck Results:
    • μ range: [{config['fp']['mu_min']:.4f}, {config['fp']['mu_max']:.4f}]
    • Mean rel. error: {config['fp']['rel_error_mean']:.2%}

    BGK Results:
    • τ range: [{config['bgk']['tau_min']:.4f}, {config['bgk']['tau_max']:.4f}]
    • Mean rel. error: {config['bgk']['rel_error_mean']:.2%}

    ═══════════════════════════════════
    CONCLUSION: {conclusion}
    Mean improvement: {config['improvement_mean_percent']:.1f}%
    ═══════════════════════════════════
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f'FP vs BGK Comparison (N_x={config["N_x"]}, N_v={config["N_v"]}, N_t={config["N_t"]})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = f"figures/landau_1d/{base_name}_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {fig_path}")
    plt.close()

    # Create a simplified publication-quality figure
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Error comparison over time
    ax = axes2[0]
    ax.semilogy(times, np.sqrt(bgk_error_L2), 'b-o', label='BGK', markersize=6, linewidth=2)
    ax.semilogy(times, np.sqrt(fp_error_L2), 'r-s', label='Fokker-Planck', markersize=6, linewidth=2)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(r'$\|Q_{Landau} - Q_{approx}\|_{L^2}$', fontsize=12)
    ax.set_title('Approximation Error vs Time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: Improvement percentage
    ax = axes2[1]
    ax.bar(range(len(improvement)), improvement,
           color=['green' if x > 0 else 'red' for x in improvement],
           alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhline(y=improvement.mean(), color='darkgreen', linestyle='--', linewidth=2,
               label=f'Mean: {improvement.mean():.1f}%')
    ax.set_xlabel('Snapshot Index', fontsize=12)
    ax.set_ylabel('FP Improvement over BGK (%)', fontsize=12)
    ax.set_title(f'Is Fokker-Planck Fundamentally Better?\n({conclusion})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig2_path = f"figures/landau_1d/{base_name}_summary.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"Summary figure saved to {fig2_path}")
    plt.close()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='FP vs BGK comparison for Landau collision')
    parser.add_argument('--N_x', type=int, default=65536, help='Spatial grid points')
    parser.add_argument('--N_v', type=int, default=1024, help='Velocity grid points')
    parser.add_argument('--N_t', type=int, default=8192, help='Time steps')
    parser.add_argument('--X', type=float, default=0.5, help='Spatial domain half-width')
    parser.add_argument('--V', type=float, default=6.0, help='Velocity domain half-width')
    parser.add_argument('--T_final', type=float, default=0.1, help='Final time')
    parser.add_argument('--lambda_D', type=float, default=10.0, help='Debye length')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs')
    parser.add_argument('--save_every', type=int, default=None, help='Save interval')
    args = parser.parse_args()

    run_fp_vs_bgk_comparison(
        N_x=args.N_x,
        N_v=args.N_v,
        N_t=args.N_t,
        X=args.X,
        V=args.V,
        T_final=args.T_final,
        lambda_D=args.lambda_D,
        num_gpus=args.num_gpus,
        save_every=args.save_every,
    )
