# 2026-02-06: Initial creation - Diagnostic script to test sign consistency between BGK and Landau operators
#!/usr/bin/env python3
"""
Diagnostic script to test if there's a sign error in BGK or Landau operator.

Two tests are performed:
1. H-theorem test: Both operators should satisfy entropy dissipation
   - <Q, ln(f/f_eq)> <= 0 for proper collision operators

2. Direct relaxation test: Both operators should drive f toward equilibrium
   - Starting from f, a small step df/dt = Q should decrease ||f - f_eq||

If one operator violates these conditions, it likely has a sign error.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import json
import matplotlib.pyplot as plt
from datetime import datetime

print(f"JAX devices: {jax.devices()}")


# =============================================================================
# Operator Implementations (copied from existing code for consistency)
# =============================================================================

@jit
def compute_maxwellian(rho, u, T, v):
    """Compute Maxwellian distribution."""
    T_safe = jnp.maximum(T, 1e-10)
    return (rho[:, None] / jnp.sqrt(2 * jnp.pi * T_safe[:, None]) *
            jnp.exp(-(v[None, :] - u[:, None])**2 / (2 * T_safe[:, None])))


@partial(jit, static_argnums=(2,))
def compute_moments(f, v, dv):
    """Compute moments (rho, u, T) from distribution function."""
    N_v = len(v)
    w = jnp.ones(N_v) * dv
    w = w.at[0].set(dv / 2)
    w = w.at[-1].set(dv / 2)

    rho = jnp.sum(f * w, axis=1)
    rho = jnp.maximum(rho, 1e-30)
    u = jnp.sum(f * v[None, :] * w, axis=1) / rho
    c_sq = (v[None, :] - u[:, None])**2
    T = jnp.sum(f * c_sq * w, axis=1) / rho
    T = jnp.maximum(T, 1e-10)

    return rho, u, T


def create_landau_operator(N_v, dv, lambda_D):
    """Create JIT-compiled Landau collision operator."""
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
    def landau_operator(f):
        """
        Compute the 1D Landau collision operator Q_L(f,f).
        Q_L = d/dv [ A[f] df/dv - B[f] f ]
        """
        # Compute df/dv using central differences
        df_dv = jnp.zeros_like(f)
        df_dv = df_dv.at[:, 1:-1].set((f[:, 2:] - f[:, :-2]) / (2 * dv))
        df_dv = df_dv.at[:, 0].set((f[:, 1] - f[:, 0]) / dv)
        df_dv = df_dv.at[:, -1].set((f[:, -1] - f[:, -2]) / dv)

        # Compute A[f] and B[f] via convolution
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

    return landau_operator


@partial(jit, static_argnums=(2,))
def bgk_operator(f, f_eq, tau):
    """
    Compute BGK collision operator.
    Q_BGK = (f_eq - f) / tau
    """
    return (f_eq - f) / tau


# =============================================================================
# Diagnostic Tests
# =============================================================================

def test_h_theorem(f, Q, f_eq, v, dv, operator_name):
    """
    Test H-theorem: <Q, ln(f/f_eq)> should be <= 0

    The entropy production rate is:
        dH/dt = integral( Q * ln(f) ) dv

    For a proper collision operator driving toward equilibrium:
        integral( Q * ln(f/f_eq) ) dv <= 0

    Returns the entropy production (should be negative for correct sign).
    """
    # Compute ln(f/f_eq) safely
    f_safe = jnp.maximum(f, 1e-30)
    f_eq_safe = jnp.maximum(f_eq, 1e-30)
    log_ratio = jnp.log(f_safe / f_eq_safe)

    # Trapezoidal weights for velocity integration
    N_v = len(v)
    w_v = jnp.ones(N_v) * dv
    w_v = w_v.at[0].set(dv / 2)
    w_v = w_v.at[-1].set(dv / 2)

    # Compute <Q, ln(f/f_eq)> at each spatial point
    # This is integral over v of Q * ln(f/f_eq)
    entropy_production_x = jnp.sum(Q * log_ratio * w_v[None, :], axis=1)

    # Average over spatial domain
    entropy_production = jnp.mean(entropy_production_x)

    # Also compute at each spatial point for detailed analysis
    return {
        'name': operator_name,
        'entropy_production': float(entropy_production),
        'entropy_production_per_x': np.array(entropy_production_x),
        'sign_correct': float(entropy_production) <= 0,
        'interpretation': 'CORRECT (dissipative)' if float(entropy_production) <= 0
                          else 'WRONG SIGN (entropy increasing!)'
    }


def test_relaxation(f, Q, f_eq, dt_test, operator_name):
    """
    Test direct relaxation: Does df/dt = Q drive f toward f_eq?

    Compute:
        f_new = f + dt * Q
        ||f_new - f_eq|| vs ||f - f_eq||

    For correct sign, ||f_new - f_eq|| < ||f - f_eq||
    """
    # Current distance to equilibrium
    dist_before = jnp.sqrt(jnp.mean((f - f_eq)**2))

    # Take a small step
    f_new = f + dt_test * Q

    # New distance to equilibrium
    dist_after = jnp.sqrt(jnp.mean((f_new - f_eq)**2))

    # Change in distance (should be negative for relaxation toward equilibrium)
    delta_dist = float(dist_after - dist_before)

    return {
        'name': operator_name,
        'dist_before': float(dist_before),
        'dist_after': float(dist_after),
        'delta_dist': delta_dist,
        'relative_change': delta_dist / float(dist_before),
        'relaxing_toward_eq': delta_dist < 0,
        'interpretation': 'CORRECT (relaxing toward equilibrium)' if delta_dist < 0
                          else 'WRONG SIGN (moving away from equilibrium!)'
    }


def test_inner_product(Q_bgk, Q_landau, operator_name="BGK vs Landau"):
    """
    Test inner product between BGK and Landau operators.
    <Q_BGK, Q_Landau> should be positive if both point toward equilibrium.
    """
    inner_prod = jnp.mean(Q_bgk * Q_landau)
    norm_bgk = jnp.sqrt(jnp.mean(Q_bgk**2))
    norm_landau = jnp.sqrt(jnp.mean(Q_landau**2))

    # Cosine similarity
    cos_sim = inner_prod / (norm_bgk * norm_landau + 1e-30)

    return {
        'name': operator_name,
        'inner_product': float(inner_prod),
        'norm_bgk': float(norm_bgk),
        'norm_landau': float(norm_landau),
        'cosine_similarity': float(cos_sim),
        'same_direction': float(inner_prod) > 0,
        'interpretation': 'ALIGNED (same direction)' if float(inner_prod) > 0
                          else 'OPPOSITE (different directions!)'
    }


# =============================================================================
# Main Diagnostic Function
# =============================================================================

def run_diagnostics(data_file, grid_file, config_file, snapshot_indices=None):
    """
    Run all diagnostic tests on the given data.

    Args:
        data_file: Path to f_history .npy file
        grid_file: Path to grid .npz file
        config_file: Path to config .json file
        snapshot_indices: List of snapshot indices to test (default: [0, middle, last])
    """
    print("=" * 70)
    print("SIGN ERROR DIAGNOSTIC FOR BGK AND LANDAU OPERATORS")
    print("=" * 70)

    # Load config
    print("\nLoading configuration...")
    with open(config_file, 'r') as f:
        config = json.load(f)

    N_x = config['N_x']
    N_v = config['N_v']
    X = config['X']
    V = config['V']
    lambda_D = config['lambda_D']

    print(f"  N_x = {N_x}, N_v = {N_v}")
    print(f"  X = {X}, V = {V}")
    print(f"  lambda_D = {lambda_D}")

    # Load grid
    print("\nLoading grid...")
    grid = np.load(grid_file)
    x = grid['x']
    v = grid['v']
    times = grid['times']

    dx = x[1] - x[0]
    dv = v[1] - v[0]
    print(f"  dx = {dx:.6e}, dv = {dv:.6e}")
    print(f"  Time range: {times[0]:.6f} to {times[-1]:.6f}")
    print(f"  Number of snapshots: {len(times)}")

    # Load distribution function
    print("\nLoading distribution function...")
    f_history = np.load(data_file)
    print(f"  Shape: {f_history.shape}")

    # Select snapshots to test
    n_snapshots = len(times)
    if snapshot_indices is None:
        # Test early, middle, and late times
        snapshot_indices = [0, n_snapshots // 4, n_snapshots // 2,
                           3 * n_snapshots // 4, n_snapshots - 1]
        snapshot_indices = list(set(snapshot_indices))  # Remove duplicates
        snapshot_indices.sort()

    print(f"\nTesting snapshots: {snapshot_indices}")
    print(f"Corresponding times: {[times[i] for i in snapshot_indices]}")

    # Create operators
    print("\nCreating JIT-compiled operators...")
    v_jax = jnp.array(v)
    landau_op = create_landau_operator(N_v, dv, lambda_D)

    # Warm up JIT
    f_test = jnp.array(f_history[0])
    _ = landau_op(f_test)
    rho, u, T = compute_moments(f_test, v_jax, dv)
    f_eq = compute_maxwellian(rho, u, T, v_jax)
    _ = bgk_operator(f_test, f_eq, 1.0)
    jax.block_until_ready(_)
    print("JIT compilation done.")

    # Storage for results
    all_results = {
        'config': config,
        'snapshot_indices': snapshot_indices,
        'times_tested': [times[i] for i in snapshot_indices],
        'h_theorem_landau': [],
        'h_theorem_bgk': [],
        'relaxation_landau': [],
        'relaxation_bgk': [],
        'inner_product': [],
    }

    # Run tests for each snapshot
    print("\n" + "=" * 70)
    print("RUNNING DIAGNOSTIC TESTS")
    print("=" * 70)

    for idx in snapshot_indices:
        t = times[idx]
        print(f"\n{'─' * 70}")
        print(f"Snapshot {idx}, t = {t:.6f}")
        print('─' * 70)

        # Get distribution function
        f = jnp.array(f_history[idx])

        # Compute moments and equilibrium
        rho, u, T = compute_moments(f, v_jax, dv)
        f_eq = compute_maxwellian(rho, u, T, v_jax)

        # Compute operators
        Q_landau = landau_op(f)
        Q_bgk = bgk_operator(f, f_eq, 1.0)  # tau=1 for comparison

        jax.block_until_ready(Q_landau)

        # Compute norms for reference
        norm_f = float(jnp.sqrt(jnp.mean(f**2)))
        norm_f_neq = float(jnp.sqrt(jnp.mean((f - f_eq)**2)))
        norm_Q_landau = float(jnp.sqrt(jnp.mean(Q_landau**2)))
        norm_Q_bgk = float(jnp.sqrt(jnp.mean(Q_bgk**2)))

        print(f"\n  Norms:")
        print(f"    ||f|| = {norm_f:.6e}")
        print(f"    ||f - f_eq|| = {norm_f_neq:.6e}")
        print(f"    ||Q_Landau|| = {norm_Q_landau:.6e}")
        print(f"    ||Q_BGK|| = {norm_Q_bgk:.6e}")

        # Choose dt for relaxation test (small enough to be in linear regime)
        dt_test = 0.01 * norm_f_neq / (norm_Q_landau + 1e-30)
        print(f"    dt_test = {dt_test:.6e}")

        # Test 1: H-theorem
        print(f"\n  TEST 1: H-theorem (entropy dissipation)")

        h_landau = test_h_theorem(f, Q_landau, f_eq, v_jax, dv, "Landau")
        print(f"    Landau: entropy_production = {h_landau['entropy_production']:.6e}")
        print(f"            → {h_landau['interpretation']}")

        h_bgk = test_h_theorem(f, Q_bgk, f_eq, v_jax, dv, "BGK")
        print(f"    BGK:    entropy_production = {h_bgk['entropy_production']:.6e}")
        print(f"            → {h_bgk['interpretation']}")

        all_results['h_theorem_landau'].append(h_landau)
        all_results['h_theorem_bgk'].append(h_bgk)

        # Test 2: Direct relaxation
        print(f"\n  TEST 2: Direct relaxation (does Q drive f toward f_eq?)")

        relax_landau = test_relaxation(f, Q_landau, f_eq, dt_test, "Landau")
        print(f"    Landau: ||f-f_eq|| before = {relax_landau['dist_before']:.6e}")
        print(f"            ||f-f_eq|| after  = {relax_landau['dist_after']:.6e}")
        print(f"            change = {relax_landau['delta_dist']:.6e} ({relax_landau['relative_change']*100:.2f}%)")
        print(f"            → {relax_landau['interpretation']}")

        relax_bgk = test_relaxation(f, Q_bgk, f_eq, dt_test, "BGK")
        print(f"    BGK:    ||f-f_eq|| before = {relax_bgk['dist_before']:.6e}")
        print(f"            ||f-f_eq|| after  = {relax_bgk['dist_after']:.6e}")
        print(f"            change = {relax_bgk['delta_dist']:.6e} ({relax_bgk['relative_change']*100:.2f}%)")
        print(f"            → {relax_bgk['interpretation']}")

        all_results['relaxation_landau'].append(relax_landau)
        all_results['relaxation_bgk'].append(relax_bgk)

        # Test 3: Inner product
        print(f"\n  TEST 3: Inner product <Q_BGK, Q_Landau>")

        inner = test_inner_product(Q_bgk, Q_landau)
        print(f"    <Q_BGK, Q_Landau> = {inner['inner_product']:.6e}")
        print(f"    cosine similarity = {inner['cosine_similarity']:.4f}")
        print(f"    → {inner['interpretation']}")

        all_results['inner_product'].append(inner)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Count violations
    h_landau_violations = sum(1 for r in all_results['h_theorem_landau'] if not r['sign_correct'])
    h_bgk_violations = sum(1 for r in all_results['h_theorem_bgk'] if not r['sign_correct'])
    relax_landau_violations = sum(1 for r in all_results['relaxation_landau'] if not r['relaxing_toward_eq'])
    relax_bgk_violations = sum(1 for r in all_results['relaxation_bgk'] if not r['relaxing_toward_eq'])
    inner_violations = sum(1 for r in all_results['inner_product'] if not r['same_direction'])

    n_tests = len(snapshot_indices)

    print(f"\nH-theorem violations:")
    print(f"  Landau: {h_landau_violations}/{n_tests}")
    print(f"  BGK:    {h_bgk_violations}/{n_tests}")

    print(f"\nRelaxation violations:")
    print(f"  Landau: {relax_landau_violations}/{n_tests}")
    print(f"  BGK:    {relax_bgk_violations}/{n_tests}")

    print(f"\nInner product (opposite direction):")
    print(f"  {inner_violations}/{n_tests} snapshots show opposite directions")

    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    if h_landau_violations > 0 and h_bgk_violations == 0:
        print("\n*** LANDAU OPERATOR HAS WRONG SIGN ***")
        print("The Landau operator is INCREASING entropy, which violates H-theorem.")
        print("Suggested fix: Negate Q_Landau (multiply by -1).")
    elif h_bgk_violations > 0 and h_landau_violations == 0:
        print("\n*** BGK OPERATOR HAS WRONG SIGN ***")
        print("The BGK operator is INCREASING entropy, which violates H-theorem.")
        print("Suggested fix: Use Q_BGK = (f - f_eq)/tau instead of (f_eq - f)/tau.")
    elif h_landau_violations > 0 and h_bgk_violations > 0:
        print("\n*** BOTH OPERATORS MAY HAVE ISSUES ***")
        print("Both operators show entropy increase. Check the implementations.")
    elif inner_violations > 0 and h_landau_violations == 0 and h_bgk_violations == 0:
        print("\n*** PHYSICS INSIGHT: Operators are locally opposite but both dissipative ***")
        print("Both operators satisfy H-theorem but point in different directions.")
        print("This could mean:")
        print("  1. They agree on the EQUILIBRIUM but disagree on the PATH to reach it")
        print("  2. The Landau operator captures physics that BGK cannot represent")
        print("  3. Check if both operators are using the SAME equilibrium definition")
    else:
        print("\n*** BOTH OPERATORS APPEAR CORRECT ***")
        print("Both satisfy H-theorem and drive toward equilibrium.")

    return all_results


def plot_diagnostics(results, output_dir):
    """Create diagnostic plots."""
    os.makedirs(output_dir, exist_ok=True)

    times = results['times_tested']
    n_tests = len(times)

    # Extract data
    h_landau = [r['entropy_production'] for r in results['h_theorem_landau']]
    h_bgk = [r['entropy_production'] for r in results['h_theorem_bgk']]

    relax_landau = [r['relative_change'] * 100 for r in results['relaxation_landau']]
    relax_bgk = [r['relative_change'] * 100 for r in results['relaxation_bgk']]

    inner_prod = [r['inner_product'] for r in results['inner_product']]
    cos_sim = [r['cosine_similarity'] for r in results['inner_product']]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) H-theorem test
    ax = axes[0, 0]
    x_pos = np.arange(n_tests)
    width = 0.35
    ax.bar(x_pos - width/2, h_landau, width, label='Landau', color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, h_bgk, width, label='BGK', color='red', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Snapshot')
    ax.set_ylabel('Entropy Production <Q, ln(f/f_eq)>')
    ax.set_title('H-theorem Test\n(Should be ≤ 0 for correct sign)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f't={t:.3f}' for t in times], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Color regions
    ymin, ymax = ax.get_ylim()
    ax.fill_between([-0.5, n_tests-0.5], 0, ymax, alpha=0.1, color='red', label='Wrong sign region')
    ax.fill_between([-0.5, n_tests-0.5], ymin, 0, alpha=0.1, color='green', label='Correct region')

    # (0,1) Relaxation test
    ax = axes[0, 1]
    ax.bar(x_pos - width/2, relax_landau, width, label='Landau', color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, relax_bgk, width, label='BGK', color='red', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Snapshot')
    ax.set_ylabel('Change in ||f - f_eq|| (%)')
    ax.set_title('Relaxation Test\n(Should be < 0 for correct sign)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f't={t:.3f}' for t in times], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ymin, ymax = ax.get_ylim()
    ax.fill_between([-0.5, n_tests-0.5], 0, ymax, alpha=0.1, color='red')
    ax.fill_between([-0.5, n_tests-0.5], ymin, 0, alpha=0.1, color='green')

    # (1,0) Inner product
    ax = axes[1, 0]
    colors = ['green' if ip > 0 else 'red' for ip in inner_prod]
    ax.bar(x_pos, inner_prod, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Snapshot')
    ax.set_ylabel('<Q_BGK, Q_Landau>')
    ax.set_title('Inner Product Test\n(Should be > 0 if both point toward equilibrium)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f't={t:.3f}' for t in times], rotation=45)
    ax.grid(True, alpha=0.3)

    # (1,1) Cosine similarity
    ax = axes[1, 1]
    colors = ['green' if cs > 0 else 'red' for cs in cos_sim]
    ax.bar(x_pos, cos_sim, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=1, color='gray', linestyle=':', linewidth=1, label='Perfect alignment')
    ax.axhline(y=-1, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Snapshot')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Cosine Similarity between Q_BGK and Q_Landau\n(+1: same direction, -1: opposite)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f't={t:.3f}' for t in times], rotation=45)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = f"{output_dir}/sign_diagnostic_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic plot saved to {fig_path}")
    plt.close()

    return fig_path


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Diagnose sign errors in BGK/Landau operators')
    parser.add_argument('--data_file', type=str,
                        default='data/landau_1d/landau_Nx65536_Nv1024_Nt16384_20260129_152948_f.npy',
                        help='Path to f_history .npy file')
    parser.add_argument('--grid_file', type=str,
                        default='data/landau_1d/landau_Nx65536_Nv1024_Nt16384_20260129_152948_grid.npz',
                        help='Path to grid .npz file')
    parser.add_argument('--config_file', type=str,
                        default='data/landau_1d/landau_Nx65536_Nv1024_Nt16384_20260129_152948_config.json',
                        help='Path to config .json file')
    parser.add_argument('--snapshots', type=int, nargs='+', default=None,
                        help='Snapshot indices to test (default: auto-select)')
    args = parser.parse_args()

    # Run diagnostics
    results = run_diagnostics(
        data_file=args.data_file,
        grid_file=args.grid_file,
        config_file=args.config_file,
        snapshot_indices=args.snapshots
    )

    # Create plots
    plot_diagnostics(results, output_dir='figures/landau_1d')

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
