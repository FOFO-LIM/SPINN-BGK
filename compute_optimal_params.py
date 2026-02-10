"""
Compute optimal collision rate μ and relaxation time τ for SPINN-BGK model.

This script loads a trained SPINN model, evaluates the distribution function f(t,r,v)
at multiple time steps, and finds optimal parameters that minimize:
- |Q(f,f) - μ P(f)|  (Landau vs Fokker-Planck)
- |Q(f,f) - (M[f] - f)/τ|  (Landau vs BGK)

Uses the normalized operators from collision_operators.py.
"""

import os
import sys
import numpy as np
from datetime import datetime

# Set GPU before importing JAX
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import jax
import jax.numpy as jnp

# Add SPINN-BGK path
sys.path.insert(0, '/home/seongjaelim/SPINN-BGK-main(original)/SPINN-BGK-main')

from src.nn import Siren
from src.x3v3 import x3v3, smooth
from utils.transform import trapezoidal_rule

# Import collision operators
sys.path.insert(0, '/home/seongjaelim')
from collision_operators import (
    normalized_landau_operator_fast,
    normalized_fokker_planck_operator,
    compute_local_maxwellian,
    create_velocity_grid,
    create_position_grid,
)


class SPINNModel(x3v3):
    """SPINN model for evaluating f(t,x,y,z,vx,vy,vz)."""

    def __init__(self, T=5.0, X=0.5, V=6.0, Kn=0.01, rank=256, width=128, depth=3, w0=10.0):
        super().__init__(T, X, V, Kn)
        layers = [1] + [width for _ in range(depth - 1)] + [rank]
        self.init, self.apply = Siren(layers, w0)
        self.rank = rank
        self.ic = smooth(X, V)
        Nv = 256
        self.v, self.w = trapezoidal_rule(Nv, -V, V)
        self.wv = self.w * self.v
        self.wvv = self.wv * self.v


def load_spinn_model(params_path, config):
    """Load SPINN model with saved parameters."""
    model = SPINNModel(
        T=config['T'],
        X=config['X'],
        V=config['V'],
        Kn=config['Kn'],
        rank=config['rank'],
    )
    params = np.load(params_path, allow_pickle=True)
    return model, params


def evaluate_f_on_grid(model, params, t_val, x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid):
    """Evaluate f at a specific time on the given grid."""
    t = jnp.array([t_val])
    x = jnp.array(x_grid)
    y = jnp.array(y_grid)
    z = jnp.array(z_grid)
    vx = jnp.array(vx_grid)
    vy = jnp.array(vy_grid)
    vz = jnp.array(vz_grid)

    f = model.f(params, t, x, y, z, vx, vy, vz)
    return np.array(f)


def main():
    print("=" * 60)
    print("Optimal Collision Parameter Computation")
    print("=" * 60)

    # File paths
    base_path = '/home/seongjaelim/SPINN-BGK-main(original)/SPINN-BGK-main/data/x3v3/smooth/'
    params_file = 'spinn_Kn0.01_rank256_ngrid16_gpu4_20260114_175452_params.npy'
    config_file = 'spinn_Kn0.01_rank256_ngrid16_gpu4_20260114_175452_config.npy'

    params_path = base_path + params_file
    config_path = base_path + config_file

    # Load config
    config = np.load(config_path, allow_pickle=True).item()
    print(f"Loaded config: Kn={config['Kn']}, rank={config['rank']}, T={config['T']}")

    # Load model
    print("Loading SPINN model...")
    model, params = load_spinn_model(params_path, config)
    print(f"Model loaded. Device: {jax.devices()[0]}")

    # Grid parameters for evaluation
    # Use smaller grid for computational feasibility
    nr = 4   # Spatial grid points per dimension
    nv = 16  # Velocity grid points per dimension

    X = config['X']  # 0.5
    V = config['V']  # 6.0
    T_max = config['T']  # 5.0

    # Create evaluation grids
    x_grid = np.linspace(-X, X, nr)
    y_grid = np.linspace(-X, X, nr)
    z_grid = np.linspace(-X, X, nr)

    vx_grid = np.linspace(-V, V, nv)
    vy_grid = np.linspace(-V, V, nv)
    vz_grid = np.linspace(-V, V, nv)

    dv = vx_grid[1] - vx_grid[0]
    dr = x_grid[1] - x_grid[0]

    # Time steps to evaluate
    n_times = 11  # 0, 0.5, 1.0, ..., 5.0
    time_steps = np.linspace(0, T_max, n_times)

    print(f"\nGrid: {nr}³ spatial × {nv}³ velocity")
    print(f"Time steps: {n_times} from t=0 to t={T_max}")
    print(f"Spatial domain: [-{X}, {X}]³")
    print(f"Velocity domain: [-{V}, {V}]³")

    # Parameter search settings
    n_samples = 20
    param_min = 0.01
    param_max = 10.0
    mu_values = np.logspace(np.log10(param_min), np.log10(param_max), n_samples)
    tau_values = np.logspace(np.log10(param_min), np.log10(param_max), n_samples)

    print(f"\nSearching {n_samples} log-uniform values from {param_min} to {param_max}")

    # Results storage
    results = {
        'Kn': config['Kn'],
        'time_steps': time_steps,
        'mu_values_searched': mu_values,
        'tau_values_searched': tau_values,
        'mu_opt': np.zeros(n_times),
        'mu_norms': np.zeros((n_times, n_samples)),
        'tau_opt': np.zeros(n_times),
        'tau_norms': np.zeros((n_times, n_samples)),
        'grid_params': {
            'nr': nr, 'nv': nv, 'X': X, 'V': V, 'dv': dv, 'dr': dr
        }
    }

    # Create evaluation function (params is pytree, not JIT-able directly)
    print("\nPreparing model evaluation...")

    # Convert params to JAX arrays where possible
    def convert_params(p):
        if isinstance(p, np.ndarray):
            return jnp.array(p)
        elif isinstance(p, list):
            return [convert_params(x) for x in p]
        elif isinstance(p, tuple):
            return tuple(convert_params(x) for x in p)
        else:
            return p

    params_jax = convert_params(list(params))
    print("Model ready.")

    # Phase space volume element
    dV = dv**3 * dr**3

    # Process each time step
    print("\n" + "-" * 60)
    for i, t_val in enumerate(time_steps):
        print(f"\nTime step {i+1}/{n_times}: t = {t_val:.2f}")

        # Evaluate f on grid
        print("  Evaluating f(t,r,v)...")
        f = model.f(params_jax, jnp.array([t_val]), jnp.array(x_grid), jnp.array(y_grid),
                  jnp.array(z_grid), jnp.array(vx_grid), jnp.array(vy_grid), jnp.array(vz_grid))
        f = np.array(f)

        # f has shape (1, nr, nr, nr, nv, nv, nv) - squeeze the time dimension
        if f.ndim == 7:
            f = f[0]  # Now (nr, nr, nr, nv, nv, nv)

        print(f"  f shape: {f.shape}, range: [{f.min():.2e}, {f.max():.2e}]")

        # Ensure non-negative
        f = np.maximum(f, 1e-30)

        # Compute normalized Landau operator Q(f,f)
        print("  Computing Landau operator Q(f,f)...")
        Q = normalized_landau_operator_fast(f, vx_grid, vy_grid, vz_grid, dv)
        print(f"  Q range: [{Q.min():.2e}, {Q.max():.2e}]")

        # Compute normalized Fokker-Planck operator P(f)
        print("  Computing Fokker-Planck operator P(f)...")
        P = normalized_fokker_planck_operator(f, vx_grid, vy_grid, vz_grid, dv)
        print(f"  P range: [{P.min():.2e}, {P.max():.2e}]")

        # Compute local Maxwellian M[f]
        print("  Computing local Maxwellian M[f]...")
        M = compute_local_maxwellian(f, vx_grid, vy_grid, vz_grid, dv)

        # Find optimal μ
        print("  Finding optimal μ...")
        mu_norms = np.zeros(n_samples)
        for j, mu in enumerate(mu_values):
            diff = Q - mu * P
            mu_norms[j] = np.sum(np.abs(diff)) * dV

        idx_mu = np.argmin(mu_norms)
        results['mu_opt'][i] = mu_values[idx_mu]
        results['mu_norms'][i] = mu_norms
        print(f"  Optimal μ = {mu_values[idx_mu]:.4f}, L1 norm = {mu_norms[idx_mu]:.4e}")

        # Find optimal τ
        print("  Finding optimal τ...")
        tau_norms = np.zeros(n_samples)
        for j, tau in enumerate(tau_values):
            bgk = (M - f) / tau
            diff = Q - bgk
            tau_norms[j] = np.sum(np.abs(diff)) * dV

        idx_tau = np.argmin(tau_norms)
        results['tau_opt'][i] = tau_values[idx_tau]
        results['tau_norms'][i] = tau_norms
        print(f"  Optimal τ = {tau_values[idx_tau]:.4f}, L1 norm = {tau_norms[idx_tau]:.4e}")

    # Save results
    date_str = datetime.now().strftime("%Y%m%d")
    output_file = f"/home/seongjaelim/optimal_params_Kn0.01_{date_str}.npy"
    np.save(output_file, results)
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Time':>8} | {'μ_opt':>10} | {'τ_opt':>10}")
    print("-" * 35)
    for i, t_val in enumerate(time_steps):
        print(f"{t_val:>8.2f} | {results['mu_opt'][i]:>10.4f} | {results['tau_opt'][i]:>10.4f}")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
