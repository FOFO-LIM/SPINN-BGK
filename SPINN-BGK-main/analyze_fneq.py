"""
Analyze f_neq magnitude evolution over time.

For the BGK equation, we expect:
1. f_neq initially increases due to spatial non-equilibrium (density gradient creates velocity distribution deviation)
2. f_neq then decreases due to collisions driving the system toward equilibrium

The decay time scale is τ = Kn, so for Kn=0.01, we should see significant decay by t=0.05-0.1.
"""

import jax
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import argparse
import os
from glob import glob

from src.nn import Siren
from src.x3v3 import x3v3, smooth
from utils.transform import trapezoidal_rule


class spinn_eval(x3v3):
    """Minimal spinn class for evaluation only."""
    def __init__(self, T=0.1, X=0.5, V=6.0, Nv=256, width=128, depth=3, rank=256, w0=10.0, Kn=0.01):
        super().__init__(T, X, V, Kn)
        layers = [1] + [width for _ in range(depth - 1)] + [rank]
        self.init, self.apply = Siren(layers, w0)
        self.rank = rank
        self.ic = smooth(X, V)
        self.v, self.w = trapezoidal_rule(Nv, -V, V)
        self.wv = self.w * self.v
        self.wvv = self.wv * self.v

    def compute_f_neq_magnitude(self, params, t_values, n_spatial=16, n_velocity=32):
        """
        Compute the magnitude of f_neq at different time points.

        Returns:
            t_values: array of time points
            f_neq_magnitudes: array of L2 norms of f_neq at each time
            f_neq_mean_abs: array of mean absolute values of f_neq
        """
        # Create spatial and velocity grids
        x = np.linspace(-self.X[1], self.X[1], n_spatial)
        y = np.linspace(-self.X[1], self.X[1], n_spatial)
        z = np.linspace(-self.X[1], self.X[1], n_spatial)
        vx = np.linspace(-self.V[1], self.V[1], n_velocity)
        vy = np.linspace(-self.V[1], self.V[1], n_velocity)
        vz = np.linspace(-self.V[1], self.V[1], n_velocity)

        f_neq_l2_norms = []
        f_neq_mean_abs = []
        f_eq_l2_norms = []
        f_total_l2_norms = []

        for t in t_values:
            t_arr = np.array([t])

            # Compute f_neq
            _f_neq = self._f_neq(params, t_arr, x, y, z, vx, vy, vz)
            f_neq = np.einsum("az,bz,cz,dz,ez,fz,gz->abcdefg", *_f_neq)
            f_neq = f_neq.squeeze()  # Remove singleton t dimension

            # Compute f_eq for comparison
            rho, u, temp = self._f_eq(params, t_arr, x, y, z)
            f_eq = self.maxwellian(rho, u, temp, vx, vy, vz).squeeze()

            # Compute total f
            f_total = f_eq + self.alpha * f_neq

            # Compute magnitudes
            f_neq_l2 = np.sqrt(np.mean(f_neq**2))
            f_neq_abs = np.mean(np.abs(f_neq))
            f_eq_l2 = np.sqrt(np.mean(f_eq**2))
            f_total_l2 = np.sqrt(np.mean(f_total**2))

            f_neq_l2_norms.append(float(f_neq_l2))
            f_neq_mean_abs.append(float(f_neq_abs))
            f_eq_l2_norms.append(float(f_eq_l2))
            f_total_l2_norms.append(float(f_total_l2))

            print(f"t={t:.4f}: |f_neq|_L2={f_neq_l2:.6e}, |f_eq|_L2={f_eq_l2:.6e}, ratio={f_neq_l2/f_eq_l2:.6e}")

        return {
            't': onp.array(t_values),
            'f_neq_l2': onp.array(f_neq_l2_norms),
            'f_neq_mean_abs': onp.array(f_neq_mean_abs),
            'f_eq_l2': onp.array(f_eq_l2_norms),
            'f_total_l2': onp.array(f_total_l2_norms),
        }


def find_latest_params(data_dir, Kn=None):
    """Find the most recent parameter file."""
    pattern = f"{data_dir}/spinn_Kn{Kn}*_params.npy" if Kn else f"{data_dir}/spinn_*_params.npy"
    files = glob(pattern)
    if not files:
        raise FileNotFoundError(f"No parameter files found matching {pattern}")
    # Sort by modification time, get newest
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def main(params_file: str = None, Kn: float = 0.01, T: float = 0.1, n_time_points: int = 21,
         n_spatial: int = 16, n_velocity: int = 32, rank: int = 256):
    """
    Analyze f_neq magnitude evolution.

    Args:
        params_file: Path to parameters file. If None, finds latest for given Kn.
        Kn: Knudsen number
        T: Final time
        n_time_points: Number of time points to evaluate
        n_spatial: Spatial grid resolution
        n_velocity: Velocity grid resolution
        rank: Model rank
    """
    data_dir = "data/x3v3/smooth"

    if params_file is None:
        params_file = find_latest_params(data_dir, Kn)

    print(f"Loading parameters from: {params_file}")
    params = onp.load(params_file, allow_pickle=True)

    # Extract config from corresponding config file
    config_file = params_file.replace("_params.npy", "_config.json")
    if os.path.exists(config_file):
        import json
        with open(config_file) as f:
            config = json.load(f)
        T = config.get('T', T)
        Kn = config.get('Kn', Kn)
        rank = config.get('rank', rank)
        print(f"Config: Kn={Kn}, T={T}, rank={rank}")

    # Create model
    model = spinn_eval(T=T, X=0.5, V=6.0, Kn=Kn, rank=rank)

    # Time points
    t_values = np.linspace(0, T, n_time_points)

    print(f"\nAnalyzing f_neq magnitude over t=[0, {T}]")
    print(f"Relaxation time τ = Kn = {Kn}")
    print(f"T/τ = {T/Kn:.2f}")
    print("-" * 60)

    # Compute magnitudes
    results = model.compute_f_neq_magnitude(params, t_values, n_spatial, n_velocity)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: f_neq L2 norm vs time
    ax = axes[0, 0]
    ax.plot(results['t'], results['f_neq_l2'], 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Time t')
    ax.set_ylabel('$||f_{neq}||_{L^2}$')
    ax.set_title(f'Non-equilibrium Magnitude (Kn={Kn})')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=Kn, color='r', linestyle='--', alpha=0.5, label=f'τ=Kn={Kn}')
    ax.axvline(x=3*Kn, color='orange', linestyle='--', alpha=0.5, label=f'3τ={3*Kn}')
    ax.legend()

    # Plot 2: f_neq / f_eq ratio vs time
    ax = axes[0, 1]
    ratio = results['f_neq_l2'] / results['f_eq_l2']
    ax.plot(results['t'], ratio, 'g-o', linewidth=2, markersize=4)
    ax.set_xlabel('Time t')
    ax.set_ylabel('$||f_{neq}||_{L^2} / ||f_{eq}||_{L^2}$')
    ax.set_title('Non-equilibrium to Equilibrium Ratio')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=Kn, color='r', linestyle='--', alpha=0.5, label=f'τ=Kn={Kn}')
    ax.legend()

    # Plot 3: Semi-log plot of f_neq
    ax = axes[1, 0]
    ax.semilogy(results['t'], results['f_neq_l2'], 'b-o', linewidth=2, markersize=4, label='$||f_{neq}||_{L^2}$')
    ax.semilogy(results['t'], results['f_eq_l2'], 'r-s', linewidth=2, markersize=4, label='$||f_{eq}||_{L^2}$')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Magnitude (log scale)')
    ax.set_title('Distribution Function Magnitudes')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: f_neq normalized by initial value
    ax = axes[1, 1]
    # Skip t=0 if f_neq is zero there
    if results['f_neq_l2'][0] > 1e-10:
        f_neq_normalized = results['f_neq_l2'] / results['f_neq_l2'][0]
        ax.plot(results['t'], f_neq_normalized, 'b-o', linewidth=2, markersize=4, label='Normalized $||f_{neq}||$')
    else:
        # Normalize by max value instead
        f_neq_normalized = results['f_neq_l2'] / np.max(results['f_neq_l2'])
        ax.plot(results['t'], f_neq_normalized, 'b-o', linewidth=2, markersize=4, label='Normalized $||f_{neq}||$ (by max)')

    # Theoretical decay exp(-t/τ) from peak
    t_theory = results['t']
    peak_idx = onp.argmax(results['f_neq_l2'])
    if peak_idx > 0:
        decay_theory = onp.exp(-(t_theory - t_theory[peak_idx]) / Kn)
        decay_theory[t_theory < t_theory[peak_idx]] = onp.nan
        ax.plot(t_theory, decay_theory * f_neq_normalized[peak_idx], 'r--', linewidth=2,
                alpha=0.7, label=f'$\\exp(-t/\\tau)$ from peak')

    ax.set_xlabel('Time t')
    ax.set_ylabel('Normalized Magnitude')
    ax.set_title('f_neq Decay Dynamics')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save figure
    fig_file = params_file.replace("_params.npy", "_fneq_analysis.png")
    plt.savefig(fig_file, dpi=150)
    print(f"\nFigure saved: {fig_file}")

    # Save results
    results_file = params_file.replace("_params.npy", "_fneq_results.npy")
    onp.save(results_file, results)
    print(f"Results saved: {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    peak_idx = onp.argmax(results['f_neq_l2'])
    print(f"Peak f_neq at t = {results['t'][peak_idx]:.4f}")
    print(f"Peak f_neq magnitude = {results['f_neq_l2'][peak_idx]:.6e}")
    print(f"Initial f_neq = {results['f_neq_l2'][0]:.6e}")
    print(f"Final f_neq = {results['f_neq_l2'][-1]:.6e}")
    if peak_idx > 0:
        decay_ratio = results['f_neq_l2'][-1] / results['f_neq_l2'][peak_idx]
        print(f"Decay from peak to final = {decay_ratio:.4f} ({100*(1-decay_ratio):.1f}% reduction)")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    import fire
    fire.Fire(main)
