"""
Plot f_neq magnitude evolution over time.

Hypothesis: f_neq increases initially (diffusion from spatially non-uniform IC)
           then decreases (reaching spatial equilibrium via collisions)
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from src.nn import Siren
from src.x3v3 import x3v3, smooth
from utils.transform import trapezoidal_rule


class SpinnEvaluator(x3v3):
    """SPINN model for evaluation."""
    def __init__(self, T=0.1, X=0.5, V=6.0, Nv=64, width=128, depth=3, rank=256, w0=10.0, Kn=0.01):
        super().__init__(T, X, V, Kn)
        layers = [1] + [width for _ in range(depth - 1)] + [rank]
        self.init, self.apply = Siren(layers, w0)
        self.rank = rank
        self.ic = smooth(X, V)
        self.v, self.w = trapezoidal_rule(Nv, -V, V)
        self.wv = self.w * self.v
        self.wvv = self.wv * self.v


def main():
    # Load the trained parameters (T=1.0 model)
    params_file = "data/x3v3/smooth/spinn_Kn0.01_rank256_ngrid12_gpu4_20260107_010026_params.npy"
    print(f"Loading parameters from: {params_file}")
    params = np.load(params_file, allow_pickle=True)

    # Model parameters
    Kn = 0.01
    T = 1.0
    rank = 256

    # Create model
    model = SpinnEvaluator(T=T, Kn=Kn, rank=rank, Nv=64)

    # Evaluation grids (smaller for efficiency)
    n_spatial = 12
    n_velocity = 24
    n_time = 41

    x = jnp.linspace(-0.5, 0.5, n_spatial)
    y = jnp.linspace(-0.5, 0.5, n_spatial)
    z = jnp.linspace(-0.5, 0.5, n_spatial)
    vx = jnp.linspace(-6.0, 6.0, n_velocity)
    vy = jnp.linspace(-6.0, 6.0, n_velocity)
    vz = jnp.linspace(-6.0, 6.0, n_velocity)

    t_values = np.linspace(0, T, n_time)

    print(f"Evaluating at {n_time} time points, spatial={n_spatial}^3, velocity={n_velocity}^3")
    print(f"Kn = {Kn}, T = {T}, T/Kn = {T/Kn}")
    print("-" * 60)

    # Store results
    f_neq_rms = []
    f_neq_max = []
    f_eq_rms = []
    f_ratio = []

    for i, t in enumerate(t_values):
        t_arr = jnp.array([t])

        # Compute f_neq (raw, before alpha scaling)
        _f_neq = model._f_neq(params, t_arr, x, y, z, vx, vy, vz)
        f_neq = jnp.einsum("az,bz,cz,dz,ez,fz,gz->abcdefg", *_f_neq)
        f_neq = model.alpha * f_neq.squeeze()  # Apply alpha and remove t dimension

        # Compute f_eq
        rho, u, temp = model._f_eq(params, t_arr, x, y, z)
        f_eq = model.maxwellian(rho, u, temp, vx, vy, vz).squeeze()

        # Compute metrics
        neq_rms = float(jnp.sqrt(jnp.mean(f_neq**2)))
        neq_max = float(jnp.max(jnp.abs(f_neq)))
        eq_rms = float(jnp.sqrt(jnp.mean(f_eq**2)))
        ratio = neq_rms / (eq_rms + 1e-10)

        f_neq_rms.append(neq_rms)
        f_neq_max.append(neq_max)
        f_eq_rms.append(eq_rms)
        f_ratio.append(ratio)

        if i % 10 == 0:
            print(f"t={t:.4f}: |f_neq|_rms={neq_rms:.4e}, |f_neq|/|f_eq|={ratio:.4e}")

    # Convert to arrays
    t_values = np.array(t_values)
    f_neq_rms = np.array(f_neq_rms)
    f_neq_max = np.array(f_neq_max)
    f_eq_rms = np.array(f_eq_rms)
    f_ratio = np.array(f_ratio)

    # Find peak
    peak_idx = np.argmax(f_neq_rms)
    peak_t = t_values[peak_idx]
    peak_val = f_neq_rms[peak_idx]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Initial |f_neq|_rms: {f_neq_rms[0]:.6e}")
    print(f"Peak |f_neq|_rms:    {peak_val:.6e} at t = {peak_t:.4f}")
    print(f"Final |f_neq|_rms:   {f_neq_rms[-1]:.6e}")
    print(f"Peak time / Kn:      {peak_t/Kn:.2f}")
    if peak_idx > 0:
        print(f"Decay from peak:     {f_neq_rms[-1]/peak_val:.4f} ({100*(1-f_neq_rms[-1]/peak_val):.1f}% reduction)")
    print("=" * 60)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: f_neq RMS over time
    ax = axes[0, 0]
    ax.plot(t_values, f_neq_rms, 'b-', linewidth=2, marker='o', markersize=3)
    ax.axvline(peak_t, color='r', linestyle='--', alpha=0.7, label=f'Peak at t={peak_t:.4f}')
    ax.axvline(Kn, color='g', linestyle=':', alpha=0.7, label=f'$\\tau$=Kn={Kn}')
    ax.axvline(3*Kn, color='orange', linestyle=':', alpha=0.7, label=f'3$\\tau$={3*Kn}')
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel(r'$\|f_{neq}\|_{RMS}$', fontsize=12)
    ax.set_title('Non-equilibrium Magnitude Evolution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, T])

    # Plot 2: f_neq max over time
    ax = axes[0, 1]
    ax.plot(t_values, f_neq_max, 'g-', linewidth=2, marker='s', markersize=3)
    ax.axvline(peak_t, color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel(r'$\|f_{neq}\|_{max}$', fontsize=12)
    ax.set_title('Non-equilibrium Max Magnitude', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, T])

    # Plot 3: Ratio |f_neq| / |f_eq|
    ax = axes[1, 0]
    ax.plot(t_values, f_ratio, 'm-', linewidth=2, marker='^', markersize=3)
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel(r'$\|f_{neq}\|_{RMS} / \|f_{eq}\|_{RMS}$', fontsize=12)
    ax.set_title('Non-equilibrium Fraction', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, T])

    # Plot 4: Semi-log with theoretical decay
    ax = axes[1, 1]
    ax.semilogy(t_values, f_neq_rms, 'b-', linewidth=2, marker='o', markersize=3, label=r'$\|f_{neq}\|_{RMS}$')

    # Theoretical exp(-t/tau) decay from peak
    if peak_idx > 0 and peak_idx < len(t_values) - 1:
        t_decay = t_values[peak_idx:]
        theory_decay = peak_val * np.exp(-(t_decay - peak_t) / Kn)
        ax.semilogy(t_decay, theory_decay, 'r--', linewidth=2, alpha=0.7, label=r'$\exp(-(t-t_{peak})/\tau)$')

    ax.axvline(peak_t, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('Magnitude (log scale)', fontsize=12)
    ax.set_title('Decay Comparison with BGK Relaxation', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, T])

    plt.suptitle(f'$f_{{neq}}$ Evolution Analysis (Kn={Kn}, T={T})', fontsize=16, y=1.02)
    plt.tight_layout()

    # Save
    output_file = "fneq_evolution_Kn0.01_T1.0.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    plt.show()


if __name__ == "__main__":
    main()
