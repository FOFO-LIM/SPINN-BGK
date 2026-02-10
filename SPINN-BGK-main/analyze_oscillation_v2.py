"""
Analyze f_neq oscillation using peak envelope method for robust β and ω estimation
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from src.nn import Siren
from src.x3v3 import x3v3, smooth
from utils.transform import trapezoidal_rule


class SpinnEvaluator(x3v3):
    """SPINN model for evaluation."""
    def __init__(self, T=5.0, X=0.5, V=6.0, Nv=64, width=128, depth=3, rank=256, w0=10.0, Kn=0.01):
        super().__init__(T, X, V, Kn)
        layers = [1] + [width for _ in range(depth - 1)] + [rank]
        self.init, self.apply = Siren(layers, w0)
        self.rank = rank
        self.ic = smooth(X, V)
        self.v, self.w = trapezoidal_rule(Nv, -V, V)
        self.wv = self.w * self.v
        self.wvv = self.wv * self.v


def exp_decay(t, A, beta):
    """Exponential decay: A * exp(-beta*t)"""
    return A * np.exp(-beta * t)


def main():
    # Load parameters
    params_file = "data/x3v3/smooth/spinn_Kn0.01_rank256_ngrid16_gpu4_20260114_175452_params.npy"
    print(f"Loading parameters from: {params_file}")
    params = np.load(params_file, allow_pickle=True)

    Kn = 0.01
    T = 5.0
    rank = 256
    model = SpinnEvaluator(T=T, Kn=Kn, rank=rank, Nv=64)

    # High resolution sampling
    n_spatial = 12
    n_velocity = 24
    n_time = 501

    x = jnp.linspace(-0.5, 0.5, n_spatial)
    y = jnp.linspace(-0.5, 0.5, n_spatial)
    z = jnp.linspace(-0.5, 0.5, n_spatial)
    vx = jnp.linspace(-6.0, 6.0, n_velocity)
    vy = jnp.linspace(-6.0, 6.0, n_velocity)
    vz = jnp.linspace(-6.0, 6.0, n_velocity)

    t_values = np.linspace(0, T, n_time)

    print(f"Evaluating at {n_time} time points...")

    f_neq_rms = []
    for i, t in enumerate(t_values):
        t_arr = jnp.array([t])
        _f_neq = model._f_neq(params, t_arr, x, y, z, vx, vy, vz)
        f_neq = jnp.einsum("az,bz,cz,dz,ez,fz,gz->abcdefg", *_f_neq)
        f_neq = model.alpha * f_neq.squeeze()
        neq_rms = float(jnp.sqrt(jnp.mean(f_neq**2)))
        f_neq_rms.append(neq_rms)
        if i % 100 == 0:
            print(f"  {i}/{n_time} done")

    t_values = np.array(t_values)
    f_neq_rms = np.array(f_neq_rms)

    # Find peaks and troughs
    peaks, _ = find_peaks(f_neq_rms, distance=10)
    troughs, _ = find_peaks(-f_neq_rms, distance=10)

    # Filter to only use significant peaks (first few where signal is strong)
    peak_times = t_values[peaks]
    peak_vals = f_neq_rms[peaks]

    # Use only peaks where signal is above noise floor (> 1e-6)
    good_peaks = peak_vals > 1e-6
    peak_times_good = peak_times[good_peaks]
    peak_vals_good = peak_vals[good_peaks]

    trough_times = t_values[troughs]
    trough_vals = f_neq_rms[troughs]

    print("\n" + "=" * 70)
    print("OSCILLATION ANALYSIS")
    print("=" * 70)

    # ============================================
    # METHOD 1: Period from peak spacing
    # ============================================
    print("\n--- METHOD 1: Period from Peak Spacing ---")
    if len(peak_times_good) >= 2:
        periods = np.diff(peak_times_good)
        print(f"Peak times: {peak_times_good[:8]}")
        print(f"Periods between peaks: {periods[:7]}")

        # Use first few periods (more reliable)
        n_periods = min(5, len(periods))
        avg_period = np.mean(periods[:n_periods])
        std_period = np.std(periods[:n_periods])

        omega_from_peaks = 2 * np.pi / avg_period
        print(f"\nAverage period (first {n_periods}): T = {avg_period:.4f} ± {std_period:.4f}")
        print(f"Angular frequency: ω = 2π/T = {omega_from_peaks:.4f} rad/s")

    # ============================================
    # METHOD 2: Decay rate from peak envelope
    # ============================================
    print("\n--- METHOD 2: Decay Rate from Peak Envelope ---")
    if len(peak_times_good) >= 3:
        # Fit exponential to peaks
        try:
            popt, pcov = curve_fit(exp_decay, peak_times_good, peak_vals_good,
                                   p0=[peak_vals_good[0], 1.0],
                                   bounds=([0, 0], [np.inf, 100]))
            A_fit, beta_from_env = popt
            perr = np.sqrt(np.diag(pcov))

            print(f"Exponential fit to peaks: A·exp(-β·t)")
            print(f"  A = {A_fit:.4e} ± {perr[0]:.2e}")
            print(f"  β = {beta_from_env:.4f} ± {perr[1]:.4f}")
        except Exception as e:
            print(f"Exponential fit failed: {e}")
            # Fallback: linear fit to log(peaks)
            log_peaks = np.log(peak_vals_good)
            coeffs = np.polyfit(peak_times_good, log_peaks, 1)
            beta_from_env = -coeffs[0]
            A_fit = np.exp(coeffs[1])
            print(f"Linear fit to log(peaks):")
            print(f"  β = {beta_from_env:.4f}")

    # ============================================
    # METHOD 3: Half-period from peak-to-trough
    # ============================================
    print("\n--- METHOD 3: Half-Period from Peak-Trough ---")
    half_periods = []
    for i, pt in enumerate(peak_times_good[:6]):
        # Find nearest trough after this peak
        later_troughs = trough_times[trough_times > pt]
        if len(later_troughs) > 0:
            nearest_trough = later_troughs[0]
            half_period = nearest_trough - pt
            half_periods.append(half_period)
            if i < 5:
                print(f"  Peak at t={pt:.3f} → Trough at t={nearest_trough:.3f}, half-period = {half_period:.4f}")

    if half_periods:
        avg_half_period = np.mean(half_periods[:4])
        full_period_from_half = 2 * avg_half_period
        omega_from_half = 2 * np.pi / full_period_from_half
        print(f"\nAverage half-period: {avg_half_period:.4f}")
        print(f"Full period: T = {full_period_from_half:.4f}")
        print(f"Angular frequency: ω = {omega_from_half:.4f} rad/s")

    # ============================================
    # FINAL ESTIMATES
    # ============================================
    print("\n" + "=" * 70)
    print("FINAL ESTIMATES")
    print("=" * 70)

    # Best estimates
    beta_best = beta_from_env
    omega_best = omega_from_peaks
    period_best = avg_period

    print(f"\n  β = {beta_best:.4f}  (decay rate)")
    print(f"  ω = {omega_best:.4f}  (angular frequency, rad/s)")
    print(f"")
    print(f"  Period T = {period_best:.4f}")
    print(f"  Frequency f = {omega_best/(2*np.pi):.4f} Hz")
    print(f"  Decay time 1/β = {1/beta_best:.4f}")
    print(f"  Quality factor Q = ω/(2β) = {omega_best/(2*beta_best):.2f}")
    print(f"")
    print(f"  f_neq ~ exp(-{beta_best:.2f}·t) · cos({omega_best:.2f}·t)")
    print(f"")
    print(f"  Relation to Kn = {Kn}:")
    print(f"    β·Kn = {beta_best*Kn:.4f}")
    print(f"    ω·Kn = {omega_best*Kn:.4f}")
    print(f"    T/Kn = {period_best/Kn:.1f}")
    print("=" * 70)

    # ============================================
    # VISUALIZATION
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Data with peaks marked
    ax = axes[0, 0]
    ax.plot(t_values, f_neq_rms, 'b-', linewidth=1, alpha=0.8, label='$|f_{neq}|_{RMS}$')
    ax.plot(peak_times_good, peak_vals_good, 'ro', markersize=8, label='Peaks')
    ax.plot(trough_times[:len(peak_times_good)], trough_vals[:len(peak_times_good)], 'g^', markersize=6, label='Troughs')

    # Damped cosine fit for visualization
    t_fit = np.linspace(0, T, 1000)
    # Shift time so first peak is at phase 0
    t_shift = peak_times_good[0]
    f_fit = A_fit * np.exp(-beta_best * t_fit) * np.abs(np.cos(omega_best * (t_fit - t_shift)))
    ax.plot(t_fit, f_fit, 'r--', linewidth=2, alpha=0.7,
            label=f'$A e^{{-\\beta t}} |\\cos(\\omega t)|$\n$\\beta$={beta_best:.2f}, $\\omega$={omega_best:.2f}')

    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel(r'$\|f_{neq}\|_{RMS}$', fontsize=12)
    ax.set_title('Data with Oscillation Fit', fontsize=14)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 3])  # Focus on first 3 time units

    # Plot 2: Log scale with envelope
    ax = axes[0, 1]
    ax.semilogy(t_values, f_neq_rms, 'b-', linewidth=1, alpha=0.8)
    ax.semilogy(peak_times_good, peak_vals_good, 'ro', markersize=8, label='Peaks')
    t_env = np.linspace(0, T, 100)
    envelope = A_fit * np.exp(-beta_best * t_env)
    ax.semilogy(t_env, envelope, 'r--', linewidth=2, label=f'Envelope: $e^{{-{beta_best:.2f}t}}$')
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel(r'$\|f_{neq}\|_{RMS}$ (log)', fontsize=12)
    ax.set_title('Exponential Decay Envelope', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([1e-7, 1e-3])

    # Plot 3: Period analysis
    ax = axes[1, 0]
    if len(periods) > 0:
        ax.bar(range(len(periods[:10])), periods[:10], color='steelblue', alpha=0.7)
        ax.axhline(avg_period, color='r', linestyle='--', linewidth=2, label=f'Mean = {avg_period:.4f}')
        ax.set_xlabel('Peak-to-Peak Index', fontsize=12)
        ax.set_ylabel('Period', fontsize=12)
        ax.set_title('Oscillation Period Between Successive Peaks', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')

    summary = f"""
    ╔══════════════════════════════════════════════════════╗
    ║         DAMPED OSCILLATION PARAMETERS                ║
    ╠══════════════════════════════════════════════════════╣
    ║                                                      ║
    ║   Model:  f_neq ~ A·exp(-β·t)·cos(ω·t)              ║
    ║                                                      ║
    ╠══════════════════════════════════════════════════════╣
    ║   DECAY RATE                                         ║
    ║   ─────────────────────────────────                  ║
    ║   β = {beta_best:.4f}                                       ║
    ║   Decay time τ = 1/β = {1/beta_best:.4f}                    ║
    ║                                                      ║
    ╠══════════════════════════════════════════════════════╣
    ║   OSCILLATION FREQUENCY                              ║
    ║   ─────────────────────────────────                  ║
    ║   ω = {omega_best:.4f} rad/s                               ║
    ║   Period T = 2π/ω = {period_best:.4f}                       ║
    ║   Frequency f = {omega_best/(2*np.pi):.4f} Hz                       ║
    ║                                                      ║
    ╠══════════════════════════════════════════════════════╣
    ║   QUALITY FACTOR                                     ║
    ║   ─────────────────────────────────                  ║
    ║   Q = ω/(2β) = {omega_best/(2*beta_best):.2f}                            ║
    ║                                                      ║
    ╠══════════════════════════════════════════════════════╣
    ║   RELATION TO Kn = {Kn}                             ║
    ║   ─────────────────────────────────                  ║
    ║   β·Kn = {beta_best*Kn:.4f}                                 ║
    ║   ω·Kn = {omega_best*Kn:.4f}                                 ║
    ║   T/Kn = {period_best/Kn:.1f}                                     ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'$f_{{neq}}$ Oscillation Analysis (Kn={Kn}, T={T})', fontsize=16, y=1.02)
    plt.tight_layout()

    output_file = "fneq_oscillation_analysis_v2_Kn0.01_T5.0.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")


if __name__ == "__main__":
    main()
