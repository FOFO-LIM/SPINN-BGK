"""
Analyze f_neq oscillation to estimate β and ω for f ~ e^(-βt) cos(ωt)
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

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


def damped_oscillation(t, A, beta, omega, phi, C):
    """Damped oscillation: A * exp(-beta*t) * cos(omega*t + phi) + C"""
    return A * np.exp(-beta * t) * np.cos(omega * t + phi) + C


def main():
    # Load the trained parameters (T=5.0 model)
    params_file = "data/x3v3/smooth/spinn_Kn0.01_rank256_ngrid16_gpu4_20260114_175452_params.npy"
    print(f"Loading parameters from: {params_file}")
    params = np.load(params_file, allow_pickle=True)

    # Model parameters
    Kn = 0.01
    T = 5.0
    rank = 256

    # Create model
    model = SpinnEvaluator(T=T, Kn=Kn, rank=rank, Nv=64)

    # Higher resolution for better oscillation analysis
    n_spatial = 12
    n_velocity = 24
    n_time = 201  # More points for better fitting

    x = jnp.linspace(-0.5, 0.5, n_spatial)
    y = jnp.linspace(-0.5, 0.5, n_spatial)
    z = jnp.linspace(-0.5, 0.5, n_spatial)
    vx = jnp.linspace(-6.0, 6.0, n_velocity)
    vy = jnp.linspace(-6.0, 6.0, n_velocity)
    vz = jnp.linspace(-6.0, 6.0, n_velocity)

    t_values = np.linspace(0, T, n_time)

    print(f"Evaluating at {n_time} time points for oscillation analysis...")
    print(f"Kn = {Kn}, T = {T}")
    print("-" * 60)

    # Collect data
    f_neq_rms = []

    for i, t in enumerate(t_values):
        t_arr = jnp.array([t])
        _f_neq = model._f_neq(params, t_arr, x, y, z, vx, vy, vz)
        f_neq = jnp.einsum("az,bz,cz,dz,ez,fz,gz->abcdefg", *_f_neq)
        f_neq = model.alpha * f_neq.squeeze()
        neq_rms = float(jnp.sqrt(jnp.mean(f_neq**2)))
        f_neq_rms.append(neq_rms)

        if i % 50 == 0:
            print(f"  t={t:.3f}: |f_neq|_rms={neq_rms:.4e}")

    t_values = np.array(t_values)
    f_neq_rms = np.array(f_neq_rms)

    # Find peaks for initial estimates
    peaks, _ = find_peaks(f_neq_rms)
    troughs, _ = find_peaks(-f_neq_rms)

    print("\n" + "=" * 60)
    print("PEAK ANALYSIS")
    print("=" * 60)

    if len(peaks) > 0:
        print(f"Peaks found at t = {t_values[peaks]}")
        print(f"Peak values: {f_neq_rms[peaks]}")

        if len(peaks) >= 2:
            # Estimate period from peak spacing
            peak_times = t_values[peaks]
            periods = np.diff(peak_times)
            avg_period = np.mean(periods)
            omega_est = 2 * np.pi / avg_period
            print(f"\nPeak-to-peak periods: {periods}")
            print(f"Average period: {avg_period:.4f}")
            print(f"Estimated ω from peaks: {omega_est:.4f} rad/s")

    if len(troughs) > 0:
        print(f"\nTroughs found at t = {t_values[troughs]}")

    # Estimate decay rate from envelope
    if len(peaks) >= 2:
        peak_vals = f_neq_rms[peaks]
        peak_times = t_values[peaks]
        # Fit exponential to peaks: log(peak) = log(A) - beta*t
        log_peaks = np.log(peak_vals)
        coeffs = np.polyfit(peak_times, log_peaks, 1)
        beta_est = -coeffs[0]
        print(f"\nEstimated β from peak envelope: {beta_est:.4f}")

    # Fit damped oscillation model
    print("\n" + "=" * 60)
    print("CURVE FITTING: f(t) = A * exp(-β*t) * cos(ω*t + φ) + C")
    print("=" * 60)

    # Initial guesses
    A0 = f_neq_rms.max() - f_neq_rms.min()
    C0 = f_neq_rms.min()

    if len(peaks) >= 2:
        omega0 = omega_est
        beta0 = beta_est
    else:
        omega0 = 10.0  # Default guess
        beta0 = 1.0

    phi0 = 0.0

    try:
        # Fit the model
        popt, pcov = curve_fit(
            damped_oscillation,
            t_values,
            f_neq_rms,
            p0=[A0, beta0, omega0, phi0, C0],
            bounds=([0, 0, 0, -np.pi, 0], [np.inf, 100, 100, np.pi, np.inf]),
            maxfev=10000
        )

        A_fit, beta_fit, omega_fit, phi_fit, C_fit = popt
        perr = np.sqrt(np.diag(pcov))

        print(f"\nFitted parameters:")
        print(f"  A     = {A_fit:.6e} ± {perr[0]:.2e}")
        print(f"  β     = {beta_fit:.4f} ± {perr[1]:.4f}")
        print(f"  ω     = {omega_fit:.4f} ± {perr[2]:.4f} rad/s")
        print(f"  φ     = {phi_fit:.4f} ± {perr[3]:.4f} rad")
        print(f"  C     = {C_fit:.6e} ± {perr[4]:.2e}")

        # Derived quantities
        period = 2 * np.pi / omega_fit
        freq = omega_fit / (2 * np.pi)
        decay_time = 1 / beta_fit

        print(f"\nDerived quantities:")
        print(f"  Period T = 2π/ω = {period:.4f}")
        print(f"  Frequency f = ω/(2π) = {freq:.4f} Hz")
        print(f"  Decay time τ = 1/β = {decay_time:.4f}")
        print(f"  Quality factor Q = ω/(2β) = {omega_fit/(2*beta_fit):.4f}")

        # Comparison with Kn
        print(f"\nComparison with Kn = {Kn}:")
        print(f"  β / (1/Kn) = β * Kn = {beta_fit * Kn:.4f}")
        print(f"  ω * Kn = {omega_fit * Kn:.4f}")
        print(f"  Period / Kn = {period / Kn:.2f}")

        fit_success = True

    except Exception as e:
        print(f"Curve fitting failed: {e}")
        fit_success = False
        beta_fit = beta_est if len(peaks) >= 2 else 1.0
        omega_fit = omega_est if len(peaks) >= 2 else 10.0

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Data with fit
    ax = axes[0, 0]
    ax.plot(t_values, f_neq_rms, 'b-', linewidth=1.5, label='Data', alpha=0.8)
    if fit_success:
        t_fine = np.linspace(0, T, 500)
        fit_curve = damped_oscillation(t_fine, A_fit, beta_fit, omega_fit, phi_fit, C_fit)
        ax.plot(t_fine, fit_curve, 'r--', linewidth=2, label=f'Fit: β={beta_fit:.3f}, ω={omega_fit:.3f}')
    if len(peaks) > 0:
        ax.plot(t_values[peaks], f_neq_rms[peaks], 'go', markersize=8, label='Peaks')
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel(r'$\|f_{neq}\|_{RMS}$', fontsize=12)
    ax.set_title('Data with Damped Oscillation Fit', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Log scale with envelope
    ax = axes[0, 1]
    ax.semilogy(t_values, f_neq_rms, 'b-', linewidth=1.5, label='Data')
    if fit_success:
        envelope = A_fit * np.exp(-beta_fit * t_fine) + C_fit
        ax.semilogy(t_fine, envelope, 'r--', linewidth=2, label=f'Envelope: exp(-{beta_fit:.3f}t)')
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel(r'$\|f_{neq}\|_{RMS}$ (log)', fontsize=12)
    ax.set_title('Exponential Decay Envelope', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Residuals
    ax = axes[1, 0]
    if fit_success:
        residuals = f_neq_rms - damped_oscillation(t_values, A_fit, beta_fit, omega_fit, phi_fit, C_fit)
        ax.plot(t_values, residuals, 'g-', linewidth=1)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_ylabel('Residual', fontsize=12)
        ax.set_title(f'Fit Residuals (RMS={np.sqrt(np.mean(residuals**2)):.2e})', fontsize=14)
    else:
        ax.text(0.5, 0.5, 'Fit failed', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Time t', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    DAMPED OSCILLATION ANALYSIS
    ═══════════════════════════════════════

    Model: f(t) = A·exp(-βt)·cos(ωt + φ) + C

    FITTED PARAMETERS:
    ──────────────────
    β (decay rate)    = {beta_fit:.4f}
    ω (angular freq)  = {omega_fit:.4f} rad/s

    DERIVED QUANTITIES:
    ──────────────────
    Period T = 2π/ω   = {2*np.pi/omega_fit:.4f}
    Frequency f       = {omega_fit/(2*np.pi):.4f} Hz
    Decay time 1/β    = {1/beta_fit:.4f}
    Quality factor Q  = {omega_fit/(2*beta_fit):.4f}

    RELATION TO Kn = {Kn}:
    ──────────────────
    β·Kn              = {beta_fit*Kn:.4f}
    ω·Kn              = {omega_fit*Kn:.4f}
    Period/Kn         = {2*np.pi/omega_fit/Kn:.1f}
    """

    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'$f_{{neq}}$ Oscillation Analysis (Kn={Kn}, T={T})', fontsize=16, y=1.02)
    plt.tight_layout()

    output_file = "fneq_oscillation_analysis_Kn0.01_T5.0.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"β = {beta_fit:.4f}  (decay rate)")
    print(f"ω = {omega_fit:.4f}  (angular frequency)")
    print(f"")
    print(f"f_neq ~ exp(-{beta_fit:.4f}·t) · cos({omega_fit:.4f}·t)")
    print("=" * 60)


if __name__ == "__main__":
    main()
