#!/usr/bin/env python3
"""
Optimize Shakhov model to approximate Landau collision.

Shakhov model:
    Q_Shakhov = (f_S - f) / τ

where f_S is the Shakhov distribution that corrects the heat flux:
    f_S = f_M * [1 + (1-Pr) * c * q / (p * T) * (c²/(2T) - 3/2)]

In 1D velocity space:
    - c = v - u (peculiar velocity)
    - q = ∫ c * (c²/2) * f dv (heat flux)
    - p = ρ * T (pressure)

The Shakhov model introduces:
    - Pr: Prandtl number (physical value ~2/3 for monatomic gases)

Optimization finds optimal (τ, Pr) that minimize:
    ||Q_Shakhov - Q_Landau||

When Pr = 1, Shakhov reduces to standard BGK.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt


def compute_maxwellian(rho: np.ndarray, u: np.ndarray, T: np.ndarray,
                       v: np.ndarray) -> np.ndarray:
    """
    Compute Maxwellian distribution f_M(x, v).

    Args:
        rho: Density, shape (N_x,)
        u: Mean velocity, shape (N_x,)
        T: Temperature, shape (N_x,)
        v: Velocity grid, shape (N_v,)

    Returns:
        f_M: Maxwellian, shape (N_x, N_v)
    """
    T_safe = np.maximum(T, 1e-10)
    return (rho[:, None] / np.sqrt(2 * np.pi * T_safe[:, None]) *
            np.exp(-(v[None, :] - u[:, None])**2 / (2 * T_safe[:, None])))


def compute_moments(f: np.ndarray, v: np.ndarray, dv: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute moments (ρ, u, T) from distribution function.

    Args:
        f: Distribution function, shape (N_x, N_v)
        v: Velocity grid, shape (N_v,)
        dv: Velocity grid spacing

    Returns:
        rho: Density, shape (N_x,)
        u: Mean velocity, shape (N_x,)
        T: Temperature, shape (N_x,)
    """
    # Trapezoidal weights
    w = np.ones(len(v)) * dv
    w[0] = dv / 2
    w[-1] = dv / 2

    rho = np.sum(f * w, axis=1)
    rho = np.maximum(rho, 1e-16)

    momentum = np.sum(f * v * w, axis=1)
    u = momentum / rho

    energy = np.sum(f * v**2 * w, axis=1)
    T = np.maximum((energy / rho) - u**2, 1e-10)

    return rho, u, T


def compute_heat_flux(f: np.ndarray, v: np.ndarray, dv: float,
                      u: np.ndarray) -> np.ndarray:
    """
    Compute heat flux q = ∫ c * (c²/2) * f dv in 1D.

    Args:
        f: Distribution function, shape (N_x, N_v)
        v: Velocity grid, shape (N_v,)
        dv: Velocity grid spacing
        u: Mean velocity, shape (N_x,)

    Returns:
        q: Heat flux, shape (N_x,)
    """
    # Trapezoidal weights
    w = np.ones(len(v)) * dv
    w[0] = dv / 2
    w[-1] = dv / 2

    # Peculiar velocity: c = v - u
    c = v[None, :] - u[:, None]  # Shape (N_x, N_v)

    # Heat flux: q = ∫ c * (c²/2) * f dv = (1/2) ∫ c³ * f dv
    q = 0.5 * np.sum(c**3 * f * w, axis=1)

    return q


def compute_shakhov_distribution(rho: np.ndarray, u: np.ndarray, T: np.ndarray,
                                  q: np.ndarray, v: np.ndarray,
                                  Pr: float) -> np.ndarray:
    """
    Compute Shakhov distribution with heat flux correction.

    f_S = f_M * [1 + (1-Pr) * c * q / (p * T) * (c²/(2T) - 3/2)]

    For 1D, using the form that preserves mass, momentum, energy,
    and gives the correct heat flux.

    Args:
        rho: Density, shape (N_x,)
        u: Mean velocity, shape (N_x,)
        T: Temperature, shape (N_x,)
        q: Heat flux, shape (N_x,)
        v: Velocity grid, shape (N_v,)
        Pr: Prandtl number

    Returns:
        f_S: Shakhov distribution, shape (N_x, N_v)
    """
    T_safe = np.maximum(T, 1e-10)
    p = rho * T_safe  # Pressure

    # Maxwellian
    f_M = compute_maxwellian(rho, u, T_safe, v)

    # Peculiar velocity
    c = v[None, :] - u[:, None]  # Shape (N_x, N_v)

    # Shakhov correction factor
    # S = (1-Pr) * c * q / (p * T) * (c²/(2T) - 3/2)
    # In 1D, the factor 3/2 comes from (D+2)/2 where D=1
    correction = ((1 - Pr) * c * q[:, None] / (p[:, None] * T_safe[:, None]) *
                  (c**2 / (2 * T_safe[:, None]) - 1.5))

    f_S = f_M * (1 + correction)

    # Ensure non-negative (Shakhov can go negative for large deviations)
    # f_S = np.maximum(f_S, 0)

    return f_S


def compute_bgk_error(f: np.ndarray, Q_target: np.ndarray,
                      rho: np.ndarray, u: np.ndarray, T: np.ndarray,
                      v: np.ndarray, tau: float,
                      norm: str = 'L2') -> float:
    """
    Compute error between BGK operator and target collision operator.
    """
    f_eq = compute_maxwellian(rho, u, T, v)
    Q_bgk = (f_eq - f) / tau
    residual = Q_bgk - Q_target

    if norm == 'L2':
        return np.mean(residual**2)
    elif norm == 'L1':
        return np.mean(np.abs(residual))
    elif norm == 'Linf':
        return np.max(np.abs(residual))
    else:
        raise ValueError(f"Unknown norm: {norm}")


def compute_shakhov_error(f: np.ndarray, Q_target: np.ndarray,
                          rho: np.ndarray, u: np.ndarray, T: np.ndarray,
                          q: np.ndarray, v: np.ndarray,
                          tau: float, Pr: float,
                          norm: str = 'L2') -> float:
    """
    Compute error between Shakhov operator and target collision operator.

    Args:
        f: Distribution function, shape (N_x, N_v)
        Q_target: Target collision operator (e.g., Landau), shape (N_x, N_v)
        rho, u, T: Moments
        q: Heat flux
        v: Velocity grid
        tau: Relaxation time
        Pr: Prandtl number
        norm: 'L2', 'L1', or 'Linf'

    Returns:
        error: Scalar error value
    """
    f_S = compute_shakhov_distribution(rho, u, T, q, v, Pr)
    Q_shakhov = (f_S - f) / tau
    residual = Q_shakhov - Q_target

    if norm == 'L2':
        return np.mean(residual**2)
    elif norm == 'L1':
        return np.mean(np.abs(residual))
    elif norm == 'Linf':
        return np.max(np.abs(residual))
    else:
        raise ValueError(f"Unknown norm: {norm}")


def optimize_bgk_tau(f: np.ndarray, Q_target: np.ndarray,
                     rho: np.ndarray, u: np.ndarray, T: np.ndarray,
                     v: np.ndarray, tau_range: Tuple[float, float] = (0.01, 1000),
                     norm: str = 'L2') -> Dict:
    """
    Find optimal τ for standard BGK.
    """
    f_eq = compute_maxwellian(rho, u, T, v)
    f_neq = f_eq - f

    if norm == 'L2':
        A = np.mean(f_neq**2)
        B = np.mean(f_neq * Q_target)

        if B > 0 and A > 0:
            tau_opt_analytical = A / B
            tau_opt = np.clip(tau_opt_analytical, tau_range[0], tau_range[1])
        else:
            err_low = compute_bgk_error(f, Q_target, rho, u, T, v, tau_range[0], norm)
            err_high = compute_bgk_error(f, Q_target, rho, u, T, v, tau_range[1], norm)
            tau_opt = tau_range[0] if err_low < err_high else tau_range[1]

        error = compute_bgk_error(f, Q_target, rho, u, T, v, tau_opt, norm)
        return {'tau_opt': tau_opt, 'error': error, 'norm': norm}
    else:
        def objective(log_tau):
            tau = np.exp(log_tau)
            return compute_bgk_error(f, Q_target, rho, u, T, v, tau, norm)

        result = minimize_scalar(objective,
                                 bounds=(np.log(tau_range[0]), np.log(tau_range[1])),
                                 method='bounded')
        tau_opt = np.exp(result.x)
        error = result.fun
        return {'tau_opt': tau_opt, 'error': error, 'norm': norm}


def optimize_shakhov(f: np.ndarray, Q_target: np.ndarray,
                     rho: np.ndarray, u: np.ndarray, T: np.ndarray,
                     v: np.ndarray, dv: float,
                     tau_range: Tuple[float, float] = (0.01, 1000),
                     Pr_range: Tuple[float, float] = (0.1, 2.0),
                     norm: str = 'L2',
                     n_grid: int = 50) -> Dict:
    """
    Find optimal (τ, Pr) for Shakhov model.

    Args:
        f: Distribution function, shape (N_x, N_v)
        Q_target: Target collision operator (e.g., Landau), shape (N_x, N_v)
        rho, u, T: Moments
        v: Velocity grid
        dv: Velocity spacing
        tau_range: Search range for τ
        Pr_range: Search range for Prandtl number
        norm: 'L2', 'L1', or 'Linf'
        n_grid: Number of grid points for initial search

    Returns:
        Dictionary with optimal (τ, Pr), error, and comparison with BGK
    """
    # Compute heat flux
    q = compute_heat_flux(f, v, dv, u)

    # Grid search for initial guess
    tau_grid = np.logspace(np.log10(tau_range[0]), np.log10(tau_range[1]), n_grid)
    Pr_grid = np.linspace(Pr_range[0], Pr_range[1], n_grid)

    error_grid = np.zeros((n_grid, n_grid))
    for i, tau in enumerate(tau_grid):
        for j, Pr in enumerate(Pr_grid):
            error_grid[i, j] = compute_shakhov_error(f, Q_target, rho, u, T, q,
                                                      v, tau, Pr, norm)

    # Find best grid point
    min_idx = np.unravel_index(np.argmin(error_grid), error_grid.shape)
    tau_init = tau_grid[min_idx[0]]
    Pr_init = Pr_grid[min_idx[1]]

    # Refine with local optimization
    def objective(params):
        log_tau, Pr = params
        tau = np.exp(log_tau)
        return compute_shakhov_error(f, Q_target, rho, u, T, q, v, tau, Pr, norm)

    result = minimize(objective,
                      x0=[np.log(tau_init), Pr_init],
                      bounds=[(np.log(tau_range[0]), np.log(tau_range[1])),
                              (Pr_range[0], Pr_range[1])],
                      method='L-BFGS-B')

    tau_opt = np.exp(result.x[0])
    Pr_opt = result.x[1]
    error_shakhov = result.fun

    # Compare with standard BGK (Pr = 1)
    bgk_result = optimize_bgk_tau(f, Q_target, rho, u, T, v, tau_range, norm)

    return {
        'tau_opt': tau_opt,
        'Pr_opt': Pr_opt,
        'error_shakhov': error_shakhov,
        'error_bgk': bgk_result['error'],
        'tau_bgk': bgk_result['tau_opt'],
        'improvement': (bgk_result['error'] - error_shakhov) / bgk_result['error'] * 100,
        'error_grid': error_grid,
        'tau_grid': tau_grid,
        'Pr_grid': Pr_grid,
        'heat_flux': q,
        'norm': norm
    }


def optimize_shakhov_trajectory(f_trajectory: np.ndarray,
                                 Q_trajectory: np.ndarray,
                                 v: np.ndarray, dv: float,
                                 times: np.ndarray,
                                 tau_range: Tuple[float, float] = (0.01, 1000),
                                 Pr_range: Tuple[float, float] = (0.1, 2.0),
                                 norm: str = 'L2') -> Dict:
    """
    Find optimal Shakhov parameters along a trajectory.

    Args:
        f_trajectory: Distribution at multiple times, shape (N_t, N_x, N_v)
        Q_trajectory: Target collision operator, shape (N_t, N_x, N_v)
        v: Velocity grid, shape (N_v,)
        dv: Velocity spacing
        times: Time values, shape (N_t,)
        tau_range, Pr_range: Search ranges
        norm: Error norm

    Returns:
        Dictionary with trajectories of optimal parameters
    """
    N_t = len(times)

    results = {
        'times': times,
        'tau_shakhov': np.zeros(N_t),
        'Pr_shakhov': np.zeros(N_t),
        'tau_bgk': np.zeros(N_t),
        'error_shakhov': np.zeros(N_t),
        'error_bgk': np.zeros(N_t),
        'improvement': np.zeros(N_t),
        'heat_flux_mean': np.zeros(N_t),
        'heat_flux_max': np.zeros(N_t),
        'norm': norm
    }

    for t_idx in range(N_t):
        f = f_trajectory[t_idx]
        Q_target = Q_trajectory[t_idx]

        rho, u, T = compute_moments(f, v, dv)

        opt_result = optimize_shakhov(f, Q_target, rho, u, T, v, dv,
                                       tau_range, Pr_range, norm)

        results['tau_shakhov'][t_idx] = opt_result['tau_opt']
        results['Pr_shakhov'][t_idx] = opt_result['Pr_opt']
        results['tau_bgk'][t_idx] = opt_result['tau_bgk']
        results['error_shakhov'][t_idx] = opt_result['error_shakhov']
        results['error_bgk'][t_idx] = opt_result['error_bgk']
        results['improvement'][t_idx] = opt_result['improvement']
        results['heat_flux_mean'][t_idx] = np.mean(np.abs(opt_result['heat_flux']))
        results['heat_flux_max'][t_idx] = np.max(np.abs(opt_result['heat_flux']))

    return results


def plot_shakhov_optimization(result: Dict, save_path: Optional[str] = None):
    """
    Plot Shakhov optimization results for a single time snapshot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error landscape
    ax = axes[0]
    im = ax.pcolormesh(result['Pr_grid'], result['tau_grid'],
                       np.log10(result['error_grid'] + 1e-20),
                       shading='auto', cmap='viridis')
    ax.set_yscale('log')
    ax.set_xlabel('Pr (Prandtl number)')
    ax.set_ylabel('τ (relaxation time)')
    ax.set_title(f'Shakhov Error Landscape ({result["norm"]} norm)')
    plt.colorbar(im, ax=ax, label='log₁₀(Error)')

    # Mark optima
    ax.plot(result['Pr_opt'], result['tau_opt'], 'r*', markersize=15,
            label=f'Shakhov opt: τ={result["tau_opt"]:.3f}, Pr={result["Pr_opt"]:.3f}')
    ax.plot(1.0, result['tau_bgk'], 'wo', markersize=10, markeredgecolor='black',
            label=f'BGK opt: τ={result["tau_bgk"]:.3f}, Pr=1')
    ax.axvline(1.0, color='white', linestyle='--', alpha=0.5, label='Pr=1 (BGK)')
    ax.axvline(2/3, color='cyan', linestyle=':', alpha=0.7, label='Pr=2/3 (physical)')
    ax.legend(loc='upper right')

    # Comparison bar chart
    ax = axes[1]
    models = ['BGK', 'Shakhov']
    errors = [result['error_bgk'], result['error_shakhov']]
    colors = ['blue', 'orange']
    bars = ax.bar(models, errors, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel(f'Error ({result["norm"]} norm)')
    ax.set_title(f'BGK vs Shakhov (improvement: {result["improvement"]:.2f}%)')
    ax.set_yscale('log')

    # Add value labels
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{err:.2e}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    return fig


def plot_shakhov_trajectory(results: Dict, save_path: Optional[str] = None):
    """
    Plot Shakhov optimization results along a trajectory.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    times = results['times']

    # τ comparison
    ax = axes[0, 0]
    ax.semilogy(times, results['tau_bgk'], 'b-', label='BGK τ_opt', linewidth=2)
    ax.semilogy(times, results['tau_shakhov'], 'orange', linestyle='-',
                label='Shakhov τ_opt', linewidth=2)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Optimal τ')
    ax.set_title('Optimal Relaxation Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pr trajectory
    ax = axes[0, 1]
    ax.plot(times, results['Pr_shakhov'], 'orange', linestyle='-', linewidth=2)
    ax.axhline(1.0, color='b', linestyle='--', label='Pr=1 (BGK)')
    ax.axhline(2/3, color='green', linestyle=':', linewidth=2, label='Pr=2/3 (physical)')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Optimal Pr')
    ax.set_title('Shakhov Prandtl Number')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Heat flux
    ax = axes[0, 2]
    ax.semilogy(times, results['heat_flux_mean'], 'purple', linestyle='-',
                label='Mean |q|', linewidth=2)
    ax.semilogy(times, results['heat_flux_max'], 'purple', linestyle='--',
                label='Max |q|', linewidth=2)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Heat flux |q|')
    ax.set_title('Heat Flux Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error comparison
    ax = axes[1, 0]
    ax.semilogy(times, results['error_bgk'], 'b-', label='BGK error', linewidth=2)
    ax.semilogy(times, results['error_shakhov'], 'orange', linestyle='-',
                label='Shakhov error', linewidth=2)
    ax.set_xlabel('Time t')
    ax.set_ylabel(f'Error ({results["norm"]} norm)')
    ax.set_title('Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Improvement
    ax = axes[1, 1]
    ax.plot(times, results['improvement'], 'orange', linestyle='-', linewidth=2)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.fill_between(times, 0, results['improvement'],
                    where=results['improvement'] > 0, alpha=0.3, color='orange',
                    label='Shakhov better')
    ax.fill_between(times, 0, results['improvement'],
                    where=results['improvement'] < 0, alpha=0.3, color='red',
                    label='BGK better')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Shakhov Improvement over BGK')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pr vs Heat flux correlation
    ax = axes[1, 2]
    sc = ax.scatter(results['heat_flux_mean'], results['Pr_shakhov'],
                    c=times, cmap='viridis', s=50, edgecolor='black')
    ax.axhline(2/3, color='green', linestyle=':', linewidth=2, label='Pr=2/3')
    ax.axhline(1.0, color='blue', linestyle='--', label='Pr=1')
    ax.set_xlabel('Mean |q| (heat flux)')
    ax.set_ylabel('Optimal Pr')
    ax.set_title('Pr vs Heat Flux')
    plt.colorbar(sc, ax=ax, label='Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    return fig


# Example usage and testing
if __name__ == "__main__":
    print("Shakhov Model Optimization Module")
    print("=" * 60)

    # Create synthetic test data
    np.random.seed(42)

    N_x = 64
    N_v = 128
    V = 6.0
    dv = 2 * V / (N_v - 1)
    v = np.linspace(-V, V, N_v)
    x = np.linspace(0, 1, N_x)

    # Synthetic distribution (perturbed Maxwellian)
    rho = 1.0 + 0.1 * np.sin(2 * np.pi * x)
    u = 0.1 * np.cos(2 * np.pi * x)
    T = 1.0 + 0.05 * np.sin(4 * np.pi * x)

    f_eq = compute_maxwellian(rho, u, T, v)
    perturbation = 0.1 * np.random.randn(N_x, N_v) * f_eq
    f = f_eq + perturbation
    f = np.maximum(f, 0)  # Ensure non-negative

    # Compute heat flux of the perturbed distribution
    q = compute_heat_flux(f, v, dv, u)
    print(f"Heat flux: mean={np.mean(np.abs(q)):.4e}, max={np.max(np.abs(q)):.4e}")

    # Synthetic "Landau" collision operator (for testing)
    Q_landau = -0.5 * (f - f_eq) + 0.1 * np.gradient(f, dv, axis=1)

    print("\nTest 1: Single snapshot optimization")
    print("-" * 40)

    # Test BGK optimization
    bgk_result = optimize_bgk_tau(f, Q_landau, rho, u, T, v, norm='L2')
    print(f"BGK optimal τ: {bgk_result['tau_opt']:.4f}")
    print(f"BGK error (L2): {bgk_result['error']:.4e}")

    # Test Shakhov optimization
    shakhov_result = optimize_shakhov(f, Q_landau, rho, u, T, v, dv, norm='L2')
    print(f"\nShakhov optimal τ: {shakhov_result['tau_opt']:.4f}")
    print(f"Shakhov optimal Pr: {shakhov_result['Pr_opt']:.4f}")
    print(f"Shakhov error (L2): {shakhov_result['error_shakhov']:.4e}")
    print(f"Improvement over BGK: {shakhov_result['improvement']:.2f}%")

    # Test different norms
    print("\nTest 2: Different norms")
    print("-" * 40)
    for norm in ['L2', 'L1', 'Linf']:
        result = optimize_shakhov(f, Q_landau, rho, u, T, v, dv, norm=norm)
        print(f"{norm}: τ={result['tau_opt']:.4f}, Pr={result['Pr_opt']:.4f}, "
              f"improvement={result['improvement']:.2f}%")

    # Plot results
    fig = plot_shakhov_optimization(shakhov_result, 'figures/landau_1d/shakhov_test.png')
    plt.close(fig)

    print("\n" + "=" * 60)
    print("Module ready for use with Landau simulation data.")
    print("Usage:")
    print("  from optimize_shakhov import optimize_shakhov, compute_moments")
    print("  result = optimize_shakhov(f, Q_landau, rho, u, T, v, dv)")
