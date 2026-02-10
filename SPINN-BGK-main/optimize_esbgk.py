#!/usr/bin/env python3
"""
Optimize ES-BGK (Ellipsoidal Statistical BGK) operator to approximate Landau collision.

ES-BGK model:
    Q_ES-BGK = (f_ES - f) / τ

where f_ES is an anisotropic Gaussian (ellipsoidal distribution):
    f_ES = ρ / sqrt(2π T_eff) * exp(-(v-u)² / (2 T_eff))

In 1D velocity space, we parameterize:
    T_eff = α * T  (α is a scaling factor)

So the optimization finds optimal (τ, α) that minimize:
    ||Q_ES-BGK - Q_Landau||

When α = 1, ES-BGK reduces to standard BGK.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from typing import Tuple, Dict, Optional, Callable
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


def compute_es_distribution(rho: np.ndarray, u: np.ndarray, T: np.ndarray,
                            v: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute ES (Ellipsoidal Statistical) distribution with scaled temperature.

    Args:
        rho: Density, shape (N_x,)
        u: Mean velocity, shape (N_x,)
        T: Temperature, shape (N_x,)
        v: Velocity grid, shape (N_v,)
        alpha: Temperature scaling factor (T_eff = α * T)

    Returns:
        f_ES: ES distribution, shape (N_x, N_v)
    """
    T_eff = alpha * np.maximum(T, 1e-10)
    return (rho[:, None] / np.sqrt(2 * np.pi * T_eff[:, None]) *
            np.exp(-(v[None, :] - u[:, None])**2 / (2 * T_eff[:, None])))


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


def compute_bgk_error(f: np.ndarray, Q_target: np.ndarray,
                      rho: np.ndarray, u: np.ndarray, T: np.ndarray,
                      v: np.ndarray, tau: float,
                      norm: str = 'L2') -> float:
    """
    Compute error between BGK operator and target collision operator.

    Args:
        f: Distribution function, shape (N_x, N_v)
        Q_target: Target collision operator (e.g., Landau), shape (N_x, N_v)
        rho, u, T: Moments
        v: Velocity grid
        tau: Relaxation time
        norm: 'L2', 'L1', or 'Linf'

    Returns:
        error: Scalar error value
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


def compute_esbgk_error(f: np.ndarray, Q_target: np.ndarray,
                        rho: np.ndarray, u: np.ndarray, T: np.ndarray,
                        v: np.ndarray, tau: float, alpha: float,
                        norm: str = 'L2') -> float:
    """
    Compute error between ES-BGK operator and target collision operator.

    Args:
        f: Distribution function, shape (N_x, N_v)
        Q_target: Target collision operator (e.g., Landau), shape (N_x, N_v)
        rho, u, T: Moments
        v: Velocity grid
        tau: Relaxation time
        alpha: Temperature scaling factor
        norm: 'L2', 'L1', or 'Linf'

    Returns:
        error: Scalar error value
    """
    f_es = compute_es_distribution(rho, u, T, v, alpha)
    Q_esbgk = (f_es - f) / tau
    residual = Q_esbgk - Q_target

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

    For L2 norm, uses analytical formula: τ_opt = A/B
    For other norms, uses numerical optimization.

    Args:
        f: Distribution function, shape (N_x, N_v)
        Q_target: Target collision operator, shape (N_x, N_v)
        rho, u, T: Moments
        v: Velocity grid
        tau_range: Search range for τ
        norm: 'L2', 'L1', or 'Linf'

    Returns:
        Dictionary with optimal τ, error, and diagnostic info
    """
    f_eq = compute_maxwellian(rho, u, T, v)
    f_neq = f_eq - f

    if norm == 'L2':
        # Analytical solution: τ_opt = A/B
        # Error(τ) = A/τ² - 2B/τ + C
        A = np.mean(f_neq**2)
        B = np.mean(f_neq * Q_target)
        C = np.mean(Q_target**2)

        if B > 0 and A > 0:
            tau_opt_analytical = A / B
            # Clip to search range
            tau_opt = np.clip(tau_opt_analytical, tau_range[0], tau_range[1])
        else:
            # No interior minimum, check boundaries
            err_low = compute_bgk_error(f, Q_target, rho, u, T, v, tau_range[0], norm)
            err_high = compute_bgk_error(f, Q_target, rho, u, T, v, tau_range[1], norm)
            tau_opt = tau_range[0] if err_low < err_high else tau_range[1]
            tau_opt_analytical = A / B if B != 0 else np.inf

        error = compute_bgk_error(f, Q_target, rho, u, T, v, tau_opt, norm)

        return {
            'tau_opt': tau_opt,
            'tau_analytical': tau_opt_analytical,
            'error': error,
            'A': A, 'B': B, 'C': C,
            'norm': norm
        }
    else:
        # Numerical optimization for L1, Linf
        def objective(log_tau):
            tau = np.exp(log_tau)
            return compute_bgk_error(f, Q_target, rho, u, T, v, tau, norm)

        result = minimize_scalar(objective,
                                 bounds=(np.log(tau_range[0]), np.log(tau_range[1])),
                                 method='bounded')
        tau_opt = np.exp(result.x)
        error = result.fun

        return {
            'tau_opt': tau_opt,
            'error': error,
            'norm': norm
        }


def optimize_esbgk(f: np.ndarray, Q_target: np.ndarray,
                   rho: np.ndarray, u: np.ndarray, T: np.ndarray,
                   v: np.ndarray,
                   tau_range: Tuple[float, float] = (0.01, 1000),
                   alpha_range: Tuple[float, float] = (0.1, 10.0),
                   norm: str = 'L2',
                   n_grid: int = 50) -> Dict:
    """
    Find optimal (τ, α) for ES-BGK.

    Args:
        f: Distribution function, shape (N_x, N_v)
        Q_target: Target collision operator, shape (N_x, N_v)
        rho, u, T: Moments
        v: Velocity grid
        tau_range: Search range for τ
        alpha_range: Search range for α (temperature scaling)
        norm: 'L2', 'L1', or 'Linf'
        n_grid: Number of grid points for initial search

    Returns:
        Dictionary with optimal (τ, α), error, and comparison with BGK
    """
    # Grid search for initial guess
    tau_grid = np.logspace(np.log10(tau_range[0]), np.log10(tau_range[1]), n_grid)
    alpha_grid = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_grid)

    error_grid = np.zeros((n_grid, n_grid))
    for i, tau in enumerate(tau_grid):
        for j, alpha in enumerate(alpha_grid):
            error_grid[i, j] = compute_esbgk_error(f, Q_target, rho, u, T, v,
                                                    tau, alpha, norm)

    # Find best grid point
    min_idx = np.unravel_index(np.argmin(error_grid), error_grid.shape)
    tau_init = tau_grid[min_idx[0]]
    alpha_init = alpha_grid[min_idx[1]]

    # Refine with local optimization
    def objective(params):
        log_tau, log_alpha = params
        tau = np.exp(log_tau)
        alpha = np.exp(log_alpha)
        return compute_esbgk_error(f, Q_target, rho, u, T, v, tau, alpha, norm)

    result = minimize(objective,
                      x0=[np.log(tau_init), np.log(alpha_init)],
                      bounds=[(np.log(tau_range[0]), np.log(tau_range[1])),
                              (np.log(alpha_range[0]), np.log(alpha_range[1]))],
                      method='L-BFGS-B')

    tau_opt = np.exp(result.x[0])
    alpha_opt = np.exp(result.x[1])
    error_esbgk = result.fun

    # Compare with standard BGK (α = 1)
    bgk_result = optimize_bgk_tau(f, Q_target, rho, u, T, v, tau_range, norm)

    return {
        'tau_opt': tau_opt,
        'alpha_opt': alpha_opt,
        'error_esbgk': error_esbgk,
        'error_bgk': bgk_result['error'],
        'tau_bgk': bgk_result['tau_opt'],
        'improvement': (bgk_result['error'] - error_esbgk) / bgk_result['error'] * 100,
        'error_grid': error_grid,
        'tau_grid': tau_grid,
        'alpha_grid': alpha_grid,
        'norm': norm
    }


def optimize_esbgk_trajectory(f_trajectory: np.ndarray,
                               Q_trajectory: np.ndarray,
                               v: np.ndarray, dv: float,
                               times: np.ndarray,
                               tau_range: Tuple[float, float] = (0.01, 1000),
                               alpha_range: Tuple[float, float] = (0.1, 10.0),
                               norm: str = 'L2') -> Dict:
    """
    Find optimal ES-BGK parameters along a trajectory.

    Args:
        f_trajectory: Distribution at multiple times, shape (N_t, N_x, N_v)
        Q_trajectory: Target collision operator, shape (N_t, N_x, N_v)
        v: Velocity grid, shape (N_v,)
        dv: Velocity spacing
        times: Time values, shape (N_t,)
        tau_range, alpha_range: Search ranges
        norm: Error norm

    Returns:
        Dictionary with trajectories of optimal parameters
    """
    N_t = len(times)

    results = {
        'times': times,
        'tau_esbgk': np.zeros(N_t),
        'alpha_esbgk': np.zeros(N_t),
        'tau_bgk': np.zeros(N_t),
        'error_esbgk': np.zeros(N_t),
        'error_bgk': np.zeros(N_t),
        'improvement': np.zeros(N_t),
        'norm': norm
    }

    for t_idx in range(N_t):
        f = f_trajectory[t_idx]
        Q_target = Q_trajectory[t_idx]

        rho, u, T = compute_moments(f, v, dv)

        opt_result = optimize_esbgk(f, Q_target, rho, u, T, v,
                                     tau_range, alpha_range, norm)

        results['tau_esbgk'][t_idx] = opt_result['tau_opt']
        results['alpha_esbgk'][t_idx] = opt_result['alpha_opt']
        results['tau_bgk'][t_idx] = opt_result['tau_bgk']
        results['error_esbgk'][t_idx] = opt_result['error_esbgk']
        results['error_bgk'][t_idx] = opt_result['error_bgk']
        results['improvement'][t_idx] = opt_result['improvement']

    return results


def plot_esbgk_optimization(result: Dict, save_path: Optional[str] = None):
    """
    Plot ES-BGK optimization results for a single time snapshot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error landscape
    ax = axes[0]
    im = ax.pcolormesh(result['alpha_grid'], result['tau_grid'],
                       np.log10(result['error_grid'] + 1e-20),
                       shading='auto', cmap='viridis')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('α (temperature scaling)')
    ax.set_ylabel('τ (relaxation time)')
    ax.set_title(f'ES-BGK Error Landscape ({result["norm"]} norm)')
    plt.colorbar(im, ax=ax, label='log₁₀(Error)')

    # Mark optima
    ax.plot(result['alpha_opt'], result['tau_opt'], 'r*', markersize=15,
            label=f'ES-BGK opt: τ={result["tau_opt"]:.3f}, α={result["alpha_opt"]:.3f}')
    ax.plot(1.0, result['tau_bgk'], 'wo', markersize=10, markeredgecolor='black',
            label=f'BGK opt: τ={result["tau_bgk"]:.3f}, α=1')
    ax.axvline(1.0, color='white', linestyle='--', alpha=0.5, label='α=1 (BGK)')
    ax.legend(loc='upper right')

    # Comparison bar chart
    ax = axes[1]
    models = ['BGK', 'ES-BGK']
    errors = [result['error_bgk'], result['error_esbgk']]
    colors = ['blue', 'green']
    bars = ax.bar(models, errors, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel(f'Error ({result["norm"]} norm)')
    ax.set_title(f'BGK vs ES-BGK (improvement: {result["improvement"]:.2f}%)')
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


def plot_esbgk_trajectory(results: Dict, save_path: Optional[str] = None):
    """
    Plot ES-BGK optimization results along a trajectory.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    times = results['times']

    # τ comparison
    ax = axes[0, 0]
    ax.semilogy(times, results['tau_bgk'], 'b-', label='BGK τ_opt', linewidth=2)
    ax.semilogy(times, results['tau_esbgk'], 'g-', label='ES-BGK τ_opt', linewidth=2)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Optimal τ')
    ax.set_title('Optimal Relaxation Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # α trajectory
    ax = axes[0, 1]
    ax.plot(times, results['alpha_esbgk'], 'g-', linewidth=2)
    ax.axhline(1.0, color='b', linestyle='--', label='α=1 (BGK)')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Optimal α')
    ax.set_title('ES-BGK Temperature Scaling Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error comparison
    ax = axes[1, 0]
    ax.semilogy(times, results['error_bgk'], 'b-', label='BGK error', linewidth=2)
    ax.semilogy(times, results['error_esbgk'], 'g-', label='ES-BGK error', linewidth=2)
    ax.set_xlabel('Time t')
    ax.set_ylabel(f'Error ({results["norm"]} norm)')
    ax.set_title('Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Improvement
    ax = axes[1, 1]
    ax.plot(times, results['improvement'], 'g-', linewidth=2)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.fill_between(times, 0, results['improvement'],
                    where=results['improvement'] > 0, alpha=0.3, color='green',
                    label='ES-BGK better')
    ax.fill_between(times, 0, results['improvement'],
                    where=results['improvement'] < 0, alpha=0.3, color='red',
                    label='BGK better')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('ES-BGK Improvement over BGK')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    return fig


# Example usage and testing
if __name__ == "__main__":
    print("ES-BGK Optimization Module")
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

    # Synthetic "Landau" collision operator (for testing)
    # In practice, this would come from the simulation
    Q_landau = -0.5 * (f - f_eq) + 0.1 * np.gradient(f, dv, axis=1)

    print("\nTest 1: Single snapshot optimization")
    print("-" * 40)

    # Test BGK optimization
    bgk_result = optimize_bgk_tau(f, Q_landau, rho, u, T, v, norm='L2')
    print(f"BGK optimal τ: {bgk_result['tau_opt']:.4f}")
    print(f"BGK error (L2): {bgk_result['error']:.4e}")

    # Test ES-BGK optimization
    esbgk_result = optimize_esbgk(f, Q_landau, rho, u, T, v, norm='L2')
    print(f"\nES-BGK optimal τ: {esbgk_result['tau_opt']:.4f}")
    print(f"ES-BGK optimal α: {esbgk_result['alpha_opt']:.4f}")
    print(f"ES-BGK error (L2): {esbgk_result['error_esbgk']:.4e}")
    print(f"Improvement over BGK: {esbgk_result['improvement']:.2f}%")

    # Test different norms
    print("\nTest 2: Different norms")
    print("-" * 40)
    for norm in ['L2', 'L1', 'Linf']:
        result = optimize_esbgk(f, Q_landau, rho, u, T, v, norm=norm)
        print(f"{norm}: τ={result['tau_opt']:.4f}, α={result['alpha_opt']:.4f}, "
              f"improvement={result['improvement']:.2f}%")

    # Plot results
    fig = plot_esbgk_optimization(esbgk_result, 'figures/landau_1d/esbgk_test.png')
    plt.close(fig)

    print("\n" + "=" * 60)
    print("Module ready for use with Landau simulation data.")
    print("Usage:")
    print("  from optimize_esbgk import optimize_esbgk, compute_moments")
    print("  result = optimize_esbgk(f, Q_landau, rho, u, T, v)")
