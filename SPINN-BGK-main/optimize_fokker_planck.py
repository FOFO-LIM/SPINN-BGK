#!/usr/bin/env python3
"""
Optimize Fokker-Planck operator to approximate Landau collision.

Fokker-Planck model:
    Q_FP = mu * P(f)

where P(f) is the normalized Fokker-Planck operator:
    P(f) = d/dv [ M[f] d/dv (f / M[f]) ]

In 1D velocity space:
    - M[f] is the local Maxwellian with the same moments as f
    - mu is the effective collision rate

Optimization finds optimal mu that minimizes:
    ||mu * P(f) - Q_Landau||
"""

import numpy as np
from scipy.optimize import minimize_scalar
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
    Compute moments (rho, u, T) from distribution function.

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
    rho = np.maximum(rho, 1e-30)

    u = np.sum(f * v[None, :] * w, axis=1) / rho

    c_sq = (v[None, :] - u[:, None])**2
    T = np.sum(f * c_sq * w, axis=1) / rho

    return rho, u, T


def compute_fokker_planck_operator(f: np.ndarray, v: np.ndarray, dv: float,
                                    rho: np.ndarray, u: np.ndarray,
                                    T: np.ndarray) -> np.ndarray:
    """
    Compute the normalized 1D Fokker-Planck operator P(f).

    P(f) = d/dv [ M[f] d/dv (f / M[f]) ]

    Args:
        f: Distribution function, shape (N_x, N_v)
        v: Velocity grid, shape (N_v,)
        dv: Velocity grid spacing
        rho: Density, shape (N_x,)
        u: Mean velocity, shape (N_x,)
        T: Temperature, shape (N_x,)

    Returns:
        P: Fokker-Planck operator, shape (N_x, N_v)
    """
    # Local Maxwellian
    M = compute_maxwellian(rho, u, T, v)
    M = np.maximum(M, 1e-30)

    # f / M
    f_over_M = f / M

    # d/dv (f / M) using central differences
    grad_f_over_M = np.gradient(f_over_M, dv, axis=1)

    # M * d/dv (f / M)
    flux = M * grad_f_over_M

    # d/dv [ M * d/dv (f / M) ]
    P = np.gradient(flux, dv, axis=1)

    return P


def compute_fp_error(Q_target: np.ndarray, P: np.ndarray,
                     mu: float, norm: str = 'L2') -> float:
    """
    Compute error between mu * P(f) and target collision operator.

    Args:
        Q_target: Target collision operator, shape (N_x, N_v)
        P: Fokker-Planck operator P(f), shape (N_x, N_v)
        mu: Collision rate
        norm: 'L2', 'L1', or 'Linf'

    Returns:
        error: Scalar error value
    """
    residual = mu * P - Q_target

    if norm == 'L2':
        return np.mean(residual**2)
    elif norm == 'L1':
        return np.mean(np.abs(residual))
    elif norm == 'Linf':
        return np.max(np.abs(residual))
    else:
        raise ValueError(f"Unknown norm: {norm}")


def compute_bgk_error(f: np.ndarray, Q_target: np.ndarray,
                      rho: np.ndarray, u: np.ndarray, T: np.ndarray,
                      v: np.ndarray, tau: float, norm: str = 'L2') -> float:
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


def optimize_bgk_tau(f: np.ndarray, Q_target: np.ndarray,
                     rho: np.ndarray, u: np.ndarray, T: np.ndarray,
                     v: np.ndarray, tau_range: Tuple[float, float] = (0.01, 1000),
                     norm: str = 'L2') -> Dict:
    """
    Find optimal tau for standard BGK (used for comparison).
    """
    f_eq = compute_maxwellian(rho, u, T, v)
    f_neq = f_eq - f

    if norm == 'L2':
        A = np.mean(f_neq**2)
        B = np.mean(f_neq * Q_target)

        if B > 0 and A > 0:
            tau_opt = np.clip(A / B, tau_range[0], tau_range[1])
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
        return {'tau_opt': np.exp(result.x), 'error': result.fun, 'norm': norm}


def optimize_fokker_planck(f: np.ndarray, Q_target: np.ndarray,
                           rho: np.ndarray, u: np.ndarray, T: np.ndarray,
                           v: np.ndarray, dv: float,
                           mu_range: Tuple[float, float] = (0.001, 100.0),
                           norm: str = 'L2',
                           n_grid: int = 50) -> Dict:
    """
    Find optimal mu for Fokker-Planck operator.

    Args:
        f: Distribution function, shape (N_x, N_v)
        Q_target: Target collision operator (e.g., Landau), shape (N_x, N_v)
        rho, u, T: Moments
        v: Velocity grid
        dv: Velocity spacing
        mu_range: Search range for mu
        norm: 'L2', 'L1', or 'Linf'
        n_grid: Number of grid points for initial search

    Returns:
        Dictionary with optimal mu, error, and comparison with BGK
    """
    # Compute Fokker-Planck operator P(f)
    P = compute_fokker_planck_operator(f, v, dv, rho, u, T)

    if norm == 'L2':
        # Analytical solution for L2: mu_opt = <P, Q_target> / <P, P>
        numerator = np.mean(P * Q_target)
        denominator = np.mean(P**2)

        if denominator > 0 and numerator > 0:
            mu_opt = np.clip(numerator / denominator, mu_range[0], mu_range[1])
        else:
            # Fall back to grid search
            mu_grid = np.logspace(np.log10(mu_range[0]), np.log10(mu_range[1]), n_grid)
            errors = np.array([compute_fp_error(Q_target, P, mu, norm) for mu in mu_grid])
            mu_opt = mu_grid[np.argmin(errors)]

        error_fp = compute_fp_error(Q_target, P, mu_opt, norm)
    else:
        # Numerical optimization for L1, Linf
        def objective(log_mu):
            mu = np.exp(log_mu)
            return compute_fp_error(Q_target, P, mu, norm)

        result = minimize_scalar(objective,
                                 bounds=(np.log(mu_range[0]), np.log(mu_range[1])),
                                 method='bounded')
        mu_opt = np.exp(result.x)
        error_fp = result.fun

    # Compare with standard BGK
    bgk_result = optimize_bgk_tau(f, Q_target, rho, u, T, v, norm=norm)

    return {
        'mu_opt': mu_opt,
        'error_fp': error_fp,
        'error_bgk': bgk_result['error'],
        'tau_bgk': bgk_result['tau_opt'],
        'improvement': (bgk_result['error'] - error_fp) / bgk_result['error'] * 100,
        'norm': norm
    }


def plot_fokker_planck_optimization(result: Dict, save_path: Optional[str] = None):
    """
    Plot Fokker-Planck optimization result (error comparison bar chart).

    Args:
        result: Dictionary from optimize_fokker_planck
        save_path: Optional path to save figure

    Returns:
        fig: matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    models = ['BGK', 'Fokker-Planck']
    errors = [result['error_bgk'], result['error_fp']]
    colors = ['steelblue', 'crimson']

    ax.bar(models, errors, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel(f'Error ({result["norm"]})')
    ax.set_title(f'Fokker-Planck vs BGK\n'
                 f'mu_opt={result["mu_opt"]:.4f}, '
                 f'improvement={result["improvement"]:.1f}%')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved Fokker-Planck plot to {save_path}")

    return fig
