"""
Collision Operators for Kinetic Theory

This module implements three collision operators for distribution functions f(r, v)
in 6-dimensional phase space (3D position + 3D velocity):

1. Landau Collision Operator - for Coulomb collisions in plasmas
2. Fokker-Planck Collision Operator - diffusion-drift in velocity space
3. BGK (Bhatnagar-Gross-Krook) Operator - relaxation-time approximation

All operators assume f is discretized on a grid with shape:
    (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
"""

import numpy as np
from scipy import constants
from typing import Tuple, Optional, Callable
from functools import lru_cache


# Physical constants
EPSILON_0 = constants.epsilon_0
K_B = constants.k
M_E = constants.m_e
E_CHARGE = constants.e


def create_velocity_grid(
    v_max: float,
    nv: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Create a 3D velocity grid.

    Parameters
    ----------
    v_max : float
        Maximum velocity in each direction
    nv : int
        Number of grid points in each velocity direction

    Returns
    -------
    vx, vy, vz : ndarray
        1D arrays of velocity coordinates
    dv : float
        Velocity grid spacing
    """
    vx = np.linspace(-v_max, v_max, nv)
    vy = np.linspace(-v_max, v_max, nv)
    vz = np.linspace(-v_max, v_max, nv)
    dv = vx[1] - vx[0]
    return vx, vy, vz, dv


def create_position_grid(
    L: float,
    nr: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Create a 3D position grid.

    Parameters
    ----------
    L : float
        Domain size in each direction
    nr : int
        Number of grid points in each position direction

    Returns
    -------
    rx, ry, rz : ndarray
        1D arrays of position coordinates
    dr : float
        Position grid spacing
    """
    rx = np.linspace(0, L, nr)
    ry = np.linspace(0, L, nr)
    rz = np.linspace(0, L, nr)
    dr = rx[1] - rx[0]
    return rx, ry, rz, dr


def gradient_v(f: np.ndarray, dv: float, axis: int) -> np.ndarray:
    """
    Compute gradient of f with respect to velocity component.

    Parameters
    ----------
    f : ndarray
        Distribution function with shape (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    dv : float
        Velocity grid spacing
    axis : int
        Velocity axis (3, 4, or 5 for vx, vy, vz)

    Returns
    -------
    ndarray
        Gradient of f along specified velocity axis
    """
    return np.gradient(f, dv, axis=axis)


def divergence_v(
    Fx: np.ndarray,
    Fy: np.ndarray,
    Fz: np.ndarray,
    dv: float
) -> np.ndarray:
    """
    Compute divergence of vector field F in velocity space.

    Parameters
    ----------
    Fx, Fy, Fz : ndarray
        Components of vector field
    dv : float
        Velocity grid spacing

    Returns
    -------
    ndarray
        Divergence of F
    """
    div_x = np.gradient(Fx, dv, axis=3)
    div_y = np.gradient(Fy, dv, axis=4)
    div_z = np.gradient(Fz, dv, axis=5)
    return div_x + div_y + div_z


def maxwellian_3d(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    n: float,
    T: float,
    m: float,
    u: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute 3D Maxwellian distribution.

    Parameters
    ----------
    vx, vy, vz : ndarray
        1D velocity arrays
    n : float
        Number density
    T : float
        Temperature in Kelvin
    m : float
        Particle mass
    u : ndarray, optional
        Bulk velocity [ux, uy, uz], default is [0, 0, 0]

    Returns
    -------
    ndarray
        Maxwellian distribution with shape (nv_x, nv_y, nv_z)
    """
    if u is None:
        u = np.array([0.0, 0.0, 0.0])

    vth2 = 2 * K_B * T / m
    norm = n / (np.pi * vth2) ** 1.5

    VX, VY, VZ = np.meshgrid(vx, vy, vz, indexing='ij')
    v_sq = (VX - u[0])**2 + (VY - u[1])**2 + (VZ - u[2])**2

    return norm * np.exp(-v_sq / vth2)


def compute_moments(
    f: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute density, bulk velocity, and temperature from distribution.

    Parameters
    ----------
    f : ndarray
        Distribution function (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    vx, vy, vz : ndarray
        1D velocity arrays
    dv : float
        Velocity grid spacing

    Returns
    -------
    n : ndarray
        Number density (nr_x, nr_y, nr_z)
    u : ndarray
        Bulk velocity (nr_x, nr_y, nr_z, 3)
    T : ndarray
        Temperature (nr_x, nr_y, nr_z)
    """
    dv3 = dv ** 3

    # Create velocity meshgrids
    VX, VY, VZ = np.meshgrid(vx, vy, vz, indexing='ij')

    # Density: integral of f over velocity
    n = np.sum(f, axis=(3, 4, 5)) * dv3

    # Bulk velocity: (1/n) * integral of v*f over velocity
    u = np.zeros(f.shape[:3] + (3,))
    n_safe = np.where(n > 1e-30, n, 1e-30)

    u[..., 0] = np.sum(f * VX, axis=(3, 4, 5)) * dv3 / n_safe
    u[..., 1] = np.sum(f * VY, axis=(3, 4, 5)) * dv3 / n_safe
    u[..., 2] = np.sum(f * VZ, axis=(3, 4, 5)) * dv3 / n_safe

    # Temperature from second moment (assuming mass = 1 for simplicity)
    # T = (m / 3kB * n) * integral of |v - u|^2 * f dv
    v_rel_sq = ((VX - u[..., 0:1, None, None])**2 +
                (VY - u[..., 1:2, None, None])**2 +
                (VZ - u[..., 2:3, None, None])**2)

    # For now return kinetic energy per particle (multiply by m/3kB for temperature)
    T = np.sum(f * v_rel_sq, axis=(3, 4, 5)) * dv3 / (3 * n_safe)

    return n, u, T


# =============================================================================
# BGK (Bhatnagar-Gross-Krook) Collision Operator
# =============================================================================

def bgk_operator(
    f: np.ndarray,
    f_eq: np.ndarray,
    nu: float
) -> np.ndarray:
    """
    Compute the BGK collision operator.

    C_BGK[f] = -ν(f - f_eq)

    This is the simplest collision operator, modeling relaxation toward
    an equilibrium distribution at rate ν.

    Parameters
    ----------
    f : ndarray
        Distribution function (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    f_eq : ndarray
        Equilibrium distribution (same shape as f, or broadcastable)
    nu : float
        Collision frequency (1/s)

    Returns
    -------
    ndarray
        BGK collision operator evaluated at each point
    """
    return -nu * (f - f_eq)


def bgk_operator_maxwellian(
    f: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
    nu: float,
    m: float
) -> np.ndarray:
    """
    Compute BGK operator with self-consistent Maxwellian equilibrium.

    The equilibrium is computed from the local moments (n, u, T) of f.

    Parameters
    ----------
    f : ndarray
        Distribution function (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    vx, vy, vz : ndarray
        1D velocity arrays
    dv : float
        Velocity grid spacing
    nu : float
        Collision frequency
    m : float
        Particle mass

    Returns
    -------
    ndarray
        BGK collision operator
    """
    # Compute moments
    n, u, T_kin = compute_moments(f, vx, vy, vz, dv)

    # Convert kinetic energy to temperature
    T = m * T_kin / K_B
    T = np.maximum(T, 1e-10)  # Prevent division by zero

    # Build local Maxwellian at each spatial point
    VX, VY, VZ = np.meshgrid(vx, vy, vz, indexing='ij')

    vth2 = 2 * K_B * T / m  # Shape: (nr_x, nr_y, nr_z)
    vth2 = vth2[..., np.newaxis, np.newaxis, np.newaxis]
    n_exp = n[..., np.newaxis, np.newaxis, np.newaxis]

    u_exp = u[..., np.newaxis, np.newaxis, np.newaxis, :]

    v_rel_sq = ((VX - u_exp[..., 0])**2 +
                (VY - u_exp[..., 1])**2 +
                (VZ - u_exp[..., 2])**2)

    f_eq = n_exp / (np.pi * vth2) ** 1.5 * np.exp(-v_rel_sq / vth2)

    return bgk_operator(f, f_eq, nu)


# =============================================================================
# Fokker-Planck Collision Operator
# =============================================================================

def fokker_planck_operator(
    f: np.ndarray,
    A: Tuple[np.ndarray, np.ndarray, np.ndarray],
    D: np.ndarray,
    dv: float
) -> np.ndarray:
    """
    Compute the Fokker-Planck collision operator.

    C_FP[f] = -∂/∂v_i (A_i f) + (1/2) ∂²/∂v_i∂v_j (D_ij f)

    Parameters
    ----------
    f : ndarray
        Distribution function (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    A : tuple of 3 ndarrays
        Friction/drag coefficient vector (Ax, Ay, Az), each same shape as f
    D : ndarray
        Diffusion tensor, shape (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z, 3, 3)
        D[..., i, j] is the (i,j) component
    dv : float
        Velocity grid spacing

    Returns
    -------
    ndarray
        Fokker-Planck collision operator
    """
    Ax, Ay, Az = A

    # Friction term: -div_v(A * f)
    friction = -divergence_v(Ax * f, Ay * f, Az * f, dv)

    # Diffusion term: (1/2) * sum_ij d^2/dv_i dv_j (D_ij * f)
    diffusion = np.zeros_like(f)

    # Velocity axis mapping
    v_axes = [3, 4, 5]

    for i in range(3):
        for j in range(3):
            Dij_f = D[..., i, j] * f
            # Second derivative with respect to v_i and v_j
            first_deriv = np.gradient(Dij_f, dv, axis=v_axes[j])
            second_deriv = np.gradient(first_deriv, dv, axis=v_axes[i])
            diffusion += 0.5 * second_deriv

    return friction + diffusion


def fokker_planck_isotropic(
    f: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
    gamma: float,
    D_coeff: float
) -> np.ndarray:
    """
    Compute Fokker-Planck operator with isotropic coefficients.

    Uses friction A = -γv and isotropic diffusion D = D_coeff * I

    Parameters
    ----------
    f : ndarray
        Distribution function (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    vx, vy, vz : ndarray
        1D velocity arrays
    dv : float
        Velocity grid spacing
    gamma : float
        Friction coefficient
    D_coeff : float
        Diffusion coefficient

    Returns
    -------
    ndarray
        Fokker-Planck collision operator
    """
    VX, VY, VZ = np.meshgrid(vx, vy, vz, indexing='ij')

    # Broadcast to full shape
    shape = f.shape
    VX_full = np.broadcast_to(VX, shape)
    VY_full = np.broadcast_to(VY, shape)
    VZ_full = np.broadcast_to(VZ, shape)

    # Friction: A = -gamma * v
    Ax = -gamma * VX_full
    Ay = -gamma * VY_full
    Az = -gamma * VZ_full

    # Isotropic diffusion tensor
    D = np.zeros(shape + (3, 3))
    D[..., 0, 0] = D_coeff
    D[..., 1, 1] = D_coeff
    D[..., 2, 2] = D_coeff

    return fokker_planck_operator(f, (Ax, Ay, Az), D, dv)


def fokker_planck_maxwellian_background(
    f: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
    n_b: float,
    T_b: float,
    m: float,
    m_b: float,
    q: float,
    q_b: float,
    ln_Lambda: float = 10.0
) -> np.ndarray:
    """
    Compute Fokker-Planck operator for test particles in Maxwellian background.

    Uses Rosenbluth potentials for a Maxwellian background distribution.

    Parameters
    ----------
    f : ndarray
        Test particle distribution (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    vx, vy, vz : ndarray
        1D velocity arrays
    dv : float
        Velocity grid spacing
    n_b : float
        Background density
    T_b : float
        Background temperature (K)
    m : float
        Test particle mass
    m_b : float
        Background particle mass
    q : float
        Test particle charge
    q_b : float
        Background particle charge
    ln_Lambda : float
        Coulomb logarithm

    Returns
    -------
    ndarray
        Fokker-Planck collision operator
    """
    from scipy.special import erf

    VX, VY, VZ = np.meshgrid(vx, vy, vz, indexing='ij')
    v_mag = np.sqrt(VX**2 + VY**2 + VZ**2)
    v_mag = np.maximum(v_mag, 1e-30)  # Avoid division by zero

    # Thermal velocity of background
    v_th = np.sqrt(2 * K_B * T_b / m_b)
    x = v_mag / v_th

    # Chandrasekhar function G(x)
    G = (erf(x) - 2*x/np.sqrt(np.pi) * np.exp(-x**2)) / (2 * x**2)

    # Collision frequency factor
    nu_0 = (q**2 * q_b**2 * n_b * ln_Lambda) / (4 * np.pi * EPSILON_0**2 * m**2)

    # Friction coefficient (slowing down)
    nu_s = (1 + m/m_b) * 2 * G
    gamma = nu_0 * nu_s / v_mag

    # Parallel and perpendicular diffusion
    nu_parallel = 2 * G / v_mag**2
    nu_perp = (erf(x) - G) / v_mag**2

    # Build friction vector A = -gamma * v
    shape = f.shape
    VX_full = np.broadcast_to(VX, shape)
    VY_full = np.broadcast_to(VY, shape)
    VZ_full = np.broadcast_to(VZ, shape)
    gamma_full = np.broadcast_to(gamma, shape)

    Ax = -gamma_full * VX_full
    Ay = -gamma_full * VY_full
    Az = -gamma_full * VZ_full

    # Build diffusion tensor
    # D_ij = D_perp * delta_ij + (D_parallel - D_perp) * v_i*v_j / v^2
    D = np.zeros(shape + (3, 3))

    D_parallel = nu_0 * np.broadcast_to(nu_parallel, shape)
    D_perp = nu_0 * np.broadcast_to(nu_perp, shape)

    v_mag_full = np.broadcast_to(v_mag, shape)
    v_mag_sq = v_mag_full**2

    # Perpendicular part (isotropic)
    D[..., 0, 0] = D_perp
    D[..., 1, 1] = D_perp
    D[..., 2, 2] = D_perp

    # Parallel correction
    D_diff = D_parallel - D_perp
    D[..., 0, 0] += D_diff * VX_full**2 / v_mag_sq
    D[..., 0, 1] += D_diff * VX_full * VY_full / v_mag_sq
    D[..., 0, 2] += D_diff * VX_full * VZ_full / v_mag_sq
    D[..., 1, 0] += D_diff * VY_full * VX_full / v_mag_sq
    D[..., 1, 1] += D_diff * VY_full**2 / v_mag_sq
    D[..., 1, 2] += D_diff * VY_full * VZ_full / v_mag_sq
    D[..., 2, 0] += D_diff * VZ_full * VX_full / v_mag_sq
    D[..., 2, 1] += D_diff * VZ_full * VY_full / v_mag_sq
    D[..., 2, 2] += D_diff * VZ_full**2 / v_mag_sq

    return fokker_planck_operator(f, (Ax, Ay, Az), D, dv)


# =============================================================================
# Landau Collision Operator
# =============================================================================

def landau_tensor(u: np.ndarray) -> np.ndarray:
    """
    Compute the Landau tensor U_ij(u) = (|u|^2 * delta_ij - u_i*u_j) / |u|^3

    Parameters
    ----------
    u : ndarray
        Relative velocity vector, shape (..., 3)

    Returns
    -------
    ndarray
        Landau tensor, shape (..., 3, 3)
    """
    u_mag = np.linalg.norm(u, axis=-1, keepdims=True)
    u_mag = np.maximum(u_mag, 1e-30)  # Regularization
    u_mag_cubed = u_mag ** 3
    u_mag_sq = u_mag ** 2

    # Identity part
    U = np.zeros(u.shape[:-1] + (3, 3))
    U[..., 0, 0] = u_mag_sq[..., 0]
    U[..., 1, 1] = u_mag_sq[..., 0]
    U[..., 2, 2] = u_mag_sq[..., 0]

    # Subtract u_i * u_j
    for i in range(3):
        for j in range(3):
            U[..., i, j] -= u[..., i] * u[..., j]

    # Divide by |u|^3
    U /= u_mag_cubed[..., np.newaxis]

    return U


def landau_operator(
    f_a: np.ndarray,
    f_b: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
    m_a: float,
    m_b: float,
    q_a: float,
    q_b: float,
    ln_Lambda: float = 10.0
) -> np.ndarray:
    """
    Compute the Landau collision operator for species a colliding with species b.

    C_ab[f_a, f_b] = Γ_ab * ∂/∂v_i ∫ d³v' U_ij(v-v') *
                     [f_b(v') ∂f_a(v)/∂v_j - (m_a/m_b) f_a(v) ∂f_b(v')/∂v'_j]

    where U_ij(u) = (|u|² δ_ij - u_i u_j) / |u|³
    and Γ_ab = q_a² q_b² ln(Λ) / (8π ε₀² m_a²)

    Parameters
    ----------
    f_a : ndarray
        Distribution of species a (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    f_b : ndarray
        Distribution of species b (same shape)
    vx, vy, vz : ndarray
        1D velocity arrays
    dv : float
        Velocity grid spacing
    m_a, m_b : float
        Masses of species a and b
    q_a, q_b : float
        Charges of species a and b
    ln_Lambda : float
        Coulomb logarithm

    Returns
    -------
    ndarray
        Landau collision operator C_ab
    """
    # Collision coefficient
    Gamma = (q_a**2 * q_b**2 * ln_Lambda) / (8 * np.pi * EPSILON_0**2 * m_a**2)

    dv3 = dv ** 3
    nv = len(vx)
    shape = f_a.shape
    nr = shape[:3]

    # Compute velocity gradients of f_a and f_b
    grad_f_a = np.stack([
        gradient_v(f_a, dv, 3),
        gradient_v(f_a, dv, 4),
        gradient_v(f_a, dv, 5)
    ], axis=-1)  # Shape: (..., 3)

    grad_f_b = np.stack([
        gradient_v(f_b, dv, 3),
        gradient_v(f_b, dv, 4),
        gradient_v(f_b, dv, 5)
    ], axis=-1)

    # Create velocity grids
    VX, VY, VZ = np.meshgrid(vx, vy, vz, indexing='ij')

    # Initialize the flux vector Q_i that we'll take divergence of
    Q = np.zeros(shape + (3,))

    # Loop over spatial points (can be parallelized)
    for ix in range(nr[0]):
        for iy in range(nr[1]):
            for iz in range(nr[2]):
                # For this spatial point, compute the velocity-space integral
                f_a_local = f_a[ix, iy, iz]  # (nv, nv, nv)
                f_b_local = f_b[ix, iy, iz]
                grad_f_a_local = grad_f_a[ix, iy, iz]  # (nv, nv, nv, 3)
                grad_f_b_local = grad_f_b[ix, iy, iz]

                # For each v, integrate over v'
                for ivx in range(nv):
                    for ivy in range(nv):
                        for ivz in range(nv):
                            v = np.array([vx[ivx], vy[ivy], vz[ivz]])

                            # Relative velocity u = v - v'
                            U_rel = np.stack([
                                v[0] - VX,
                                v[1] - VY,
                                v[2] - VZ
                            ], axis=-1)  # (nv, nv, nv, 3)

                            # Compute Landau tensor
                            U_tensor = landau_tensor(U_rel)  # (nv, nv, nv, 3, 3)

                            # First term: U_ij * f_b(v') * ∂f_a/∂v_j
                            # Sum over j: U_ij * grad_f_a_j at point (ivx, ivy, ivz)
                            grad_f_a_at_v = grad_f_a_local[ivx, ivy, ivz]  # (3,)
                            term1 = np.einsum('...ij,j,...->...i',
                                            U_tensor, grad_f_a_at_v, f_b_local)

                            # Second term: -m_a/m_b * U_ij * f_a(v) * ∂f_b/∂v'_j
                            f_a_at_v = f_a_local[ivx, ivy, ivz]
                            term2 = -m_a/m_b * np.einsum('...ij,...j,...->...i',
                                                         U_tensor, grad_f_b_local,
                                                         np.full_like(f_b_local, f_a_at_v))

                            # Integrate over v'
                            integrand = term1 + term2  # (nv, nv, nv, 3)
                            Q[ix, iy, iz, ivx, ivy, ivz, :] = np.sum(
                                integrand, axis=(0, 1, 2)
                            ) * dv3

    # Take divergence of Q to get the collision operator
    C = Gamma * divergence_v(Q[..., 0], Q[..., 1], Q[..., 2], dv)

    return C


def landau_operator_optimized(
    f_a: np.ndarray,
    f_b: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
    m_a: float,
    m_b: float,
    q_a: float,
    q_b: float,
    ln_Lambda: float = 10.0
) -> np.ndarray:
    """
    Optimized Landau collision operator using Rosenbluth potentials.

    Instead of direct 6D integration, uses the Rosenbluth potential formulation:
    C[f_a] = Γ * ∂/∂v_i [ ∂H/∂v_i * f_a - (1/2) * ∂²G/∂v_i∂v_j * ∂f_a/∂v_j ]

    where H and G are Rosenbluth potentials satisfying:
    ∇²H = -8π f_b
    ∇²G = 2H

    Parameters
    ----------
    f_a : ndarray
        Distribution of species a (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    f_b : ndarray
        Distribution of species b (same shape)
    vx, vy, vz : ndarray
        1D velocity arrays
    dv : float
        Velocity grid spacing
    m_a, m_b : float
        Masses of species a and b
    q_a, q_b : float
        Charges of species a and b
    ln_Lambda : float
        Coulomb logarithm

    Returns
    -------
    ndarray
        Landau collision operator C_ab
    """
    from scipy.ndimage import laplace
    from scipy.linalg import solve

    Gamma = (q_a**2 * q_b**2 * ln_Lambda) / (8 * np.pi * EPSILON_0**2 * m_a**2)

    shape = f_a.shape
    nr = shape[:3]
    nv = shape[3:]

    C = np.zeros_like(f_a)

    # Process each spatial point
    for ix in range(nr[0]):
        for iy in range(nr[1]):
            for iz in range(nr[2]):
                f_b_local = f_b[ix, iy, iz]
                f_a_local = f_a[ix, iy, iz]

                # Solve Poisson equations for Rosenbluth potentials
                # Using spectral method for efficiency
                H, G = _solve_rosenbluth_potentials(f_b_local, dv)

                # Compute gradients
                grad_H = np.stack([
                    np.gradient(H, dv, axis=0),
                    np.gradient(H, dv, axis=1),
                    np.gradient(H, dv, axis=2)
                ], axis=-1)

                grad_f_a = np.stack([
                    np.gradient(f_a_local, dv, axis=0),
                    np.gradient(f_a_local, dv, axis=1),
                    np.gradient(f_a_local, dv, axis=2)
                ], axis=-1)

                # Compute Hessian of G
                hess_G = np.zeros(nv + (3, 3))
                for i in range(3):
                    G_i = np.gradient(G, dv, axis=i)
                    for j in range(3):
                        hess_G[..., i, j] = np.gradient(G_i, dv, axis=j)

                # Friction term: ∂H/∂v_i * f_a
                friction_flux = grad_H * f_a_local[..., np.newaxis]

                # Diffusion term: (1/2) * ∂²G/∂v_i∂v_j * ∂f_a/∂v_j
                diffusion_flux = 0.5 * np.einsum('...ij,...j->...i', hess_G, grad_f_a)

                # Total flux
                flux = (1 + m_a/m_b) * friction_flux - diffusion_flux

                # Divergence
                div_flux = (np.gradient(flux[..., 0], dv, axis=0) +
                           np.gradient(flux[..., 1], dv, axis=1) +
                           np.gradient(flux[..., 2], dv, axis=2))

                C[ix, iy, iz] = Gamma * div_flux

    return C


def _solve_rosenbluth_potentials(
    f: np.ndarray,
    dv: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve for Rosenbluth potentials H and G using FFT-based Poisson solver.

    ∇²H = -8π f
    ∇²G = 2H

    Parameters
    ----------
    f : ndarray
        Distribution function in velocity space (nv, nv, nv)
    dv : float
        Velocity grid spacing

    Returns
    -------
    H, G : ndarray
        Rosenbluth potentials
    """
    nv = f.shape[0]

    # Wavenumbers
    k = np.fft.fftfreq(nv, d=dv) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    K_sq = KX**2 + KY**2 + KZ**2
    K_sq[0, 0, 0] = 1  # Avoid division by zero

    # Solve ∇²H = -8π f
    f_hat = np.fft.fftn(f)
    H_hat = 8 * np.pi * f_hat / K_sq
    H_hat[0, 0, 0] = 0  # Zero mean
    H = np.real(np.fft.ifftn(H_hat))

    # Solve ∇²G = 2H
    H_hat_for_G = np.fft.fftn(H)
    G_hat = -2 * H_hat_for_G / K_sq
    G_hat[0, 0, 0] = 0
    G = np.real(np.fft.ifftn(G_hat))

    return H, G


# =============================================================================
# Normalized Operators for Parameter Optimization
# =============================================================================

def normalized_landau_operator(
    f: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float
) -> np.ndarray:
    """
    Compute the normalized Landau collision operator Q(f,f).

    Q(f,f)(v) = ∇_v · ∫ Φ(v - v*) (∇_v f · f* - ∇_{v*} f* · f) dv*

    where Φ(v) = |v|^(γ+2) S(v) with γ = -3, so |v|^(-1) S(v)
    and S(v) = I - v⊗v / |v|^2 (projection tensor perpendicular to v)

    This is the dimensionless/normalized form without physical constants.

    Parameters
    ----------
    f : ndarray
        Distribution function (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    vx, vy, vz : ndarray
        1D velocity arrays
    dv : float
        Velocity grid spacing

    Returns
    -------
    ndarray
        Normalized Landau collision operator
    """
    dv3 = dv ** 3
    shape = f.shape
    nr = shape[:3]
    nv = shape[3]

    # Compute velocity gradients of f
    grad_f = np.stack([
        gradient_v(f, dv, 3),
        gradient_v(f, dv, 4),
        gradient_v(f, dv, 5)
    ], axis=-1)  # Shape: (..., 3)

    # Create velocity grids
    VX, VY, VZ = np.meshgrid(vx, vy, vz, indexing='ij')

    # Initialize the flux vector Q_i
    Q = np.zeros(shape + (3,))

    # Loop over spatial points
    for ix in range(nr[0]):
        for iy in range(nr[1]):
            for iz in range(nr[2]):
                f_local = f[ix, iy, iz]  # (nv, nv, nv)
                grad_f_local = grad_f[ix, iy, iz]  # (nv, nv, nv, 3)

                # For each v, integrate over v*
                for ivx in range(nv):
                    for ivy in range(nv):
                        for ivz in range(nv):
                            v = np.array([vx[ivx], vy[ivy], vz[ivz]])

                            # Relative velocity u = v - v*
                            U_rel = np.stack([
                                v[0] - VX,
                                v[1] - VY,
                                v[2] - VZ
                            ], axis=-1)  # (nv, nv, nv, 3)

                            # Compute Φ(u) = |u|^(-1) * S(u)
                            # S(u) = I - u⊗u / |u|^2
                            u_mag = np.linalg.norm(U_rel, axis=-1)
                            u_mag = np.maximum(u_mag, 1e-30)
                            u_mag_inv = 1.0 / u_mag

                            # Φ_ij = (δ_ij - u_i u_j / |u|^2) / |u|
                            Phi = np.zeros(U_rel.shape[:-1] + (3, 3))
                            Phi[..., 0, 0] = u_mag_inv
                            Phi[..., 1, 1] = u_mag_inv
                            Phi[..., 2, 2] = u_mag_inv

                            u_outer = U_rel[..., :, np.newaxis] * U_rel[..., np.newaxis, :]
                            u_mag_cubed = u_mag[..., np.newaxis, np.newaxis] ** 3
                            Phi -= u_outer / u_mag_cubed

                            # First term: Φ_ij * (∇_v f)_j * f*
                            grad_f_at_v = grad_f_local[ivx, ivy, ivz]  # (3,)
                            term1 = np.einsum('...ij,j,...->...i',
                                            Phi, grad_f_at_v, f_local)

                            # Second term: -Φ_ij * (∇_{v*} f*)_j * f(v)
                            f_at_v = f_local[ivx, ivy, ivz]
                            term2 = -np.einsum('...ij,...j->...i',
                                              Phi, grad_f_local) * f_at_v

                            # Integrate over v*
                            integrand = term1 + term2  # (nv, nv, nv, 3)
                            Q[ix, iy, iz, ivx, ivy, ivz, :] = np.sum(
                                integrand, axis=(0, 1, 2)
                            ) * dv3

    # Take divergence of Q
    C = divergence_v(Q[..., 0], Q[..., 1], Q[..., 2], dv)

    return C


def normalized_landau_operator_fast(
    f: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float
) -> np.ndarray:
    """
    Fast version of normalized Landau operator using Rosenbluth potentials.

    Uses FFT-based Poisson solver for efficiency.

    Parameters
    ----------
    f : ndarray
        Distribution function (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    vx, vy, vz : ndarray
        1D velocity arrays
    dv : float
        Velocity grid spacing

    Returns
    -------
    ndarray
        Normalized Landau collision operator
    """
    shape = f.shape
    nr = shape[:3]
    nv = shape[3:]

    C = np.zeros_like(f)

    for ix in range(nr[0]):
        for iy in range(nr[1]):
            for iz in range(nr[2]):
                f_local = f[ix, iy, iz]

                # Solve for Rosenbluth potentials
                H, G = _solve_rosenbluth_potentials(f_local, dv)

                # Compute gradients
                grad_H = np.stack([
                    np.gradient(H, dv, axis=0),
                    np.gradient(H, dv, axis=1),
                    np.gradient(H, dv, axis=2)
                ], axis=-1)

                grad_f = np.stack([
                    np.gradient(f_local, dv, axis=0),
                    np.gradient(f_local, dv, axis=1),
                    np.gradient(f_local, dv, axis=2)
                ], axis=-1)

                # Compute Hessian of G
                hess_G = np.zeros(nv + (3, 3))
                for i in range(3):
                    G_i = np.gradient(G, dv, axis=i)
                    for j in range(3):
                        hess_G[..., i, j] = np.gradient(G_i, dv, axis=j)

                # Friction flux: 2 * ∇H * f
                friction_flux = 2 * grad_H * f_local[..., np.newaxis]

                # Diffusion flux: ∂²G/∂v_i∂v_j * ∂f/∂v_j
                diffusion_flux = np.einsum('...ij,...j->...i', hess_G, grad_f)

                # Total flux
                flux = friction_flux - diffusion_flux

                # Divergence
                div_flux = (np.gradient(flux[..., 0], dv, axis=0) +
                           np.gradient(flux[..., 1], dv, axis=1) +
                           np.gradient(flux[..., 2], dv, axis=2))

                C[ix, iy, iz] = div_flux

    return C


def normalized_fokker_planck_operator(
    f: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float
) -> np.ndarray:
    """
    Compute the normalized Fokker-Planck operator P(f).

    P(f) = ∇_v · (M[f] ∇_v (f / M[f]))

    where M[f] is the local Maxwellian with the same moments as f.

    This can be rewritten as:
    P(f) = ∇_v · (∇_v f - f ∇_v ln(M[f]))

    Parameters
    ----------
    f : ndarray
        Distribution function (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    vx, vy, vz : ndarray
        1D velocity arrays
    dv : float
        Velocity grid spacing

    Returns
    -------
    ndarray
        Normalized Fokker-Planck operator
    """
    shape = f.shape
    nr = shape[:3]

    # Compute moments
    n, u, T_kin = compute_moments(f, vx, vy, vz, dv)

    # Temperature (in velocity^2 units, since we use normalized form)
    # T_kin = <|v-u|^2> / 3, so thermal velocity squared is 2*T_kin
    theta = np.maximum(T_kin, 1e-30)  # theta = T/m in normalized units

    # Build local Maxwellian
    VX, VY, VZ = np.meshgrid(vx, vy, vz, indexing='ij')

    # Expand moments to full 6D shape
    n_exp = n[..., np.newaxis, np.newaxis, np.newaxis]
    theta_exp = theta[..., np.newaxis, np.newaxis, np.newaxis]
    u_exp = u[..., np.newaxis, np.newaxis, np.newaxis, :]

    # Relative velocity
    v_rel_x = VX - u_exp[..., 0]
    v_rel_y = VY - u_exp[..., 1]
    v_rel_z = VZ - u_exp[..., 2]
    v_rel_sq = v_rel_x**2 + v_rel_y**2 + v_rel_z**2

    # Local Maxwellian: M = n / (2π θ)^(3/2) * exp(-|v-u|^2 / (2θ))
    M = n_exp / (2 * np.pi * theta_exp) ** 1.5 * np.exp(-v_rel_sq / (2 * theta_exp))
    M = np.maximum(M, 1e-30)  # Avoid division by zero

    # Compute f / M
    f_over_M = f / M

    # Compute ∇_v (f / M)
    grad_f_over_M_x = np.gradient(f_over_M, dv, axis=3)
    grad_f_over_M_y = np.gradient(f_over_M, dv, axis=4)
    grad_f_over_M_z = np.gradient(f_over_M, dv, axis=5)

    # Compute M * ∇_v (f / M)
    flux_x = M * grad_f_over_M_x
    flux_y = M * grad_f_over_M_y
    flux_z = M * grad_f_over_M_z

    # Compute divergence: ∇_v · (M ∇_v (f/M))
    P = divergence_v(flux_x, flux_y, flux_z, dv)

    return P


def compute_local_maxwellian(
    f: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float
) -> np.ndarray:
    """
    Compute the local Maxwellian M[f] with same moments as f.

    Parameters
    ----------
    f : ndarray
        Distribution function (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    vx, vy, vz : ndarray
        1D velocity arrays
    dv : float
        Velocity grid spacing

    Returns
    -------
    ndarray
        Local Maxwellian distribution (same shape as f)
    """
    # Compute moments
    n, u, T_kin = compute_moments(f, vx, vy, vz, dv)
    theta = np.maximum(T_kin, 1e-30)

    # Build local Maxwellian
    VX, VY, VZ = np.meshgrid(vx, vy, vz, indexing='ij')

    n_exp = n[..., np.newaxis, np.newaxis, np.newaxis]
    theta_exp = theta[..., np.newaxis, np.newaxis, np.newaxis]
    u_exp = u[..., np.newaxis, np.newaxis, np.newaxis, :]

    v_rel_x = VX - u_exp[..., 0]
    v_rel_y = VY - u_exp[..., 1]
    v_rel_z = VZ - u_exp[..., 2]
    v_rel_sq = v_rel_x**2 + v_rel_y**2 + v_rel_z**2

    M = n_exp / (2 * np.pi * theta_exp) ** 1.5 * np.exp(-v_rel_sq / (2 * theta_exp))

    return M


def normalized_bgk_operator(
    f: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
    tau: float
) -> np.ndarray:
    """
    Compute the normalized BGK operator (M[f] - f) / τ.

    Parameters
    ----------
    f : ndarray
        Distribution function (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    vx, vy, vz : ndarray
        1D velocity arrays
    dv : float
        Velocity grid spacing
    tau : float
        Relaxation time

    Returns
    -------
    ndarray
        Normalized BGK operator
    """
    M = compute_local_maxwellian(f, vx, vy, vz, dv)
    return (M - f) / tau


# =============================================================================
# Optimal Parameter Finding
# =============================================================================

def find_optimal_collision_rate(
    f: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
    dr: float,
    n_samples: int = 20,
    mu_min: float = 0.01,
    mu_max: float = 10.0,
    use_fast_landau: bool = True
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Find optimal effective collision rate μ that minimizes |Q(f,f) - μ P(f)|.

    Searches over log-uniformly spaced values of μ and returns the one
    that minimizes the L1 norm over full 6D phase space.

    Parameters
    ----------
    f : ndarray
        Distribution function (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    vx, vy, vz : ndarray
        1D velocity arrays
    dv : float
        Velocity grid spacing
    dr : float
        Position grid spacing
    n_samples : int
        Number of μ values to test (default 20)
    mu_min : float
        Minimum μ value (default 0.01)
    mu_max : float
        Maximum μ value (default 10.0)
    use_fast_landau : bool
        Use fast Rosenbluth-based Landau operator (default True)

    Returns
    -------
    mu_opt : float
        Optimal collision rate
    min_norm : float
        L1 norm at optimal μ
    mu_values : ndarray
        Array of tested μ values
    norms : ndarray
        L1 norms for each μ value
    """
    # Compute Q(f,f) - the normalized Landau operator
    if use_fast_landau:
        Q = normalized_landau_operator_fast(f, vx, vy, vz, dv)
    else:
        Q = normalized_landau_operator(f, vx, vy, vz, dv)

    # Compute P(f) - the normalized Fokker-Planck operator
    P = normalized_fokker_planck_operator(f, vx, vy, vz, dv)

    # Log-uniformly spaced μ values
    mu_values = np.logspace(np.log10(mu_min), np.log10(mu_max), n_samples)

    # Phase space volume element
    dV = dv**3 * dr**3

    # Compute L1 norm for each μ
    norms = np.zeros(n_samples)
    for i, mu in enumerate(mu_values):
        diff = Q - mu * P
        norms[i] = np.sum(np.abs(diff)) * dV

    # Find optimal
    idx_opt = np.argmin(norms)
    mu_opt = mu_values[idx_opt]
    min_norm = norms[idx_opt]

    return mu_opt, min_norm, mu_values, norms


def find_optimal_relaxation_time(
    f: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
    dr: float,
    n_samples: int = 20,
    tau_min: float = 0.01,
    tau_max: float = 10.0,
    use_fast_landau: bool = True
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Find optimal relaxation time τ that minimizes |Q(f,f) - (M[f] - f)/τ|.

    Searches over log-uniformly spaced values of τ and returns the one
    that minimizes the L1 norm over full 6D phase space.

    Parameters
    ----------
    f : ndarray
        Distribution function (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    vx, vy, vz : ndarray
        1D velocity arrays
    dv : float
        Velocity grid spacing
    dr : float
        Position grid spacing
    n_samples : int
        Number of τ values to test (default 20)
    tau_min : float
        Minimum τ value (default 0.01)
    tau_max : float
        Maximum τ value (default 10.0)
    use_fast_landau : bool
        Use fast Rosenbluth-based Landau operator (default True)

    Returns
    -------
    tau_opt : float
        Optimal relaxation time
    min_norm : float
        L1 norm at optimal τ
    tau_values : ndarray
        Array of tested τ values
    norms : ndarray
        L1 norms for each τ value
    """
    # Compute Q(f,f) - the normalized Landau operator
    if use_fast_landau:
        Q = normalized_landau_operator_fast(f, vx, vy, vz, dv)
    else:
        Q = normalized_landau_operator(f, vx, vy, vz, dv)

    # Compute M[f] - the local Maxwellian
    M = compute_local_maxwellian(f, vx, vy, vz, dv)

    # Log-uniformly spaced τ values
    tau_values = np.logspace(np.log10(tau_min), np.log10(tau_max), n_samples)

    # Phase space volume element
    dV = dv**3 * dr**3

    # Compute L1 norm for each τ
    norms = np.zeros(n_samples)
    for i, tau in enumerate(tau_values):
        bgk = (M - f) / tau
        diff = Q - bgk
        norms[i] = np.sum(np.abs(diff)) * dV

    # Find optimal
    idx_opt = np.argmin(norms)
    tau_opt = tau_values[idx_opt]
    min_norm = norms[idx_opt]

    return tau_opt, min_norm, tau_values, norms


def find_optimal_parameters(
    f: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
    dr: float,
    n_samples: int = 20,
    param_min: float = 0.01,
    param_max: float = 10.0,
    use_fast_landau: bool = True
) -> dict:
    """
    Find both optimal collision rate μ and relaxation time τ.

    This is a convenience function that calls both find_optimal_collision_rate
    and find_optimal_relaxation_time.

    Parameters
    ----------
    f : ndarray
        Distribution function (nr_x, nr_y, nr_z, nv_x, nv_y, nv_z)
    vx, vy, vz : ndarray
        1D velocity arrays
    dv : float
        Velocity grid spacing
    dr : float
        Position grid spacing
    n_samples : int
        Number of parameter values to test (default 20)
    param_min : float
        Minimum parameter value (default 0.01)
    param_max : float
        Maximum parameter value (default 10.0)
    use_fast_landau : bool
        Use fast Rosenbluth-based Landau operator (default True)

    Returns
    -------
    dict
        Dictionary containing:
        - 'mu_opt': optimal collision rate
        - 'mu_norm': L1 norm at optimal μ
        - 'mu_values': tested μ values
        - 'mu_norms': L1 norms for FP comparison
        - 'tau_opt': optimal relaxation time
        - 'tau_norm': L1 norm at optimal τ
        - 'tau_values': tested τ values
        - 'tau_norms': L1 norms for BGK comparison
    """
    # Compute Q(f,f) once (expensive operation)
    if use_fast_landau:
        Q = normalized_landau_operator_fast(f, vx, vy, vz, dv)
    else:
        Q = normalized_landau_operator(f, vx, vy, vz, dv)

    # Compute P(f) and M[f]
    P = normalized_fokker_planck_operator(f, vx, vy, vz, dv)
    M = compute_local_maxwellian(f, vx, vy, vz, dv)

    # Parameter values
    mu_values = np.logspace(np.log10(param_min), np.log10(param_max), n_samples)
    tau_values = np.logspace(np.log10(param_min), np.log10(param_max), n_samples)

    # Phase space volume element
    dV = dv**3 * dr**3

    # Compute L1 norms for μ (FP comparison)
    mu_norms = np.zeros(n_samples)
    for i, mu in enumerate(mu_values):
        diff = Q - mu * P
        mu_norms[i] = np.sum(np.abs(diff)) * dV

    # Compute L1 norms for τ (BGK comparison)
    tau_norms = np.zeros(n_samples)
    for i, tau in enumerate(tau_values):
        bgk = (M - f) / tau
        diff = Q - bgk
        tau_norms[i] = np.sum(np.abs(diff)) * dV

    # Find optima
    idx_mu = np.argmin(mu_norms)
    idx_tau = np.argmin(tau_norms)

    return {
        'mu_opt': mu_values[idx_mu],
        'mu_norm': mu_norms[idx_mu],
        'mu_values': mu_values,
        'mu_norms': mu_norms,
        'tau_opt': tau_values[idx_tau],
        'tau_norm': tau_norms[idx_tau],
        'tau_values': tau_values,
        'tau_norms': tau_norms
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def collision_frequency_ee(n: float, T: float, ln_Lambda: float = 10.0) -> float:
    """
    Electron-electron collision frequency.

    Parameters
    ----------
    n : float
        Electron density (m^-3)
    T : float
        Temperature (K)

    Returns
    -------
    float
        Collision frequency (1/s)
    """
    v_th = np.sqrt(K_B * T / M_E)
    return (n * E_CHARGE**4 * ln_Lambda) / (4 * np.pi * EPSILON_0**2 * M_E**2 * v_th**3)


def collision_frequency_ei(
    n_e: float,
    T_e: float,
    Z: float,
    m_i: float,
    ln_Lambda: float = 10.0
) -> float:
    """
    Electron-ion collision frequency.

    Parameters
    ----------
    n_e : float
        Electron density (m^-3)
    T_e : float
        Electron temperature (K)
    Z : float
        Ion charge state
    m_i : float
        Ion mass (kg)

    Returns
    -------
    float
        Collision frequency (1/s)
    """
    v_th = np.sqrt(K_B * T_e / M_E)
    return (n_e * Z**2 * E_CHARGE**4 * ln_Lambda) / (4 * np.pi * EPSILON_0**2 * M_E**2 * v_th**3)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Compare BGK and Fokker-Planck operators
    print("Collision Operators Demo")
    print("=" * 50)

    # Grid parameters
    nv = 16  # Velocity grid points per dimension (smaller for demo)
    nr = 2   # Spatial grid points per dimension
    v_max = 3.0  # Normalized velocity units
    L = 1.0  # Normalized length

    # Create grids
    vx, vy, vz, dv = create_velocity_grid(v_max, nv)
    rx, ry, rz, dr = create_position_grid(L, nr)

    # Create initial distribution (perturbed Maxwellian in normalized units)
    # Using normalized units: n=1, theta=1
    VX, VY, VZ = np.meshgrid(vx, vy, vz, indexing='ij')
    v_sq = VX**2 + VY**2 + VZ**2
    f_maxwell = 1.0 / (2 * np.pi) ** 1.5 * np.exp(-v_sq / 2)

    # Broadcast to 6D
    f = np.broadcast_to(f_maxwell, (nr, nr, nr) + f_maxwell.shape).copy()

    # Add perturbation to make it non-equilibrium
    np.random.seed(42)
    f += 0.1 * f.max() * np.random.randn(*f.shape)
    f = np.maximum(f, 1e-30)  # Ensure non-negative

    print(f"Grid: {nr}³ spatial × {nv}³ velocity = {nr**3 * nv**3:.2e} points")
    print(f"Distribution f shape: {f.shape}")
    print(f"Distribution f range: [{f.min():.2e}, {f.max():.2e}]")

    # Demo of physical operators
    print("\n--- Physical Operators Demo ---")
    n0 = 1e20  # m^-3
    T0 = 1e6   # K
    m = M_E
    nu = collision_frequency_ee(n0, T0)
    print(f"Collision frequency: {nu:.2e} Hz")

    # Demo of optimal parameter finding
    print("\n--- Optimal Parameter Finding Demo ---")
    print("Finding optimal μ (collision rate) and τ (relaxation time)...")
    print("Searching 20 log-uniform values from 0.01 to 10.0")

    results = find_optimal_parameters(
        f, vx, vy, vz, dv, dr,
        n_samples=20,
        param_min=0.01,
        param_max=10.0,
        use_fast_landau=True
    )

    print(f"\nOptimal collision rate μ: {results['mu_opt']:.4f}")
    print(f"  L1 norm |Q - μP|: {results['mu_norm']:.6e}")

    print(f"\nOptimal relaxation time τ: {results['tau_opt']:.4f}")
    print(f"  L1 norm |Q - (M-f)/τ|: {results['tau_norm']:.6e}")

    print("\nμ values tested:", [f"{v:.3f}" for v in results['mu_values'][:5]], "...")
    print("τ values tested:", [f"{v:.3f}" for v in results['tau_values'][:5]], "...")

    print("\nDone!")
