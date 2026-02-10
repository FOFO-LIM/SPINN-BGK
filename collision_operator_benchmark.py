"""
Benchmark wall clock time for collision operators:
- BGK operator
- Fokker-Planck (FP) operator
- Landau operator

For both Maxwellian and non-Maxwellian distribution functions.

Parameters extracted from: spinn_Kn0.01_rank256_ngrid16_gpu4_20260114_175452_params.npy
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Parameters from filename
Kn = 0.01
RANK = 256
NGRID = 16  # Grid points per dimension
V_MAX = 10.0  # Velocity domain [-V, V]
X_MAX = 0.5   # Spatial domain [-X, X]
T_MAX = 0.1   # Time domain [0, T]

# Derived
NU = 1.0 / Kn  # Collision frequency


def create_grid(n_per_dim):
    """Create 7D grid: (t, x, y, z, vx, vy, vz)"""
    t = jnp.linspace(0, T_MAX, n_per_dim)
    x = jnp.linspace(-X_MAX, X_MAX, n_per_dim)
    y = jnp.linspace(-X_MAX, X_MAX, n_per_dim)
    z = jnp.linspace(-X_MAX, X_MAX, n_per_dim)
    vx = jnp.linspace(-V_MAX, V_MAX, n_per_dim)
    vy = jnp.linspace(-V_MAX, V_MAX, n_per_dim)
    vz = jnp.linspace(-V_MAX, V_MAX, n_per_dim)
    return t, x, y, z, vx, vy, vz


def create_maxwellian_field(nx, nv):
    """Create a Maxwellian distribution on the grid (smooth initial condition)."""
    x = jnp.linspace(-X_MAX, X_MAX, nx)
    y = jnp.linspace(-X_MAX, X_MAX, nx)
    z = jnp.linspace(-X_MAX, X_MAX, nx)
    vx = jnp.linspace(-V_MAX, V_MAX, nv)
    vy = jnp.linspace(-V_MAX, V_MAX, nv)
    vz = jnp.linspace(-V_MAX, V_MAX, nv)

    # Smooth initial condition: rho = 1 + 0.5*sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    rho = 1 + 0.5 * jnp.sin(2*jnp.pi*X) * jnp.sin(2*jnp.pi*Y) * jnp.sin(2*jnp.pi*Z)
    ux = jnp.zeros_like(rho)
    uy = jnp.zeros_like(rho)
    uz = jnp.zeros_like(rho)
    temp = jnp.ones_like(rho)

    return rho, (ux, uy, uz), temp, (x, y, z), (vx, vy, vz)


def create_non_maxwellian_field(nx, nv, rank):
    """Create a separable non-Maxwellian perturbation (rank-r tensor)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 7)

    # Separable representation: f_neq = sum_r g_r(x,y,z) * h_r(vx,vy,vz)
    g_x = jax.random.normal(keys[0], (nx, rank)) * 0.1
    g_y = jax.random.normal(keys[1], (nx, rank)) * 0.1
    g_z = jax.random.normal(keys[2], (nx, rank)) * 0.1
    h_vx = jax.random.normal(keys[3], (nv, rank)) * 0.1
    h_vy = jax.random.normal(keys[4], (nv, rank)) * 0.1
    h_vz = jax.random.normal(keys[5], (nv, rank)) * 0.1
    weights = jax.random.normal(keys[6], (rank,)) / rank

    return weights, (g_x, g_y, g_z), (h_vx, h_vy, h_vz)


# =============================================================================
# BGK OPERATOR
# =============================================================================

@jax.jit
def bgk_maxwellian(rho, u, temp, f_eq, vx, vy, vz):
    """
    BGK operator for Maxwellian: Q_BGK = nu * (f_eq - f)
    For f = f_eq (Maxwellian), Q_BGK = 0

    Cost: O(N_x^3 * N_v^3) = O(N^6)
    """
    # Expand dimensions for broadcasting
    rho_exp = rho[:, :, :, None, None, None]
    ux_exp = u[0][:, :, :, None, None, None]
    uy_exp = u[1][:, :, :, None, None, None]
    uz_exp = u[2][:, :, :, None, None, None]
    temp_exp = temp[:, :, :, None, None, None]

    VX, VY, VZ = jnp.meshgrid(vx, vy, vz, indexing='ij')

    f_eq_computed = (rho_exp / jnp.power(2*jnp.pi*temp_exp, 1.5) *
                    jnp.exp(-((VX - ux_exp)**2 + (VY - uy_exp)**2 + (VZ - uz_exp)**2) / (2*temp_exp)))

    # BGK collision: nu * (f_eq - f), here f = f_eq so result is 0
    Q_bgk = NU * (f_eq_computed - f_eq)
    return Q_bgk


@jax.jit
def bgk_non_maxwellian_separable(weights, g_x, g_y, g_z, h_vx, h_vy, h_vz):
    """
    BGK operator for non-Maxwellian part using separable representation.

    f_neq = sum_r w_r * g_x(x)_r * g_y(y)_r * g_z(z)_r * h_vx(vx)_r * h_vy(vy)_r * h_vz(vz)_r

    Q_BGK[f_neq] = nu * (0 - f_neq) = -nu * f_neq

    Cost with separable: O(r * N_x^3 + r * N_v^3) = O(r * N^3)
    """
    # Separable computation via einsum
    f_neq = jnp.einsum('r,ir,jr,kr,lr,mr,nr->ijklmn',
                       weights, g_x, g_y, g_z, h_vx, h_vy, h_vz)

    Q_bgk = -NU * f_neq
    return Q_bgk


@jax.jit
def bgk_full(rho, u, temp, weights, g_x, g_y, g_z, h_vx, h_vy, h_vz, vx, vy, vz, alpha):
    """
    BGK operator for full distribution f = f_eq + alpha * f_neq
    """
    # Compute f_eq
    rho_exp = rho[:, :, :, None, None, None]
    ux_exp = u[0][:, :, :, None, None, None]
    uy_exp = u[1][:, :, :, None, None, None]
    uz_exp = u[2][:, :, :, None, None, None]
    temp_exp = temp[:, :, :, None, None, None]

    VX, VY, VZ = jnp.meshgrid(vx, vy, vz, indexing='ij')

    f_eq = (rho_exp / jnp.power(2*jnp.pi*temp_exp, 1.5) *
            jnp.exp(-((VX - ux_exp)**2 + (VY - uy_exp)**2 + (VZ - uz_exp)**2) / (2*temp_exp)))

    # Compute f_neq (separable)
    f_neq = jnp.einsum('r,ir,jr,kr,lr,mr,nr->ijklmn',
                       weights, g_x, g_y, g_z, h_vx, h_vy, h_vz)

    f = f_eq + alpha * f_neq

    # For BGK, Q = nu * (f_eq - f)
    Q_bgk = NU * (f_eq - f)
    return Q_bgk


# =============================================================================
# FOKKER-PLANCK OPERATOR
# =============================================================================

@jax.jit
def fp_maxwellian(rho, u, temp, vx, vy, vz):
    """
    Fokker-Planck operator for Maxwellian.

    FP operator: Q_FP = div_v(A*f + D*grad_v(f))

    Cost for Maxwellian: O(N_x^3 * N_v^3) = O(N^6) with analytical forms
    """
    dv = vx[1] - vx[0]

    # Compute f_eq
    rho_exp = rho[:, :, :, None, None, None]
    ux_exp = u[0][:, :, :, None, None, None]
    uy_exp = u[1][:, :, :, None, None, None]
    uz_exp = u[2][:, :, :, None, None, None]
    temp_exp = temp[:, :, :, None, None, None]

    VX, VY, VZ = jnp.meshgrid(vx, vy, vz, indexing='ij')

    f_eq = (rho_exp / jnp.power(2*jnp.pi*temp_exp, 1.5) *
            jnp.exp(-((VX - ux_exp)**2 + (VY - uy_exp)**2 + (VZ - uz_exp)**2) / (2*temp_exp)))

    # Compute velocity derivatives using finite differences
    df_dvx = jnp.gradient(f_eq, dv, axis=3)
    df_dvy = jnp.gradient(f_eq, dv, axis=4)
    df_dvz = jnp.gradient(f_eq, dv, axis=5)

    # For Maxwellian, Rosenbluth potentials have known forms
    # Simplified: drag coefficient A ~ -nu*v, diffusion D ~ nu*T
    A_x = -NU * (VX - ux_exp)
    A_y = -NU * (VY - uy_exp)
    A_z = -NU * (VZ - uz_exp)
    D = NU * temp_exp

    # Drag term: div_v(A*f)
    drag_x = jnp.gradient(A_x * f_eq, dv, axis=3)
    drag_y = jnp.gradient(A_y * f_eq, dv, axis=4)
    drag_z = jnp.gradient(A_z * f_eq, dv, axis=5)

    # Diffusion term: div_v(D*grad_v(f))
    diff_x = jnp.gradient(D * df_dvx, dv, axis=3)
    diff_y = jnp.gradient(D * df_dvy, dv, axis=4)
    diff_z = jnp.gradient(D * df_dvz, dv, axis=5)

    Q_fp = drag_x + drag_y + drag_z + diff_x + diff_y + diff_z
    return Q_fp


@jax.jit
def fp_non_maxwellian_separable(weights, g_x, g_y, g_z, h_vx, h_vy, h_vz, vx, vy, vz):
    """
    Fokker-Planck operator for non-Maxwellian part using separable representation.

    Cost with separable: O(r^2 * N_v^3 * log(N_v)) for Rosenbluth potentials
                        + O(r * N_x^3 * N_v^3) for applying FP
    Simplified to: O(r^2 * N^3 * log(N))
    """
    dv = vx[1] - vx[0]

    # Reconstruct f_neq
    f_neq = jnp.einsum('r,ir,jr,kr,lr,mr,nr->ijklmn',
                       weights, g_x, g_y, g_z, h_vx, h_vy, h_vz)

    # Simplified FP: use linearized approximation around Maxwellian (T=1, u=0)
    VX, VY, VZ = jnp.meshgrid(vx, vy, vz, indexing='ij')

    df_dvx = jnp.gradient(f_neq, dv, axis=3)
    df_dvy = jnp.gradient(f_neq, dv, axis=4)
    df_dvz = jnp.gradient(f_neq, dv, axis=5)

    # Linearized FP around Maxwellian with T=1
    drag_x = jnp.gradient(-NU * VX * f_neq, dv, axis=3)
    drag_y = jnp.gradient(-NU * VY * f_neq, dv, axis=4)
    drag_z = jnp.gradient(-NU * VZ * f_neq, dv, axis=5)

    diff_x = jnp.gradient(NU * df_dvx, dv, axis=3)
    diff_y = jnp.gradient(NU * df_dvy, dv, axis=4)
    diff_z = jnp.gradient(NU * df_dvz, dv, axis=5)

    Q_fp = drag_x + drag_y + drag_z + diff_x + diff_y + diff_z
    return Q_fp


# =============================================================================
# LANDAU OPERATOR
# =============================================================================

@jax.jit
def landau_maxwellian(rho, u, temp, vx, vy, vz):
    """
    Landau collision operator for Maxwellian.

    For Maxwellian, Landau = FP (they're equivalent)

    Cost for Maxwellian: O(N_x^3 * N_v^3) = O(N^6)
    """
    return fp_maxwellian(rho, u, temp, vx, vy, vz)


def landau_non_maxwellian_naive(weights, g_x, g_y, g_z, h_vx, h_vy, h_vz, vx_grid):
    """
    Landau collision operator for non-Maxwellian (NAIVE implementation).

    Full double integral: Q = int int U(v-v') * [...] dv dv'

    Cost: O(N_x^3 * N_v^6) = O(N^9) NAIVE

    WARNING: This is extremely expensive! For benchmarking only with small N.
    """
    nv = len(vx_grid)
    dv = vx_grid[1] - vx_grid[0]

    # Use only subset of data for tractability
    nv_sub = min(nv, 8)
    g_x_sub = g_x[:nv_sub, :]
    g_y_sub = g_y[:nv_sub, :]
    g_z_sub = g_z[:nv_sub, :]
    h_vx_sub = h_vx[:nv_sub, :]
    h_vy_sub = h_vy[:nv_sub, :]
    h_vz_sub = h_vz[:nv_sub, :]

    vx = jnp.linspace(-V_MAX, V_MAX, nv_sub)

    # Reconstruct f on subset grid
    f = jnp.einsum('r,ir,jr,kr,lr,mr,nr->ijklmn',
                   weights, g_x_sub, g_y_sub, g_z_sub, h_vx_sub, h_vy_sub, h_vz_sub)

    # Compute gradients
    dv_sub = vx[1] - vx[0]
    df_dvx = jnp.gradient(f, dv_sub, axis=3)
    df_dvy = jnp.gradient(f, dv_sub, axis=4)
    df_dvz = jnp.gradient(f, dv_sub, axis=5)

    VX, VY, VZ = jnp.meshgrid(vx, vx, vx, indexing='ij')

    # Naive implementation: double loop over velocity space
    Q = jnp.zeros_like(f)

    for lp in range(nv_sub):
        for mp in range(nv_sub):
            for np_ in range(nv_sub):
                # Relative velocity
                gx = VX - vx[lp]
                gy = VY - vx[mp]
                gz = VZ - vx[np_]
                g_mag = jnp.sqrt(gx**2 + gy**2 + gz**2 + 1e-10)

                # Simplified Landau kernel (isotropic part)
                U = 1.0 / g_mag

                f_prime = f[:, :, :, lp, mp, np_]
                Q = Q + (U * f_prime[:, :, :, None, None, None] * dv_sub**3)

    return Q


@jax.jit
def landau_non_maxwellian_spectral(weights, g_x, g_y, g_z, h_vx, h_vy, h_vz, vx, vy, vz):
    """
    Landau collision operator using spectral acceleration.

    Uses FFT-based convolution for the velocity integral.

    Cost: O(r^2 * N_v^3 * log(N_v) + N_x^3) = O(r^2 * N^3 * log(N))
    """
    nv = len(vx)
    dv = vx[1] - vx[0]

    # Reconstruct f_neq
    f_neq = jnp.einsum('r,ir,jr,kr,lr,mr,nr->ijklmn',
                       weights, g_x, g_y, g_z, h_vx, h_vy, h_vz)

    # Compute gradients
    df_dvx = jnp.gradient(f_neq, dv, axis=3)
    df_dvy = jnp.gradient(f_neq, dv, axis=4)
    df_dvz = jnp.gradient(f_neq, dv, axis=5)

    # Spectral convolution for Landau kernel
    kx = jnp.fft.fftfreq(nv, dv) * 2 * jnp.pi
    KX, KY, KZ = jnp.meshgrid(kx, kx, kx, indexing='ij')
    k_sq = KX**2 + KY**2 + KZ**2 + 1e-10

    # Fourier transform of 1/|v| ~ 1/k^2
    U_hat = 1.0 / k_sq

    # Convolution via FFT for each spatial point
    f_hat = jnp.fft.fftn(f_neq, axes=(3, 4, 5))

    # Rosenbluth-like potential
    phi_hat = U_hat * f_hat
    phi = jnp.fft.ifftn(phi_hat, axes=(3, 4, 5)).real

    # Gradient of potential
    dphi_dvx = jnp.fft.ifftn(1j * KX * phi_hat, axes=(3, 4, 5)).real
    dphi_dvy = jnp.fft.ifftn(1j * KY * phi_hat, axes=(3, 4, 5)).real
    dphi_dvz = jnp.fft.ifftn(1j * KZ * phi_hat, axes=(3, 4, 5)).real

    # Landau collision term (simplified)
    Q_landau = (jnp.gradient(f_neq * dphi_dvx, dv, axis=3) +
                jnp.gradient(f_neq * dphi_dvy, dv, axis=4) +
                jnp.gradient(f_neq * dphi_dvz, dv, axis=5))

    return NU * Q_landau


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def benchmark_function(func, args, name, n_warmup=3, n_runs=10, use_jit=True):
    """Benchmark a function with warmup and multiple runs."""
    # Warmup (JIT compilation)
    for _ in range(n_warmup):
        result = func(*args)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    mean_time = np.mean(times)
    std_time = np.std(times)

    return mean_time, std_time, result.shape


def run_benchmarks(nx, nv, rank):
    """Run all collision operator benchmarks."""
    print(f"\n{'='*70}")
    print(f"COLLISION OPERATOR BENCHMARKS")
    print(f"Grid: N_x = {nx}, N_v = {nv}, Rank = {rank}")
    print(f"Total phase space points: {nx**3 * nv**3:,} = {nx}^3 × {nv}^3")
    print(f"Knudsen number: Kn = {Kn}, Collision frequency: ν = {NU}")
    print(f"{'='*70}\n")

    # Create test data
    print("Creating test data...")
    rho, u, temp, (x, y, z), (vx, vy, vz) = create_maxwellian_field(nx, nv)
    weights, (g_x, g_y, g_z), (h_vx, h_vy, h_vz) = create_non_maxwellian_field(nx, nv, rank)

    # Compute f_eq for BGK
    rho_exp = rho[:, :, :, None, None, None]
    ux_exp = u[0][:, :, :, None, None, None]
    uy_exp = u[1][:, :, :, None, None, None]
    uz_exp = u[2][:, :, :, None, None, None]
    temp_exp = temp[:, :, :, None, None, None]
    VX, VY, VZ = jnp.meshgrid(vx, vy, vz, indexing='ij')
    f_eq = (rho_exp / jnp.power(2*jnp.pi*temp_exp, 1.5) *
            jnp.exp(-((VX - ux_exp)**2 + (VY - uy_exp)**2 + (VZ - uz_exp)**2) / (2*temp_exp)))

    alpha = jnp.array(1e-3)

    results = {}

    # ==========================================================================
    # BGK OPERATOR
    # ==========================================================================
    print("\n" + "-"*50)
    print("BGK OPERATOR")
    print("-"*50)

    # BGK Maxwellian
    print("\n[1] BGK - Maxwellian part...")
    mean_t, std_t, shape = benchmark_function(
        bgk_maxwellian, (rho, u, temp, f_eq, vx, vy, vz), "BGK-Maxwellian"
    )
    results['BGK_Maxwellian'] = (mean_t, std_t)
    print(f"    Time: {mean_t*1000:.3f} ± {std_t*1000:.3f} ms")
    print(f"    Output shape: {shape}")
    print(f"    Complexity: O(N_x³ × N_v³) = O({nx**3 * nv**3:,})")

    # BGK Non-Maxwellian (Separable)
    print("\n[2] BGK - Non-Maxwellian part (separable)...")
    mean_t, std_t, shape = benchmark_function(
        bgk_non_maxwellian_separable, (weights, g_x, g_y, g_z, h_vx, h_vy, h_vz),
        "BGK-NonMaxwellian"
    )
    results['BGK_NonMaxwellian_Separable'] = (mean_t, std_t)
    print(f"    Time: {mean_t*1000:.3f} ± {std_t*1000:.3f} ms")
    print(f"    Output shape: {shape}")
    print(f"    Complexity: O(r × N⁶) for einsum reconstruction")

    # BGK Full
    print("\n[3] BGK - Full distribution (f_eq + α·f_neq)...")
    mean_t, std_t, shape = benchmark_function(
        bgk_full, (rho, u, temp, weights, g_x, g_y, g_z, h_vx, h_vy, h_vz, vx, vy, vz, alpha),
        "BGK-Full"
    )
    results['BGK_Full'] = (mean_t, std_t)
    print(f"    Time: {mean_t*1000:.3f} ± {std_t*1000:.3f} ms")
    print(f"    Output shape: {shape}")

    # ==========================================================================
    # FOKKER-PLANCK OPERATOR
    # ==========================================================================
    print("\n" + "-"*50)
    print("FOKKER-PLANCK OPERATOR")
    print("-"*50)

    # FP Maxwellian
    print("\n[4] FP - Maxwellian part...")
    mean_t, std_t, shape = benchmark_function(
        fp_maxwellian, (rho, u, temp, vx, vy, vz), "FP-Maxwellian"
    )
    results['FP_Maxwellian'] = (mean_t, std_t)
    print(f"    Time: {mean_t*1000:.3f} ± {std_t*1000:.3f} ms")
    print(f"    Output shape: {shape}")
    print(f"    Complexity: O(N_x³ × N_v³) = O({nx**3 * nv**3:,})")

    # FP Non-Maxwellian (Separable)
    print("\n[5] FP - Non-Maxwellian part (separable)...")
    mean_t, std_t, shape = benchmark_function(
        fp_non_maxwellian_separable, (weights, g_x, g_y, g_z, h_vx, h_vy, h_vz, vx, vy, vz),
        "FP-NonMaxwellian"
    )
    results['FP_NonMaxwellian_Separable'] = (mean_t, std_t)
    print(f"    Time: {mean_t*1000:.3f} ± {std_t*1000:.3f} ms")
    print(f"    Output shape: {shape}")
    print(f"    Complexity: O(r × N⁶ + N⁶ for gradients)")

    # ==========================================================================
    # LANDAU OPERATOR
    # ==========================================================================
    print("\n" + "-"*50)
    print("LANDAU OPERATOR")
    print("-"*50)

    # Landau Maxwellian (same as FP)
    print("\n[6] Landau - Maxwellian part (≈ FP for Maxwellian)...")
    mean_t, std_t, shape = benchmark_function(
        landau_maxwellian, (rho, u, temp, vx, vy, vz), "Landau-Maxwellian"
    )
    results['Landau_Maxwellian'] = (mean_t, std_t)
    print(f"    Time: {mean_t*1000:.3f} ± {std_t*1000:.3f} ms")
    print(f"    Output shape: {shape}")
    print(f"    Complexity: O(N_x³ × N_v³) = O({nx**3 * nv**3:,})")

    # Landau Non-Maxwellian NAIVE (very expensive!)
    nv_naive = min(nv, 8)  # Limit for tractability
    print(f"\n[7] Landau - Non-Maxwellian part (NAIVE, N_v={nv_naive})...")
    print(f"    WARNING: Naive O(N^9) implementation - using reduced grid")

    # Create smaller data for naive benchmark
    _, (g_x_small, g_y_small, g_z_small), (h_vx_small, h_vy_small, h_vz_small) = \
        create_non_maxwellian_field(nv_naive, nv_naive, rank)
    vx_small = jnp.linspace(-V_MAX, V_MAX, nv_naive)

    mean_t, std_t, shape = benchmark_function(
        landau_non_maxwellian_naive,
        (weights, g_x_small, g_y_small, g_z_small, h_vx_small, h_vy_small, h_vz_small, vx_small),
        "Landau-NonMaxwellian-Naive", n_warmup=1, n_runs=3
    )
    results['Landau_NonMaxwellian_Naive'] = (mean_t, std_t)
    print(f"    Time: {mean_t*1000:.3f} ± {std_t*1000:.3f} ms (N_v={nv_naive})")
    print(f"    Output shape: {shape}")
    print(f"    Complexity: O(N_x³ × N_v⁶) [subset only]")
    if nv > nv_naive:
        extrapolated = mean_t * (nv/nv_naive)**6
        print(f"    Extrapolated for N_v={nv}: ~{extrapolated*1000:.1f} ms")

    # Landau Non-Maxwellian Spectral (fast)
    print("\n[8] Landau - Non-Maxwellian part (SPECTRAL/FFT)...")
    mean_t, std_t, shape = benchmark_function(
        landau_non_maxwellian_spectral,
        (weights, g_x, g_y, g_z, h_vx, h_vy, h_vz, vx, vy, vz),
        "Landau-NonMaxwellian-Spectral"
    )
    results['Landau_NonMaxwellian_Spectral'] = (mean_t, std_t)
    print(f"    Time: {mean_t*1000:.3f} ± {std_t*1000:.3f} ms")
    print(f"    Output shape: {shape}")
    print(f"    Complexity: O(r × N⁶ + N⁶ log N) for FFT convolution")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "="*70)
    print("SUMMARY (Wall Clock Time per Evaluation)")
    print("="*70)
    print(f"\n{'Operator':<40} {'Time (ms)':<20}")
    print("-"*60)

    for name, (mean_t, std_t) in results.items():
        print(f"{name:<40} {mean_t*1000:>8.3f} ± {std_t*1000:>6.3f}")

    print("\n" + "="*70)
    print("COMPLEXITY COMPARISON (Theoretical)")
    print("="*70)
    print(f"""
    | Operator              | Maxwellian      | Non-Maxwellian (Full) | Non-Maxwellian (Separable) |
    |-----------------------|-----------------|----------------------|---------------------------|
    | BGK                   | O(N⁶)           | O(N⁶)                | O(r·N⁶) [einsum]          |
    | Fokker-Planck         | O(N⁶)           | O(N⁹) → O(N⁶ log N)  | O(r·N⁶ + N⁶)              |
    | Landau (naive)        | O(N⁶)           | O(N⁹)                | O(r·N⁹)                   |
    | Landau (spectral)     | O(N⁶)           | O(N⁶ log N)          | O(r·N⁶ log N)             |

    Parameters: N_x = {nx}, N_v = {nv}, r = {rank}

    Note: The separable representation helps with storage (O(rN) vs O(N^6)) but
    for collision operators, the full 6D tensor must typically be reconstructed,
    giving O(r·N⁶) complexity for the einsum operation.
    """)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark collision operators")
    parser.add_argument("--nx", type=int, default=NGRID, help="Spatial grid points per dimension")
    parser.add_argument("--nv", type=int, default=NGRID, help="Velocity grid points per dimension")
    parser.add_argument("--rank", type=int, default=RANK, help="Separable rank")
    args = parser.parse_args()

    # Check JAX backend
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    results = run_benchmarks(args.nx, args.nv, args.rank)
