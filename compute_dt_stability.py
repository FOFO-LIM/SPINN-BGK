# 2026-02-08: Compute D_max and explicit time-stepping stability bound for 2D Landau operator at Nv=64
"""
Compute the maximum diffusion coefficient D_max from A_ij(v) = conv(a_ij, f)(v)
for the 2D Landau operator with Nv=64, V=6.0, and f = Maxwellian(rho=1, u=0, T=1).

Stability bound for explicit Euler: Δt ≤ (Δv)² / (2d · D_max)
where d=2 for 2D.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import sys
import time

# Import the operator from the existing file
sys.path.insert(0, '/home/seongjaelim')
from verify_true_landau import LandauOperator2D

def compute_Dmax_and_stability():
    print("=" * 70)
    print("Computing D_max and Δt stability bound for 2D Landau operator")
    print("=" * 70)

    Nv = 64
    V = 6.0
    lambda_D = 10.0

    op = LandauOperator2D(Nv=Nv, V=V, lambda_D=lambda_D)
    dv = op.dv

    print(f"  Nv = {Nv}, V = {V}, λ_D = {lambda_D}")
    print(f"  Δv = 2V/(Nv-1) = {dv:.6f}")
    print(f"  (Δv)² = {dv**2:.6e}")
    print()

    # Maxwellian f
    f = op.maxwellian_2d(rho=1.0, ux=0.0, uy=0.0, T=1.0)
    print(f"  max|f| = {float(jnp.max(f)):.6e}")
    print()

    # Compute A_ij(v) = conv(a_ij, f)(v) for all v
    t0 = time.time()
    A_xx = op._conv2d(op.a_xx_fft, f)
    A_yy = op._conv2d(op.a_yy_fft, f)
    A_xy = op._conv2d(op.a_xy_fft, f)
    elapsed = time.time() - t0
    print(f"  A_ij convolutions computed in {elapsed:.3f} s")

    # Also compute B terms for advection constraint
    df_dvx, df_dvy = op._gradient_v(f)
    B_xx = op._conv2d(op.a_xx_fft, df_dvx)
    B_xy = op._conv2d(op.a_xy_fft, df_dvy)
    B_yx = op._conv2d(op.a_xy_fft, df_dvx)
    B_yy = op._conv2d(op.a_yy_fft, df_dvy)

    # Effective advection: b_x = B_xx + B_xy, b_y = B_yx + B_yy
    b_x = B_xx + B_xy
    b_y = B_yx + B_yy
    b_mag = jnp.sqrt(b_x**2 + b_y**2)
    B_max = float(jnp.max(b_mag))

    # At each grid point, the diffusion matrix is:
    #   D(v) = [[A_xx(v), A_xy(v)],
    #           [A_xy(v), A_yy(v)]]
    #
    # Eigenvalues of 2x2 symmetric matrix:
    #   λ± = (tr ± sqrt(tr² - 4·det)) / 2
    #   tr = A_xx + A_yy
    #   det = A_xx·A_yy - A_xy²

    tr = A_xx + A_yy
    det = A_xx * A_yy - A_xy**2
    discriminant = tr**2 - 4 * det
    discriminant = jnp.maximum(discriminant, 0.0)  # numerical safety

    lambda_max = (tr + jnp.sqrt(discriminant)) / 2.0
    lambda_min = (tr - jnp.sqrt(discriminant)) / 2.0

    # D_max = max over all grid points of the largest eigenvalue
    D_max = float(jnp.max(lambda_max))
    D_min_global = float(jnp.min(lambda_min))

    # Where does D_max occur?
    idx = jnp.unravel_index(jnp.argmax(lambda_max), lambda_max.shape)
    vx_at_max = float(op.vx[idx])
    vy_at_max = float(op.vy[idx])

    print()
    print("  Diffusion matrix A_ij(v) analysis:")
    print(f"    max eigenvalue (D_max) = {D_max:.6e}")
    print(f"    min eigenvalue (global) = {D_min_global:.6e}")
    print(f"    D_max occurs at (vx, vy) = ({vx_at_max:.3f}, {vy_at_max:.3f})")
    print()

    # Print A_ij statistics
    print(f"    max|A_xx| = {float(jnp.max(jnp.abs(A_xx))):.6e}")
    print(f"    max|A_yy| = {float(jnp.max(jnp.abs(A_yy))):.6e}")
    print(f"    max|A_xy| = {float(jnp.max(jnp.abs(A_xy))):.6e}")
    print(f"    max|b|    = {B_max:.6e}  (advection coefficient)")
    print()

    # Stability bounds
    print("  Stability bounds (explicit Euler):")
    print("  " + "-" * 50)

    # Diffusion constraint: Δt ≤ (Δv)² / (2d · D_max)
    d = 2
    dt_diffusion = dv**2 / (2 * d * D_max)
    print(f"    Diffusion:  Δt ≤ (Δv)² / (2d·D_max)")
    print(f"              = {dv**2:.6e} / (4 × {D_max:.6e})")
    print(f"              = {dt_diffusion:.6e}")

    # Advection constraint: Δt ≤ Δv / |B_max|
    if B_max > 0:
        dt_advection = dv / B_max
        print(f"    Advection:  Δt ≤ Δv / |b_max|")
        print(f"              = {dv:.6e} / {B_max:.6e}")
        print(f"              = {dt_advection:.6e}")
    else:
        dt_advection = float('inf')
        print(f"    Advection:  no constraint (B_max ≈ 0)")

    # Combined
    dt_stable = min(dt_diffusion, dt_advection)
    limiting = "diffusion" if dt_diffusion <= dt_advection else "advection"

    print()
    print(f"    >>> Most restrictive: Δt ≤ {dt_stable:.6e}  (limited by {limiting})")
    print()

    # Safety factor recommendations
    print("  Practical recommendations (with safety factor):")
    for sf in [0.5, 0.25, 0.1]:
        print(f"    Safety {sf}: Δt = {sf * dt_stable:.6e}")

    print()

    # Also show how this scales with Nv
    print("  Scaling with Nv (D_max assumed constant ≈ {:.4e}):".format(D_max))
    print(f"  {'Nv':>6} {'Δv':>10} {'Δt_max (diffusion)':>20}")
    print("  " + "-" * 40)
    for Nv_test in [16, 32, 48, 64, 96, 128]:
        dv_test = 2 * V / (Nv_test - 1)
        dt_test = dv_test**2 / (2 * d * D_max)
        print(f"  {Nv_test:6d} {dv_test:10.6f} {dt_test:20.6e}")

    print("=" * 70)


if __name__ == "__main__":
    compute_Dmax_and_stability()
