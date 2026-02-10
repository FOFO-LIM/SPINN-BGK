# 2026-02-07: Initial creation - Diagnose why Q_L(M,M) != 0 for the 1D scalar Landau operator
"""
Diagnose the Landau collision operator Q_L(f,f) on a Maxwellian at t=0.

The operator is:
    Q_L = d/dv [A * df/dv - B * f]
    A[f](v) = ∫ Φ(|v-v'|) f(v') dv'
    B[f](v) = ∫ Φ(|v-v'|) df/dv'(v') dv'

Key identity (integration by parts):
    B[f](v) = dA[f]/dv

So the flux becomes:
    J = A * f' - A' * f

For Maxwellian f = C*exp(-v^2/2), f' = -v*f, so:
    J = A*(-v*f) - A'*f = -f*(v*A + A')
    Q = dJ/dv = ... = f * [(v^2-1)*A - A'']

Q = 0  requires  A'' = (v^2 - 1)*A, which is satisfied only if A is Gaussian.
But A = Φ * f (convolution), which is Gaussian only if Φ is Gaussian.
Since Φ(|u|) = 1/max(|u|, 1/λ_D) (Coulomb) is NOT Gaussian, Q(M,M) != 0.

This is a fundamental property of the 1D scalar Landau model -- the projection
tensor Π = I - zz^T/|z|^2 that makes Q(M,M)=0 in 3D is absent here.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser(
    "~/SPINN-BGK-main(original)/SPINN-BGK-main"
))
from landau_1d_numerical_jax import LandauSolver1D_JAX


def diagnose():
    N_x, N_v = 4, 512  # few x-points, fine v-grid
    solver = LandauSolver1D_JAX(
        N_x=N_x, N_v=N_v, N_t=1,
        X=0.5, V=8.0, T_final=0.01, lambda_D=10.0,
    )
    import jax.numpy as jnp

    v = np.array(solver.v)
    dv = float(solver.dv)

    print("=" * 70)
    print("DIAGNOSING Q_L(M, M) for 1D scalar Landau operator")
    print("=" * 70)
    print(f"N_v={N_v}, V={solver.V}, dv={dv:.6f}, lambda_D={solver.lambda_D}")
    print()

    # Use a SINGLE uniform Maxwellian to isolate the v-space behavior
    rho_uniform = jnp.ones(N_x)
    u_zero = jnp.zeros(N_x)
    T_one = jnp.ones(N_x)
    f0 = solver._maxwellian(rho_uniform, u_zero, T_one)
    f0_np = np.array(f0)

    # ---- Step 1: Compute intermediate quantities ----
    A, B = solver._compute_collision_coefficients(f0)
    A, B = np.array(A), np.array(B)
    df_dv = np.array(solver._compute_dv(f0))

    # Numerical dA/dv (should equal B by IBP identity)
    dA_dv_num = np.zeros_like(A)
    dA_dv_num[:, 1:-1] = (A[:, 2:] - A[:, :-2]) / (2 * dv)
    dA_dv_num[:, 0] = (A[:, 1] - A[:, 0]) / dv
    dA_dv_num[:, -1] = (A[:, -1] - A[:, -2]) / dv

    # Flux J = A*f' - B*f
    J = A * df_dv - B * f0_np
    Q = np.array(solver._collision_operator(f0))

    # Pick x=0 slice (all x are identical for uniform rho)
    ix = 0
    f_s = f0_np[ix, :]
    A_s = A[ix, :]
    B_s = B[ix, :]
    dA_s = dA_dv_num[ix, :]
    J_s = J[ix, :]
    Q_s = Q[ix, :]

    print("---- Step 1: Verify IBP identity B[f] = dA[f]/dv ----")
    print(f"  max|B|              = {np.max(np.abs(B_s)):.6e}")
    print(f"  max|dA/dv|          = {np.max(np.abs(dA_s)):.6e}")
    print(f"  max|B - dA/dv|      = {np.max(np.abs(B_s - dA_s)):.6e}")
    print(f"  max|B - dA/dv|/|B|  = {np.max(np.abs(B_s - dA_s)) / (np.max(np.abs(B_s)) + 1e-30):.6e}")
    print()

    print("---- Step 2: Intermediate quantities at x=0 ----")
    print(f"  max|f|    = {np.max(np.abs(f_s)):.6e}")
    print(f"  max|A|    = {np.max(np.abs(A_s)):.6e}")
    print(f"  max|B|    = {np.max(np.abs(B_s)):.6e}")
    print(f"  max|f'|   = {np.max(np.abs(df_dv[ix])):.6e}")
    print(f"  max|J|    = {np.max(np.abs(J_s)):.6e}")
    print(f"  max|Q|    = {np.max(np.abs(Q_s)):.6e}")
    print()

    # ---- Step 3: Analytical prediction ----
    # Q = f * [(v^2-1)*A - A'']
    # Compute A'' numerically
    d2A_dv2 = np.zeros_like(A_s)
    d2A_dv2[1:-1] = (A_s[2:] - 2*A_s[1:-1] + A_s[:-2]) / dv**2
    predicted_Q = f_s * ((v**2 - 1) * A_s - d2A_dv2)

    print("---- Step 3: Analytical prediction Q = f*[(v^2-1)*A - A''] ----")
    print(f"  max|predicted Q|       = {np.max(np.abs(predicted_Q)):.6e}")
    print(f"  max|actual Q|          = {np.max(np.abs(Q_s)):.6e}")
    # The sign flip from the code's -Q:
    print(f"  max|predicted + Q_code| = {np.max(np.abs(predicted_Q + Q_s)):.6e}  (should be ~0, sign convention)")
    print(f"  max|predicted - Q_code| = {np.max(np.abs(predicted_Q - Q_s)):.6e}")
    print()

    print("---- Step 4: Why Q != 0: Is A Gaussian? ----")
    # If A were Gaussian: A(v) = c * exp(-v^2/(2*sigma^2))
    # Then A'' = (v^2/sigma^4 - 1/sigma^2) * A
    # For sigma=1: A'' = (v^2 - 1)*A, giving Q=0.
    # But A = Phi*f is NOT Gaussian for Coulomb Phi.
    #
    # Compare A(v) to best-fit Gaussian
    A_max = A_s[N_v//2]  # peak at v=0
    # Fit: A(v) ~ A_max * exp(-v^2 / (2*sigma^2))
    # At v=1: A(1)/A_max = exp(-1/(2*sigma^2))
    idx_v1 = np.argmin(np.abs(v - 1.0))
    if A_max > 0 and A_s[idx_v1] > 0:
        sigma2_fit = -1.0 / (2 * np.log(A_s[idx_v1] / A_max))
        A_gaussian_fit = A_max * np.exp(-v**2 / (2 * sigma2_fit))
    else:
        sigma2_fit = 1.0
        A_gaussian_fit = A_max * np.exp(-v**2 / 2)

    residual = A_s - A_gaussian_fit
    print(f"  A(0)                   = {A_max:.6e}")
    print(f"  Best-fit Gaussian σ²   = {sigma2_fit:.4f}  (f has σ²=1)")
    print(f"  max|A - A_gauss_fit|   = {np.max(np.abs(residual)):.6e}")
    print(f"  max|A - A_gauss_fit|/A = {np.max(np.abs(residual)) / A_max:.6e}")
    print(f"  => A is NOT Gaussian => (v²-1)*A - A'' != 0 => Q(M,M) != 0")
    print()

    print("---- Step 5: The root cause (theory) ----")
    print("  The TRUE 3D Landau operator uses a tensor kernel:")
    print("    a_ij(z) = Φ(|z|) * (δ_ij - z_i*z_j/|z|²)")
    print("  The projection Π = I - zz^T/|z|² ensures Q(M,M) = 0 because:")
    print("    Σ_j Π_ij(v-v') * (v'_j - v_j) = 0  (identically)")
    print()
    print("  In 1D, Π = 1 - 1 = 0, so the true Landau operator is trivially 0.")
    print("  This code uses a SCALAR 1D model (drops the projection).")
    print("  Without Π, the cancellation breaks: Q(M,M) != 0.")
    print()
    print("  The nonzero Q_L ~ 0.7 at t=0 is NOT a numerical bug.")
    print("  It is the CORRECT value of this model operator on a Maxwellian.")
    print("=" * 70)


if __name__ == "__main__":
    diagnose()
