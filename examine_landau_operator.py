# 2026-02-07: Added advection term comparison and N_v convergence test; Examine Q_L(f,f) at t=0
"""
Examine the Landau collision operator Q_L(f,f) at t=0 (no time evolution).

At t=0, f = local Maxwellian, so:
  - BGK operator:  (M[f] - f) / τ = 0  (because f = M[f])
  - Landau Q_L(M, M) should also = 0 for a Maxwellian

This matters for optimal τ = <f_neq, f_neq> / <Q_L, f_neq>:
if both Q_L and f_neq are zero at t=0, τ_opt is indeterminate (0/0).

Uses the LandauSolver1D_JAX class from the SPINN-BGK project.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser(
    "~/SPINN-BGK-main(original)/SPINN-BGK-main"
))
from landau_1d_numerical_jax import LandauSolver1D_JAX


def examine():
    N_x, N_v = 64, 128
    solver = LandauSolver1D_JAX(
        N_x=N_x, N_v=N_v, N_t=1,
        X=0.5, V=6.0, T_final=0.01, lambda_D=10.0,
    )
    import jax.numpy as jnp

    print("=" * 65)
    print("Q_L(f,f) at t=0  (f = local Maxwellian IC)")
    print("=" * 65)
    print(f"Grid: N_x={N_x}, N_v={N_v}, V={solver.V}, lambda_D={solver.lambda_D}")
    print(f"IC:   rho = 1 + 0.5*sin(2*pi*x),  u = 0,  T = 1")
    print()

    # IC: local Maxwellian with rho=1+0.5*sin(2*pi*x), u=0, T=1
    f0 = solver.initial_condition()
    f0_np = np.array(f0)

    # Landau operator at t=0
    Q_L = np.array(solver._collision_operator(f0))

    # BGK operator at t=0: (M[f] - f) = 0 since f is already Maxwellian
    rho, u, T = solver._compute_moments(f0)
    f_eq = np.array(solver._maxwellian(rho, u, T))
    f_neq = f_eq - f0_np  # should be ~0

    print("--- Landau operator Q_L(f0, f0) ---")
    print(f"  max|Q_L|       = {np.max(np.abs(Q_L)):.6e}")
    print(f"  mean|Q_L|      = {np.mean(np.abs(Q_L)):.6e}")
    print(f"  L2(Q_L)        = {np.sqrt(np.mean(Q_L**2)):.6e}")
    print()

    print("--- BGK operator: f_neq = M[f] - f ---")
    print(f"  max|f_neq|     = {np.max(np.abs(f_neq)):.6e}")
    print(f"  mean|f_neq|    = {np.mean(np.abs(f_neq)):.6e}")
    print(f"  L2(f_neq)      = {np.sqrt(np.mean(f_neq**2)):.6e}")
    print()

    print("--- Reference scale ---")
    print(f"  max|f0|        = {np.max(np.abs(f0_np)):.6e}")
    print()

    print("--- Relative magnitudes ---")
    print(f"  |Q_L|/|f0|     = {np.max(np.abs(Q_L)) / np.max(np.abs(f0_np)):.6e}")
    print(f"  |f_neq|/|f0|   = {np.max(np.abs(f_neq)) / np.max(np.abs(f0_np)):.6e}")
    print()

    # Advection term: v * df/dx at t=0
    # f(0,x,v) = rho(x) * g(v), so df/dx = rho'(x) * g(v)
    # Use spectral derivative for df/dx
    x_np = np.array(solver.x)
    v_np = np.array(solver.v)
    kx = np.array(solver.kx)
    f0_hat = np.fft.fft(f0_np, axis=0)
    df_dx = np.real(np.fft.ifft(1j * kx[:, None] * f0_hat, axis=0))
    advection = v_np[None, :] * df_dx  # (N_x, N_v)

    print("--- Advection term: v * df/dx at t=0 ---")
    print(f"  max|v df/dx|   = {np.max(np.abs(advection)):.6e}")
    print(f"  mean|v df/dx|  = {np.mean(np.abs(advection)):.6e}")
    print(f"  L2(v df/dx)    = {np.sqrt(np.mean(advection**2)):.6e}")
    print()

    print("--- Comparison: spurious Q_L vs physical advection ---")
    print(f"  |Q_L| / |v df/dx|  = {np.max(np.abs(Q_L)) / np.max(np.abs(advection)):.6e}")
    print()

    # ------------------------------------------------------------------
    # Convergence test: does Q_L -> 0 as N_v increases?
    # ------------------------------------------------------------------
    print("=" * 65)
    print("CONVERGENCE TEST: max|Q_L(M,M)| vs N_v")
    print("=" * 65)
    print(f"  {'N_v':>6}  {'dv':>10}  {'max|Q_L|':>12}  {'max|Q_L|/|f|':>14}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*14}")
    for nv in [32, 64, 128, 256, 512]:
        s = LandauSolver1D_JAX(
            N_x=64, N_v=nv, N_t=1,
            X=0.5, V=6.0, T_final=0.01, lambda_D=10.0,
        )
        f_test = s.initial_condition()
        Q_test = np.array(s._collision_operator(f_test))
        f_max = np.max(np.abs(np.array(f_test)))
        Q_max = np.max(np.abs(Q_test))
        print(f"  {nv:>6}  {s.dv:>10.6f}  {Q_max:>12.6e}  {Q_max/f_max:>14.6e}")
    print("=" * 65)


if __name__ == "__main__":
    examine()
