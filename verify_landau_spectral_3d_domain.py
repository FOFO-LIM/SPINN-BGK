# 2026-02-08: Test 3D spectral Landau with varying V to show domain truncation is the bottleneck
"""
The 3D spectral operator plateaus at ~10^-5 with V=4.0.
This is because f(±V) is not negligible, breaking the periodicity assumption.

Test: fix Nv, vary V to show that larger V → smaller Q(M,M).
Also test: fix V, vary Nv.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import time

import sys
sys.path.insert(0, '/home/seongjaelim')
from verify_landau_spectral import LandauOperator3D_Spectral


def test_vary_V():
    """Fix Nv, vary V: show domain truncation is the bottleneck."""
    print("=" * 70)
    print("TEST A: 3D Spectral — Fix Nv, Vary V (domain size)")
    print("=" * 70)
    print("  Larger V → f(±V) smaller → better periodicity → smaller Q(M,M)")
    print()
    print(f"{'Nv':>6} {'V':>6} {'Δv':>8} {'f(±V)/f(0)':>12} {'max|Q|/|f|':>14} {'time':>8}")
    print("-" * 58)

    for Nv in [16, 24]:
        for V in [3.0, 4.0, 5.0, 6.0]:
            t0 = time.time()
            op = LandauOperator3D_Spectral(Nv=Nv, V=V, lambda_D=10.0)
            f = op.maxwellian_3d(rho=1.0, ux=0.0, uy=0.0, uz=0.0, T=1.0)
            Q = op.collision_operator(f)
            elapsed = time.time() - t0

            max_Q = float(jnp.max(jnp.abs(Q)))
            max_f = float(jnp.max(jnp.abs(f)))
            rel = max_Q / max_f

            # f(V,0,0) / f(0,0,0) = exp(-V²/2)
            boundary_ratio = np.exp(-V**2 / 2)

            print(f"{Nv:6d} {V:6.1f} {float(op.dv):8.4f} {boundary_ratio:12.4e} {rel:14.4e} {elapsed:8.2f}s")

        print()

    print()


def test_vary_Nv_fixed_V():
    """Fix V=6.0, vary Nv: show spectral convergence in 3D with adequate domain."""
    print("=" * 70)
    print("TEST B: 3D Spectral — Fix V=6.0, Vary Nv")
    print("=" * 70)
    print("  With V=6.0, f(±V)/f(0) ≈ 1.5e-8 — adequate for spectral convergence")
    print()
    print(f"{'Nv':>6} {'Nv³':>8} {'Δv':>8} {'max|Q|/|f|':>14} {'time':>8}")
    print("-" * 50)

    Nv_list = [12, 16, 20, 24]

    for Nv in Nv_list:
        t0 = time.time()
        op = LandauOperator3D_Spectral(Nv=Nv, V=6.0, lambda_D=10.0)
        f = op.maxwellian_3d(rho=1.0, ux=0.0, uy=0.0, uz=0.0, T=1.0)
        Q = op.collision_operator(f)
        elapsed = time.time() - t0

        max_Q = float(jnp.max(jnp.abs(Q)))
        max_f = float(jnp.max(jnp.abs(f)))
        rel = max_Q / max_f

        print(f"{Nv:6d} {Nv**3:8d} {float(op.dv):8.4f} {rel:14.4e} {elapsed:8.2f}s")

    print()


def test_3d_spatially_varying_V6():
    """3D spatially-varying IC with V=6.0."""
    print("=" * 70)
    print("TEST C: 3D Spatially-Varying IC (Spectral, V=6.0)")
    print("         ρ = 1 + 0.5 sin(2πx)sin(2πy)")
    print("=" * 70)

    Nx, Ny = 4, 4
    Nv = 16

    x = np.linspace(-0.5, 0.5, Nx, endpoint=False)
    y = np.linspace(-0.5, 0.5, Ny, endpoint=False)

    op = LandauOperator3D_Spectral(Nv=Nv, V=6.0, lambda_D=10.0)

    max_Q_vals = []
    max_f_vals = []

    t0 = time.time()
    for ix in range(Nx):
        for iy in range(Ny):
            rho = 1.0 + 0.5 * np.sin(2 * np.pi * x[ix]) * np.sin(2 * np.pi * y[iy])
            f = op.maxwellian_3d(rho=rho, ux=0.0, uy=0.0, uz=0.0, T=1.0)
            Q = op.collision_operator(f)

            max_Q_vals.append(float(jnp.max(jnp.abs(Q))))
            max_f_vals.append(float(jnp.max(jnp.abs(f))))

    elapsed = time.time() - t0

    max_Q_all = np.max(max_Q_vals)
    max_f_all = np.max(max_f_vals)

    print(f"  Nx×Ny = {Nx}×{Ny}, Nv = {Nv}, V = 6.0")
    print(f"  max|Q| over all spatial points: {max_Q_all:.4e}")
    print(f"  max|f| over all spatial points: {max_f_all:.4e}")
    print(f"  max|Q|/max|f|:                  {max_Q_all / max_f_all:.4e}")
    print(f"  Time: {elapsed:.2f} s")
    print()


def main():
    print("\n" + "=" * 70)
    print("3D SPECTRAL LANDAU: Domain Size vs Resolution Analysis")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")
    print()

    total_start = time.time()

    test_vary_V()
    test_vary_Nv_fixed_V()
    test_3d_spatially_varying_V6()

    total_elapsed = time.time() - total_start

    print("=" * 70)
    print(f"Total wall time: {total_elapsed:.1f} s")


if __name__ == "__main__":
    main()
