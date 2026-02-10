# 2026-02-07: Initial creation - Verify true 2D/3D Landau collision operator Q(M,M)=0; V=4.0 for 3D resolution
"""
Verify that the TRUE tensor Landau collision operator satisfies Q(M,M) = 0.

The Landau operator in d >= 2 dimensions:
    Q_i = Σ_j ∂/∂v_i [ A_ij · ∂f/∂v_j  -  B_ij · f ]
    Q = Σ_i ∂Q_i/∂v_i   (divergence of flux)

where:
    A_ij(v) = (a_ij * f)(v)        convolution with f
    B_ij(v) = (a_ij * ∂_j f)(v)   convolution with ∂f/∂v_j

    a_ij(z) = Φ(|z|) · Π_ij(z)   tensor kernel
    Π_ij(z) = δ_ij - z_i z_j/|z|²  projection tensor

    Φ(|z|) = 1/max(|z|, 1/λ_D)   Coulomb with Debye cutoff

Key property: Π_ij(z) · z_j = 0 ∀i, which ensures Q(M,M) = 0.
In 1D, Π = 1-1 = 0 (trivial). Starting from 2D, Π is non-trivial.

This script:
1. Implements LandauOperator2D (2D velocity space)
2. Implements LandauOperator3D (3D velocity space)
3. Verifies Q(M,M) → 0 with grid refinement (2D and 3D)
4. Tests spatially-varying IC: ρ(x,y) = 1 + 0.5 sin(2πx)sin(2πy)
5. Compares with the 1D scalar operator (negative control: Q ≠ 0)
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import numpy as np
import time


# =============================================================================
# 2D Landau Collision Operator
# =============================================================================

class LandauOperator2D:
    """
    True 2D Landau collision operator with tensor kernel.

    Kernel: a_ij(z) = Φ(|z|) · (δ_ij - z_i z_j / |z|²)
    3 unique components: a_xx, a_yy, a_xy (since a_yx = a_xy)
    7 FFT convolutions per spatial point.
    """

    def __init__(self, Nv, V=6.0, lambda_D=10.0):
        self.Nv = Nv
        self.V = V
        self.lambda_D = lambda_D
        self.cutoff = 1.0 / lambda_D

        # Velocity grid
        self.dv = 2 * V / (Nv - 1)
        self.v = jnp.linspace(-V, V, Nv)
        self.vx, self.vy = jnp.meshgrid(self.v, self.v, indexing='ij')

        # Padded convolution size
        self.N_conv = 2 * Nv - 1

        # Precompute kernel FFTs
        self._precompute_kernel_fft()

    def _precompute_kernel_fft(self):
        """Precompute FFT of tensor kernel components a_xx, a_yy, a_xy."""
        Nv = self.Nv
        dv = self.dv
        N_conv = self.N_conv

        # Relative velocity grid z = (zx, zy)
        z1d = jnp.arange(-(Nv - 1), Nv) * dv  # length 2*Nv-1
        zx, zy = jnp.meshgrid(z1d, z1d, indexing='ij')
        z_mag = jnp.sqrt(zx**2 + zy**2)

        # Coulomb kernel with Debye cutoff
        Phi = 1.0 / jnp.maximum(z_mag, self.cutoff)

        # Projection tensor Π_ij = δ_ij - z_i z_j / |z|²
        # Regularize at z=0: use angular average Π_ij(0) = δ_ij/d = δ_ij/2
        z_mag_sq = z_mag**2
        z_mag_sq_safe = jnp.where(z_mag_sq > 0, z_mag_sq, 1.0)

        Pi_xx = jnp.where(z_mag > 0, 1.0 - zx**2 / z_mag_sq_safe, 0.5)
        Pi_yy = jnp.where(z_mag > 0, 1.0 - zy**2 / z_mag_sq_safe, 0.5)
        Pi_xy = jnp.where(z_mag > 0, -zx * zy / z_mag_sq_safe, 0.0)

        # Tensor kernel components
        a_xx = Phi * Pi_xx
        a_yy = Phi * Pi_yy
        a_xy = Phi * Pi_xy

        # Store FFTs
        self.a_xx_fft = jnp.fft.fft2(a_xx)
        self.a_yy_fft = jnp.fft.fft2(a_yy)
        self.a_xy_fft = jnp.fft.fft2(a_xy)

    def _conv2d(self, kernel_fft, g):
        """
        Compute 2D convolution of kernel with g using FFT.

        Parameters:
            kernel_fft: FFT of kernel on (2*Nv-1, 2*Nv-1) grid
            g: function on (Nv, Nv) velocity grid

        Returns:
            conv: convolution result on (Nv, Nv) grid
        """
        Nv = self.Nv
        dv = self.dv

        # Zero-pad g
        g_padded = jnp.zeros((self.N_conv, self.N_conv))
        g_padded = g_padded.at[:Nv, :Nv].set(g)

        # Convolution via FFT
        conv_full = jnp.real(jnp.fft.ifft2(kernel_fft * jnp.fft.fft2(g_padded)))

        # Extract valid part (linear convolution)
        start = Nv - 1
        conv = conv_full[start:start + Nv, start:start + Nv] * dv**2

        return conv

    def _gradient_v(self, f):
        """
        Compute (∂f/∂vx, ∂f/∂vy) using central differences.

        Parameters:
            f: shape (Nv, Nv)

        Returns:
            df_dvx, df_dvy: each shape (Nv, Nv)
        """
        dv = self.dv

        # ∂f/∂vx (axis=0)
        df_dvx = jnp.zeros_like(f)
        df_dvx = df_dvx.at[1:-1, :].set((f[2:, :] - f[:-2, :]) / (2 * dv))
        df_dvx = df_dvx.at[0, :].set((f[1, :] - f[0, :]) / dv)
        df_dvx = df_dvx.at[-1, :].set((f[-1, :] - f[-2, :]) / dv)

        # ∂f/∂vy (axis=1)
        df_dvy = jnp.zeros_like(f)
        df_dvy = df_dvy.at[:, 1:-1].set((f[:, 2:] - f[:, :-2]) / (2 * dv))
        df_dvy = df_dvy.at[:, 0].set((f[:, 1] - f[:, 0]) / dv)
        df_dvy = df_dvy.at[:, -1].set((f[:, -1] - f[:, -2]) / dv)

        return df_dvx, df_dvy

    def collision_operator(self, f):
        """
        Compute Q(f,f) for distribution f on 2D velocity grid.

        Parameters:
            f: shape (Nv, Nv)

        Returns:
            Q: shape (Nv, Nv)
        """
        # Step 1: Velocity gradients of f
        df_dvx, df_dvy = self._gradient_v(f)

        # Step 2: A_ij = conv(a_ij, f)  [3 convolutions]
        A_xx = self._conv2d(self.a_xx_fft, f)
        A_yy = self._conv2d(self.a_yy_fft, f)
        A_xy = self._conv2d(self.a_xy_fft, f)

        # Step 3: B_ij = conv(a_ij, ∂_j f)  [4 convolutions]
        B_xx = self._conv2d(self.a_xx_fft, df_dvx)  # a_xx * ∂_x f
        B_xy = self._conv2d(self.a_xy_fft, df_dvy)   # a_xy * ∂_y f
        B_yx = self._conv2d(self.a_xy_fft, df_dvx)   # a_yx * ∂_x f  (a_yx = a_xy)
        B_yy = self._conv2d(self.a_yy_fft, df_dvy)   # a_yy * ∂_y f

        # Step 4: Fluxes
        # J_x = A_xx * ∂_x f + A_xy * ∂_y f - f * (B_xx + B_xy)
        # J_y = A_xy * ∂_x f + A_yy * ∂_y f - f * (B_yx + B_yy)
        J_x = A_xx * df_dvx + A_xy * df_dvy - f * (B_xx + B_xy)
        J_y = A_xy * df_dvx + A_yy * df_dvy - f * (B_yx + B_yy)

        # Step 5: Divergence Q = ∂J_x/∂vx + ∂J_y/∂vy
        dJx_dvx, _ = self._gradient_v(J_x)
        _, dJy_dvy = self._gradient_v(J_y)

        Q = dJx_dvx + dJy_dvy

        return Q

    def maxwellian_2d(self, rho, ux, uy, T):
        """
        2D Maxwellian: f = ρ/(2πT) · exp(-((vx-ux)² + (vy-uy)²)/(2T))

        Parameters:
            rho, ux, uy, T: scalars

        Returns:
            f: shape (Nv, Nv)
        """
        return rho / (2 * jnp.pi * T) * jnp.exp(
            -((self.vx - ux)**2 + (self.vy - uy)**2) / (2 * T)
        )


# =============================================================================
# 3D Landau Collision Operator
# =============================================================================

class LandauOperator3D:
    """
    True 3D Landau collision operator with tensor kernel.

    Kernel: a_ij(z) = Φ(|z|) · (δ_ij - z_i z_j / |z|²)
    6 unique components: a_xx, a_yy, a_zz, a_xy, a_xz, a_yz
    15 FFT convolutions per spatial point.
    """

    def __init__(self, Nv, V=6.0, lambda_D=10.0):
        self.Nv = Nv
        self.V = V
        self.lambda_D = lambda_D
        self.cutoff = 1.0 / lambda_D

        # Velocity grid
        self.dv = 2 * V / (Nv - 1)
        self.v = jnp.linspace(-V, V, Nv)
        vx, vy, vz = jnp.meshgrid(self.v, self.v, self.v, indexing='ij')
        self.vx = vx
        self.vy = vy
        self.vz = vz

        # Padded convolution size
        self.N_conv = 2 * Nv - 1

        # Precompute kernel FFTs
        self._precompute_kernel_fft()

    def _precompute_kernel_fft(self):
        """Precompute FFT of tensor kernel components."""
        Nv = self.Nv
        dv = self.dv
        N_conv = self.N_conv

        # Relative velocity grid
        z1d = jnp.arange(-(Nv - 1), Nv) * dv
        zx, zy, zz = jnp.meshgrid(z1d, z1d, z1d, indexing='ij')
        z_mag = jnp.sqrt(zx**2 + zy**2 + zz**2)

        # Coulomb kernel with Debye cutoff
        Phi = 1.0 / jnp.maximum(z_mag, self.cutoff)

        # Projection tensor with regularization at z=0
        # Angular average in 3D: Π_ij(0) = δ_ij * (d-1)/d = δ_ij * 2/3
        z_mag_sq = z_mag**2
        z_mag_sq_safe = jnp.where(z_mag_sq > 0, z_mag_sq, 1.0)

        Pi_xx = jnp.where(z_mag > 0, 1.0 - zx**2 / z_mag_sq_safe, 2.0/3.0)
        Pi_yy = jnp.where(z_mag > 0, 1.0 - zy**2 / z_mag_sq_safe, 2.0/3.0)
        Pi_zz = jnp.where(z_mag > 0, 1.0 - zz**2 / z_mag_sq_safe, 2.0/3.0)
        Pi_xy = jnp.where(z_mag > 0, -zx * zy / z_mag_sq_safe, 0.0)
        Pi_xz = jnp.where(z_mag > 0, -zx * zz / z_mag_sq_safe, 0.0)
        Pi_yz = jnp.where(z_mag > 0, -zy * zz / z_mag_sq_safe, 0.0)

        # Store FFTs of tensor kernel
        self.a_xx_fft = jnp.fft.fftn(Phi * Pi_xx)
        self.a_yy_fft = jnp.fft.fftn(Phi * Pi_yy)
        self.a_zz_fft = jnp.fft.fftn(Phi * Pi_zz)
        self.a_xy_fft = jnp.fft.fftn(Phi * Pi_xy)
        self.a_xz_fft = jnp.fft.fftn(Phi * Pi_xz)
        self.a_yz_fft = jnp.fft.fftn(Phi * Pi_yz)

    def _conv3d(self, kernel_fft, g):
        """Compute 3D convolution of kernel with g using FFT."""
        Nv = self.Nv
        dv = self.dv
        N_conv = self.N_conv

        g_padded = jnp.zeros((N_conv, N_conv, N_conv))
        g_padded = g_padded.at[:Nv, :Nv, :Nv].set(g)

        conv_full = jnp.real(jnp.fft.ifftn(kernel_fft * jnp.fft.fftn(g_padded)))

        start = Nv - 1
        conv = conv_full[start:start+Nv, start:start+Nv, start:start+Nv] * dv**3

        return conv

    def _gradient_v(self, f):
        """
        Compute (∂f/∂vx, ∂f/∂vy, ∂f/∂vz) using central differences.

        Parameters:
            f: shape (Nv, Nv, Nv)

        Returns:
            df_dvx, df_dvy, df_dvz: each shape (Nv, Nv, Nv)
        """
        dv = self.dv

        # ∂f/∂vx (axis=0)
        df_dvx = jnp.zeros_like(f)
        df_dvx = df_dvx.at[1:-1, :, :].set((f[2:, :, :] - f[:-2, :, :]) / (2 * dv))
        df_dvx = df_dvx.at[0, :, :].set((f[1, :, :] - f[0, :, :]) / dv)
        df_dvx = df_dvx.at[-1, :, :].set((f[-1, :, :] - f[-2, :, :]) / dv)

        # ∂f/∂vy (axis=1)
        df_dvy = jnp.zeros_like(f)
        df_dvy = df_dvy.at[:, 1:-1, :].set((f[:, 2:, :] - f[:, :-2, :]) / (2 * dv))
        df_dvy = df_dvy.at[:, 0, :].set((f[:, 1, :] - f[:, 0, :]) / dv)
        df_dvy = df_dvy.at[:, -1, :].set((f[:, -1, :] - f[:, -2, :]) / dv)

        # ∂f/∂vz (axis=2)
        df_dvz = jnp.zeros_like(f)
        df_dvz = df_dvz.at[:, :, 1:-1].set((f[:, :, 2:] - f[:, :, :-2]) / (2 * dv))
        df_dvz = df_dvz.at[:, :, 0].set((f[:, :, 1] - f[:, :, 0]) / dv)
        df_dvz = df_dvz.at[:, :, -1].set((f[:, :, -1] - f[:, :, -2]) / dv)

        return df_dvx, df_dvy, df_dvz

    def collision_operator(self, f):
        """
        Compute Q(f,f) for distribution f on 3D velocity grid.

        Parameters:
            f: shape (Nv, Nv, Nv)

        Returns:
            Q: shape (Nv, Nv, Nv)
        """
        # Step 1: Velocity gradients of f
        df_dvx, df_dvy, df_dvz = self._gradient_v(f)

        # Step 2: A_ij = conv(a_ij, f)  [6 convolutions]
        A_xx = self._conv3d(self.a_xx_fft, f)
        A_yy = self._conv3d(self.a_yy_fft, f)
        A_zz = self._conv3d(self.a_zz_fft, f)
        A_xy = self._conv3d(self.a_xy_fft, f)
        A_xz = self._conv3d(self.a_xz_fft, f)
        A_yz = self._conv3d(self.a_yz_fft, f)

        # Step 3: B_ij = conv(a_ij, ∂_j f)  [9 convolutions]
        # B_ij = Σ_j' conv(a_ij', ∂_{j'} f)... but actually B is defined per (i,j):
        # B_ij(v) = (a_ij * ∂_j f)(v), so for the flux:
        # J_i = Σ_j [A_ij * ∂_j f - B_ij * f]
        #     = Σ_j [A_ij * ∂_j f - (a_ij * ∂_j f) * f]
        # Wait, re-derive: B is the sum over j of convolutions.
        #
        # From the plan:
        # J_i = Σ_j [ A_ij · ∂f/∂v_j - f · B_ij ]
        # where B_ij = conv(a_ij, ∂_j f)
        #
        # So we need B_ij for all (i,j) pairs, but a_ij = a_ji, so:
        # B_xx = conv(a_xx, ∂_x f)
        # B_xy = conv(a_xy, ∂_y f)
        # B_xz = conv(a_xz, ∂_z f)
        # B_yx = conv(a_xy, ∂_x f)  (a_yx = a_xy)
        # B_yy = conv(a_yy, ∂_y f)
        # B_yz = conv(a_yz, ∂_z f)
        # B_zx = conv(a_xz, ∂_x f)  (a_zx = a_xz)
        # B_zy = conv(a_yz, ∂_y f)  (a_zy = a_yz)
        # B_zz = conv(a_zz, ∂_z f)
        # = 9 convolutions total

        B_xx = self._conv3d(self.a_xx_fft, df_dvx)
        B_xy = self._conv3d(self.a_xy_fft, df_dvy)
        B_xz = self._conv3d(self.a_xz_fft, df_dvz)
        B_yx = self._conv3d(self.a_xy_fft, df_dvx)
        B_yy = self._conv3d(self.a_yy_fft, df_dvy)
        B_yz = self._conv3d(self.a_yz_fft, df_dvz)
        B_zx = self._conv3d(self.a_xz_fft, df_dvx)
        B_zy = self._conv3d(self.a_yz_fft, df_dvy)
        B_zz = self._conv3d(self.a_zz_fft, df_dvz)

        # Step 4: Fluxes
        # J_x = Σ_j [A_xj · ∂_j f - f · B_xj]
        J_x = (A_xx * df_dvx + A_xy * df_dvy + A_xz * df_dvz
                - f * (B_xx + B_xy + B_xz))
        J_y = (A_xy * df_dvx + A_yy * df_dvy + A_yz * df_dvz
                - f * (B_yx + B_yy + B_yz))
        J_z = (A_xz * df_dvx + A_yz * df_dvy + A_zz * df_dvz
                - f * (B_zx + B_zy + B_zz))

        # Step 5: Divergence Q = ∂J_x/∂vx + ∂J_y/∂vy + ∂J_z/∂vz
        dJx_dvx, _, _ = self._gradient_v(J_x)
        _, dJy_dvy, _ = self._gradient_v(J_y)
        _, _, dJz_dvz = self._gradient_v(J_z)

        Q = dJx_dvx + dJy_dvy + dJz_dvz

        return Q

    def maxwellian_3d(self, rho, ux, uy, uz, T):
        """
        3D Maxwellian: f = ρ/(2πT)^{3/2} · exp(-((vx-ux)²+(vy-uy)²+(vz-uz)²)/(2T))
        """
        return rho / (2 * jnp.pi * T)**1.5 * jnp.exp(
            -((self.vx - ux)**2 + (self.vy - uy)**2 + (self.vz - uz)**2) / (2 * T)
        )


# =============================================================================
# 1D Scalar Operator (negative control - does NOT have projection tensor)
# =============================================================================

class LandauOperator1D_Scalar:
    """
    1D scalar Landau operator (WITHOUT projection tensor).
    Q = d/dv [ A·df/dv - B·f ]
    A = Φ * f,  B = Φ * df/dv
    Φ(|z|) = 1/max(|z|, 1/λ_D)

    This does NOT satisfy Q(M,M) = 0.
    """

    def __init__(self, Nv, V=6.0, lambda_D=10.0):
        self.Nv = Nv
        self.V = V
        self.lambda_D = lambda_D
        self.cutoff = 1.0 / lambda_D
        self.dv = 2 * V / (Nv - 1)
        self.v = jnp.linspace(-V, V, Nv)

        # Padded convolution size
        self.N_conv = 2 * Nv - 1

        # Kernel
        u = jnp.arange(-(Nv - 1), Nv) * self.dv
        Phi = 1.0 / jnp.maximum(jnp.abs(u), self.cutoff)
        self.Phi_fft = jnp.fft.fft(Phi)

    def _conv1d(self, f_row):
        """Compute 1D convolution via FFT."""
        Nv = self.Nv
        dv = self.dv
        f_padded = jnp.zeros(self.N_conv)
        f_padded = f_padded.at[:Nv].set(f_row)
        conv_full = jnp.real(jnp.fft.ifft(self.Phi_fft * jnp.fft.fft(f_padded)))
        start = Nv - 1
        return conv_full[start:start + Nv] * dv

    def _gradient_v(self, f):
        """Central differences for 1D."""
        dv = self.dv
        df = jnp.zeros_like(f)
        df = df.at[1:-1].set((f[2:] - f[:-2]) / (2 * dv))
        df = df.at[0].set((f[1] - f[0]) / dv)
        df = df.at[-1].set((f[-1] - f[-2]) / dv)
        return df

    def collision_operator(self, f):
        """Q = d/dv [A·f' - B·f], A = Φ*f, B = Φ*f'"""
        df_dv = self._gradient_v(f)
        A = self._conv1d(f)
        B = self._conv1d(df_dv)
        J = A * df_dv - B * f
        Q = self._gradient_v(J)
        return Q

    def maxwellian_1d(self, rho, u, T):
        """1D Maxwellian."""
        return rho / jnp.sqrt(2 * jnp.pi * T) * jnp.exp(-(self.v - u)**2 / (2 * T))


# =============================================================================
# Verification Tests
# =============================================================================

def test_2d_convergence():
    """Test 1: Q(M,M) for uniform 2D Maxwellian converges to 0 with grid refinement."""
    print("=" * 70)
    print("TEST 1: 2D Convergence — Q(M,M) for uniform Maxwellian")
    print("=" * 70)
    print(f"{'Nv':>6} {'max|Q|':>12} {'max|Q|/max|f|':>16} {'time (s)':>10}")
    print("-" * 50)

    Nv_list = [16, 32, 48, 64]
    results = []

    for Nv in Nv_list:
        t0 = time.time()
        op = LandauOperator2D(Nv=Nv, V=6.0, lambda_D=10.0)
        f = op.maxwellian_2d(rho=1.0, ux=0.0, uy=0.0, T=1.0)
        Q = op.collision_operator(f)
        elapsed = time.time() - t0

        max_Q = float(jnp.max(jnp.abs(Q)))
        max_f = float(jnp.max(jnp.abs(f)))
        rel = max_Q / max_f

        print(f"{Nv:6d} {max_Q:12.4e} {rel:16.4e} {elapsed:10.2f}")
        results.append((Nv, max_Q, rel))

    # Check convergence
    if len(results) >= 2:
        r0 = results[0][2]
        r_last = results[-1][2]
        if r_last < r0:
            print(f"\n  ✓ Convergence: relative error decreased from {r0:.4e} to {r_last:.4e}")
        else:
            print(f"\n  ✗ No convergence: {r0:.4e} -> {r_last:.4e}")

    print()
    return results


def test_2d_spatially_varying():
    """Test 2: Q(M,M) ≈ 0 for spatially-varying IC ρ(x,y) = 1 + 0.5 sin(2πx)sin(2πy)."""
    print("=" * 70)
    print("TEST 2: 2D Spatially-Varying IC — ρ = 1 + 0.5 sin(2πx)sin(2πy)")
    print("=" * 70)

    Nx, Ny = 8, 8
    Nv = 32
    X, Y = 0.5, 0.5

    # Spatial grid
    x = np.linspace(-X, X, Nx, endpoint=False)
    y = np.linspace(-Y, Y, Ny, endpoint=False)

    op = LandauOperator2D(Nv=Nv, V=6.0, lambda_D=10.0)

    max_Q_vals = []
    max_f_vals = []

    t0 = time.time()
    for ix in range(Nx):
        for iy in range(Ny):
            rho = 1.0 + 0.5 * np.sin(2 * np.pi * x[ix]) * np.sin(2 * np.pi * y[iy])
            f = op.maxwellian_2d(rho=rho, ux=0.0, uy=0.0, T=1.0)
            Q = op.collision_operator(f)

            max_Q_vals.append(float(jnp.max(jnp.abs(Q))))
            max_f_vals.append(float(jnp.max(jnp.abs(f))))

    elapsed = time.time() - t0

    max_Q_all = np.max(max_Q_vals)
    max_f_all = np.max(max_f_vals)
    mean_Q = np.mean(max_Q_vals)

    print(f"  Nx×Ny = {Nx}×{Ny}, Nv = {Nv}")
    print(f"  max|Q| over all spatial points: {max_Q_all:.4e}")
    print(f"  mean max|Q| per spatial point:  {mean_Q:.4e}")
    print(f"  max|f| over all spatial points: {max_f_all:.4e}")
    print(f"  max|Q|/max|f|:                  {max_Q_all / max_f_all:.4e}")
    print(f"  Time: {elapsed:.2f} s")
    print()

    return max_Q_all, max_f_all


def test_3d_convergence():
    """Test 3: Q(M,M) for uniform 3D Maxwellian converges to 0."""
    print("=" * 70)
    print("TEST 3: 3D Convergence — Q(M,M) for uniform Maxwellian")
    print("=" * 70)
    print(f"  Using V=4.0 (smaller domain for better resolution at feasible Nv)")
    print(f"{'Nv':>6} {'dv':>8} {'pts/σ':>8} {'max|Q|':>12} {'max|Q|/max|f|':>16} {'time (s)':>10}")
    print("-" * 66)

    Nv_list = [12, 16, 24, 32]
    V_3d = 4.0
    results = []

    for Nv in Nv_list:
        t0 = time.time()
        op = LandauOperator3D(Nv=Nv, V=V_3d, lambda_D=10.0)
        f = op.maxwellian_3d(rho=1.0, ux=0.0, uy=0.0, uz=0.0, T=1.0)
        Q = op.collision_operator(f)
        elapsed = time.time() - t0

        max_Q = float(jnp.max(jnp.abs(Q)))
        max_f = float(jnp.max(jnp.abs(f)))
        rel = max_Q / max_f
        dv = float(op.dv)
        pts_per_sigma = 1.0 / dv

        print(f"{Nv:6d} {dv:8.4f} {pts_per_sigma:8.2f} {max_Q:12.4e} {rel:16.4e} {elapsed:10.2f}")
        results.append((Nv, max_Q, rel))

    if len(results) >= 2:
        r0 = results[0][2]
        r_last = results[-1][2]
        if r_last < r0:
            print(f"\n  ✓ Convergence: relative error decreased from {r0:.4e} to {r_last:.4e}")
        else:
            print(f"\n  ✗ No convergence: {r0:.4e} -> {r_last:.4e}")

    print()
    return results


def test_3d_spatially_varying():
    """Test 4: Q(M,M) ≈ 0 for spatially-varying 3D IC."""
    print("=" * 70)
    print("TEST 4: 3D Spatially-Varying IC — ρ = 1 + 0.5 sin(2πx)sin(2πy)")
    print("=" * 70)

    Nx, Ny = 4, 4
    Nv = 16
    V_3d = 4.0

    x = np.linspace(-0.5, 0.5, Nx, endpoint=False)
    y = np.linspace(-0.5, 0.5, Ny, endpoint=False)

    op = LandauOperator3D(Nv=Nv, V=V_3d, lambda_D=10.0)

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
    mean_Q = np.mean(max_Q_vals)

    print(f"  Nx×Ny = {Nx}×{Ny}, Nv = {Nv}")
    print(f"  max|Q| over all spatial points: {max_Q_all:.4e}")
    print(f"  mean max|Q| per spatial point:  {mean_Q:.4e}")
    print(f"  max|f| over all spatial points: {max_f_all:.4e}")
    print(f"  max|Q|/max|f|:                  {max_Q_all / max_f_all:.4e}")
    print(f"  Time: {elapsed:.2f} s")
    print()

    return max_Q_all, max_f_all


def test_1d_scalar_comparison():
    """Test 5: 1D scalar operator does NOT converge to 0 (negative control)."""
    print("=" * 70)
    print("TEST 5: 1D Scalar Operator (Negative Control) — Q(M,M) ≠ 0")
    print("=" * 70)
    print(f"{'Nv':>6} {'max|Q|':>12} {'max|Q|/max|f|':>16}")
    print("-" * 40)

    Nv_list = [32, 64, 128, 256, 512]
    results = []

    for Nv in Nv_list:
        op = LandauOperator1D_Scalar(Nv=Nv, V=8.0, lambda_D=10.0)
        f = op.maxwellian_1d(rho=1.0, u=0.0, T=1.0)
        Q = op.collision_operator(f)

        max_Q = float(jnp.max(jnp.abs(Q)))
        max_f = float(jnp.max(jnp.abs(f)))
        rel = max_Q / max_f

        print(f"{Nv:6d} {max_Q:12.4e} {rel:16.4e}")
        results.append((Nv, max_Q, rel))

    # The relative error should NOT converge to 0
    r_last = results[-1][2]
    if r_last > 0.1:
        print(f"\n  ✓ Confirmed: 1D scalar operator gives |Q|/|f| ≈ {r_last:.2f} (does NOT → 0)")
    else:
        print(f"\n  ⚠ Unexpected: |Q|/|f| = {r_last:.4e} is small")

    print()
    return results


# =============================================================================
# Summary
# =============================================================================

def print_summary(res_2d, res_3d, res_1d):
    """Print comparison summary."""
    print("=" * 70)
    print("SUMMARY: True Landau Operator vs 1D Scalar Model")
    print("=" * 70)

    print("\n  2D Tensor Operator (finest grid):")
    Nv, _, rel = res_2d[-1]
    print(f"    Nv={Nv}: max|Q(M,M)|/max|f| = {rel:.4e}")

    print("\n  3D Tensor Operator (finest grid):")
    Nv, _, rel = res_3d[-1]
    print(f"    Nv={Nv}: max|Q(M,M)|/max|f| = {rel:.4e}")

    print("\n  1D Scalar Operator (finest grid, negative control):")
    Nv, _, rel = res_1d[-1]
    print(f"    Nv={Nv}: max|Q(M,M)|/max|f| = {rel:.4e}")

    print()
    print("  Conclusion:")
    r2d = res_2d[-1][2]
    r3d = res_3d[-1][2]
    r1d = res_1d[-1][2]
    if r2d < 0.1 and r3d < 0.1 and r1d > 0.1:
        print("    ✓ True tensor Landau operator satisfies Q(M,M) ≈ 0 in 2D and 3D")
        print("    ✓ 1D scalar model correctly does NOT satisfy Q(M,M) = 0")
        print("    ✓ This confirms the projection tensor Π is essential")
    else:
        print(f"    Results: 2D={r2d:.4e}, 3D={r3d:.4e}, 1D={r1d:.4e}")
        print("    Check convergence trends above for details")

    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("VERIFICATION: True 2D/3D Landau Collision Operator Q(M,M) = 0")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")
    print(f"Float precision: float64 (jax_enable_x64 = True)")
    print()

    total_start = time.time()

    # Test 1: 2D convergence
    res_2d = test_2d_convergence()

    # Test 2: 2D spatially-varying
    test_2d_spatially_varying()

    # Test 3: 3D convergence
    res_3d = test_3d_convergence()

    # Test 4: 3D spatially-varying
    test_3d_spatially_varying()

    # Test 5: 1D scalar comparison
    res_1d = test_1d_scalar_comparison()

    # Summary
    print_summary(res_2d, res_3d, res_1d)

    total_elapsed = time.time() - total_start
    print(f"\nTotal wall time: {total_elapsed:.1f} s")


if __name__ == "__main__":
    main()
