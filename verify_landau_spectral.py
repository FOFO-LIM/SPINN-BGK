# 2026-02-08: Landau 2D/3D with spectral (FFT) derivatives — verify Q(M,M)=0
"""
Landau collision operator using spectral derivatives instead of finite differences.

Key change: _gradient_v now uses FFT-based differentiation:
    ∂f/∂v = IFFT( i·k · FFT(f) )

This gives exponential convergence for smooth functions (Maxwellians),
compared to O(Δv²) from 2nd-order central differences.

Memory usage is identical — same Nv grids, just different derivative computation.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import time


# =============================================================================
# 2D Landau with Spectral Derivatives
# =============================================================================

class LandauOperator2D_Spectral:
    """
    2D Landau collision operator with tensor kernel and spectral derivatives.
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

        # Precompute wavenumbers for spectral derivatives
        self._precompute_wavenumbers()

        # Precompute kernel FFTs
        self._precompute_kernel_fft()

    def _precompute_wavenumbers(self):
        """Precompute wavenumbers for spectral differentiation."""
        Nv = self.Nv
        dv = self.dv

        # Wavenumbers: k = 2π · fftfreq(Nv, d=dv)
        k = 2 * jnp.pi * jnp.fft.fftfreq(Nv, d=dv)

        # Zero out Nyquist mode for even Nv (avoids aliasing in derivative)
        if Nv % 2 == 0:
            k = k.at[Nv // 2].set(0.0)

        # Store as 2D arrays for broadcasting
        self.kx = k[:, None]  # shape (Nv, 1) — for axis 0
        self.ky = k[None, :]  # shape (1, Nv) — for axis 1

    def _precompute_kernel_fft(self):
        """Precompute FFT of tensor kernel components a_xx, a_yy, a_xy."""
        Nv = self.Nv
        dv = self.dv

        # Relative velocity grid z = (zx, zy)
        z1d = jnp.arange(-(Nv - 1), Nv) * dv
        zx, zy = jnp.meshgrid(z1d, z1d, indexing='ij')
        z_mag = jnp.sqrt(zx**2 + zy**2)

        # Coulomb kernel with Debye cutoff
        Phi = 1.0 / jnp.maximum(z_mag, self.cutoff)

        # Projection tensor Π_ij = δ_ij - z_i z_j / |z|²
        z_mag_sq = z_mag**2
        z_mag_sq_safe = jnp.where(z_mag_sq > 0, z_mag_sq, 1.0)

        Pi_xx = jnp.where(z_mag > 0, 1.0 - zx**2 / z_mag_sq_safe, 0.5)
        Pi_yy = jnp.where(z_mag > 0, 1.0 - zy**2 / z_mag_sq_safe, 0.5)
        Pi_xy = jnp.where(z_mag > 0, -zx * zy / z_mag_sq_safe, 0.0)

        # Store FFTs
        self.a_xx_fft = jnp.fft.fft2(Phi * Pi_xx)
        self.a_yy_fft = jnp.fft.fft2(Phi * Pi_yy)
        self.a_xy_fft = jnp.fft.fft2(Phi * Pi_xy)

    def _conv2d(self, kernel_fft, g):
        """Compute 2D convolution via FFT."""
        Nv = self.Nv
        dv = self.dv

        g_padded = jnp.zeros((self.N_conv, self.N_conv))
        g_padded = g_padded.at[:Nv, :Nv].set(g)

        conv_full = jnp.real(jnp.fft.ifft2(kernel_fft * jnp.fft.fft2(g_padded)))

        start = Nv - 1
        conv = conv_full[start:start + Nv, start:start + Nv] * dv**2

        return conv

    def _gradient_v(self, f):
        """
        Compute (∂f/∂vx, ∂f/∂vy) using SPECTRAL derivatives.

        df/dvx = IFFT2( i·kx · FFT2(f) )  — differentiate along axis 0
        df/dvy = IFFT2( i·ky · FFT2(f) )  — differentiate along axis 1
        """
        f_hat = jnp.fft.fft2(f)

        df_dvx = jnp.real(jnp.fft.ifft2(1j * self.kx * f_hat))
        df_dvy = jnp.real(jnp.fft.ifft2(1j * self.ky * f_hat))

        return df_dvx, df_dvy

    def collision_operator(self, f):
        """Compute Q(f,f) for distribution f on 2D velocity grid."""
        # Step 1: Velocity gradients of f (spectral)
        df_dvx, df_dvy = self._gradient_v(f)

        # Step 2: A_ij = conv(a_ij, f)
        A_xx = self._conv2d(self.a_xx_fft, f)
        A_yy = self._conv2d(self.a_yy_fft, f)
        A_xy = self._conv2d(self.a_xy_fft, f)

        # Step 3: B_ij = conv(a_ij, ∂_j f)
        B_xx = self._conv2d(self.a_xx_fft, df_dvx)
        B_xy = self._conv2d(self.a_xy_fft, df_dvy)
        B_yx = self._conv2d(self.a_xy_fft, df_dvx)
        B_yy = self._conv2d(self.a_yy_fft, df_dvy)

        # Step 4: Fluxes
        J_x = A_xx * df_dvx + A_xy * df_dvy - f * (B_xx + B_xy)
        J_y = A_xy * df_dvx + A_yy * df_dvy - f * (B_yx + B_yy)

        # Step 5: Divergence (spectral)
        dJx_dvx, _ = self._gradient_v(J_x)
        _, dJy_dvy = self._gradient_v(J_y)

        Q = dJx_dvx + dJy_dvy

        return Q

    def maxwellian_2d(self, rho, ux, uy, T):
        """2D Maxwellian."""
        return rho / (2 * jnp.pi * T) * jnp.exp(
            -((self.vx - ux)**2 + (self.vy - uy)**2) / (2 * T)
        )


# =============================================================================
# 3D Landau with Spectral Derivatives
# =============================================================================

class LandauOperator3D_Spectral:
    """
    3D Landau collision operator with tensor kernel and spectral derivatives.
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

        # Precompute wavenumbers
        self._precompute_wavenumbers()

        # Precompute kernel FFTs
        self._precompute_kernel_fft()

    def _precompute_wavenumbers(self):
        """Precompute wavenumbers for spectral differentiation in 3D."""
        Nv = self.Nv
        dv = self.dv

        k = 2 * jnp.pi * jnp.fft.fftfreq(Nv, d=dv)

        if Nv % 2 == 0:
            k = k.at[Nv // 2].set(0.0)

        self.kx = k[:, None, None]
        self.ky = k[None, :, None]
        self.kz = k[None, None, :]

    def _precompute_kernel_fft(self):
        """Precompute FFT of tensor kernel components."""
        Nv = self.Nv
        dv = self.dv

        z1d = jnp.arange(-(Nv - 1), Nv) * dv
        zx, zy, zz = jnp.meshgrid(z1d, z1d, z1d, indexing='ij')
        z_mag = jnp.sqrt(zx**2 + zy**2 + zz**2)

        Phi = 1.0 / jnp.maximum(z_mag, self.cutoff)

        z_mag_sq = z_mag**2
        z_mag_sq_safe = jnp.where(z_mag_sq > 0, z_mag_sq, 1.0)

        Pi_xx = jnp.where(z_mag > 0, 1.0 - zx**2 / z_mag_sq_safe, 2.0/3.0)
        Pi_yy = jnp.where(z_mag > 0, 1.0 - zy**2 / z_mag_sq_safe, 2.0/3.0)
        Pi_zz = jnp.where(z_mag > 0, 1.0 - zz**2 / z_mag_sq_safe, 2.0/3.0)
        Pi_xy = jnp.where(z_mag > 0, -zx * zy / z_mag_sq_safe, 0.0)
        Pi_xz = jnp.where(z_mag > 0, -zx * zz / z_mag_sq_safe, 0.0)
        Pi_yz = jnp.where(z_mag > 0, -zy * zz / z_mag_sq_safe, 0.0)

        self.a_xx_fft = jnp.fft.fftn(Phi * Pi_xx)
        self.a_yy_fft = jnp.fft.fftn(Phi * Pi_yy)
        self.a_zz_fft = jnp.fft.fftn(Phi * Pi_zz)
        self.a_xy_fft = jnp.fft.fftn(Phi * Pi_xy)
        self.a_xz_fft = jnp.fft.fftn(Phi * Pi_xz)
        self.a_yz_fft = jnp.fft.fftn(Phi * Pi_yz)

    def _conv3d(self, kernel_fft, g):
        """Compute 3D convolution via FFT."""
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
        Compute (∂f/∂vx, ∂f/∂vy, ∂f/∂vz) using SPECTRAL derivatives.
        """
        f_hat = jnp.fft.fftn(f)

        df_dvx = jnp.real(jnp.fft.ifftn(1j * self.kx * f_hat))
        df_dvy = jnp.real(jnp.fft.ifftn(1j * self.ky * f_hat))
        df_dvz = jnp.real(jnp.fft.ifftn(1j * self.kz * f_hat))

        return df_dvx, df_dvy, df_dvz

    def collision_operator(self, f):
        """Compute Q(f,f) for distribution f on 3D velocity grid."""
        df_dvx, df_dvy, df_dvz = self._gradient_v(f)

        # A_ij = conv(a_ij, f)
        A_xx = self._conv3d(self.a_xx_fft, f)
        A_yy = self._conv3d(self.a_yy_fft, f)
        A_zz = self._conv3d(self.a_zz_fft, f)
        A_xy = self._conv3d(self.a_xy_fft, f)
        A_xz = self._conv3d(self.a_xz_fft, f)
        A_yz = self._conv3d(self.a_yz_fft, f)

        # B_ij = conv(a_ij, ∂_j f)
        B_xx = self._conv3d(self.a_xx_fft, df_dvx)
        B_xy = self._conv3d(self.a_xy_fft, df_dvy)
        B_xz = self._conv3d(self.a_xz_fft, df_dvz)
        B_yx = self._conv3d(self.a_xy_fft, df_dvx)
        B_yy = self._conv3d(self.a_yy_fft, df_dvy)
        B_yz = self._conv3d(self.a_yz_fft, df_dvz)
        B_zx = self._conv3d(self.a_xz_fft, df_dvx)
        B_zy = self._conv3d(self.a_yz_fft, df_dvy)
        B_zz = self._conv3d(self.a_zz_fft, df_dvz)

        # Fluxes
        J_x = (A_xx * df_dvx + A_xy * df_dvy + A_xz * df_dvz
                - f * (B_xx + B_xy + B_xz))
        J_y = (A_xy * df_dvx + A_yy * df_dvy + A_yz * df_dvz
                - f * (B_yx + B_yy + B_yz))
        J_z = (A_xz * df_dvx + A_yz * df_dvy + A_zz * df_dvz
                - f * (B_zx + B_zy + B_zz))

        # Divergence (spectral)
        dJx_dvx, _, _ = self._gradient_v(J_x)
        _, dJy_dvy, _ = self._gradient_v(J_y)
        _, _, dJz_dvz = self._gradient_v(J_z)

        Q = dJx_dvx + dJy_dvy + dJz_dvz

        return Q

    def maxwellian_3d(self, rho, ux, uy, uz, T):
        """3D Maxwellian."""
        return rho / (2 * jnp.pi * T)**1.5 * jnp.exp(
            -((self.vx - ux)**2 + (self.vy - uy)**2 + (self.vz - uz)**2) / (2 * T)
        )


# =============================================================================
# Comparison: import original FD operator
# =============================================================================

import sys
sys.path.insert(0, '/home/seongjaelim')
from verify_true_landau import LandauOperator2D, LandauOperator3D


# =============================================================================
# Tests
# =============================================================================

def test_2d_comparison():
    """Compare spectral vs finite-difference derivatives for 2D Q(M,M)."""
    print("=" * 70)
    print("TEST 1: 2D Convergence — Spectral vs Finite Difference")
    print("=" * 70)
    print(f"{'Nv':>6} {'Nv²':>8} {'FD max|Q|/|f|':>16} {'Spec max|Q|/|f|':>18} {'Improvement':>14}")
    print("-" * 66)

    Nv_list = [16, 32, 48, 64]

    for Nv in Nv_list:
        # Finite difference (original)
        t0 = time.time()
        op_fd = LandauOperator2D(Nv=Nv, V=6.0, lambda_D=10.0)
        f_fd = op_fd.maxwellian_2d(rho=1.0, ux=0.0, uy=0.0, T=1.0)
        Q_fd = op_fd.collision_operator(f_fd)
        t_fd = time.time() - t0

        max_Q_fd = float(jnp.max(jnp.abs(Q_fd)))
        max_f_fd = float(jnp.max(jnp.abs(f_fd)))
        rel_fd = max_Q_fd / max_f_fd

        # Spectral
        t0 = time.time()
        op_sp = LandauOperator2D_Spectral(Nv=Nv, V=6.0, lambda_D=10.0)
        f_sp = op_sp.maxwellian_2d(rho=1.0, ux=0.0, uy=0.0, T=1.0)
        Q_sp = op_sp.collision_operator(f_sp)
        t_sp = time.time() - t0

        max_Q_sp = float(jnp.max(jnp.abs(Q_sp)))
        max_f_sp = float(jnp.max(jnp.abs(f_sp)))
        rel_sp = max_Q_sp / max_f_sp

        ratio = rel_fd / rel_sp if rel_sp > 0 else float('inf')
        print(f"{Nv:6d} {Nv**2:8d} {rel_fd:16.4e} {rel_sp:18.4e} {ratio:14.1f}x")

    print()


def test_2d_spatially_varying():
    """Test Q(M,M) ≈ 0 for spatially-varying IC with spectral derivatives."""
    print("=" * 70)
    print("TEST 2: 2D Spatially-Varying IC (Spectral) — ρ = 1 + 0.5 sin(2πx)sin(2πy)")
    print("=" * 70)

    Nx, Ny = 8, 8
    Nv = 32
    x = np.linspace(-0.5, 0.5, Nx, endpoint=False)
    y = np.linspace(-0.5, 0.5, Ny, endpoint=False)

    op = LandauOperator2D_Spectral(Nv=Nv, V=6.0, lambda_D=10.0)

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

    print(f"  Nx×Ny = {Nx}×{Ny}, Nv = {Nv}")
    print(f"  max|Q| over all spatial points: {max_Q_all:.4e}")
    print(f"  max|f| over all spatial points: {max_f_all:.4e}")
    print(f"  max|Q|/max|f|:                  {max_Q_all / max_f_all:.4e}")
    print(f"  Time: {elapsed:.2f} s")
    print()


def test_3d_comparison():
    """Compare spectral vs FD for 3D Q(M,M)."""
    print("=" * 70)
    print("TEST 3: 3D Convergence — Spectral vs Finite Difference")
    print("=" * 70)
    print(f"{'Nv':>6} {'Nv³':>8} {'FD max|Q|/|f|':>16} {'Spec max|Q|/|f|':>18} {'Improvement':>14}")
    print("-" * 66)

    Nv_list = [12, 16, 24, 32]
    V_3d = 4.0

    for Nv in Nv_list:
        # Finite difference (original)
        t0 = time.time()
        op_fd = LandauOperator3D(Nv=Nv, V=V_3d, lambda_D=10.0)
        f_fd = op_fd.maxwellian_3d(rho=1.0, ux=0.0, uy=0.0, uz=0.0, T=1.0)
        Q_fd = op_fd.collision_operator(f_fd)
        t_fd = time.time() - t0

        max_Q_fd = float(jnp.max(jnp.abs(Q_fd)))
        max_f_fd = float(jnp.max(jnp.abs(f_fd)))
        rel_fd = max_Q_fd / max_f_fd

        # Spectral
        t0 = time.time()
        op_sp = LandauOperator3D_Spectral(Nv=Nv, V=V_3d, lambda_D=10.0)
        f_sp = op_sp.maxwellian_3d(rho=1.0, ux=0.0, uy=0.0, uz=0.0, T=1.0)
        Q_sp = op_sp.collision_operator(f_sp)
        t_sp = time.time() - t0

        max_Q_sp = float(jnp.max(jnp.abs(Q_sp)))
        max_f_sp = float(jnp.max(jnp.abs(f_sp)))
        rel_sp = max_Q_sp / max_f_sp

        ratio = rel_fd / rel_sp if rel_sp > 0 else float('inf')
        print(f"{Nv:6d} {Nv**3:8d} {rel_fd:16.4e} {rel_sp:18.4e} {ratio:14.1f}x")

    print()


def test_3d_spatially_varying():
    """Test 3D Q(M,M) ≈ 0 for spatially-varying IC with spectral derivatives."""
    print("=" * 70)
    print("TEST 4: 3D Spatially-Varying IC (Spectral) — ρ = 1 + 0.5 sin(2πx)sin(2πy)")
    print("=" * 70)

    Nx, Ny = 4, 4
    Nv = 16
    V_3d = 4.0

    x = np.linspace(-0.5, 0.5, Nx, endpoint=False)
    y = np.linspace(-0.5, 0.5, Ny, endpoint=False)

    op = LandauOperator3D_Spectral(Nv=Nv, V=V_3d, lambda_D=10.0)

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

    print(f"  Nx×Ny = {Nx}×{Ny}, Nv = {Nv}")
    print(f"  max|Q| over all spatial points: {max_Q_all:.4e}")
    print(f"  max|f| over all spatial points: {max_f_all:.4e}")
    print(f"  max|Q|/max|f|:                  {max_Q_all / max_f_all:.4e}")
    print(f"  Time: {elapsed:.2f} s")
    print()


def main():
    print("\n" + "=" * 70)
    print("VERIFICATION: Landau Operator with SPECTRAL Derivatives")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")
    print(f"Float precision: float64")
    print()

    total_start = time.time()

    test_2d_comparison()
    test_2d_spatially_varying()
    test_3d_comparison()
    test_3d_spatially_varying()

    total_elapsed = time.time() - total_start

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("  Spectral derivatives replace O(Δv²) finite differences with")
    print("  exponentially convergent FFT differentiation.")
    print("  Memory usage: identical (same Nv grids).")
    print("  Compute cost: ~1 extra FFT per derivative (negligible vs convolutions).")
    print(f"\nTotal wall time: {total_elapsed:.1f} s")


if __name__ == "__main__":
    main()
