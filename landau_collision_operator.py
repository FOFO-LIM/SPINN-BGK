# 2026-02-09: Extracted from verify_landau_spectral.py — reusable 2D/3D Landau collision operators with spectral derivatives
"""
Landau collision operator with spectral (FFT) derivatives for 2D and 3D.

Uses the full tensor formulation:
    Q(f,f) = ∇_v · [ A[f] · ∇_v f  -  B[f] · f ]

where:
    a_ij(z) = Φ(|z|) · Π_ij(z)
    Π_ij(z) = δ_ij - z_i z_j / |z|²   (projection tensor)
    Φ(|z|)  = 1 / max(|z|, 1/λ_D)      (Coulomb kernel with Debye cutoff)

    A_ij[f](v) = ∫ a_ij(v-v') f(v') dv'
    B_i[f](v)  = Σ_j ∫ a_ij(v-v') ∂f(v')/∂v'_j dv'

Velocity derivatives use spectral differentiation:
    ∂f/∂v = IFFT( i·k · FFT(f) )

This gives exponential convergence for smooth functions (e.g. Maxwellians),
compared to O(Δv²) from 2nd-order central finite differences.

Verified: Q(M,M) ≈ 0 for Maxwellian inputs.
    - 2D: max|Q|/max|f| ~ 10^-10  (Nv=32, V=6.0)
    - 3D: max|Q|/max|f| ~ 5×10^-8 (Nv=24, V=6.0)

Source: verify_landau_spectral.py (2026-02-08)
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


# =============================================================================
# 2D Landau Collision Operator (Spectral)
# =============================================================================

class LandauOperator2D_Spectral:
    """
    2D Landau collision operator with tensor kernel and spectral derivatives.

    Parameters
    ----------
    Nv : int
        Number of velocity grid points per dimension.
    V : float
        Velocity domain is [-V, V] in each direction.
    lambda_D : float
        Debye length for Coulomb cutoff: Φ(|z|) = 1/max(|z|, 1/λ_D).
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

        k = 2 * jnp.pi * jnp.fft.fftfreq(Nv, d=dv)

        # Zero out Nyquist mode for even Nv (avoids aliasing in derivative)
        if Nv % 2 == 0:
            k = k.at[Nv // 2].set(0.0)

        self.kx = k[:, None]  # shape (Nv, 1)
        self.ky = k[None, :]  # shape (1, Nv)

    def _precompute_kernel_fft(self):
        """Precompute FFT of tensor kernel components a_xx, a_yy, a_xy."""
        Nv = self.Nv
        dv = self.dv

        z1d = jnp.arange(-(Nv - 1), Nv) * dv
        zx, zy = jnp.meshgrid(z1d, z1d, indexing='ij')
        z_mag = jnp.sqrt(zx**2 + zy**2)

        Phi = 1.0 / jnp.maximum(z_mag, self.cutoff)

        z_mag_sq = z_mag**2
        z_mag_sq_safe = jnp.where(z_mag_sq > 0, z_mag_sq, 1.0)

        Pi_xx = jnp.where(z_mag > 0, 1.0 - zx**2 / z_mag_sq_safe, 0.5)
        Pi_yy = jnp.where(z_mag > 0, 1.0 - zy**2 / z_mag_sq_safe, 0.5)
        Pi_xy = jnp.where(z_mag > 0, -zx * zy / z_mag_sq_safe, 0.0)

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
        Compute (∂f/∂vx, ∂f/∂vy) using spectral derivatives.

        df/dvx = IFFT2( i·kx · FFT2(f) )
        df/dvy = IFFT2( i·ky · FFT2(f) )
        """
        f_hat = jnp.fft.fft2(f)

        df_dvx = jnp.real(jnp.fft.ifft2(1j * self.kx * f_hat))
        df_dvy = jnp.real(jnp.fft.ifft2(1j * self.ky * f_hat))

        return df_dvx, df_dvy

    def collision_operator(self, f):
        """
        Compute Q(f,f) for distribution f on 2D velocity grid.

        Parameters
        ----------
        f : jnp.ndarray of shape (Nv, Nv)

        Returns
        -------
        Q : jnp.ndarray of shape (Nv, Nv)
        """
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

        # Step 4: Fluxes J_i = Σ_j [ A_ij ∂f/∂v_j - f · B_ij ]
        J_x = A_xx * df_dvx + A_xy * df_dvy - f * (B_xx + B_xy)
        J_y = A_xy * df_dvx + A_yy * df_dvy - f * (B_yx + B_yy)

        # Step 5: Divergence (spectral)
        dJx_dvx, _ = self._gradient_v(J_x)
        _, dJy_dvy = self._gradient_v(J_y)

        Q = dJx_dvx + dJy_dvy

        return Q

    def maxwellian_2d(self, rho, ux, uy, T):
        """
        Compute 2D Maxwellian distribution.

        M = ρ / (2πT) · exp( -(|v-u|²) / (2T) )
        """
        return rho / (2 * jnp.pi * T) * jnp.exp(
            -((self.vx - ux)**2 + (self.vy - uy)**2) / (2 * T)
        )


# =============================================================================
# 3D Landau Collision Operator (Spectral)
# =============================================================================

class LandauOperator3D_Spectral:
    """
    3D Landau collision operator with tensor kernel and spectral derivatives.

    Parameters
    ----------
    Nv : int
        Number of velocity grid points per dimension.
    V : float
        Velocity domain is [-V, V] in each direction.
    lambda_D : float
        Debye length for Coulomb cutoff: Φ(|z|) = 1/max(|z|, 1/λ_D).
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
        Compute (∂f/∂vx, ∂f/∂vy, ∂f/∂vz) using spectral derivatives.
        """
        f_hat = jnp.fft.fftn(f)

        df_dvx = jnp.real(jnp.fft.ifftn(1j * self.kx * f_hat))
        df_dvy = jnp.real(jnp.fft.ifftn(1j * self.ky * f_hat))
        df_dvz = jnp.real(jnp.fft.ifftn(1j * self.kz * f_hat))

        return df_dvx, df_dvy, df_dvz

    def collision_operator(self, f):
        """
        Compute Q(f,f) for distribution f on 3D velocity grid.

        Parameters
        ----------
        f : jnp.ndarray of shape (Nv, Nv, Nv)

        Returns
        -------
        Q : jnp.ndarray of shape (Nv, Nv, Nv)
        """
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
        """
        Compute 3D Maxwellian distribution.

        M = ρ / (2πT)^(3/2) · exp( -(|v-u|²) / (2T) )
        """
        return rho / (2 * jnp.pi * T)**1.5 * jnp.exp(
            -((self.vx - ux)**2 + (self.vy - uy)**2 + (self.vz - uz)**2) / (2 * T)
        )
