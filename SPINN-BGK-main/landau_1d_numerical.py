"""
Numerical solver for 1D Boltzmann-Landau equation using operator splitting.

Equation:
    ∂f/∂t + v ∂f/∂x = Q_L(f,f)

where Q_L is the 1D Landau collision operator:
    Q_L(f,f) = ∂/∂v [ A[f] ∂f/∂v - B[f] f ]

with:
    A[f](v) = (Φ * f)(v) = ∫ Φ(|v-v'|) f(v') dv'
    B[f](v) = (Φ * ∂_v f)(v) = ∫ Φ(|v-v'|) ∂f(v')/∂v' dv'

Coulomb kernel with Debye cutoff:
    Φ(|u|) = 1 / max(|u|, 1/λ_D)

Numerical method:
    - Strang splitting: advection (dt/2) -> collision (dt) -> advection (dt/2)
    - Advection: spectral method (exact in Fourier space)
    - Collision: FFT-based convolution + RK4 time integration

Initial condition:
    ρ(0,x) = 1 + 0.5 sin(2πx)
    u(0,x) = 0
    T(0,x) = 1
    f(0,x,v) = Maxwellian(ρ, u, T)

Domain:
    x ∈ [-X, X] with periodic BC
    v ∈ [-V, V] (truncated)
    t ∈ [0, T_final]
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import time
import os
import json
from datetime import datetime

# Optional tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def get_memory_usage_gib():
    """Get current process memory usage in GiB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    except ImportError:
        return 0.0


class LandauSolver1D:
    """
    Solves the 1D Boltzmann-Landau equation using Strang splitting.
    """

    def __init__(self, N_x, N_v, N_t, X=0.5, V=6.0, T_final=0.1, lambda_D=10.0):
        """
        Initialize the solver.

        Parameters:
        -----------
        N_x : int - Number of spatial grid points
        N_v : int - Number of velocity grid points
        N_t : int - Number of time steps
        X : float - Spatial domain is [-X, X]
        V : float - Velocity domain is [-V, V]
        T_final : float - Final time
        lambda_D : float - Debye length for cutoff (Φ = 1/max(|u|, 1/λ_D))
        """
        self.N_x = N_x
        self.N_v = N_v
        self.N_t = N_t
        self.X = X
        self.V = V
        self.T_final = T_final
        self.lambda_D = lambda_D
        self.cutoff = 1.0 / lambda_D

        # Spatial grid (periodic, exclude right endpoint)
        self.dx = 2 * X / N_x
        self.x = np.linspace(-X, X - self.dx, N_x)

        # Velocity grid
        self.dv = 2 * V / (N_v - 1)
        self.v = np.linspace(-V, V, N_v)

        # Time step
        self.dt = T_final / N_t

        # Wavenumbers for spatial derivatives (spectral method)
        self.kx = 2 * np.pi * fftfreq(N_x, self.dx)

        # Precompute Coulomb kernel FFT for convolutions
        self._precompute_kernel_fft()

        # Print CFL info
        cfl_advection = np.max(np.abs(self.v)) * self.dt / self.dx
        print(f"CFL number (advection): {cfl_advection:.4f}")

    def _precompute_kernel_fft(self):
        """
        Precompute FFT of the Coulomb kernel with cutoff for convolution.

        Uses zero-padding for linear (non-circular) convolution.
        """
        # For linear convolution, pad to size 2*N_v - 1
        self.N_conv = 2 * self.N_v - 1

        # Velocity difference grid for kernel (centered at 0)
        # Range: [-(N_v-1)*dv, (N_v-1)*dv]
        u = np.arange(-(self.N_v - 1), self.N_v) * self.dv

        # Coulomb kernel with cutoff: Φ(|u|) = 1/max(|u|, 1/λ_D)
        self.Phi = 1.0 / np.maximum(np.abs(u), self.cutoff)

        # FFT of kernel (for use in convolution)
        self.Phi_fft = fft(self.Phi)

    def compute_collision_coefficients(self, f):
        """
        Compute A[f] and B[f] using FFT-based convolution.

        A[f](v) = ∫ Φ(|v-v'|) f(v') dv'
        B[f](v) = ∫ Φ(|v-v'|) ∂f(v')/∂v' dv'

        Parameters:
        -----------
        f : ndarray of shape (N_x, N_v)

        Returns:
        --------
        A, B : ndarrays of shape (N_x, N_v)
        """
        # Compute ∂f/∂v using central differences (second-order)
        df_dv = np.zeros_like(f)
        df_dv[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * self.dv)
        # One-sided differences at boundaries
        df_dv[:, 0] = (f[:, 1] - f[:, 0]) / self.dv
        df_dv[:, -1] = (f[:, -1] - f[:, -2]) / self.dv

        A = np.zeros_like(f)
        B = np.zeros_like(f)

        for i in range(self.N_x):
            # Zero-pad for linear convolution
            f_padded = np.zeros(self.N_conv)
            f_padded[:self.N_v] = f[i, :]

            df_padded = np.zeros(self.N_conv)
            df_padded[:self.N_v] = df_dv[i, :]

            # Convolution via FFT: (Φ * f)[n] = IFFT(FFT(Φ) · FFT(f))
            A_full = np.real(ifft(self.Phi_fft * fft(f_padded)))
            B_full = np.real(ifft(self.Phi_fft * fft(df_padded)))

            # Extract valid part of convolution
            # For convolution of size N_v with kernel size 2*N_v-1,
            # valid output starts at index N_v-1
            start = self.N_v - 1
            A[i, :] = A_full[start:start + self.N_v] * self.dv
            B[i, :] = B_full[start:start + self.N_v] * self.dv

        return A, B

    def collision_operator(self, f):
        """
        Compute the Landau collision operator Q_L(f,f).

        Q_L = ∂/∂v [ A[f] ∂f/∂v - B[f] f ]

        Parameters:
        -----------
        f : ndarray of shape (N_x, N_v)

        Returns:
        --------
        Q : ndarray of shape (N_x, N_v)
        """
        A, B = self.compute_collision_coefficients(f)

        # Compute ∂f/∂v (central differences)
        df_dv = np.zeros_like(f)
        df_dv[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * self.dv)
        df_dv[:, 0] = (f[:, 1] - f[:, 0]) / self.dv
        df_dv[:, -1] = (f[:, -1] - f[:, -2]) / self.dv

        # Flux: J = A ∂f/∂v - B f
        J = A * df_dv - B * f

        # Q = ∂J/∂v (central differences)
        Q = np.zeros_like(f)
        Q[:, 1:-1] = (J[:, 2:] - J[:, :-2]) / (2 * self.dv)
        Q[:, 0] = (J[:, 1] - J[:, 0]) / self.dv
        Q[:, -1] = (J[:, -1] - J[:, -2]) / self.dv

        return Q

    def advection_step(self, f, dt):
        """
        Solve advection equation ∂f/∂t + v ∂f/∂x = 0 using spectral method.

        Exact solution in Fourier space:
            f̂(k, v, t+dt) = f̂(k, v, t) · exp(-i k v dt)

        Parameters:
        -----------
        f : ndarray of shape (N_x, N_v)
        dt : float

        Returns:
        --------
        f_new : ndarray of shape (N_x, N_v)
        """
        # FFT in x direction
        f_hat = fft(f, axis=0)

        # Phase shift: exp(-i k v dt)
        phase = np.exp(-1j * np.outer(self.kx, self.v) * dt)

        # Apply phase shift and inverse FFT
        f_new = np.real(ifft(f_hat * phase, axis=0))

        return f_new

    def collision_step(self, f, dt):
        """
        Solve collision equation ∂f/∂t = Q_L(f,f) using RK4.

        Parameters:
        -----------
        f : ndarray of shape (N_x, N_v)
        dt : float

        Returns:
        --------
        f_new : ndarray of shape (N_x, N_v)
        """
        k1 = self.collision_operator(f)
        k2 = self.collision_operator(f + 0.5 * dt * k1)
        k3 = self.collision_operator(f + 0.5 * dt * k2)
        k4 = self.collision_operator(f + dt * k3)

        f_new = f + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Ensure non-negativity (distribution function must be >= 0)
        f_new = np.maximum(f_new, 0.0)

        return f_new

    def strang_splitting_step(self, f):
        """
        Perform one Strang splitting step:
            1. Advection for dt/2
            2. Collision for dt
            3. Advection for dt/2
        """
        f = self.advection_step(f, self.dt / 2)
        f = self.collision_step(f, self.dt)
        f = self.advection_step(f, self.dt / 2)
        return f

    def maxwellian(self, rho, u, T):
        """
        Compute Maxwellian distribution.

        M(x,v) = ρ(x) / √(2πT(x)) · exp(-(v - u(x))² / (2T(x)))

        Parameters:
        -----------
        rho : ndarray of shape (N_x,) - density
        u : ndarray of shape (N_x,) - bulk velocity
        T : ndarray of shape (N_x,) - temperature

        Returns:
        --------
        f : ndarray of shape (N_x, N_v)
        """
        rho = rho[:, np.newaxis]
        u = u[:, np.newaxis]
        T = T[:, np.newaxis]
        v = self.v[np.newaxis, :]

        return rho / np.sqrt(2 * np.pi * T) * np.exp(-(v - u)**2 / (2 * T))

    def compute_moments(self, f):
        """
        Compute macroscopic moments from distribution function.

        ρ = ∫ f dv           (density)
        ρu = ∫ v f dv        (momentum)
        ρE = ∫ v² f dv / 2   (energy)
        T = (ρE - ρu²/2) / ρ (temperature, for 1D: no factor of 1/d)

        Returns:
        --------
        rho, u, T : ndarrays of shape (N_x,)
        """
        # Use trapezoidal rule for integration
        rho = np.trapezoid(f, self.v, axis=1)
        momentum = np.trapezoid(f * self.v, self.v, axis=1)
        energy = np.trapezoid(f * self.v**2, self.v, axis=1)

        u = momentum / (rho + 1e-16)
        T = (energy / (rho + 1e-16)) - u**2

        return rho, u, T

    def compute_conservation(self, f):
        """
        Compute conserved quantities (integrated over x and v).

        Returns: total_mass, total_momentum, total_energy
        """
        rho, u, T = self.compute_moments(f)

        # Integrate over x
        total_mass = np.trapezoid(rho, self.x)
        total_momentum = np.trapezoid(rho * u, self.x)
        total_energy = np.trapezoid(rho * (T + u**2), self.x) / 2

        return total_mass, total_momentum, total_energy

    def initial_condition(self):
        """
        Set initial condition:
            ρ(0,x) = 1 + 0.5 sin(2πx)
            u(0,x) = 0
            T(0,x) = 1
        """
        rho0 = 1 + 0.5 * np.sin(2 * np.pi * self.x)
        u0 = np.zeros(self.N_x)
        T0 = np.ones(self.N_x)

        return self.maxwellian(rho0, u0, T0)

    def solve(self, save_every=None, verbose=True):
        """
        Solve the Boltzmann-Landau equation.

        Parameters:
        -----------
        save_every : int or None - save solution every N steps
        verbose : bool - print progress

        Returns:
        --------
        results : dict containing solution and diagnostics
        """
        if verbose:
            print(f"\n{'='*60}")
            print("1D Boltzmann-Landau Equation Solver")
            print('='*60)
            print(f"Grid: N_x={self.N_x}, N_v={self.N_v}, N_t={self.N_t}")
            print(f"Domain: x ∈ [-{self.X}, {self.X}], v ∈ [-{self.V}, {self.V}]")
            print(f"Time: t ∈ [0, {self.T_final}], dt = {self.dt:.6e}")
            print(f"Debye length: λ_D = {self.lambda_D}, cutoff = {self.cutoff:.4f}")
            print('='*60)

        # Initialize
        f = self.initial_condition()

        # Initial conservation quantities
        mass0, mom0, energy0 = self.compute_conservation(f)

        # Storage
        times = [0.0]
        rho_history = []
        u_history = []
        T_history = []
        conservation_history = [(mass0, mom0, energy0)]

        rho, u, T = self.compute_moments(f)
        rho_history.append(rho.copy())
        u_history.append(u.copy())
        T_history.append(T.copy())

        if save_every is not None:
            f_history = [f.copy()]

        # Time stepping
        start_time = time.time()

        if verbose and HAS_TQDM:
            iterator = tqdm(range(self.N_t), desc="Time stepping")
        else:
            iterator = range(self.N_t)
            if verbose:
                print("Time stepping...")

        for n in iterator:
            f = self.strang_splitting_step(f)

            # Save at specified intervals
            if save_every is not None and (n + 1) % save_every == 0:
                times.append((n + 1) * self.dt)
                rho, u, T = self.compute_moments(f)
                rho_history.append(rho.copy())
                u_history.append(u.copy())
                T_history.append(T.copy())
                conservation_history.append(self.compute_conservation(f))

                if save_every is not None:
                    f_history.append(f.copy())

        elapsed_time = time.time() - start_time

        # Final state
        rho_final, u_final, T_final = self.compute_moments(f)
        mass_final, mom_final, energy_final = self.compute_conservation(f)

        if verbose:
            print(f"\nCompleted in {elapsed_time:.2f} seconds")
            print(f"\nFinal moments:")
            print(f"  ρ: min={rho_final.min():.6f}, max={rho_final.max():.6f}, mean={rho_final.mean():.6f}")
            print(f"  u: min={u_final.min():.6f}, max={u_final.max():.6f}, mean={u_final.mean():.6f}")
            print(f"  T: min={T_final.min():.6f}, max={T_final.max():.6f}, mean={T_final.mean():.6f}")
            print(f"\nConservation errors:")
            print(f"  Mass:     {abs(mass_final - mass0) / abs(mass0):.2e} (relative)")
            if abs(mom0) > 1e-10:
                print(f"  Momentum: {abs(mom_final - mom0) / abs(mom0):.2e} (relative)")
            else:
                print(f"  Momentum: {abs(mom_final - mom0):.2e} (absolute, initial ≈ 0)")
            print(f"  Energy:   {abs(energy_final - energy0) / abs(energy0):.2e} (relative)")

        results = {
            'f': f,
            'x': self.x,
            'v': self.v,
            'rho': rho_final,
            'u': u_final,
            'T': T_final,
            'times': np.array(times),
            'rho_history': np.array(rho_history),
            'u_history': np.array(u_history),
            'T_history': np.array(T_history),
            'conservation_history': np.array(conservation_history),
            'elapsed_time': elapsed_time,
            'params': {
                'N_x': self.N_x,
                'N_v': self.N_v,
                'N_t': self.N_t,
                'X': self.X,
                'V': self.V,
                'T_final': self.T_final,
                'lambda_D': self.lambda_D,
            }
        }

        if save_every is not None:
            results['f_history'] = np.array(f_history)

        return results


def main(N_i: int = 64, N_t: int = None, X: float = 0.5, V: float = 6.0, T_final: float = 0.1,
         lambda_D: float = 10.0, save_every: int = 10, plot: bool = True):
    """
    Run the 1D Boltzmann-Landau solver.

    Parameters:
    -----------
    N_i : int - Grid size (N_x = N_v = N_i)
    N_t : int - Number of time steps (default: N_i)
    X : float - Spatial domain half-width [-X, X]
    V : float - Velocity domain half-width [-V, V]
    T_final : float - Final simulation time
    lambda_D : float - Debye length for Coulomb cutoff
    save_every : int - Save solution every N time steps
    plot : bool - Generate plots
    """
    if N_t is None:
        N_t = N_i

    # Record start time
    start_datetime = datetime.now()
    start_time_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")

    print(f"Starting simulation with N_x=N_v={N_i}, N_t={N_t}")
    print(f"Start time: {start_time_str}")

    # Create solver
    solver = LandauSolver1D(
        N_x=N_i,
        N_v=N_i,
        N_t=N_t,
        X=X,
        V=V,
        T_final=T_final,
        lambda_D=lambda_D
    )

    # Solve
    results = solver.solve(save_every=save_every)

    # Record end time
    end_datetime = datetime.now()
    end_time_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # Get memory usage
    memory_usage_gib = get_memory_usage_gib()

    # Create output directories
    os.makedirs("data/landau_1d", exist_ok=True)
    os.makedirs("figures/landau_1d", exist_ok=True)

    # Generate filename with timestamp
    timestamp = start_datetime.strftime("%Y%m%d_%H%M%S")
    base_name = f"landau_numpy_N{N_i}_T{T_final}_lambdaD{lambda_D}_{timestamp}"

    # ========== Save metadata to JSON ==========
    config = {
        # Run info
        "run_date": start_datetime.strftime("%Y-%m-%d"),
        "run_start_time": start_time_str,
        "run_end_time": end_time_str,
        "elapsed_time_sec": results['elapsed_time'],

        # Equation info
        "equation": "1D Boltzmann-Landau",
        "equation_form": "∂f/∂t + v ∂f/∂x = Q_L(f,f), Q_L = ∂/∂v[A ∂f/∂v - B f]",
        "collision_kernel": "Coulomb with Debye cutoff",
        "kernel_formula": "Φ(|u|) = 1 / max(|u|, 1/λ_D)",

        # Hardware info
        "num_gpus": 0,  # CPU version
        "device_type": "CPU",
        "memory_usage_gib": memory_usage_gib,

        # Simulation parameters
        "N_i": N_i,
        "N_x": N_i,
        "N_v": N_i,
        "N_t": N_t,
        "lambda_D": lambda_D,
        "cutoff": 1.0 / lambda_D,

        # Domain
        "X": X,
        "V": V,
        "T_final": T_final,
        "dx": float(solver.dx),
        "dv": float(solver.dv),
        "dt": float(solver.dt),

        # Initial condition
        "initial_condition": {
            "rho": "1 + 0.5 * sin(2πx)",
            "u": "0",
            "T": "1"
        },

        # Numerical method
        "method": "Strang operator splitting",
        "advection_solver": "Spectral (FFT)",
        "collision_solver": "RK4 with FFT-based convolution",

        # Final state summary
        "final_rho_min": float(results['rho'].min()),
        "final_rho_max": float(results['rho'].max()),
        "final_rho_mean": float(results['rho'].mean()),
        "final_T_min": float(results['T'].min()),
        "final_T_max": float(results['T'].max()),
        "final_T_mean": float(results['T'].mean()),

        # Conservation
        "mass_error_relative": float(abs(results['conservation_history'][-1, 0] - results['conservation_history'][0, 0]) / abs(results['conservation_history'][0, 0])),
        "energy_error_relative": float(abs(results['conservation_history'][-1, 2] - results['conservation_history'][0, 2]) / abs(results['conservation_history'][0, 2])),

        # File references
        "data_file": f"{base_name}_f.npy",
        "grid_file": f"{base_name}_grid.npz",
    }

    config_file = f"data/landau_1d/{base_name}_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_file}")

    # ========== Save distribution function f(x,v,t) to NPY ==========
    if 'f_history' in results:
        f_data = results['f_history']
    else:
        f_data = results['f'][np.newaxis, :, :]

    f_file = f"data/landau_1d/{base_name}_f.npy"
    np.save(f_file, f_data)
    print(f"Distribution f(x,v,t) saved to {f_file}")
    print(f"  Shape: {f_data.shape} (N_times, N_x, N_v)")

    # ========== Save grid info to NPZ ==========
    grid_file = f"data/landau_1d/{base_name}_grid.npz"
    np.savez(grid_file,
             x=results['x'],
             v=results['v'],
             times=results['times'],
             rho_history=results['rho_history'],
             u_history=results['u_history'],
             T_history=results['T_history'],
             conservation_history=results['conservation_history'])
    print(f"Grid and moments saved to {grid_file}")

    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))

        # Initial condition
        rho0 = 1 + 0.5 * np.sin(2 * np.pi * solver.x)

        # (0,0) Density evolution
        ax = axes[0, 0]
        for i, t in enumerate(results['times']):
            alpha = 0.3 + 0.7 * i / max(1, len(results['times']) - 1)
            label = f't={t:.3f}' if i == 0 or i == len(results['times'])-1 else None
            ax.plot(solver.x, results['rho_history'][i], alpha=alpha, label=label)
        ax.set_xlabel('x')
        ax.set_ylabel('ρ')
        ax.set_title('Density Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (0,1) Temperature evolution
        ax = axes[0, 1]
        for i, t in enumerate(results['times']):
            alpha = 0.3 + 0.7 * i / max(1, len(results['times']) - 1)
            label = f't={t:.3f}' if i == 0 or i == len(results['times'])-1 else None
            ax.plot(solver.x, results['T_history'][i], alpha=alpha, label=label)
        ax.set_xlabel('x')
        ax.set_ylabel('T')
        ax.set_title('Temperature Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (0,2) Bulk velocity evolution
        ax = axes[0, 2]
        for i, t in enumerate(results['times']):
            alpha = 0.3 + 0.7 * i / max(1, len(results['times']) - 1)
            ax.plot(solver.x, results['u_history'][i], alpha=alpha)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title('Bulk Velocity Evolution')
        ax.grid(True, alpha=0.3)

        # (1,0) Final distribution at x=0
        ax = axes[1, 0]
        mid_x = N_i // 2
        f0 = solver.initial_condition()
        ax.plot(solver.v, f0[mid_x, :], 'b--', label='Initial', alpha=0.7)
        ax.plot(solver.v, results['f'][mid_x, :], 'r-', label='Final')
        ax.set_xlabel('v')
        ax.set_ylabel('f')
        ax.set_title(f'Distribution at x=0')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (1,1) Distribution function heatmap (final)
        ax = axes[1, 1]
        im = ax.pcolormesh(solver.x, solver.v, results['f'].T, shading='auto', cmap='hot')
        plt.colorbar(im, ax=ax, label='f(x,v)')
        ax.set_xlabel('x')
        ax.set_ylabel('v')
        ax.set_title('Final Distribution f(x,v)')

        # (1,2) Conservation errors
        ax = axes[1, 2]
        cons = np.array(results['conservation_history'])
        mass_err = np.abs(cons[:, 0] - cons[0, 0]) / np.abs(cons[0, 0])
        energy_err = np.abs(cons[:, 2] - cons[0, 2]) / np.abs(cons[0, 2])
        ax.semilogy(results['times'], mass_err + 1e-16, 'b-', label='Mass')
        ax.semilogy(results['times'], energy_err + 1e-16, 'r-', label='Energy')
        ax.set_xlabel('t')
        ax.set_ylabel('Relative Error')
        ax.set_title('Conservation Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_file = f"figures/landau_1d/{base_name}.png"
        plt.savefig(fig_file, dpi=150)
        print(f"Figure saved to {fig_file}")
        plt.show()

    return results


if __name__ == "__main__":
    try:
        import fire
        fire.Fire(main)
    except ImportError:
        import argparse
        parser = argparse.ArgumentParser(description='1D Boltzmann-Landau solver')
        parser.add_argument('--N_i', type=int, default=64, help='Grid size (N_x = N_v)')
        parser.add_argument('--N_t', type=int, default=None, help='Number of time steps (default: N_i)')
        parser.add_argument('--X', type=float, default=0.5, help='Spatial domain half-width')
        parser.add_argument('--V', type=float, default=6.0, help='Velocity domain half-width')
        parser.add_argument('--T_final', type=float, default=0.1, help='Final time')
        parser.add_argument('--lambda_D', type=float, default=10.0, help='Debye length')
        parser.add_argument('--save_every', type=int, default=10, help='Save interval')
        parser.add_argument('--no_plot', action='store_true', help='Disable plotting')
        args = parser.parse_args()
        main(N_i=args.N_i, N_t=args.N_t, X=args.X, V=args.V, T_final=args.T_final,
             lambda_D=args.lambda_D, save_every=args.save_every, plot=not args.no_plot)
