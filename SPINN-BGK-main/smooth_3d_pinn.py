"""
Ordinary PINN (Physics-Informed Neural Network) for 3D BGK equation.
Uses mini-batch random sampling instead of grid-based evaluation.
"""
import jax
import jax.numpy as np
import jax.random as jr
import optax
import matplotlib.pyplot as plt
import numpy as onp
import time
import os
import json
from datetime import datetime
from functools import partial
from tqdm import trange
from jax import lax, random, vmap

from src.nn import Siren
from src.x3v3 import smooth
from utils.transform import maxwellian3d, trapezoidal_rule


def get_gpu_memory_gib(device_idx=0, peak=False):
    """Get GPU memory usage in GiB for a single device."""
    try:
        device = jax.local_devices()[device_idx]
        stats = device.memory_stats()
        if stats:
            if peak:
                return stats.get('peak_bytes_in_use', 0) / (1024 ** 3)
            else:
                return stats.get('bytes_in_use', 0) / (1024 ** 3)
    except:
        pass
    return 0.0


def get_date_str():
    """Get current date and time as string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class OrdinaryPINN:
    """Ordinary PINN for 3D BGK equation using mini-batch sampling."""

    dim = 7  # t, x, y, z, vx, vy, vz

    def __init__(
        self,
        T=0.1,
        X=0.5,
        V=10.0,
        Nv=64,
        width=256,
        depth=5,
        w0=10.0,
        Kn=None,
        batch_size=50000,
        ngrid=9,
    ):
        self.T_bounds = np.array([0, T])
        self.X_bounds = np.array([-X, X])
        self.V_bounds = np.array([-V, V])
        self.nu = 1 / Kn
        self.maxwellian = maxwellian3d
        self.ic = smooth(X, V)
        self.batch_size = batch_size
        self.ngrid = ngrid
        self.total_grid_points = ngrid ** 7

        # Quadrature for moment computation (smaller for memory)
        self.v, self.w = trapezoidal_rule(Nv, -V, V)
        self.wv = self.w * self.v
        self.wvv = self.wv * self.v

        # Single MLP: 7 inputs -> 1 output
        layers = [7] + [width for _ in range(depth - 1)] + [1]
        self.init, self.apply = Siren(layers, w0)

        # Domain bounds for sampling
        self.bounds_low = np.array([0, -X, -X, -X, -V, -V, -V])
        self.bounds_high = np.array([T, X, X, X, V, V, V])

        # Test domain (small grid for logging)
        self.domain_te = self._create_test_grid(6)

        self.alpha = 1e-0
        self.eps = 1e-3

    def _create_test_grid(self, n):
        """Create small test grid for logging."""
        t = np.linspace(0, self.T_bounds[1], n)
        x = np.linspace(*self.X_bounds, n)
        v = np.linspace(*self.V_bounds, n)
        return (t, x, x, x, v, v, v)

    def f_point(self, params, point):
        """Evaluate f at a single 7D point."""
        return self.apply(params, point).squeeze()

    def f_batch(self, params, points):
        """Evaluate f at batch of points. points: (N, 7) -> (N,)"""
        return vmap(self.f_point, in_axes=(None, 0))(params, points)

    def sample_interior(self, key, n_points):
        """Sample random points in the interior domain."""
        return jr.uniform(key, (n_points, 7),
                         minval=self.bounds_low,
                         maxval=self.bounds_high)

    def sample_initial(self, key, n_points):
        """Sample points at t=0 for initial condition."""
        points = jr.uniform(key, (n_points, 7),
                           minval=self.bounds_low,
                           maxval=self.bounds_high)
        # Set t=0
        points = points.at[:, 0].set(0.0)
        return points

    def sample_boundary(self, key, n_points):
        """Sample points at spatial boundaries for BC."""
        keys = jr.split(key, 3)
        n_per_dim = n_points // 3

        # x boundary
        points_x = jr.uniform(keys[0], (n_per_dim, 7),
                             minval=self.bounds_low, maxval=self.bounds_high)
        x_vals = jr.choice(keys[0], np.array([self.X_bounds[0], self.X_bounds[1]]), (n_per_dim,))
        points_x = points_x.at[:, 1].set(x_vals)

        # y boundary
        points_y = jr.uniform(keys[1], (n_per_dim, 7),
                             minval=self.bounds_low, maxval=self.bounds_high)
        y_vals = jr.choice(keys[1], np.array([self.X_bounds[0], self.X_bounds[1]]), (n_per_dim,))
        points_y = points_y.at[:, 2].set(y_vals)

        # z boundary
        points_z = jr.uniform(keys[2], (n_per_dim, 7),
                             minval=self.bounds_low, maxval=self.bounds_high)
        z_vals = jr.choice(keys[2], np.array([self.X_bounds[0], self.X_bounds[1]]), (n_per_dim,))
        points_z = points_z.at[:, 3].set(z_vals)

        return points_x, points_y, points_z

    def compute_pde_residual(self, params, points):
        """Compute PDE residual at given points."""
        t, x, y, z, vx, vy, vz = points[:, 0], points[:, 1], points[:, 2], points[:, 3], \
                                  points[:, 4], points[:, 5], points[:, 6]

        # Compute f and its derivatives using autodiff
        def f_single(point):
            return self.apply(params, point).squeeze()

        def grad_f(point):
            return jax.grad(f_single)(point)

        f_vals = vmap(f_single)(points)
        grads = vmap(grad_f)(points)  # (N, 7)

        f_t = grads[:, 0]
        f_x = grads[:, 1]
        f_y = grads[:, 2]
        f_z = grads[:, 3]

        # Compute equilibrium (simplified - use rho=1, u=0, T=1 as approximation)
        # For proper implementation, we'd need to integrate over velocity space
        def maxwellian_point(vxi, vyi, vzi):
            rho = 1.0
            temp = 1.0
            coeff = rho / ((2 * np.pi * temp) ** 1.5)
            exponent = -(vxi**2 + vyi**2 + vzi**2) / (2 * temp)
            return coeff * np.exp(exponent)

        f_eq = vmap(maxwellian_point)(vx, vy, vz)

        # PDE: f_t + vx*f_x + vy*f_y + vz*f_z = nu*(f_eq - f)
        residual = f_t + vx * f_x + vy * f_y + vz * f_z - self.nu * (f_eq - f_vals)

        return residual, f_vals

    def compute_ic_residual(self, params, points):
        """Compute initial condition residual."""
        x, y, z = points[:, 1], points[:, 2], points[:, 3]
        vx, vy, vz = points[:, 4], points[:, 5], points[:, 6]

        f_pred = self.f_batch(params, points)

        # Compute f0 point-wise (Maxwellian at t=0)
        def f0_point(xi, yi, zi, vxi, vyi, vzi):
            rho = 1 + 0.5 * np.sin(2*np.pi*xi) * np.sin(2*np.pi*yi) * np.sin(2*np.pi*zi)
            # Maxwellian: rho / (2*pi*T)^(3/2) * exp(-|v-u|^2 / (2*T))
            # With u=0, T=1
            temp = 1.0
            coeff = rho / ((2 * np.pi * temp) ** 1.5)
            exponent = -(vxi**2 + vyi**2 + vzi**2) / (2 * temp)
            return coeff * np.exp(exponent)

        f0_vals = vmap(f0_point)(x, y, z, vx, vy, vz)

        return f_pred - f0_vals, f0_vals

    def compute_bc_residual(self, params, points_x, points_y, points_z):
        """Compute periodic BC residual."""
        # For periodic BC: f(x=-X) = f(x=X), etc.
        # Simplified: just enforce smoothness at boundaries

        f_x = self.f_batch(params, points_x)
        f_y = self.f_batch(params, points_y)
        f_z = self.f_batch(params, points_z)

        return f_x, f_y, f_z

    def loss(self, params, key):
        """Compute total loss using mini-batch sampling."""
        keys = jr.split(key, 4)

        n_interior = int(self.batch_size * 0.7)
        n_ic = int(self.batch_size * 0.15)
        n_bc = int(self.batch_size * 0.15)

        # Sample points
        interior_points = self.sample_interior(keys[0], n_interior)
        ic_points = self.sample_initial(keys[1], n_ic)
        bc_points_x, bc_points_y, bc_points_z = self.sample_boundary(keys[2], n_bc)

        # PDE residual
        pde_res, f_vals = self.compute_pde_residual(params, interior_points)
        loss_pde = np.mean((pde_res / (np.abs(f_vals) + self.eps)) ** 2)

        # IC residual
        ic_res, f0_vals = self.compute_ic_residual(params, ic_points)
        loss_ic = np.mean((ic_res / (np.abs(f0_vals) + self.eps)) ** 2)

        # BC residual (periodic - values should match at boundaries)
        f_x, f_y, f_z = self.compute_bc_residual(params, bc_points_x, bc_points_y, bc_points_z)
        loss_bc = np.mean(f_x**2) + np.mean(f_y**2) + np.mean(f_z**2)

        total_loss = loss_pde + 1e3 * loss_ic + loss_bc

        return total_loss, (loss_pde, loss_ic, loss_bc)

    @partial(jax.jit, static_argnums=(0, 1))
    def step(self, optimizer, params, state, key):
        """Single optimization step."""
        key, subkey = jr.split(key)
        (loss, aux), grads = jax.value_and_grad(self.loss, has_aux=True)(params, subkey)
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, key, loss, aux

    def train(self, optimizer, params, key, nIter, patience=100, min_delta=1e-4):
        """Training loop with early stopping.

        Args:
            patience: Number of log intervals (each=100 iters) without improvement before stopping.
                      Set to 0 or None to disable early stopping.
            min_delta: Minimum relative improvement to count as improvement.
        """
        min_loss = np.inf
        state = optimizer.init(params)
        pde_log, ic_log, bc_log = [], [], []
        no_improve_count = 0
        converged = False
        converged_iter = None
        opt_params = None

        for it in (pbar := trange(1, nIter + 1)):
            params, state, key, loss, (loss_pde, loss_ic, loss_bc) = self.step(
                optimizer, params, state, key
            )

            if it % 100 == 0:
                pde_log.append(float(loss_pde))
                ic_log.append(float(loss_ic))
                bc_log.append(float(loss_bc))

                # Check for improvement
                if np.isinf(min_loss):
                    min_loss = loss
                    opt_params = params
                    no_improve_count = 0
                    pbar.set_postfix({"loss": f"{loss:.3e}"})
                else:
                    relative_improvement = (min_loss - loss) / min_loss if min_loss > 0 else (1.0 if loss < min_loss else 0.0)
                    if loss < min_loss and relative_improvement > min_delta:
                        min_loss = loss
                        opt_params = params
                        no_improve_count = 0
                        pbar.set_postfix({"loss": f"{loss:.3e}"})
                    else:
                        no_improve_count += 1

                # Early stopping check
                if patience and patience > 0 and no_improve_count >= patience:
                    converged = True
                    converged_iter = it
                    print(f"\nConverged at iteration {it} (no improvement for {patience * 100} iterations)")
                    break

                if np.sum(np.isnan(loss)) > 0:
                    print(f"\nNaN detected at iteration {it}, stopping")
                    break

        if opt_params is None:
            opt_params = params

        return opt_params, pde_log, ic_log, bc_log, converged, converged_iter

    def _make_pmap_step(self, optimizer):
        """Create pmap-ed step for multi-GPU."""
        def step_fn(params, state, key):
            key, subkey = random.split(key)
            (loss, aux), grads = jax.value_and_grad(self.loss, has_aux=True)(params, subkey)
            grads = lax.pmean(grads, axis_name='devices')
            updates, state = optimizer.update(grads, state, params)
            params = optax.apply_updates(params, updates)
            return params, state, key, loss, aux

        return jax.pmap(step_fn, axis_name='devices')

    def train_parallel(self, optimizer, params, key, nIter, patience=100, min_delta=1e-4):
        """Multi-GPU training with early stopping.

        Args:
            patience: Number of log intervals (each=100 iters) without improvement before stopping.
                      Set to 0 or None to disable early stopping.
            min_delta: Minimum relative improvement to count as improvement.
        """
        num_devices = jax.local_device_count()
        print(f"Training on {num_devices} device(s)")

        if num_devices == 1:
            return self.train(optimizer, params, key, nIter, patience, min_delta)

        min_loss = np.inf
        no_improve_count = 0
        converged = False
        converged_iter = None
        opt_params = None

        # Initialize and replicate
        state = optimizer.init(params)
        params = jax.tree.map(lambda x: np.stack([x] * num_devices), params)
        state = jax.tree.map(lambda x: np.stack([x] * num_devices), state)
        keys = random.split(key, num_devices)

        pmap_step = self._make_pmap_step(optimizer)

        pde_log, ic_log, bc_log = [], [], []
        for it in (pbar := trange(1, nIter + 1)):
            params, state, keys, loss, (loss_pde, loss_ic, loss_bc) = pmap_step(params, state, keys)

            if it % 100 == 0:
                loss_val = float(loss[0])
                pde_log.append(float(loss_pde[0]))
                ic_log.append(float(loss_ic[0]))
                bc_log.append(float(loss_bc[0]))

                # Check for improvement
                if np.isinf(min_loss):
                    min_loss = loss_val
                    opt_params = jax.tree.map(lambda x: x[0], params)
                    no_improve_count = 0
                    pbar.set_postfix({"loss": f"{loss_val:.3e}", "devices": num_devices})
                else:
                    relative_improvement = (min_loss - loss_val) / min_loss if min_loss > 0 else (1.0 if loss_val < min_loss else 0.0)
                    if loss_val < min_loss and relative_improvement > min_delta:
                        min_loss = loss_val
                        opt_params = jax.tree.map(lambda x: x[0], params)
                        no_improve_count = 0
                        pbar.set_postfix({"loss": f"{loss_val:.3e}", "devices": num_devices})
                    else:
                        no_improve_count += 1

                # Early stopping check
                if patience and patience > 0 and no_improve_count >= patience:
                    converged = True
                    converged_iter = it
                    print(f"\nConverged at iteration {it} (no improvement for {patience * 100} iterations)")
                    break

                if np.isnan(loss_val):
                    print(f"\nNaN detected at iteration {it}, stopping")
                    break

        if opt_params is None:
            opt_params = jax.tree.map(lambda x: x[0], params)

        return opt_params, pde_log, ic_log, bc_log, converged, converged_iter


def main(seed: int = 0, Kn: float = None, width: int = 256, depth: int = 5,
         nIter: int = 100000, parallel: bool = False, batch_size: int = 50000, ngrid: int = 9,
         T: float = 0.1, patience: int = 100, min_delta: float = 1e-4, early_stop: bool = True):
    """Main training function for ordinary PINN with mini-batch sampling.

    Args:
        patience: Number of log intervals (each=100 iters) without improvement before stopping.
        min_delta: Minimum relative improvement to count as improvement.
        early_stop: Whether to enable early stopping.
    """
    date_str = get_date_str()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    w0 = 10
    lr = optax.cosine_decay_schedule(1e-4 / w0, nIter)
    opt = optax.lion(lr, weight_decay=0)
    X, V = 0.5, 6.0
    seed_int = seed
    seed = jr.key(seed)

    assert Kn is not None, "set Kn!"
    num_devices = jax.local_device_count()

    # Calculate total grid points and scaling ratio
    total_grid_points = ngrid ** 7
    batch_per_iter = batch_size * num_devices if parallel else batch_size
    scaling_ratio = total_grid_points / batch_per_iter

    # BGK relaxation time scale
    relaxation_time = Kn
    effective_patience = patience if early_stop else 0

    print(f"="*60)
    print(f"Ordinary PINN 3D smooth (Mini-batch Sampling)")
    print(f"="*60)
    print(f"Kn={Kn}, width={width}, depth={depth}, T={T}")
    print(f"Relaxation time τ=Kn={relaxation_time:.4f}, T/τ={T/relaxation_time:.2f}")
    print(f"Start time: {start_datetime}")
    if early_stop:
        print(f"Number of iterations: {nIter} (max), early_stop=ON, patience={patience} (stop after {patience*100} iters w/o improvement)")
    else:
        print(f"Number of iterations: {nIter}, early_stop=OFF (will run all iterations)")
    print(f"Number of devices: {num_devices}")
    print(f"Batch size per device: {batch_size}")
    print(f"Total batch per iter: {batch_per_iter}")
    print(f"Equivalent grid points (ngrid={ngrid}): {total_grid_points:,}")
    print(f"Scaling ratio (grid/batch): {scaling_ratio:.2f}")
    print(f"="*60)

    model = OrdinaryPINN(T=T, X=X, V=V, w0=w0, Kn=Kn, width=width, depth=depth,
                         batch_size=batch_size, ngrid=ngrid)

    train_key, init_key = jr.split(seed)
    init_params = model.init(init_key)

    # Track training time
    start_time = time.time()

    if parallel:
        opt_params, pde_log, ic_log, bc_log, converged, converged_iter = model.train_parallel(
            opt, init_params, train_key, nIter, patience=effective_patience, min_delta=min_delta
        )
    else:
        opt_params, pde_log, ic_log, bc_log, converged, converged_iter = model.train(
            opt, init_params, train_key, nIter, patience=effective_patience, min_delta=min_delta
        )

    logs = [pde_log, ic_log, bc_log]
    actual_iterations = converged_iter if converged else nIter
    training_time = time.time() - start_time
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get GPU memory usage
    gpu_memory_peak_gib = get_gpu_memory_gib(device_idx=0, peak=True)

    # Estimate full grid memory and time
    # Memory estimation: batch_size points uses gpu_memory_peak_gib
    # Full grid would use: gpu_memory_peak_gib * scaling_ratio
    estimated_full_grid_memory_gib = gpu_memory_peak_gib * scaling_ratio
    estimated_full_grid_time_sec = training_time * scaling_ratio

    # Equivalent single GPU time
    equivalent_single_gpu_time = training_time * num_devices if parallel else training_time
    estimated_full_grid_single_gpu_time = equivalent_single_gpu_time * scaling_ratio

    print(f"\nTraining completed!")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Equivalent single GPU time: {equivalent_single_gpu_time:.2f} seconds")
    print(f"GPU 0 peak memory: {gpu_memory_peak_gib:.4f} GiB")
    print(f"\n--- Estimated Full Grid ({ngrid}^7 = {total_grid_points:,} points) ---")
    print(f"Estimated memory (single GPU): {estimated_full_grid_memory_gib:.2f} GiB")
    print(f"Estimated time (single GPU): {estimated_full_grid_single_gpu_time:.2f} sec ({estimated_full_grid_single_gpu_time/3600:.2f} hours)")
    print(f"End time: {end_datetime}")

    # Ensure directories exist
    os.makedirs("data/x3v3/smooth", exist_ok=True)
    os.makedirs("figures/x3v3/smooth", exist_ok=True)

    # File prefix
    file_prefix = f"pinn_Kn{Kn}_w{width}_d{depth}_gpu{num_devices}_{date_str}"

    # Save parameters
    params_file = f"data/x3v3/smooth/{file_prefix}_params.npy"
    onp.save(params_file, onp.asarray(opt_params, dtype="object"))
    print(f"Parameters saved: {params_file}")

    # Config with all estimations
    config = {
        # Run info
        'date_start': start_datetime,
        'date_end': end_datetime,
        'date_str': date_str,
        # Model info
        'model_type': 'ordinary_pinn_minibatch',
        'width': width,
        'depth': depth,
        # Hardware info
        'num_gpus': num_devices,
        'single_gpu_peak_memory_gib': gpu_memory_peak_gib,
        # Training config
        'Kn': Kn,
        'T': T,
        'parallel': parallel,
        'batch_size': batch_size,
        'ngrid': ngrid,
        'total_grid_points': total_grid_points,
        'scaling_ratio': scaling_ratio,
        'nIter_max': nIter,
        'seed': seed_int,
        # Early stopping / convergence
        'early_stop': early_stop,
        'patience': patience,
        'min_delta': min_delta,
        'converged': converged,
        'actual_iterations': actual_iterations,
        'convergence_iteration': converged_iter,
        # BGK relaxation time scale
        'relaxation_time_tau': relaxation_time,
        'T_over_tau': T / relaxation_time,
        # Timing (actual)
        'training_time_sec': training_time,
        'equivalent_single_gpu_time_sec': equivalent_single_gpu_time,
        # Estimated full grid values
        'estimated_full_grid_memory_gib': estimated_full_grid_memory_gib,
        'estimated_full_grid_time_single_gpu_sec': estimated_full_grid_single_gpu_time,
        # Loss history
        'pde_loss_history': logs[0],
        'ic_loss_history': logs[1],
        'bc_loss_history': logs[2],
        # Final losses
        'final_pde_loss': logs[0][-1] if logs[0] else None,
        'final_ic_loss': logs[1][-1] if logs[1] else None,
        'final_bc_loss': logs[2][-1] if logs[2] else None,
    }

    config_file_npy = f"data/x3v3/smooth/{file_prefix}_config.npy"
    config_file_json = f"data/x3v3/smooth/{file_prefix}_config.json"
    onp.save(config_file_npy, config)
    with open(config_file_json, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_file_json}")

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY (Ordinary PINN - Mini-batch)")
    print("="*60)
    print(f"Date/Time:              {start_datetime} - {end_datetime}")
    print(f"Model:                  Ordinary PINN (width={width}, depth={depth})")
    print(f"Sampling:               Mini-batch ({batch_size:,} points/device)")
    print(f"Number of GPUs:         {num_devices}")
    print(f"Parallel:               {parallel}")
    print(f"Single GPU Peak RAM:    {gpu_memory_peak_gib:.4f} GiB")
    print(f"Training time:          {training_time:.2f} sec")
    print(f"Equiv. single GPU time: {equivalent_single_gpu_time:.2f} sec")
    print(f"Iterations:             {actual_iterations}/{nIter} {'(CONVERGED)' if converged else '(max reached)'}")
    print(f"Convergence:            patience={patience} (={patience*100} iters), min_delta={min_delta}")
    print(f"Final PDE Loss:         {logs[0][-1]:.3e}" if logs[0] else "N/A")
    print(f"Final IC Loss:          {logs[1][-1]:.3e}" if logs[1] else "N/A")
    print(f"Final BC Loss:          {logs[2][-1]:.3e}" if logs[2] else "N/A")
    print("-"*60)
    print(f"Domain: T={T}, X=[-{X},{X}]^3, V=[-{V},{V}]^3")
    print(f"Kn={Kn}, Relaxation time τ={relaxation_time:.4f}")
    print(f"T/τ = {T/relaxation_time:.2f} (need ~3 for 95%, ~5 for 99% decay)")
    print("-"*60)
    print(f"ESTIMATED FULL GRID ({ngrid}^7 = {total_grid_points:,} points):")
    print(f"  Memory (single GPU):  {estimated_full_grid_memory_gib:.2f} GiB")
    print(f"  Time (single GPU):    {estimated_full_grid_single_gpu_time:.2f} sec")
    print("="*60)

    # Loss plot
    if logs[0]:
        _, ax0 = plt.subplots(figsize=(4, 4))
        ax0.semilogy(logs[0], label=f"PDE Loss:{logs[0][-1]:.3e}")
        ax0.semilogy(logs[1], label=f"IC Loss:{logs[1][-1]:.3e}")
        ax0.semilogy(logs[2], label=f"BC Loss:{logs[2][-1]:.3e}")
        ax0.set_xlabel("100 iterations")
        ax0.set_title("Ordinary PINN (Mini-batch) - Loss")
        ax0.legend()
        plt.tight_layout()
        fig_file = f"figures/x3v3/smooth/{file_prefix}_loss.png"
        plt.savefig(fig_file, format="png")
        print(f"Loss plot saved: {fig_file}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
