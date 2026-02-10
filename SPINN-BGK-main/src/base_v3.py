from functools import partial

import jax
from jax import random, grad, lax
import jax.numpy as np
import optax
from tqdm import trange

from utils.transform import maxwellian3d
from utils.training import unif_sampler


def get_num_devices():
    """Get the number of available devices (GPUs/TPUs)."""
    return jax.local_device_count()


def replicate(pytree, num_devices=None):
    """Replicate a pytree across devices."""
    if num_devices is None:
        num_devices = get_num_devices()
    return jax.tree.map(lambda x: np.stack([x] * num_devices), pytree)


def scale_domain_for_parallel(domain, num_devices, dim):
    """Scale down domain size per device to maintain total work.

    Since computation is a tensor product of `dim` dimensions (work ~ N^dim),
    to get num_devices× speedup, each GPU should process N' points where:
    N'^dim = N^dim / num_devices  =>  N' = N / num_devices^(1/dim)
    """
    scale_factor = num_devices ** (1.0 / dim)
    scaled_domain = []
    for d in domain:
        n_original = len(d)
        n_scaled = max(2, int(n_original / scale_factor))  # at least 2 points
        # Create new linspace with same bounds but fewer points
        scaled_domain.append(np.linspace(d[0], d[-1], n_scaled))
    return scaled_domain


def unreplicate(pytree):
    """Get the first replica of a pytree (all replicas should be identical)."""
    return jax.tree.map(lambda x: x[0], pytree)


class base_v3:
    def __init__(
        self,
        T: float = 0.1,
        X: float = 0.5,
        V: float = 10,
        Kn: float = None,
    ):
        super().__init__()
        self.T = np.array([0, T])
        self.X = np.array([-X, X])
        self.V = np.array([-V, V])
        self.nu = 1 / Kn
        self.maxwellian = maxwellian3d

    def rho_u_temp(self, params, t, x, y, z):
        moments = self.moments(params, t, x, y, z)
        m, p, E = moments[0, ...], moments[1:4, ...], moments[4:, ...]
        rho = np.maximum(m, 1e-3)
        u = p / rho
        temp = (E.sum(0) / rho - (u**2).sum(0)) / 3
        temp = np.maximum(temp, 1e-3)
        return rho, u, temp

    def sampling(self, key, domain):
        keys = random.split(key, self.dim)
        update = [unif_sampler(k, d, *b) for k, d, b in zip(keys, domain, self.bd)]
        return update

    @partial(jax.jit, static_argnums=(0, 1))
    def step(self, optimizer, params, state, key, domain, *args):
        # update params
        g, _ = grad(self.loss, has_aux=True)(params, domain, *args)
        updates, state = optimizer.update(g, state, params)
        params = optax.apply_updates(params, updates)
        # sample domain
        key, subkey = random.split(key)
        domain = self.sampling(subkey, domain)
        return params, state, key, domain

    def _make_pmap_step(self, optimizer):
        """Create a pmap-ed step function for multi-GPU training."""
        def step_fn(params, state, key, domain):
            # Compute gradients on each device's portion of domain
            g, _ = grad(self.loss, has_aux=True)(params, domain)
            # Average gradients across all devices
            g = lax.pmean(g, axis_name='devices')
            # Update using synchronized gradients
            updates, state = optimizer.update(g, state, params)
            params = optax.apply_updates(params, updates)
            # Sample new domain points (each device gets different samples)
            key, subkey = random.split(key)
            domain = self.sampling(subkey, domain)
            return params, state, key, domain

        return jax.pmap(step_fn, axis_name='devices')

    @partial(jax.jit, static_argnums=(0,))
    def logger(self, params):
        loss, (loss_r, loss_ic, loss_bc) = self.loss(params, self.domain_te)
        return loss, (loss_r, loss_ic, loss_bc)

    def train(self, optimizer, domain, params, key, nIter, patience=50, min_delta=1e-4):
        """Single-GPU training with early stopping.

        Args:
            patience: Number of log intervals (each=100 iters) without improvement before stopping.
                      Set to 0 or None to disable early stopping.
            min_delta: Minimum relative improvement (loss_old - loss_new) / loss_old to count as improvement.
        """
        min_loss = np.inf
        domain = [*domain]
        state = optimizer.init(params)
        pde_log, ic_log, bc_log = [], [], []
        no_improve_count = 0
        converged = False
        converged_iter = None
        opt_params = None  # Initialize to avoid UnboundLocalError

        for it in (pbar := trange(1, nIter + 1)):
            params, state, key, domain = self.step(
                optimizer, params, state, key, domain
            )
            if it % 100 == 0:
                loss, (loss_r, loss_ic, loss_bc) = self.logger(params)
                pde_log.append(loss_r), ic_log.append(loss_ic), bc_log.append(loss_bc)

                # Check for improvement (handle first iteration where min_loss is inf)
                if np.isinf(min_loss):
                    # First valid loss - always accept
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

        # Fallback if no params were ever saved (shouldn't happen)
        if opt_params is None:
            opt_params = params

        return opt_params, pde_log, ic_log, bc_log, converged, converged_iter

    def train_parallel(self, optimizer, domain, params, key, nIter, patience=50, min_delta=1e-4):
        """Multi-GPU training using pmap for data parallelism with early stopping.

        Each GPU processes different domain samples, gradients are averaged
        across devices using lax.pmean.

        Domain is scaled down per device so total work equals single-GPU work,
        achieving actual speedup proportional to num_devices.

        Args:
            patience: Number of log intervals (each=100 iters) without improvement before stopping.
                      Set to 0 or None to disable early stopping.
            min_delta: Minimum relative improvement (loss_old - loss_new) / loss_old to count as improvement.
        """
        num_devices = get_num_devices()
        print(f"Training on {num_devices} device(s)")

        if num_devices == 1:
            print("Single device detected, falling back to regular training")
            return self.train(optimizer, domain, params, key, nIter, patience, min_delta)

        min_loss = np.inf
        domain = [*domain]
        no_improve_count = 0
        converged = False
        converged_iter = None
        opt_params = None  # Initialize to avoid UnboundLocalError

        # Scale down domain per device to maintain total work
        # Work ~ N^dim, so N' = N / num_devices^(1/dim)
        scaled_domain = scale_domain_for_parallel(domain, num_devices, self.dim)
        original_size = [len(d) for d in domain]
        scaled_size = [len(d) for d in scaled_domain]
        print(f"Domain scaled: {original_size} -> {scaled_size} per device (total work preserved)")

        # Initialize optimizer state and replicate across devices
        state = optimizer.init(params)
        params = replicate(params, num_devices)
        state = replicate(state, num_devices)

        # Split keys for each device
        keys = random.split(key, num_devices)

        # Replicate scaled domain across devices (each device samples differently due to different keys)
        domain = replicate(scaled_domain, num_devices)

        # Create pmap-ed step function
        pmap_step = self._make_pmap_step(optimizer)

        pde_log, ic_log, bc_log = [], [], []
        for it in (pbar := trange(1, nIter + 1)):
            params, state, keys, domain = pmap_step(params, state, keys, domain)

            if it % 100 == 0:
                # Get params from first device for logging (all devices have same params after pmean)
                params_single = unreplicate(params)
                loss, (loss_r, loss_ic, loss_bc) = self.logger(params_single)
                pde_log.append(loss_r), ic_log.append(loss_ic), bc_log.append(loss_bc)

                # Check for improvement (handle first iteration where min_loss is inf)
                if np.isinf(min_loss):
                    # First valid loss - always accept
                    min_loss = loss
                    opt_params = params_single
                    no_improve_count = 0
                    pbar.set_postfix({"loss": f"{loss:.3e}", "devices": num_devices})
                else:
                    relative_improvement = (min_loss - loss) / min_loss if min_loss > 0 else (1.0 if loss < min_loss else 0.0)
                    if loss < min_loss and relative_improvement > min_delta:
                        min_loss = loss
                        opt_params = params_single
                        no_improve_count = 0
                        pbar.set_postfix({"loss": f"{loss:.3e}", "devices": num_devices})
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

        # Fallback if no params were ever saved (shouldn't happen)
        if opt_params is None:
            opt_params = unreplicate(params)

        return opt_params, pde_log, ic_log, bc_log, converged, converged_iter
