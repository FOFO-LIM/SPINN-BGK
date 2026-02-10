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

from src.nn import Siren
from src.x3v3 import x3v3, smooth
from utils.transform import trapezoidal_rule


def get_gpu_memory_gib(device_idx=0, peak=False):
    """Get GPU memory usage in GiB for a single device.

    Args:
        device_idx: GPU device index
        peak: If True, return peak memory usage; otherwise return current usage
    """
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


class spinn(x3v3):
    def __init__(
        self,
        T=0.1,
        X=0.5,
        V=10.0,
        Nv=256,
        width=128,
        depth=3,
        rank=256,
        w0=10.0,
        ic=smooth,
        Kn=None,
    ):
        super().__init__(T, X, V, Kn)
        layers = [1] + [width for _ in range(depth - 1)] + [rank]
        self.init, self.apply = Siren(layers, w0)
        self.rank = rank
        self.ic = ic(X, V)
        self.v, self.w = trapezoidal_rule(Nv, -V, V)
        self.wv = self.w * self.v
        self.wvv = self.wv * self.v

    def _loss(self, params, domain):
        t, x, y, z, vx, vy, vz = domain

        def t_mapsto_f(t):
            return self.f(params, t, x, y, z, vx, vy, vz)

        def x_mapsto_f(x):
            return self.f(params, t, x, y, z, vx, vy, vz)

        def y_mapsto_f(y):
            return self.f(params, t, x, y, z, vx, vy, vz)

        def z_mapsto_f(z):
            return self.f(params, t, x, y, z, vx, vy, vz)

        # pde
        f, f_t = jax.jvp(t_mapsto_f, (t,), (np.ones(t.shape),))
        f_x = jax.jvp(x_mapsto_f, (x,), (np.ones(x.shape),))[1]
        f_y = jax.jvp(y_mapsto_f, (y,), (np.ones(y.shape),))[1]
        f_z = jax.jvp(z_mapsto_f, (z,), (np.ones(z.shape),))[1]
        maxwellian = self.maxwellian(*self.rho_u_temp(params, t, x, y, z), vx, vy, vz)
        pde = (
            f_t
            + vx[:, None, None] * f_x
            + vy[:, None] * f_y
            + vz * f_z
            - self.nu * (maxwellian - f)
        )
        # initial condition
        f0 = self.ic.f0(x, y, z, vx, vy, vz)
        ic = self.f(params, np.array([0.0]), x, y, z, vx, vy, vz) - f0
        # boundary condition
        fx = self.f(params, t, self.X, y, z, vx, vy, vz)
        fy = self.f(params, t, x, self.X, z, vx, vy, vz)
        fz = self.f(params, t, x, y, self.X, vx, vy, vz)
        bc_x = fx[:, 1] - fx[:, 0]
        bc_y = fy[:, :, 1] - fy[:, :, 0]
        bc_z = fz[:, :, :, 1] - fz[:, :, :, 0]
        return (pde, ic, bc_x, bc_y, bc_z), (
            f,
            f0,
            np.abs(fx).mean(1),
            np.abs(fy).mean(2),
            np.abs(fz).mean(3),
        )


def main(seed: int = 0, Kn: float = None, rank: int = 128, parallel: bool = False, nIter: int = 100000, ngrid: int = 12, T: float = 0.1, patience: int = 50, min_delta: float = 1e-4, early_stop: bool = True):
    # Get date/time for filenames
    date_str = get_date_str()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    w0 = 10
    lr = optax.cosine_decay_schedule(1e-4 / w0, nIter)
    opt = optax.lion(lr, weight_decay=0)
    X, V = 0.5, 6.0  # T is now a command-line argument
    N = [ngrid] * 7
    seed_int = seed  # save original seed for config
    seed = jr.key(seed)

    assert Kn is not None, "set Kn!"
    num_devices = jax.local_device_count()

    # BGK relaxation time scale: τ = Kn (collision frequency ν = 1/Kn)
    # f_neq decays as exp(-t/τ) = exp(-t/Kn)
    # For ~95% decay: need T ≈ 3*Kn
    # For ~99% decay: need T ≈ 5*Kn
    relaxation_time = Kn
    time_for_95_decay = 3 * Kn
    time_for_99_decay = 5 * Kn

    print(f"3d smooth, Kn={Kn}, rank={rank}, ngrid={ngrid}, parallel={parallel}, devices={num_devices}")
    print(f"Final time T={T}, Relaxation time τ=Kn={relaxation_time:.4f}")
    print(f"  T/τ = {T/relaxation_time:.2f} (need ~3 for 95% decay, ~5 for 99% decay)")
    print(f"Start time: {start_datetime}")
    # Effective patience (0 disables early stopping)
    effective_patience = patience if early_stop else 0
    if early_stop:
        print(f"Number of iterations: {nIter} (max), early_stop=ON, patience={patience} (stop after {patience*100} iters w/o improvement)")
    else:
        print(f"Number of iterations: {nIter}, early_stop=OFF (will run all iterations)")
    model = spinn(T=T, X=X, V=V, w0=w0, Kn=Kn, rank=rank)
    domain = [np.linspace(*bd, n) for bd, n in zip(model.bd, N)]

    train_key, init_key = jr.split(seed)
    init_key = jr.split(init_key, model.dim + 5)
    init_params = (
        jr.uniform(init_key[0], (5, model.rank), minval=-1, maxval=1)
        * np.sqrt(6 / model.rank),
        [model.init(k) for k in init_key[1:5]],
        [model.init(k) for k in init_key[5:]],
        [np.array([0.0, 0.0, 0.0]), np.array(1.0)],
    )

    # Track training time and memory
    start_time = time.time()

    if parallel:
        opt_params, pde_log, ic_log, bc_log, converged, converged_iter = model.train_parallel(
            opt, domain, init_params, train_key, nIter, patience=effective_patience, min_delta=min_delta
        )
    else:
        opt_params, pde_log, ic_log, bc_log, converged, converged_iter = model.train(
            opt, domain, init_params, train_key, nIter, patience=effective_patience, min_delta=min_delta
        )

    # For backward compatibility, keep logs as list
    logs = [pde_log, ic_log, bc_log]
    actual_iterations = converged_iter if converged else nIter

    training_time = time.time() - start_time
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get GPU memory usage (single GPU)
    gpu_memory_peak_gib = get_gpu_memory_gib(device_idx=0, peak=True)

    # Calculate equivalent single GPU time
    equivalent_single_gpu_time = training_time * num_devices if parallel else training_time

    print(opt_params[-1])
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Equivalent single GPU time: {equivalent_single_gpu_time:.2f} seconds")
    print(f"GPU 0 peak memory: {gpu_memory_peak_gib:.4f} GiB")
    print(f"End time: {end_datetime}")

    # Ensure directories exist
    os.makedirs("data/x3v3/smooth", exist_ok=True)
    os.makedirs("figures/x3v3/smooth", exist_ok=True)

    # File prefix with date
    file_prefix = f"spinn_Kn{Kn}_rank{rank}_ngrid{ngrid}_gpu{num_devices}_{date_str}"

    # save network parameters
    params_file = f"data/x3v3/smooth/{file_prefix}_params.npy"
    onp.save(params_file, onp.asarray(opt_params, dtype="object"))
    print(f"Parameters saved: {params_file}")

    # save training configuration and stats
    config = {
        # Run info
        'date_start': start_datetime,
        'date_end': end_datetime,
        'date_str': date_str,
        # Hardware info
        'num_gpus': num_devices,
        'single_gpu_peak_memory_gib': gpu_memory_peak_gib,
        # Training config
        'model_type': 'spinn',
        'Kn': Kn,
        'rank': rank,
        'ngrid': ngrid,
        'parallel': parallel,
        'domain_size': N,
        'nIter_max': nIter,  # Maximum iterations (may stop earlier if converged)
        'seed': seed_int,
        # Early stopping / convergence
        'early_stop': early_stop,  # Whether early stopping is enabled
        'patience': patience,  # Log intervals (100 iters each) without improvement before stopping
        'min_delta': min_delta,  # Minimum relative improvement to count as improvement
        'converged': converged,  # True if training stopped early due to convergence
        'actual_iterations': actual_iterations,  # Actual number of iterations completed
        'convergence_iteration': converged_iter,  # Iteration at which convergence was detected (or None)
        # Domain bounds
        'T': T,  # Final time (time domain: [0, T])
        'X': X,  # Spatial half-width (spatial domain: [-X, X]^3)
        'V': V,  # Velocity half-width (velocity domain: [-V, V]^3)
        # BGK relaxation time scale
        'relaxation_time_tau': relaxation_time,  # τ = Kn
        'time_for_95_decay': time_for_95_decay,  # 3*Kn
        'time_for_99_decay': time_for_99_decay,  # 5*Kn
        'T_over_tau': T / relaxation_time,  # T/τ ratio (>3 for significant decay)
        # Timing
        'training_time_sec': training_time,
        'equivalent_single_gpu_time_sec': equivalent_single_gpu_time,
        # Loss history (all epochs, logged every 100 iterations)
        'pde_loss_history': [float(x) for x in logs[0]],
        'ic_loss_history': [float(x) for x in logs[1]],
        'bc_loss_history': [float(x) for x in logs[2]],
        # Final losses
        'final_pde_loss': float(logs[0][-1]),
        'final_ic_loss': float(logs[1][-1]),
        'final_bc_loss': float(logs[2][-1]),
    }
    config_file_npy = f"data/x3v3/smooth/{file_prefix}_config.npy"
    config_file_json = f"data/x3v3/smooth/{file_prefix}_config.json"
    onp.save(config_file_npy, config)
    with open(config_file_json, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_file_json}")

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Date/Time:              {start_datetime} - {end_datetime}")
    print(f"Number of GPUs:         {num_devices}")
    print(f"Single GPU Peak RAM:    {gpu_memory_peak_gib:.4f} GiB")
    print(f"Training time:          {training_time:.2f} sec")
    print(f"Equiv. single GPU time: {equivalent_single_gpu_time:.2f} sec")
    print(f"Iterations:             {actual_iterations}/{nIter} {'(CONVERGED)' if converged else '(max reached)'}")
    print(f"Convergence:            patience={patience} (={patience*100} iters), min_delta={min_delta}")
    print(f"Final PDE Loss:         {logs[0][-1]:.3e}")
    print(f"Final IC Loss:          {logs[1][-1]:.3e}")
    print(f"Final BC Loss:          {logs[2][-1]:.3e}")
    print("-"*60)
    print(f"Domain: T={T}, X=[-{X},{X}]^3, V=[-{V},{V}]^3")
    print(f"Kn={Kn}, Relaxation time τ={relaxation_time:.4f}")
    print(f"T/τ = {T/relaxation_time:.2f} (need ~3 for 95%, ~5 for 99% decay)")
    print("="*60)

    # loss trajectory
    _, ax0 = plt.subplots(figsize=(4, 4))
    ax0.semilogy(logs[0], label=f"PDE Loss:{logs[0][-1]:.3e}")
    ax0.semilogy(logs[1], label=f"IC Loss:{logs[1][-1]:.3e}")
    ax0.semilogy(logs[2], label=f"BC Loss:{logs[2][-1]:.3e}")
    ax0.set_xlabel("100 iterations")
    ax0.set_title("Test Mean Squared Loss")
    ax0.legend()
    plt.tight_layout()
    fig_file = f"figures/x3v3/smooth/{file_prefix}_loss.png"
    plt.savefig(fig_file, format="png")
    print(f"Loss plot saved: {fig_file}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
