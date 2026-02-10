#!/usr/bin/env python3
"""
Optimize τ (tau) for BGK approximation ON-THE-FLY during Landau simulation.

Uses 8 GPUs with domain decomposition:
- Collision step: local on each GPU's chunk (8192 x-points per GPU)
- Advection step: all_gather for global FFT, scatter back
- τ error computation: parallel across GPUs

Parameters:
- N_size = 32: time steps per measurement (4096 measurements total)
- N_bin = 32: measurements per output (128 outputs)
- τ grid: 0.1 to 1000 with 100 log-uniform values
"""

import os
# Use all 8 GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax, pmap
from jax.lax import all_gather, pmean
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from tqdm import tqdm
import time

print(f"JAX devices: {jax.devices()}")
num_devices = jax.local_device_count()
print(f"Number of devices: {num_devices}")


def run_tau_optimization_8gpu(N_size=32, N_bin=32):
    """
    Run Landau simulation with on-the-fly τ optimization using 8 GPUs.
    Domain decomposition for parallel collision, all_gather for advection.
    """
    # Parameters
    N_x = 2**16  # 65536
    N_v = 2**10  # 1024
    N_t = 2**17  # 131072
    X = 0.5
    V = 6.0
    T_final = 0.1
    lambda_D = 10.0
    cutoff = 1.0 / lambda_D

    # τ grid: log-uniform from 0.1 to 1000
    tau_values = jnp.logspace(jnp.log10(0.1), jnp.log10(1000), 100)
    N_tau = len(tau_values)

    # Domain decomposition
    assert N_x % num_devices == 0, f"N_x ({N_x}) must be divisible by num_devices ({num_devices})"
    chunk_size = N_x // num_devices  # 8192

    # Grids
    dx = 2 * X / N_x
    x = jnp.linspace(-X, X - dx, N_x)
    dv = 2 * V / (N_v - 1)
    v = jnp.linspace(-V, V, N_v)
    dt = T_final / N_t

    # Spectral wavenumbers
    kx = 2 * jnp.pi * jnp.fft.fftfreq(N_x, dx)

    # Coulomb kernel FFT
    N_conv = 2 * N_v - 1
    u_kernel = jnp.arange(-(N_v - 1), N_v) * dv
    Phi = 1.0 / jnp.maximum(jnp.abs(u_kernel), cutoff)
    Phi_fft = jnp.fft.fft(Phi)

    # Advection phase
    phase_half = jnp.exp(-1j * jnp.outer(kx, v) * dt / 2)

    # Number of measurements and outputs
    N_measurements = N_t // N_size  # 4096
    N_outputs = N_measurements // N_bin  # 128

    print(f"\n{'='*70}")
    print(f"On-the-fly τ Optimization ({num_devices} GPUs)")
    print('='*70)
    print(f"Grid: N_x={N_x}, N_v={N_v}, N_t={N_t}")
    print(f"Chunk size per GPU: {chunk_size} x-points")
    print(f"N_size={N_size} (time steps per measurement)")
    print(f"N_bin={N_bin} (measurements per output)")
    print(f"Total measurements: {N_measurements}")
    print(f"Output points: {N_outputs}")
    print(f"τ grid: {float(tau_values[0]):.2f} to {float(tau_values[-1]):.2f} ({N_tau} log-uniform)")
    print('='*70)

    # ===== Define functions for parallel execution =====

    def convolution_fft_inline(f_row):
        """FFT convolution for single row."""
        f_padded = jnp.zeros(N_conv)
        f_padded = f_padded.at[:N_v].set(f_row)
        conv_full = jnp.real(jnp.fft.ifft(Phi_fft * jnp.fft.fft(f_padded)))
        return conv_full[N_v - 1:N_v - 1 + N_v] * dv

    def collision_step_chunk(f_chunk):
        """Collision step for a spatial chunk. f_chunk: (chunk_size, N_v)"""
        def collision_one_step(f):
            # Compute velocity derivative
            df_dv = jnp.zeros_like(f)
            df_dv = df_dv.at[:, 1:-1].set((f[:, 2:] - f[:, :-2]) / (2 * dv))
            df_dv = df_dv.at[:, 0].set((f[:, 1] - f[:, 0]) / dv)
            df_dv = df_dv.at[:, -1].set((f[:, -1] - f[:, -2]) / dv)

            # Compute collision coefficients via vmap
            A = vmap(convolution_fft_inline)(f)
            B = vmap(convolution_fft_inline)(df_dv)

            # Flux and divergence
            J = A * df_dv - B * f
            Q = jnp.zeros_like(f)
            Q = Q.at[:, 1:-1].set((J[:, 2:] - J[:, :-2]) / (2 * dv))
            Q = Q.at[:, 0].set((J[:, 1] - J[:, 0]) / dv)
            Q = Q.at[:, -1].set((J[:, -1] - J[:, -2]) / dv)
            return Q

        # RK4 time integration
        k1 = collision_one_step(f_chunk)
        k2 = collision_one_step(f_chunk + 0.5 * dt * k1)
        k3 = collision_one_step(f_chunk + 0.5 * dt * k2)
        k4 = collision_one_step(f_chunk + dt * k3)
        f_new = f_chunk + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return jnp.maximum(f_new, 0.0)

    def advection_step_full(f_full):
        """Advection step on full domain."""
        f_hat = jnp.fft.fft(f_full, axis=0)
        return jnp.real(jnp.fft.ifft(f_hat * phase_half, axis=0))

    def strang_step_parallel(f_chunk):
        """
        One Strang splitting step distributed across GPUs.
        f_chunk has shape (chunk_size, N_v) per device.
        """
        # Gather from all devices for advection
        f_gathered = all_gather(f_chunk, axis_name='devices')
        f_full = f_gathered.reshape(N_x, N_v)

        # Advection half step (on full domain)
        f_full = advection_step_full(f_full)

        # Split back into chunks and get this device's chunk
        f_split = f_full.reshape(num_devices, chunk_size, N_v)
        device_idx = jax.lax.axis_index('devices')
        f_chunk = f_split[device_idx]

        # Collision full step (local on this device)
        f_chunk = collision_step_chunk(f_chunk)

        # Gather for second advection
        f_gathered = all_gather(f_chunk, axis_name='devices')
        f_full = f_gathered.reshape(N_x, N_v)

        # Advection half step
        f_full = advection_step_full(f_full)

        # Split and return this device's chunk
        f_split = f_full.reshape(num_devices, chunk_size, N_v)
        f_chunk = f_split[device_idx]
        return f_chunk

    def do_n_steps_parallel(f_chunk):
        """Do N_size time steps using fori_loop inside pmap."""
        return lax.fori_loop(0, N_size, lambda i, f: strang_step_parallel(f), f_chunk)

    def compute_tau_errors_chunk(f_chunk, f_prev_chunk, f_next_chunk):
        """
        Compute L2 error contribution from this chunk for all τ values.
        Returns sum of squared errors (to be averaged across devices).
        """
        dt_meas = N_size * dt

        # Trapezoidal weights
        w = jnp.ones(N_v) * dv
        w = w.at[0].set(dv / 2)
        w = w.at[-1].set(dv / 2)

        # Moments for this chunk
        rho = jnp.sum(f_chunk * w, axis=1)
        momentum = jnp.sum(f_chunk * v * w, axis=1)
        energy = jnp.sum(f_chunk * v**2 * w, axis=1)
        rho = jnp.maximum(rho, 1e-16)
        u = momentum / rho
        T = jnp.maximum((energy / rho) - u**2, 1e-10)

        # Maxwellian
        f_eq = (rho[:, None] / jnp.sqrt(2 * jnp.pi * T[:, None]) *
                jnp.exp(-(v[None, :] - u[:, None])**2 / (2 * T[:, None])))

        # Time derivative (central difference)
        df_dt = (f_next_chunk - f_prev_chunk) / (2 * dt_meas)

        # Spatial derivative - need global data for this
        # Gather full domain
        f_gathered = all_gather(f_chunk, axis_name='devices')
        f_full = f_gathered.reshape(N_x, N_v)

        # Compute spatial derivative on full domain
        df_dx_full = (jnp.roll(f_full, -1, axis=0) - jnp.roll(f_full, 1, axis=0)) / (2 * dx)

        # Get this device's chunk of the derivative
        device_idx = jax.lax.axis_index('devices')
        df_dx = df_dx_full.reshape(num_devices, chunk_size, N_v)[device_idx]

        # Collision term (what we're trying to match with BGK)
        collision_term = df_dt + v[None, :] * df_dx

        # f_neq = f_eq - f
        f_neq = f_eq - f_chunk

        # Error for each tau - compute sum of squared errors for this chunk
        def error_for_tau(tau):
            bgk = f_neq / tau
            diff = bgk - collision_term
            return jnp.sum(diff**2)  # Sum for this chunk

        chunk_errors = vmap(error_for_tau)(tau_values)

        # Average across all devices
        return pmean(chunk_errors, axis_name='devices') / N_x

    # pmap the step and error functions
    do_n_steps_pmap = pmap(do_n_steps_parallel, axis_name='devices')
    compute_tau_errors_pmap = pmap(compute_tau_errors_chunk, axis_name='devices')

    # ===== Initial condition =====
    rho0 = 1 + 0.5 * jnp.sin(2 * jnp.pi * x)
    f_full = (rho0[:, None] / jnp.sqrt(2 * jnp.pi) *
              jnp.exp(-v[None, :]**2 / 2))

    # Reshape into chunks for pmap
    f_chunks = f_full.reshape(num_devices, chunk_size, N_v)

    # ===== JIT warmup =====
    print("\nJIT compiling parallel code...", end=" ", flush=True)
    _ = do_n_steps_pmap(f_chunks)
    jax.block_until_ready(_)
    # Warmup error computation
    _ = compute_tau_errors_pmap(f_chunks, f_chunks, f_chunks)
    jax.block_until_ready(_)
    print("done")

    # ===== Main loop =====
    print(f"\nRunning {N_measurements} measurements ({N_size} steps each)...")

    all_tau_errors = []
    measurement_times = []

    # Keep 3 f_chunks values on GPU for central difference
    f_history = [f_chunks, None, None]

    start_time = time.time()

    for m in tqdm(range(N_measurements), desc="Measurements"):
        # Do N_size steps (entirely on GPUs, uses fori_loop inside pmap)
        f_chunks = do_n_steps_pmap(f_chunks)

        # Shift history
        f_history[0] = f_history[1]
        f_history[1] = f_history[2]
        f_history[2] = f_chunks

        # Compute τ errors if we have 3 values
        if m >= 2:
            errors = compute_tau_errors_pmap(f_history[1], f_history[0], f_history[2])
            # errors is replicated across devices, just take first one
            all_tau_errors.append(np.array(errors[0]))
            measurement_times.append((m - 1) * N_size * dt)

    jax.block_until_ready(f_chunks)
    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Throughput: {N_t / elapsed:.1f} steps/sec")
    print(f"τ measurements: {len(all_tau_errors)}")

    # Convert
    all_tau_errors = np.array(all_tau_errors)
    measurement_times = np.array(measurement_times)

    # ===== Average over N_bin =====
    N_actual = len(all_tau_errors)
    N_outputs = N_actual // N_bin

    optimal_taus = []
    tau_stds = []
    output_times = []
    output_errors = []

    for i in range(N_outputs):
        start_idx = i * N_bin
        end_idx = start_idx + N_bin

        bin_errors = all_tau_errors[start_idx:end_idx]
        bin_optimal_indices = np.argmin(bin_errors, axis=1)
        bin_optimal_taus = np.array(tau_values)[bin_optimal_indices]

        optimal_taus.append(np.mean(bin_optimal_taus))
        tau_stds.append(np.std(bin_optimal_taus))
        output_times.append(np.mean(measurement_times[start_idx:end_idx]))
        output_errors.append(np.mean(bin_errors, axis=0))

    optimal_taus = np.array(optimal_taus)
    tau_stds = np.array(tau_stds)
    output_times = np.array(output_times)
    output_errors = np.array(output_errors)

    print(f"\n{'='*60}")
    print("Results")
    print('='*60)
    print(f"Outputs: {len(optimal_taus)}")
    print(f"τ range: {optimal_taus.min():.2f} to {optimal_taus.max():.2f}")
    print(f"τ mean: {optimal_taus.mean():.2f} ± {optimal_taus.std():.2f}")

    return {
        'optimal_taus': optimal_taus,
        'tau_stds': tau_stds,
        'output_times': output_times,
        'output_errors': output_errors,
        'tau_values': np.array(tau_values),
        'elapsed_time': elapsed,
        'N_size': N_size,
        'N_bin': N_bin,
        'num_devices': num_devices,
    }


def main(N_size=32, N_bin=32):
    results = run_tau_optimization_8gpu(N_size, N_bin)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f"tau_online_8gpu_Nsize{N_size}_Nbin{N_bin}_{timestamp}"
    data_dir = "data/landau_1d"
    fig_dir = "figures/landau_1d"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    np.savez(f"{data_dir}/{output_base}.npz", **results)
    print(f"\nSaved to {data_dir}/{output_base}.npz")

    # Plot
    optimal_taus = results['optimal_taus']
    tau_stds = results['tau_stds']
    output_times = results['output_times']
    tau_values = results['tau_values']
    output_errors = results['output_errors']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(output_times, optimal_taus, 'b.-', markersize=4)
    ax.fill_between(output_times, optimal_taus - tau_stds, optimal_taus + tau_stds,
                    alpha=0.3, color='blue', label='± Std within bin')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Optimal τ')
    ax.set_yscale('log')
    ax.set_title(f'Optimal τ (N_size={N_size}, N_bin={N_bin}, {results["num_devices"]} GPUs)')
    ax.axhline(optimal_taus.mean(), color='r', linestyle='--',
               label=f'Mean = {optimal_taus.mean():.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(optimal_taus, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(optimal_taus.mean(), color='r', linestyle='--')
    ax.set_xlabel('Optimal τ')
    ax.set_ylabel('Count')
    ax.set_title('Distribution')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(output_times, tau_stds, 'g.-', markersize=4)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Std within bin')
    ax.set_title('Within-bin Std')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if len(output_errors) > 0:
        im = ax.pcolormesh(tau_values, output_times, np.log10(output_errors + 1e-20),
                           shading='auto', cmap='viridis')
        plt.colorbar(im, ax=ax, label='log₁₀(Error)')
        ax.set_xscale('log')
        ax.set_xlabel('τ')
        ax.set_ylabel('Time t')
        ax.set_title('Error Landscape')
        ax.plot(optimal_taus, output_times, 'r.', markersize=3)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{output_base}.png", dpi=150)
    print(f"Figure: {fig_dir}/{output_base}.png")

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(output_times, optimal_taus, 'b-o', markersize=3)
    ax2.fill_between(output_times, optimal_taus - tau_stds, optimal_taus + tau_stds,
                     alpha=0.3, color='blue', label=f'± Std (avg={tau_stds.mean():.2f})')
    ax2.axhline(optimal_taus.mean(), color='r', linestyle='--',
               label=f'Mean τ = {optimal_taus.mean():.2f}')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Optimal τ')
    ax2.set_yscale('log')
    ax2.set_title(f'Optimal BGK τ vs Time ({results["num_devices"]} GPUs)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.savefig(f"{fig_dir}/{output_base}_timeseries.png", dpi=150)
    print(f"Timeseries: {fig_dir}/{output_base}_timeseries.png")

    plt.show()
    return results


if __name__ == "__main__":
    # N_size=1024 (2^10): time steps per measurement -> 128 measurements
    # N_bin=1 (2^0): no averaging -> 128 outputs
    main(N_size=1024, N_bin=1)
