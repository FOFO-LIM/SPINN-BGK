#!/usr/bin/env python3
"""
Run 1D Boltzmann-Landau simulation with frequent saves for τ optimization.

Saves every N_batch time steps to enable proper τ optimization with:
- N_batch = 32: one measurement per 32 time steps
- N_bin = 32: average 32 measurements per output
- Total outputs = N_t / (N_batch * N_bin) = 131072 / 1024 = 128
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Use 1 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
print(f"JAX devices: {jax.devices()}")
print(f"Number of devices: {jax.local_device_count()}")

from landau_1d_numerical_jax import LandauSolver1D_JAX, get_gpu_memory_gib

# Parameters for τ optimization
N_batch = 32  # 2^5: time steps per measurement
N_bin = 32    # 2^5: measurements per output
N_output = 128  # Desired number of output τ values

# Simulation parameters (same as original)
N_x = 2**16  # 65536
N_v = 2**10  # 1024
N_t = 2**17  # 131072
X = 0.5
V = 6.0
T_final = 0.1
lambda_D = 10.0
num_gpus = 1

# Save every N_batch * N_bin steps to get N_output snapshots
# Each snapshot represents one "bin" of measurements
# 131072 / 1024 = 128 snapshots → ~34 GB file
save_every = N_batch * N_bin  # 1024

print(f"\n{'='*70}")
print("1D Boltzmann-Landau Simulation for τ Optimization")
print('='*70)
print(f"N_x = {N_x}, N_v = {N_v}, N_t = {N_t}")
print(f"N_batch = {N_batch} (time steps per τ measurement)")
print(f"N_bin = {N_bin} (measurements to average per output)")
print(f"save_every = {save_every} (= N_batch × N_bin)")
print(f"Expected snapshots: {N_t // save_every + 1}")
print(f"Expected τ outputs: ~{N_t // save_every - 2} (minus boundaries)")
print(f"Using {num_gpus} GPU")

# Estimate file size
snapshots = N_t // save_every + 1
file_size_gb = snapshots * N_x * N_v * 4 / 1e9  # float32
print(f"Estimated file size: {file_size_gb:.1f} GB")
print('='*70)

# Check GPU memory
total_gpu_mem = 48  # ~48 GB per A6000
print(f"GPU memory available: ~{total_gpu_mem} GB")

if file_size_gb > 200:
    print(f"\nWARNING: File size ({file_size_gb:.1f} GB) is very large!")
    print("Aborting to prevent disk overflow.")
    sys.exit(1)

# Record start time
start_datetime = datetime.now()
print(f"\nStart time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

# Create solver
print("\nInitializing solver...")
solver = LandauSolver1D_JAX(
    N_x=N_x,
    N_v=N_v,
    N_t=N_t,
    X=X,
    V=V,
    T_final=T_final,
    lambda_D=lambda_D
)

# Run simulation (single GPU)
print(f"\nStarting simulation (save_every={save_every})...")
results = solver.solve(save_every=save_every, verbose=True)

# Record end time
end_datetime = datetime.now()
elapsed = (end_datetime - start_datetime).total_seconds()

print(f"\n{'='*70}")
print("SIMULATION COMPLETE")
print('='*70)
print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
print(f"Throughput: {N_t / results['elapsed_time']:.1f} steps/sec")

# Get GPU memory
gpu_mem = get_gpu_memory_gib(device_idx=0, peak=True)
print(f"Peak GPU memory (device 0): {gpu_mem:.2f} GiB")

# Save results
os.makedirs("data/landau_1d", exist_ok=True)
os.makedirs("figures/landau_1d", exist_ok=True)

timestamp = start_datetime.strftime("%Y%m%d_%H%M%S")
base_name = f"landau_tau_Nx{N_x}_Nv{N_v}_Nt{N_t}_batch{N_batch}_{timestamp}"

# Save config
config = {
    "run_date": start_datetime.strftime("%Y-%m-%d"),
    "run_start_time": start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
    "run_end_time": end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
    "elapsed_time_sec": elapsed,
    "simulation_time_sec": results['elapsed_time'],
    "N_x": N_x,
    "N_v": N_v,
    "N_t": N_t,
    "X": X,
    "V": V,
    "T_final": T_final,
    "lambda_D": lambda_D,
    "num_gpus": num_gpus,
    "save_every": save_every,
    "N_batch": N_batch,
    "N_bin": N_bin,
    "gpu_memory_peak_gib": gpu_mem,
    "throughput_steps_per_sec": N_t / results['elapsed_time'],
}

config_file = f"data/landau_1d/{base_name}_config.json"
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)
print(f"\nConfig saved to {config_file}")

# Save distribution function
if 'f_history' in results:
    f_data = results['f_history']
else:
    f_data = results['f'][np.newaxis, :, :]

f_file = f"data/landau_1d/{base_name}_f.npy"
np.save(f_file, f_data.astype(np.float32))
print(f"Distribution f(x,v,t) saved to {f_file}")
print(f"  Shape: {f_data.shape}")
print(f"  Size: {f_data.nbytes / 1e9:.2f} GB")

# Save grid
grid_file = f"data/landau_1d/{base_name}_grid.npz"
np.savez(grid_file,
         x=results['x'],
         v=results['v'],
         times=results['times'],
         rho_history=results['rho_history'],
         conservation_history=results['conservation_history'])
print(f"Grid saved to {grid_file}")

print(f"\n{'='*70}")
print("ALL DONE!")
print(f"Base name for τ optimization: {base_name}")
print('='*70)
