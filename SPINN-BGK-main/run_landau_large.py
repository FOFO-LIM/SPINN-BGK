#!/usr/bin/env python3
"""
Run large-scale 1D Boltzmann-Landau simulation with separate N_x, N_v, N_t.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Set JAX to use 4 GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import jax
print(f"JAX devices: {jax.devices()}")
print(f"Number of devices: {jax.local_device_count()}")

from landau_1d_numerical_jax import LandauSolver1D_JAX, get_gpu_memory_gib

# Parameters
N_x = 2**16  # 65536
N_v = 2**10  # 1024
N_t = 2**17  # 131072
num_gpus = 4

# Other parameters (you can modify these)
X = 0.5
V = 6.0
T_final = 0.1
lambda_D = 10.0
save_every = N_t // 100  # Save 100 snapshots

print(f"\n{'='*70}")
print("Large-scale 1D Boltzmann-Landau Simulation")
print('='*70)
print(f"N_x = 2^16 = {N_x}")
print(f"N_v = 2^10 = {N_v}")
print(f"N_t = 2^19 = {N_t}")
print(f"Total grid points per time step: {N_x * N_v:,} = {N_x * N_v / 1e6:.2f}M")
print(f"Using {num_gpus} GPUs")
print('='*70)

# Check if N_x is divisible by num_gpus for parallel execution
if N_x % num_gpus != 0:
    print(f"WARNING: N_x ({N_x}) is not divisible by num_gpus ({num_gpus})")
    print("Falling back to single GPU mode")
    use_parallel = False
else:
    use_parallel = True
    print(f"Spatial domain will be split: {N_x // num_gpus} x-points per GPU")

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

# Run simulation
print(f"\nStarting simulation (save_every={save_every})...")
if use_parallel and num_gpus > 1:
    results = solver.solve_parallel(save_every=save_every, num_devices=num_gpus, verbose=True)
else:
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
base_name = f"landau_large_Nx{N_x}_Nv{N_v}_Nt{N_t}_{timestamp}"

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
    "parallel": use_parallel,
    "gpu_memory_peak_gib": gpu_mem,
    "throughput_steps_per_sec": N_t / results['elapsed_time'],
    "final_rho_min": float(results['rho'].min()),
    "final_rho_max": float(results['rho'].max()),
    "final_T_min": float(results['T'].min()),
    "final_T_max": float(results['T'].max()),
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
np.save(f_file, f_data)
print(f"Distribution f(x,v,t) saved to {f_file}")
print(f"  Shape: {f_data.shape}")

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
print('='*70)
