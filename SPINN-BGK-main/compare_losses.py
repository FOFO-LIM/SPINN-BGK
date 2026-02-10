import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Get date string for filename
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load SPINN config (ngrid=9, same as PINN)
with open('data/x3v3/smooth/spinn_Kn0.1_rank128_ngrid9_gpu4_20260101_165915_config.json', 'r') as f:
    spinn_config = json.load(f)

# Load PINN config
with open('data/x3v3/smooth/pinn_Kn0.1_w256_d5_gpu4_20260101_164656_config.json', 'r') as f:
    pinn_config = json.load(f)

# Extract loss histories
spinn_pde = spinn_config['pde_loss_history']
spinn_ic = spinn_config['ic_loss_history']
spinn_bc = spinn_config['bc_loss_history']

pinn_pde = pinn_config['pde_loss_history']
pinn_ic = pinn_config['ic_loss_history']
pinn_bc = pinn_config['bc_loss_history']

# X-axis: iterations (logged every 100 iterations)
spinn_iters = np.arange(len(spinn_pde)) * 100
pinn_iters = np.arange(len(pinn_pde)) * 100

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: PDE Loss
ax1 = axes[0]
ax1.semilogy(spinn_iters, spinn_pde, 'b-', linewidth=2, label=f'SPINN (final: {spinn_pde[-1]:.3e})')
ax1.semilogy(pinn_iters, pinn_pde, 'r-', linewidth=2, label=f'PINN (final: {pinn_pde[-1]:.3e})')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('PDE Loss')
ax1.set_title('PDE Loss Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: IC Loss
ax2 = axes[1]
ax2.semilogy(spinn_iters, spinn_ic, 'b-', linewidth=2, label=f'SPINN (final: {spinn_ic[-1]:.3e})')
ax2.semilogy(pinn_iters, pinn_ic, 'r-', linewidth=2, label=f'PINN (final: {pinn_ic[-1]:.3e})')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('IC Loss')
ax2.set_title('Initial Condition Loss Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: BC Loss
ax3 = axes[2]
ax3.semilogy(spinn_iters, spinn_bc, 'b-', linewidth=2, label=f'SPINN (final: {spinn_bc[-1]:.3e})')
ax3.semilogy(pinn_iters, pinn_bc, 'r-', linewidth=2, label=f'PINN (final: {pinn_bc[-1]:.3e})')
ax3.set_xlabel('Iterations')
ax3.set_ylabel('BC Loss')
ax3.set_title('Boundary Condition Loss Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add overall title with configuration info
fig.suptitle(f'SPINN vs PINN Loss Comparison (Kn=0.1, ngrid=9, 10000 iterations)\n'
             f'SPINN: rank=128 | PINN: width=256, depth=5 (mini-batch sampling)',
             fontsize=12, y=1.02)

plt.tight_layout()

# Save figure
output_file = f'figures/x3v3/smooth/spinn_vs_pinn_comparison_{date_str}.png'
plt.savefig(output_file, format='png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_file}")

# Also create individual plots for each loss type
for loss_name, spinn_loss, pinn_loss in [
    ('pde', spinn_pde, pinn_pde),
    ('ic', spinn_ic, pinn_ic),
    ('bc', spinn_bc, pinn_bc)
]:
    fig2, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(spinn_iters, spinn_loss, 'b-', linewidth=2, label=f'SPINN (final: {spinn_loss[-1]:.3e})')
    ax.semilogy(pinn_iters, pinn_loss, 'r-', linewidth=2, label=f'PINN (final: {pinn_loss[-1]:.3e})')
    ax.set_xlabel('Iterations', fontsize=12)
    ax.set_ylabel(f'{loss_name.upper()} Loss', fontsize=12)
    ax.set_title(f'{loss_name.upper()} Loss: SPINN vs PINN (Kn=0.1)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    individual_file = f'figures/x3v3/smooth/spinn_vs_pinn_{loss_name}_{date_str}.png'
    plt.savefig(individual_file, format='png', dpi=150, bbox_inches='tight')
    print(f"Saved: {individual_file}")
    plt.close(fig2)

plt.show()

# Print summary
print("\n" + "="*60)
print("LOSS COMPARISON SUMMARY")
print("="*60)
print(f"{'Loss Type':<15} {'SPINN Final':<15} {'PINN Final':<15} {'Ratio (PINN/SPINN)':<20}")
print("-"*60)
print(f"{'PDE':<15} {spinn_pde[-1]:<15.3e} {pinn_pde[-1]:<15.3e} {pinn_pde[-1]/spinn_pde[-1]:<20.1f}")
print(f"{'IC':<15} {spinn_ic[-1]:<15.3e} {pinn_ic[-1]:<15.3e} {pinn_ic[-1]/spinn_ic[-1]:<20.1f}")
print(f"{'BC':<15} {spinn_bc[-1]:<15.3e} {pinn_bc[-1]:<15.3e} {pinn_bc[-1]/spinn_bc[-1]:<20.1f}")
print("="*60)
