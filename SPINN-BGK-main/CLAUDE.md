# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPINN-BGK implements Separable Physics-Informed Neural Networks for solving the Boltzmann-BGK equation. The method fuses SPINNs with Gaussian functions and relative loss to handle multi-scale particle density functions efficiently. Published in SIAM Journal on Scientific Computing (2025).

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
# JAX must be installed separately per official JAX installation guide
```

**Train model (3D smooth problem):**
```bash
python smooth_3d.py --Kn=0.01 --rank=256
```
- `Kn`: Knudsen number (collision frequency nu=1/Kn)
- `rank`: Tensor decomposition rank

**Multi-GPU training:**
```bash
python smooth_3d.py --Kn=0.01 --rank=256 --parallel=True
```
Uses `jax.pmap` for data parallelism: each GPU computes gradients on different domain samples, gradients are synchronized via `lax.pmean`.

**Evaluate errors against reference:**
```bash
python error_3d.py --problem="smooth" --Kn=1.0
```

## Architecture

### Core Components

The model uses Canonical Polyadic Decomposition (CPD) to represent the 7D distribution function f(t,x,y,z,vx,vy,vz) as a sum of separable terms:

```
f = Σ_k (weight_k × φ_t(t) × φ_x(x) × φ_y(y) × φ_z(z) × ψ_vx(vx) × ψ_vy(vy) × ψ_vz(vz))
```

**Class hierarchy:**
- `smooth_3d.py::spinn` → `src/x3v3.py::x3v3` → `src/base_v3.py::base_v3`
- Neural networks: SIREN (sine-activated MLP) in `src/nn.py`

**Key modules:**
- `src/base_v3.py`: Training loop, loss computation, domain sampling
- `src/x3v3.py`: 3D spatial + 3D velocity implementation, equilibrium/non-equilibrium decomposition
- `utils/transform.py`: Maxwellian computation, Gaussian functions, quadrature rules

### Design Patterns

- **Separable structure**: Each spatial/velocity dimension has its own MLP, enabling efficient moment computation via einsum
- **Gaussian augmentation**: Non-equilibrium part modulated by Gaussians for rapid Maxwellian-like decay
- **Relative loss**: Each loss term divided by (|prediction| + ε) to capture multi-scale features
- **JAX functional style**: Heavy use of `jit`, `grad`, `vmap` for performance

### Data Flow

1. Sample 7D domain points (t, x, y, z, vx, vy, vz)
2. Compute equilibrium f_eq from Maxwellian(ρ, u, T)
3. Compute non-equilibrium f_neq with Gaussian modulation
4. Evaluate PDE residual: f_t + v·∇f - ν(f_eq - f)
5. Apply relative loss weighting and optimize with Lion optimizer

## Key Dependencies

- **JAX**: Automatic differentiation, JIT compilation (install separately)
- **optax**: Lion optimizer with cosine decay schedule
- **equinox**: Neural network definitions
- **fire**: CLI argument parsing
