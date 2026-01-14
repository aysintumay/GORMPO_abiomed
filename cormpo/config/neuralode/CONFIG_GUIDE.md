# Neural ODE Configuration Guide

This guide explains all the Neural ODE configuration files and how to use them.

## Configuration File Structure

### 1. Training Configs (in `cormpo/config/neuralode/`)

#### `abiomed.yaml` - Standard Training
Default configuration for training Neural ODE on Abiomed data.

**Use case:** Initial experiments, development, testing
**Training time:** ~2-3 hours on GPU
**Settings:** Balanced between speed and accuracy

```bash
python cormpo/neuralode_module/train_neuralode.py --config cormpo/config/neuralode/abiomed.yaml
```

**Key parameters:**
- Epochs: 200
- Batch size: 512
- Hidden dims: [512, 512]
- Learning rate: 0.001
- ODE solver: dopri5 (rtol=1e-5, atol=1e-5)

#### `real.yaml` - Production Training
High-precision configuration for production/real clinical data.

**Use case:** Final models, production deployment, publications
**Training time:** ~4-6 hours on GPU
**Settings:** Maximum accuracy, conservative thresholds

```bash
python cormpo/neuralode_module/train_neuralode.py --config cormpo/config/neuralode/real.yaml
```

**Key parameters:**
- Epochs: 300
- Batch size: 256 (smaller for better generalization)
- Hidden dims: [512, 512]
- Learning rate: 0.0005 (lower for stability)
- ODE solver: dopri5 (rtol=1e-6, atol=1e-6) - tighter tolerances
- Anomaly fraction: 0.005 (0.5% - more conservative)
- Early stopping: patience=5

#### `test_ood_abiomed.yaml` - OOD Testing
Configuration for testing trained models on OOD datasets.

**Use case:** Model evaluation, performance benchmarking

```bash
python cormpo/neuralode_module/test_neuralode_ood.py --config cormpo/config/neuralode/test_ood_abiomed.yaml
```

**Key parameters:**
- Model path: `/public/gormpo/models/abiomed/neuralODE/real_abiomed_neuralode_model.pt`
- Test distances: [0.5, 1, 2, 4, 8]
- Save directory: `cormpo/figures/neuralode`

### 2. MBPO Integration Config (in `cormpo/config/real/`)

#### `mbpo_neuralode.yaml` - Model-Based Policy Optimization
Configuration for using Neural ODE as OOD detector in MBPO.

**Use case:** Reinforcement learning with Neural ODE-based OOD penalties

```bash
python cormpo/mbpo_kde/mopo.py --config cormpo/config/real/mbpo_neuralode.yaml
```

**Key parameters:**
- Algorithm: mbpo_neuralode
- Reward penalty coefficient: 0.1
- Classifier model: `/public/gormpo/models/abiomed/neuralODE/real_abiomed_neuralode`
- Epochs: 200

### 3. Top-Level Config (in `cormpo/config/`)

#### `neuralode.yaml` - General Purpose
Comprehensive configuration covering all use cases.

**Use case:** Template for custom configs, general-purpose training

```bash
python cormpo/neuralode_module/train_neuralode.py --config cormpo/config/neuralode.yaml
```

## Configuration Comparison

| Config | Epochs | Batch | LR | Solver Tolerance | Use Case |
|--------|--------|-------|-----|------------------|----------|
| `abiomed.yaml` | 200 | 512 | 0.001 | 1e-5 | Development |
| `real.yaml` | 300 | 256 | 0.0005 | 1e-6 | Production |
| `neuralode.yaml` | 200 | 512 | 0.001 | 1e-5 | General |

## Complete Workflow

### 1. Development Phase

```bash
# Train with standard config
python cormpo/neuralode_module/train_neuralode.py \
    --config cormpo/config/neuralode/abiomed.yaml

# Test OOD detection
python cormpo/neuralode_module/test_neuralode_ood.py \
    --config cormpo/config/neuralode/test_ood_abiomed.yaml
```

### 2. Production Phase

```bash
# Train with production config
python cormpo/neuralode_module/train_neuralode.py \
    --config cormpo/config/neuralode/real.yaml

# Test thoroughly
python cormpo/neuralode_module/test_neuralode_ood.py \
    --config cormpo/config/neuralode/test_ood_abiomed.yaml
```

### 3. MBPO Integration

```bash
# Use Neural ODE in MBPO
python cormpo/mbpo_kde/mopo.py \
    --config cormpo/config/real/mbpo_neuralode.yaml
```

## Customizing Configs

### Create Custom Config

```yaml
# my_config.yaml - Example custom configuration

# Start with abiomed.yaml as template
device: cuda
hidden_dims: [1024, 1024]  # Larger model
epochs: 500                 # More training
lr: 0.0001                 # Lower learning rate

# All other settings...
```

### Command-Line Overrides

```bash
# Override specific settings
python cormpo/neuralode_module/train_neuralode.py \
    --config cormpo/config/neuralode/abiomed.yaml \
    --epochs 300 \
    --device cuda

# Multiple overrides
python cormpo/neuralode_module/train_neuralode.py \
    --config cormpo/config/neuralode/real.yaml \
    --epochs 500 \
    --verbose
```

## Config Hierarchy

1. **Base config file** - Defines all parameters
2. **Command-line arguments** - Override base config
3. **Environment defaults** - Fallback values

## All Config Files Summary

```
cormpo/config/
├── neuralode.yaml                      # Top-level general config
├── neuralode/
│   ├── abiomed.yaml                    # Standard training
│   ├── real.yaml                       # Production training
│   ├── test_ood_abiomed.yaml          # OOD testing
│   └── CONFIG_GUIDE.md                 # This file
└── real/
    ├── mbpo_neuralode.yaml             # MBPO integration
    ├── mbpo_kde.yaml                   # (KDE for comparison)
    ├── mbpo_vae.yaml                   # (VAE for comparison)
    └── mbpo_realnvp.yaml               # (RealNVP for comparison)
```

## Matching Other Methods

Neural ODE configs follow the same pattern as other methods:

| Method | Training Config | MBPO Config |
|--------|----------------|-------------|
| KDE | `cormpo/config/kde/test_ood_abiomed.yaml` | `cormpo/config/real/mbpo_kde.yaml` |
| VAE | `cormpo/config/vae/abiomed.yaml` | `cormpo/config/real/mbpo_vae.yaml` |
| RealNVP | `cormpo/config/realnvp.yaml` | `cormpo/config/real/mbpo_realnvp.yaml` |
| **Neural ODE** | `cormpo/config/neuralode/abiomed.yaml` | `cormpo/config/real/mbpo_neuralode.yaml` |

## Quick Reference

**Train (development):**
```bash
python cormpo/neuralode_module/train_neuralode.py --config cormpo/config/neuralode/abiomed.yaml
```

**Train (production):**
```bash
python cormpo/neuralode_module/train_neuralode.py --config cormpo/config/neuralode/real.yaml
```

**Test OOD:**
```bash
python cormpo/neuralode_module/test_neuralode_ood.py --config cormpo/config/neuralode/test_ood_abiomed.yaml
```

**MBPO with Neural ODE:**
```bash
python cormpo/mbpo_kde/mopo.py --config cormpo/config/real/mbpo_neuralode.yaml
```
