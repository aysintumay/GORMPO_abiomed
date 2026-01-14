# Diffusion Model for OOD Detection - Abiomed

This module provides a complete implementation of Diffusion Models (DDIM/DDPM) for Out-of-Distribution (OOD) detection in the Abiomed environment. It uses ELBO (Evidence Lower Bound) to compute log-likelihoods for density estimation and anomaly detection.

## Overview

The diffusion module is adapted from the GORMPO diffusion implementation and follows the same architecture pattern as the Neural ODE, VAE, RealNVP, and KDE modules in this project.

### Key Features

- **DDIM/DDPM Diffusion Models**: Unconditional epsilon prediction networks with sinusoidal time embeddings
- **ELBO-based Log-Likelihood**: Uses Evidence Lower Bound for density estimation
- **OOD Detection**: Threshold-based anomaly detection using log-likelihood percentiles
- **YAML Configuration**: Centralized config system matching the mbpo_kde pattern
- **Abiomed Integration**: Direct integration with Abiomed world model data

## OOD Distance Levels Testing

The diffusion module supports testing OOD detection at different distance levels, similar to KDE and RealNVP modules. This allows you to evaluate how well the model distinguishes between in-distribution (ID) and out-of-distribution (OOD) samples at varying levels of distributional shift.

**Distance Levels for Abiomed:** 0.5, 1, 2, 4, 8
- **Lower distances** (0.5, 1): Small distributional shifts - harder to detect
- **Medium distances** (2, 4): Moderate distributional shifts
- **Higher distances** (8): Large distributional shifts - easier to detect

**Test Data Format:**
- Data files: `/abiomed/downsampled/ood_test/ood-distance-{distance}.pkl`
- Each file contains: first half = ID samples, second half = OOD samples
- Metrics computed: ROC AUC, accuracy, mean log-likelihood for ID/OOD

**Output Visualizations:**
1. `ood_distance_summary.png` - 4-panel summary with metrics vs distance
2. `roc_curves.png` - ROC curves for each distance level
3. `log_likelihood_distributions.png` - ELBO distributions for ID vs OOD at each distance

## Architecture

The module consists of five main components:

1. **diffusion_density.py**: Core diffusion models and ELBO computation
   - `UnconditionalEpsilonMLP`: MLP-based epsilon predictor
   - `UnconditionalEpsilonTransformer`: Transformer-based epsilon predictor
   - `log_prob_elbo()`: ELBO computation for log-likelihood
   - `SinusoidalTimeEmbedding`: Time encoding for diffusion steps

2. **diffusion_ood.py**: OOD detection wrapper
   - `DiffusionOOD`: Main class for OOD detection
   - `score_samples()`: Compute log probabilities
   - `set_threshold()`: Set anomaly threshold from validation data
   - `predict()`: Predict anomalies based on threshold

3. **train_diffusion.py**: YAML-based training script
   - Loads Abiomed RL data (next_observations + actions)
   - Trains diffusion model with MSE loss
   - Supports both MLP and Transformer architectures
   - Early stopping and checkpointing

4. **test_diffusion_ood.py**: YAML-based OOD testing script
   - Evaluates OOD detection performance
   - Computes ROC curves and metrics
   - Generates visualization plots

5. **test_diffusion_ood_levels.py**: OOD distance levels testing script
   - Tests model at multiple OOD distance levels
   - Generates comprehensive comparison plots
   - Compatible with Abiomed and D4RL datasets

## Installation

### Requirements

```bash
# Core dependencies (already in requirements.txt)
torch>=1.10.0
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
pyyaml>=5.4.0

# Diffusion-specific
diffusers>=0.21.0  # For DDIM/DDPM schedulers
```

Add to `requirements.txt`:
```
diffusers>=0.21.0
```

Install:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Training a Diffusion Model

```bash
# Using real Abiomed data configuration
python cormpo/diffusion_module/train_diffusion.py --config cormpo/config/diffusion/real.yaml

# Using standard configuration
python cormpo/diffusion_module/train_diffusion.py --config cormpo/config/diffusion/abiomed.yaml
```

### 2. Testing OOD Detection

```bash
# Test OOD detection performance
python cormpo/diffusion_module/test_diffusion_ood.py --config cormpo/config/diffusion/test_ood_abiomed.yaml
```

### 2b. Testing OOD Detection at Different Distance Levels

```bash
# Test OOD detection at different distance levels (0.5, 1, 2, 4, 8)
CUDA_VISIBLE_DEVICES=0 python cormpo/diffusion_module/test_diffusion_ood_levels.py \
    --model_dir checkpoints/diffusion \
    --dataset_name abiomed \
    --distances 0.5 1 2 4 8 \
    --base_path /abiomed/downsampled/ood_test \
    --device cuda \
    --batch_size 100 \
    --num_inference_steps 20
```

### 3. Using with MBPO

```bash
# Run MBPO with Diffusion OOD detection
python cormpo/mbpo_kde/mopo.py --config cormpo/config/real/mbpo_diffusion.yaml
```

## Configuration Files

### Training Configurations

**cormpo/config/diffusion/abiomed.yaml** - Standard training:
- 50 epochs, batch size 256
- MLP architecture with 512 hidden units
- 1000 diffusion timesteps
- 1% anomaly fraction

**cormpo/config/diffusion/real.yaml** - Production settings:
- 30 epochs, batch size 128 (conservative)
- Lower learning rate (1e-4) with dropout (0.1)
- 0.5% anomaly fraction (more conservative)
- Early stopping after 5 epochs

### Testing Configuration

**cormpo/config/diffusion/test_ood_abiomed.yaml**:
- Batch size 100 (avoids OOM)
- 0.5% anomaly fraction
- Optional `num_inference_steps` for faster approximation

### MBPO Integration

**cormpo/config/real/mbpo_diffusion.yaml**:
- Reward penalty coefficient: 0.1
- Fast ELBO with 50 inference steps
- SAC policy parameters
- Rollout length: 5

## Model Architecture

### MLP Architecture (Default)

```
Input: x_t (noisy sample) + t (timestep)
  ↓
SinusoidalTimeEmbedding(t) → time_embed
  ↓
Concat(x_t, time_embed)
  ↓
Linear(input_dim + time_embed_dim, hidden_dim) → SiLU → [Dropout]
  ↓
Linear(hidden_dim, hidden_dim) → SiLU → [Dropout]  (×3 layers)
  ↓
Linear(hidden_dim, target_dim) → epsilon prediction
```

### Transformer Architecture (Optional)

```
Input: x_t (noisy sample) + t (timestep)
  ↓
x_t → Reshape to sequence [B, L, 1] → Linear(1, d_model)
  ↓
SinusoidalTimeEmbedding(t) → Linear(time_embed, d_model)
  ↓
Add positional embeddings
  ↓
TransformerEncoder (4 layers, 8 heads)
  ↓
Linear(d_model, 1) → Flatten → epsilon prediction
```

## ELBO Computation

The log-likelihood is computed using the Evidence Lower Bound (ELBO):

```
log p(x0) >= ELBO = E_q[log p(x0|x1)] - ∑_t KL(q(x_{t-1}|x_t,x0) || p_θ(x_{t-1}|x_t)) - KL(q(x_T|x0) || p(x_T))
```

Key properties:
- **Lower bound** on true log-likelihood
- **Monte Carlo approximation** with single trajectory
- **KL divergence** terms for each timestep
- **Reconstruction term** at t=1

Speed optimization:
- Set `num_inference_steps` (e.g., 50) to use fewer timesteps
- Trades off accuracy for speed
- Full computation uses all 1000 timesteps

## Data Format

### Input Data Structure

The diffusion model expects concatenated `next_observations + actions`:

```python
# From Abiomed world model
train_next_obs = train_dataset.labels  # Shape: [N, obs_dim]
train_actions = train_dataset.pl        # Shape: [N, 1]
train_data = torch.cat([train_next_obs, train_actions], dim=1)  # Shape: [N, 15]
```

For Abiomed:
- `obs_dim`: 14 (physiological features)
- `action_dim`: 1 (pump level)
- `input_dim`: 15 total

## Output Files

### Training Outputs

```
checkpoints/diffusion/
├── checkpoint.pt              # Best model checkpoint
├── checkpoint_epoch_N.pt      # Periodic checkpoints
├── model.pt                   # Final model weights
└── scheduler/                 # Diffusion scheduler config
    ├── scheduler_config.json
    └── ...

/public/gormpo/models/abiomed/diffusion/
└── diffusion_model.pt         # Production model
```

### Testing Outputs

```
figures/diffusion_ood/
├── diffusion_likelihood_distribution.png  # Log-likelihood histogram with KDE
├── diffusion_roc_curve.png               # ROC curve for OOD detection
├── diffusion_train_distribution.png      # Train/val distributions
└── diffusion_ood_distribution.png        # OOD vs validation distributions

diffusion_metrics.json                     # Numerical metrics
```

## Performance Considerations

### Memory Usage

Diffusion models require gradients during ELBO computation:

```python
# ELBO requires forward pass through all timesteps
# Memory scales with: batch_size × num_timesteps × model_size

# Recommended settings:
batch_size: 100-256         # For training
batch_size: 100             # For OOD testing
num_inference_steps: 50     # For fast approximation (vs 1000 full)
```

### Speed Optimization

1. **Fast ELBO**: Use `num_inference_steps=50` instead of full 1000
2. **Smaller batches**: Reduce batch size during evaluation
3. **CUDA caching**: Periodic `torch.cuda.empty_cache()` calls
4. **Deterministic DDIM**: Use `ddim_eta=0.0` for faster sampling

### Comparison with Other Methods

| Method      | Speed  | Accuracy | Memory |
|-------------|--------|----------|--------|
| KDE         | Fast   | Good     | Low    |
| RealNVP     | Fast   | Good     | Medium |
| VAE         | Fast   | Good     | Medium |
| Neural ODE  | Slow   | Best     | High   |
| **Diffusion** | **Medium** | **Best** | **High** |

## API Reference

### DiffusionOOD Class

```python
from diffusion_module.diffusion_ood import DiffusionOOD

# Initialize
ood_model = DiffusionOOD(
    model=trained_model,
    scheduler=ddim_scheduler,
    device='cuda',
    num_inference_steps=50  # Optional: faster approximation
)

# Set threshold from validation data
threshold = ood_model.set_threshold(
    val_data=val_tensor,
    anomaly_fraction=0.005,  # 0.5%
    batch_size=100
)

# Score samples (compute log probabilities)
log_probs = ood_model.score_samples(
    x=test_data,
    batch_size=100
)

# Predict anomalies
predictions = ood_model.predict(
    x=test_data,
    batch_size=100
)  # Returns: 1 for normal, -1 for anomaly

# Evaluate OOD detection
results = ood_model.evaluate_anomaly_detection(
    normal_data=test_data,
    anomaly_data=ood_data,
    plot=True,
    save_path='roc_curve.png'
)
```

### Training Function

```python
from diffusion_module.train_diffusion import train, TrainConfig

config = TrainConfig(
    task='abiomed',
    model_type='mlp',
    epochs=50,
    batch_size=256,
    lr=2e-4,
    # ... other params
)

train(config)
```

## Examples

### Example 1: Train and Test

```bash
# Step 1: Train diffusion model
python cormpo/diffusion_module/train_diffusion.py \
    --config cormpo/config/diffusion/real.yaml

# Step 2: Test OOD detection
python cormpo/diffusion_module/test_diffusion_ood.py \
    --config cormpo/config/diffusion/test_ood_abiomed.yaml
```

### Example 2: Custom Configuration

```yaml
# my_config.yaml
device: cuda
devid: 1
epochs: 100
batch_size: 128
lr: 1e-4
hidden_dim: 1024
model_type: mlp
anomaly_fraction: 0.01

task: abiomed
model_name: 10min_1hr_all_data
model_path_wm: /path/to/world_model.pth
data_path_wm: /path/to/data.pkl
```

```bash
python cormpo/diffusion_module/train_diffusion.py --config my_config.yaml
```

### Example 3: Testing OOD Distance Levels

```bash
# Test on Abiomed OOD distance levels
CUDA_VISIBLE_DEVICES=0 python cormpo/diffusion_module/test_diffusion_ood_levels.py \
    --model_dir checkpoints/diffusion \
    --dataset_name abiomed \
    --distances 0.5 1 2 4 8 \
    --base_path /abiomed/downsampled/ood_test \
    --device cuda \
    --devid 0 \
    --batch_size 100 \
    --num_inference_steps 20 \
    --save_dir figures/diffusion_ood_levels

# Output:
# - figures/diffusion_ood_levels/abiomed/ood_distance_summary.png
# - figures/diffusion_ood_levels/abiomed/roc_curves.png
# - figures/diffusion_ood_levels/abiomed/log_likelihood_distributions.png
```

### Example 4: Programmatic Usage

```python
import torch
from diffusion_module.diffusion_density import UnconditionalEpsilonMLP
from diffusion_module.diffusion_ood import DiffusionOOD
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# Load trained model
ckpt = torch.load('checkpoints/diffusion/checkpoint.pt')
model = UnconditionalEpsilonMLP(
    target_dim=ckpt['target_dim'],
    hidden_dim=512,
    time_embed_dim=128,
    num_hidden_layers=3
)
model.load_state_dict(ckpt['model_state_dict'])

# Load scheduler
scheduler = DDIMScheduler.from_pretrained('checkpoints/diffusion/scheduler')

# Create OOD detector
ood = DiffusionOOD(model, scheduler, device='cuda', num_inference_steps=50)

# Set threshold
ood.set_threshold(val_data, anomaly_fraction=0.005)

# Detect anomalies
predictions = ood.predict(test_data)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Use `num_inference_steps=50` for faster ELBO
   - Clear cache: `torch.cuda.empty_cache()`

2. **Slow Training**
   - Use MLP instead of Transformer
   - Reduce `num_train_timesteps` to 500
   - Enable early stopping with `patience=5`

3. **Poor OOD Detection**
   - Train longer (increase `epochs`)
   - Adjust `anomaly_fraction` (try 0.01 or 0.005)
   - Use more data (increase `train_ratio`)

4. **Import Errors**
   - Ensure `abiomed_env` is in PYTHONPATH
   - Install diffusers: `pip install diffusers`
   - Check YAML syntax: `python -m yaml config.yaml`

## Citation

This implementation is adapted from:
- GORMPO: "Goal-Oriented Offline Reinforcement Learning with Model Predictive Optimization"
- Diffusion Models: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- DDIM: "Denoising Diffusion Implicit Models" (Song et al., 2021)

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. NeurIPS.
- Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. ICLR.
- Neural ODE module: `cormpo/neuralode_module/`
- RealNVP module: `cormpo/realnvp_module/`
- VAE module: `cormpo/vae_module/`
- KDE classifier: `cormpo/mbpo_kde/`
