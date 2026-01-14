# Neural ODE Usage Guide for Abiomed

This guide shows you how to use the Neural ODE module with YAML configuration files, similar to the mbpo_kde pattern.

## Quick Start

### 1. Training a Neural ODE Model

**Basic usage:**
```bash
python cormpo/neuralode_module/train_neuralode.py --config cormpo/config/neuralode/abiomed.yaml
```

This will:
- Load the Abiomed environment and data
- Train a Neural ODE model with default settings
- Save the model to the specified path
- Create an OOD detector with threshold

**Output files:**
- `cormpo/checkpoints/neuralode/model.pt` - Final model
- `cormpo/checkpoints/neuralode/best_model.pt` - Best validation model
- `/public/gormpo/models/abiomed/neuralode/abiomed_neuralode_model.pt` - OOD model
- `/public/gormpo/models/abiomed/neuralode/abiomed_neuralode_metadata.pkl` - Metadata

### 2. Testing OOD Detection

**Basic usage:**
```bash
python cormpo/neuralode_module/test_neuralode_ood.py --config cormpo/config/neuralode/test_ood_abiomed.yaml
```

This will:
- Load the trained model
- Test on multiple OOD distance levels
- Generate ROC curves and performance plots
- Save results to `cormpo/figures/neuralode/`

## Configuration Files

### Available Configs

1. **`cormpo/config/neuralode/abiomed.yaml`** - Default training config
   - Standard settings for Abiomed dataset
   - 200 epochs, batch size 512
   - Hidden dims: [512, 512]
   - Learning rate: 0.001

2. **`cormpo/config/neuralode/real.yaml`** - Production config
   - Higher precision for real clinical data
   - 300 epochs, batch size 256
   - Lower learning rate: 0.0005
   - Tighter ODE tolerances (rtol=1e-6, atol=1e-6)

3. **`cormpo/config/neuralode/test_ood_abiomed.yaml`** - OOD testing config
   - Specifies model path and test distances
   - Configure which OOD levels to test

### Customizing Configs

Create your own config file based on the templates:

```yaml
# my_custom_config.yaml

# Device settings
device: cuda
devid: 0

# Model architecture
hidden_dims: [1024, 1024]  # Larger model
activation: silu
time_dependent: true

# Training parameters
epochs: 500  # More epochs
batch_size: 256
lr: 0.0001  # Lower learning rate
weight_decay: 1.0e-4

# ODE solver settings
solver: dopri5
rtol: 1.0e-6
atol: 1.0e-6

# Save paths
out_dir: cormpo/checkpoints/my_neuralode
model_save_path: /path/to/save/my_model

# Abiomed settings
task: abiomed
model_path_wm: /abiomed/downsampled/models/10min_1hr_all_data_model.pth
data_path_wm: /abiomed/downsampled/10min_1hr_all_data.pkl
```

Then run:
```bash
python cormpo/neuralode_module/train_neuralode.py --config my_custom_config.yaml
```

## Command Line Overrides

You can override config settings from the command line:

```bash
# Override epochs and device
python cormpo/neuralode_module/train_neuralode.py \
    --config cormpo/config/neuralode/abiomed.yaml \
    --epochs 300 \
    --device cuda

# Verbose output
python cormpo/neuralode_module/train_neuralode.py \
    --config cormpo/config/neuralode/abiomed.yaml \
    --verbose
```

## Advanced Usage

### Training with Different Architectures

**Larger model:**
```yaml
hidden_dims: [1024, 1024]  # More capacity
```

**Deeper model:**
```yaml
hidden_dims: [512, 512, 512]  # More layers
```

**Different activation:**
```yaml
activation: tanh  # Instead of silu
```

### ODE Solver Settings

**Faster but less accurate:**
```yaml
solver: rk4
rtol: 1.0e-4
atol: 1.0e-4
```

**Slower but more accurate:**
```yaml
solver: dopri5
rtol: 1.0e-7
atol: 1.0e-7
```

### Data Splitting

Adjust train/val/test ratios:
```yaml
train_ratio: 0.7
val_ratio: 0.15
test_ratio: 0.15
```

### OOD Detection Settings

Control threshold sensitivity:
```yaml
anomaly_fraction: 0.01  # 1% of validation data marked as anomalies
```

More conservative (fewer false positives):
```yaml
anomaly_fraction: 0.005  # 0.5%
```

More aggressive (catch more anomalies):
```yaml
anomaly_fraction: 0.05  # 5%
```

## Complete Workflow Example

### Step 1: Train the model

```bash
python cormpo/neuralode_module/train_neuralode.py \
    --config cormpo/config/neuralode/abiomed.yaml \
    --verbose
```

**Expected output:**
```
Loading configuration from: cormpo/config/neuralode/abiomed.yaml
Using device: cuda
Creating Abiomed environment...
✓ Abiomed environment created successfully
Loading RL dataset for Neural ODE training...
Data shapes - Train: torch.Size([12000, 15]), Val: torch.Size([3000, 15]), Test: torch.Size([3000, 15])
Creating Neural ODE model...
Model created with 1,055,503 parameters

Training Neural ODE model...
  Step 100  Epoch 1/200  Loss 2.456789  Mean log-prob -2.456789
  ...
[Epoch 200/200] Train NLL: 1.234567, Val NLL: 1.345678

✓ Neural ODE training completed successfully!
✓ Model saved to: cormpo/checkpoints/neuralode/model.pt
✓ OOD model saved to: /public/gormpo/models/abiomed/neuralode/abiomed_neuralode_model.pt
```

### Step 2: Test OOD detection

Update the config with your model path:
```yaml
# cormpo/config/neuralode/test_ood_abiomed.yaml
model_path: cormpo/checkpoints/neuralode/best_model.pt  # or your model path
```

Run testing:
```bash
python cormpo/neuralode_module/test_neuralode_ood.py \
    --config cormpo/config/neuralode/test_ood_abiomed.yaml
```

**Expected output:**
```
Loading configuration from: cormpo/config/neuralode/test_ood_abiomed.yaml
Loading Neural ODE model from: cormpo/checkpoints/neuralode/best_model.pt
Model loaded successfully!

Testing on 5 distance levels: [0.5, 1, 2, 4, 8]
================================================================================

Testing at OOD distance: 0.5
--------------------------------------------------------------------------------
  Total samples: 2000 (ID: 1000, OOD: 1000)
  ID mean: -1.234 ± 0.456
  OOD mean: -2.345 ± 0.567
  Separation (ID - OOD): 1.111
  ROC AUC: 0.856
  Accuracy: 0.782

...

SUMMARY
================================================================================
Distance     ROC AUC      Separation      Accuracy
--------------------------------------------------------------------------------
0.5          0.8560       1.1110          0.7820
1            0.9123       1.5678          0.8456
2            0.9567       2.1234          0.9012
4            0.9812       2.8901          0.9456
8            0.9923       3.4567          0.9678

Saved performance plot to: cormpo/figures/neuralode/neuralode_ood_performance.png
Saved ROC curves to: cormpo/figures/neuralode/neuralode_roc_curves.png
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```yaml
batch_size: 128  # Instead of 512
```

Or reduce model size:
```yaml
hidden_dims: [256, 256]  # Instead of [512, 512]
```

### Slow Training

Use faster solver:
```yaml
solver: rk4  # Instead of dopri5
rtol: 1.0e-4
atol: 1.0e-4
```

Reduce checkpointing:
```yaml
checkpoint_every: 0  # Disable checkpointing
```

### Poor OOD Detection

Increase model capacity:
```yaml
hidden_dims: [1024, 1024]
epochs: 300
```

Adjust threshold:
```yaml
anomaly_fraction: 0.02  # Try different values
```

## Comparison with Other Methods

| Method | Config File | Training Command |
|--------|-------------|------------------|
| Neural ODE | `cormpo/config/neuralode/abiomed.yaml` | `python cormpo/neuralode_module/train_neuralode.py --config ...` |
| VAE | `cormpo/config/vae/abiomed.yaml` | Similar pattern |
| RealNVP | `cormpo/config/realnvp.yaml` | Similar pattern |
| KDE | `cormpo/config/kde/test_ood_abiomed.yaml` | Similar pattern |

All methods follow the same YAML config pattern for consistency!

## Performance Tips

1. **Use GPU**: Always set `device: cuda` if available
2. **Batch size**: Larger is faster, but may need more memory
3. **ODE tolerances**: Lower tolerances = more accurate but slower
4. **Checkpointing**: Disable during production runs for speed
5. **Validation frequency**: Can reduce validation to every N epochs

## Next Steps

- Experiment with different architectures
- Try different ODE solvers
- Adjust threshold for your use case
- Compare with VAE and RealNVP results
- Fine-tune hyperparameters for best performance
