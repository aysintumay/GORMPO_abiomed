# Neural ODE Quick Start Guide

## TL;DR - Get Started in 2 Commands

### 1. Train a Neural ODE model
```bash
python cormpo/neuralode_module/train_neuralode.py --config cormpo/config/neuralode/abiomed.yaml
```

### 2. Test OOD detection
```bash
python cormpo/neuralode_module/test_neuralode_ood.py --config cormpo/config/neuralode/test_ood_abiomed.yaml
```

That's it! 🎉

---

## What You Just Did

**Training:** Created a Neural ODE (Continuous Normalizing Flow) model that learns the density of normal Abiomed data.

**Testing:** Evaluated how well the model detects out-of-distribution samples at different distance levels.

---

## Key Files

### Configuration Files
- `cormpo/config/neuralode/abiomed.yaml` - Standard training config
- `cormpo/config/neuralode/real.yaml` - Production config (higher precision)
- `cormpo/config/neuralode/test_ood_abiomed.yaml` - OOD testing config

### Training Scripts
- `cormpo/neuralode_module/train_neuralode.py` - Main training script (YAML-based)
- `cormpo/neuralode_module/neural_ode_density.py` - Core model + direct training

### Testing Scripts
- `cormpo/neuralode_module/test_neuralode_ood.py` - YAML-based OOD testing
- `cormpo/test_neuralode_ood_levels.py` - Legacy command-line testing

### Documentation
- `README.md` - Complete module documentation
- `USAGE_GUIDE.md` - Detailed usage examples
- `QUICK_START.md` - This file!

---

## Common Tasks

### Change training epochs
```bash
python cormpo/neuralode_module/train_neuralode.py \
    --config cormpo/config/neuralode/abiomed.yaml \
    --epochs 300
```

### Use CPU instead of GPU
```bash
python cormpo/neuralode_module/train_neuralode.py \
    --config cormpo/config/neuralode/abiomed.yaml \
    --device cpu
```

### Customize everything
Edit `cormpo/config/neuralode/abiomed.yaml` or create your own config file.

---

## Outputs

### Training produces:
- `cormpo/checkpoints/neuralode/model.pt` - Trained model
- `cormpo/checkpoints/neuralode/best_model.pt` - Best validation model
- `/public/gormpo/models/abiomed/neuralode/abiomed_neuralode_model.pt` - OOD model
- `/public/gormpo/models/abiomed/neuralode/abiomed_neuralode_metadata.pkl` - Metadata

### Testing produces:
- `cormpo/figures/neuralode/neuralode_ood_performance.png` - Performance plots
- `cormpo/figures/neuralode/neuralode_roc_curves.png` - ROC curves
- Console output with metrics (ROC AUC, accuracy, separation)

---

## Key Hyperparameters

Edit these in your config file:

```yaml
# Model size
hidden_dims: [512, 512]  # Larger = more capacity

# Training
epochs: 200
batch_size: 512
lr: 0.001

# ODE precision
solver: dopri5  # dopri5 (accurate) or rk4 (fast)
rtol: 1.0e-5   # Lower = more accurate, slower
atol: 1.0e-5

# OOD threshold
anomaly_fraction: 0.01  # 1% of validation data
```

---

## Architecture Summary

**Neural ODE** uses ODEs to transform a simple base distribution (Gaussian) into the complex data distribution.

**Key Features:**
- ✅ Exact likelihood computation (no approximation)
- ✅ Continuous transformations via ODEs
- ✅ Flexible architecture
- ⚠️ Slower than VAE/RealNVP due to ODE integration
- ⚠️ Requires GPU for reasonable training time

**When to use:**
- Need exact likelihoods
- Have GPU available
- Want mathematically principled density model
- Don't mind slower training

**When NOT to use:**
- Need fast inference
- Limited to CPU only
- Want approximate methods (use VAE)
- Need faster training (use RealNVP)

---

## Next Steps

1. ✅ Train your first model
2. ✅ Test OOD detection
3. 📊 Compare with VAE/RealNVP results
4. 🔧 Tune hyperparameters
5. 🚀 Deploy to production

For more details, see `USAGE_GUIDE.md` or `README.md`.
