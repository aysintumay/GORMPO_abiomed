# Neural ODE Module for Abiomed

This module provides Neural ODE (Continuous Normalizing Flow) implementations for density estimation and out-of-distribution (OOD) detection in the Abiomed environment.

## Overview

Neural ODEs use ordinary differential equations to model continuous transformations, enabling flexible density estimation. This implementation uses `torchdiffeq` for ODE integration and supports OOD detection through log-likelihood thresholding.

## Components

### Core Files

1. **neural_ode_density.py** - Core Neural ODE implementation
   - `ODEFunc`: Neural network defining the ODE dynamics
   - `ContinuousNormalizingFlow`: Main CNF model with log probability computation
   - `NPZTargetDataset`: Dataset loader for NPZ files
   - Training functionality

2. **neural_ode_ood.py** - OOD detection wrapper
   - `NeuralODEOOD`: Wrapper class for OOD detection
   - Threshold-based anomaly detection
   - Visualization utilities
   - Abiomed data loading utilities

3. **neural_ode_inference.py** - Model evaluation
   - Evaluation on test datasets
   - Percentile computation
   - Visualization of log-likelihood distributions

4. **test_neuralode_ood_levels.py** - OOD testing script (in parent directory)
   - Test models on different OOD distance levels
   - Compute ROC AUC and accuracy metrics
   - Generate performance plots

## Installation

The module requires the following dependencies:

```bash
pip install torch numpy scipy scikit-learn matplotlib seaborn torchdiffeq pyyaml
```

## Usage

### 1. Training a Neural ODE Model (Recommended: YAML Config)

**Using YAML configuration (Recommended):**

```bash
# Train with default Abiomed config
python cormpo/neuralode_module/train_neuralode.py --config cormpo/config/neuralode/abiomed.yaml

# Train with real data config (production)
python cormpo/neuralode_module/train_neuralode.py --config cormpo/config/neuralode/real.yaml

# Override config settings
python cormpo/neuralode_module/train_neuralode.py \
    --config cormpo/config/neuralode/abiomed.yaml \
    --epochs 300 \
    --device cuda
```

**Using Python API:**

```python
from cormpo.neuralode_module import train, TrainConfig

config = TrainConfig(
    npz_path="data/train_data.npz",
    out_dir="checkpoints/neuralode",
    batch_size=512,
    num_epochs=200,
    lr=1e-3,
    hidden_dims=(512, 512),
    activation="silu",
    device="cuda"
)

train(config)
```

**Direct training (for NPZ files):**

```bash
python -m cormpo.neuralode_module.neural_ode_density \
    --npz data/train_data.npz \
    --out checkpoints/neuralode \
    --epochs 200 \
    --batch 512 \
    --hidden-dims 512 512
```

### 2. OOD Detection

```python
from cormpo.neuralode_module import NeuralODEOOD, load_abiomed_data
from cormpo.neuralode_module.neural_ode_density import ODEFunc, ContinuousNormalizingFlow
import torch

# Load data
train_data, val_data, test_data, input_dim = load_abiomed_data(
    data_path="/abiomed/downsampled/replay_buffer.pkl",
    val_ratio=0.2,
    test_ratio=0.2
)

# Create model
device = "cuda" if torch.cuda.is_available() else "cpu"

odefunc = ODEFunc(
    dim=input_dim,
    hidden_dims=(512, 512),
    activation="silu",
    time_dependent=True
).to(device)

flow = ContinuousNormalizingFlow(
    func=odefunc,
    t0=0.0,
    t1=1.0,
    solver="dopri5",
    rtol=1e-5,
    atol=1e-5
).to(device)

# Load trained weights
checkpoint = torch.load("checkpoints/neuralode/model.pt")
flow.load_state_dict(checkpoint)

# Create OOD detector
ood_model = NeuralODEOOD(flow, device=device)

# Set threshold using validation data
ood_model.set_threshold(val_data, anomaly_fraction=0.01)

# Predict on test data
predictions = ood_model.predict(test_data)  # 1 for normal, -1 for anomaly
scores = ood_model.score_samples(test_data)  # log probabilities
```

### 3. Model Evaluation

```python
from cormpo.neuralode_module import evaluate, EvalConfig

config = EvalConfig(
    npz_path="data/test_data.npz",
    model_path="checkpoints/neuralode/model.pt",
    batch_size=512,
    hidden_dims=(512, 512),
    device="cuda",
    save_metrics="results/metrics.json",
    save_plot="figures/logp_distribution.png",
    percentile=1.0
)

evaluate(config)
```

### 4. Testing on OOD Levels

**Using YAML configuration (Recommended):**

```bash
# Test with config file
python cormpo/neuralode_module/test_neuralode_ood.py --config cormpo/config/neuralode/test_ood_abiomed.yaml
```

**Direct command line (legacy):**

```bash
python cormpo/test_neuralode_ood_levels.py \
    --model-path checkpoints/neuralode/model.pt \
    --dataset abiomed \
    --distances 0.5 1 2 4 8 \
    --base-path /abiomed/downsampled/ood_test \
    --input-dim 14 \
    --hidden-dims 512 512 \
    --device cuda \
    --save-dir cormpo/figures
```

## Model Architecture

### ODEFunc
- Input: State vector z and time t
- Hidden layers: Configurable (default: 2 layers of 512 units)
- Activation: SiLU (default) or Tanh
- Output: Time derivative dz/dt

### ContinuousNormalizingFlow
- Base distribution: Standard Normal N(0, I)
- ODE Solver: Dopri5 (adaptive Runge-Kutta)
- Integration: t ∈ [0, 1]
- Divergence: Computed via trace estimation

### NeuralODEOOD
- Threshold: Percentile-based (default: 1%)
- Anomaly score: Negative log-likelihood
- Prediction: Binary (normal/anomaly)

## Key Features

1. **Continuous Normalizing Flows**: Exact likelihood computation through ODE integration
2. **Flexible Architecture**: Configurable hidden dimensions and activation functions
3. **OOD Detection**: Threshold-based anomaly detection with ROC curve evaluation
4. **Visualization**: Built-in plotting for distributions and ROC curves
5. **Abiomed Integration**: Specialized data loading for Abiomed replay buffers

## Performance Notes

- Neural ODEs can be slower than other methods due to ODE integration
- GPU acceleration is recommended for training and inference
- Batch size affects ODE solver memory usage
- Consider using smaller batch sizes for score_samples() to avoid OOM

## File Structure

```
cormpo/neuralode_module/
├── __init__.py                  # Module exports
├── neural_ode_density.py        # Core Neural ODE implementation
├── neural_ode_ood.py            # OOD detection wrapper
├── neural_ode_inference.py      # Evaluation utilities
└── README.md                    # This file

cormpo/
└── test_neuralode_ood_levels.py # OOD testing script
```

## References

- Chen et al. (2018) - "Neural Ordinary Differential Equations"
- Grathwohl et al. (2018) - "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models"
- Original GORMPO implementation: `../GORMPO/neuralODE/`

## Adaptation from GORMPO

This module is adapted from the Neural ODE implementation in the `../GORMPO` folder, with modifications for:
- Abiomed environment compatibility
- Consistent interface with VAE and RealNVP modules
- Enhanced OOD testing capabilities
- Improved data loading utilities
