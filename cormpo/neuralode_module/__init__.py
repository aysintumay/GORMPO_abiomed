"""
Neural ODE module for density estimation and OOD detection.

This module provides Neural ODE (Continuous Normalizing Flow) implementations
for density estimation and out-of-distribution detection in the Abiomed environment.

Main scripts:
- train_neuralode.py: Train Neural ODE with YAML configs
- test_neuralode_ood.py: Test OOD detection with YAML configs
"""

from .neural_ode_density import (
    ODEFunc,
    ContinuousNormalizingFlow,
    NPZTargetDataset,
    TrainConfig,
    train,
)

from .neural_ode_ood import (
    NeuralODEOOD,
    plot_likelihood_distributions,
    plot_tsne,
    load_abiomed_data,
)

from .neural_ode_inference import (
    EvalConfig,
    evaluate,
    load_flow,
)

__all__ = [
    # Core components
    'ODEFunc',
    'ContinuousNormalizingFlow',
    'NPZTargetDataset',

    # Training
    'TrainConfig',
    'train',

    # OOD detection
    'NeuralODEOOD',
    'plot_likelihood_distributions',
    'plot_tsne',
    'load_abiomed_data',

    # Evaluation
    'EvalConfig',
    'evaluate',
    'load_flow',
]
