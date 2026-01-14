"""
Example usage of the Neural ODE module for Abiomed OOD detection.

This script demonstrates:
1. Loading Abiomed data
2. Training a Neural ODE model
3. Setting up OOD detection
4. Evaluating on test data
"""

import torch
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cormpo.neuralode_module import (
    ODEFunc,
    ContinuousNormalizingFlow,
    NeuralODEOOD,
    load_abiomed_data,
    plot_likelihood_distributions,
)


def example_training():
    """Example: Train a Neural ODE model on Abiomed data."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Training Neural ODE Model")
    print("="*80)

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    print("\nLoading Abiomed data...")
    train_data, val_data, test_data, input_dim = load_abiomed_data(
        data_path="/abiomed/downsampled/replay_buffer.pkl",
        val_ratio=0.2,
        test_ratio=0.2
    )

    print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    print(f"Input dimension: {input_dim}")

    # Create model
    print("\nCreating Neural ODE model...")
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

    print(f"Model created with {sum(p.numel() for p in flow.parameters())} parameters")

    # Training loop (simplified)
    print("\nTraining model (demo - just a few iterations)...")
    optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-3)

    flow.train()
    train_data = train_data.to(device)

    for step in range(10):  # Just 10 steps for demo
        # Sample a batch
        batch_size = 128
        indices = torch.randperm(len(train_data))[:batch_size]
        batch = train_data[indices]

        # Compute loss
        log_px = flow.log_prob(batch)
        loss = -log_px.mean()

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}: Loss = {loss.item():.4f}, Mean log-prob = {log_px.mean().item():.4f}")

    print("\nTraining complete (demo)!")
    print("For full training, use: python -m cormpo.neuralode_module.neural_ode_density")

    return flow, train_data, val_data, test_data, device


def example_ood_detection(flow, train_data, val_data, test_data, device):
    """Example: OOD detection with Neural ODE."""
    print("\n" + "="*80)
    print("EXAMPLE 2: OOD Detection")
    print("="*80)

    # Create OOD detector
    print("\nCreating OOD detector...")
    ood_model = NeuralODEOOD(flow, device=device)

    # Set threshold using validation data
    print("\nSetting threshold using validation data...")
    ood_model.set_threshold(val_data, anomaly_fraction=0.01)

    # Get predictions on test data
    print("\nEvaluating on test data...")
    predictions = ood_model.predict(test_data)
    scores = ood_model.score_samples(test_data)

    n_anomalies = (predictions == -1).sum()
    print(f"Detected {n_anomalies}/{len(test_data)} anomalies ({n_anomalies/len(test_data)*100:.2f}%)")
    print(f"Mean log-likelihood: {scores.mean():.4f} ± {scores.std():.4f}")

    # Create OOD test data (synthetic - just for demo)
    print("\nCreating synthetic OOD data...")
    # Add noise to make it OOD
    ood_test_data = test_data[:100] + torch.randn_like(test_data[:100]) * 0.2
    ood_test_data = ood_test_data.to(device)

    # Evaluate on OOD data
    ood_predictions = ood_model.predict(ood_test_data)
    ood_scores = ood_model.score_samples(ood_test_data)

    n_ood_detected = (ood_predictions == -1).sum()
    print(f"Detected {n_ood_detected}/{len(ood_test_data)} anomalies in OOD data ({n_ood_detected/len(ood_test_data)*100:.2f}%)")
    print(f"OOD mean log-likelihood: {ood_scores.mean():.4f} ± {ood_scores.std():.4f}")

    # Plot distributions
    print("\nPlotting likelihood distributions...")
    plot_likelihood_distributions(
        model=ood_model,
        train_data=train_data[:1000],  # Subsample for speed
        val_data=val_data[:500],
        ood_data=ood_test_data,
        save_dir="cormpo/figures"
    )

    print("\nOOD detection complete!")


def example_save_load(flow, train_data, device):
    """Example: Save and load model."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Save and Load Model")
    print("="*80)

    # Create OOD model
    ood_model = NeuralODEOOD(flow, device=device)
    ood_model.set_threshold(train_data[:500], anomaly_fraction=0.01)

    # Save model
    save_path = "cormpo/checkpoints/neuralode_example"
    print(f"\nSaving model to {save_path}...")
    ood_model.save_model(save_path, train_data=train_data[:1000])

    # Load model
    print(f"\nLoading model from {save_path}...")
    loaded_dict = NeuralODEOOD.load_model(
        save_path=save_path,
        target_dim=train_data.shape[1],
        hidden_dims=(512, 512),
        activation="silu",
        device=device
    )

    loaded_model = loaded_dict['model']
    print(f"Model loaded! Threshold: {loaded_model.threshold:.4f}")

    # Test loaded model
    test_scores = loaded_model.score_samples(train_data[:10])
    print(f"Test scores: {test_scores.mean():.4f} ± {test_scores.std():.4f}")

    print("\nSave/Load complete!")


def main():
    parser = argparse.ArgumentParser(description='Neural ODE Example Usage')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training example (faster demo)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("Neural ODE Module - Example Usage")
    print("="*80)

    if not args.skip_training:
        # Example 1: Training
        flow, train_data, val_data, test_data, device = example_training()

        # Example 2: OOD Detection
        example_ood_detection(flow, train_data, val_data, test_data, device)

        # Example 3: Save/Load
        example_save_load(flow, train_data, device)
    else:
        print("\nSkipping training examples (use --skip-training=False to run)")
        print("\nFor full examples:")
        print("1. Training: python -m cormpo.neuralode_module.neural_ode_density --help")
        print("2. Evaluation: python -m cormpo.neuralode_module.neural_ode_inference --help")
        print("3. OOD Testing: python cormpo/test_neuralode_ood_levels.py --help")

    print("\n" + "="*80)
    print("Example usage complete!")
    print("="*80)


if __name__ == '__main__':
    main()
