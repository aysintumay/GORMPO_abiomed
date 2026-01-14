"""
Test Neural ODE model on different OOD levels using YAML configuration.

This script supports testing Neural ODE models on OOD datasets using configuration files.

Usage:
    python cormpo/neuralode_module/test_neuralode_ood.py --config cormpo/config/neuralode/test_ood_abiomed.yaml
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
import yaml
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cormpo.neuralode_module import (
    ODEFunc,
    ContinuousNormalizingFlow,
    NeuralODEOOD,
)


def load_ood_test_data(dataset_name, distance, base_path='/abiomed/downsampled/ood_test'):
    """Load OOD test data from pickle file."""
    distance_str = str(int(distance)) if isinstance(distance, int) else str(distance)

    if 'abiomed' in dataset_name.lower() or base_path == '/abiomed/downsampled/ood_test':
        file_path = os.path.join(base_path, f'ood-distance-{distance_str}.pkl')
    else:
        file_path = os.path.join(base_path, dataset_name, f'ood-distance-{distance_str}.pkl')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test data file not found: {file_path}")

    print(f"Loading test data from: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        observations = data['observations']
        actions = data['actions']

        if isinstance(observations, torch.Tensor):
            observations = observations.cpu().numpy()
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        data = np.concatenate([observations, actions], axis=1)
    elif isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    return data


def evaluate_ood_at_distance(model, dataset_name, distance, base_path, device):
    """Evaluate Neural ODE model on OOD test data at a specific distance level."""
    test_data = load_ood_test_data(dataset_name, distance, base_path)

    n_samples = len(test_data)
    half_point = n_samples // 2

    labels = np.zeros(n_samples, dtype=int)
    labels[half_point:] = 1

    print(f"  Total samples: {n_samples} (ID: {half_point}, OOD: {n_samples - half_point})")

    test_data_tensor = torch.FloatTensor(test_data).to(device)

    model.flow.eval()
    with torch.no_grad():
        log_probs = model.score_samples(test_data_tensor)

    if isinstance(log_probs, torch.Tensor):
        log_probs = log_probs.cpu().numpy()

    id_mask = labels == 0
    ood_mask = labels == 1

    id_log_probs = log_probs[id_mask]
    ood_log_probs = log_probs[ood_mask]

    mean_id = id_log_probs.mean()
    std_id = id_log_probs.std()
    mean_ood = ood_log_probs.mean()
    std_ood = ood_log_probs.std()

    anomaly_scores = -log_probs
    roc_auc = roc_auc_score(labels, anomaly_scores)

    fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)

    accuracy = None
    if model.threshold is not None:
        predictions = (log_probs < model.threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)

    print(f"  ID mean: {mean_id:.4f} ± {std_id:.4f}")
    print(f"  OOD mean: {mean_ood:.4f} ± {std_ood:.4f}")
    print(f"  Separation (ID - OOD): {mean_id - mean_ood:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    if accuracy is not None:
        print(f"  Accuracy: {accuracy:.4f}")

    return {
        'distance': distance,
        'mean_id': mean_id,
        'std_id': std_id,
        'mean_ood': mean_ood,
        'std_ood': std_ood,
        'separation': mean_id - mean_ood,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'fpr': fpr,
        'tpr': tpr,
    }


def plot_ood_results(results, save_path):
    """Plot OOD detection performance across different distance levels."""
    distances = [r['distance'] for r in results]
    roc_aucs = [r['roc_auc'] for r in results]
    separations = [r['separation'] for r in results]
    accuracies = [r['accuracy'] for r in results if r['accuracy'] is not None]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(distances, roc_aucs, marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('OOD Distance', fontsize=12)
    axes[0].set_ylabel('ROC AUC', fontsize=12)
    axes[0].set_title('ROC AUC vs OOD Distance', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    axes[1].plot(distances, separations, marker='s', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('OOD Distance', fontsize=12)
    axes[1].set_ylabel('ID-OOD Separation', fontsize=12)
    axes[1].set_title('Log-Likelihood Separation vs OOD Distance', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    if accuracies:
        axes[2].plot(distances[:len(accuracies)], accuracies, marker='^', linewidth=2, markersize=8, color='red')
        axes[2].set_xlabel('OOD Distance', fontsize=12)
        axes[2].set_ylabel('Accuracy', fontsize=12)
        axes[2].set_title('Detection Accuracy vs OOD Distance', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1.05])
    else:
        axes[2].text(0.5, 0.5, 'No accuracy data\n(threshold not set)',
                    ha='center', va='center', fontsize=12, transform=axes[2].transAxes)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved performance plot to: {save_path}")
    plt.close()


def plot_roc_curves(results, save_path):
    """Plot ROC curves for all distance levels."""
    plt.figure(figsize=(10, 8))

    for r in results:
        plt.plot(r['fpr'], r['tpr'], linewidth=2,
                label=f"Distance {r['distance']} (AUC={r['roc_auc']:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Neural ODE OOD Detection', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curves to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test Neural ODE on OOD datasets')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Configuration loaded successfully")

    # Extract settings
    model_path = config['model_path']
    dataset_name = config.get('dataset_name', 'abiomed')
    distances = config.get('distances', [0.5, 1, 2, 4, 8])
    base_path = config.get('base_path', '/abiomed/downsampled/ood_test')
    save_dir = config.get('save_dir', 'cormpo/figures/neuralode')
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Model architecture settings
    input_dim = config['input_dim']
    hidden_dims = tuple(config.get('hidden_dims', [512, 512]))
    activation = config.get('activation', 'silu')
    solver = config.get('solver', 'dopri5')

    print(f"\nLoading Neural ODE model from: {model_path}")
    print(f"Device: {device}")
    print(f"Input dimension: {input_dim}")

    # Create model
    odefunc = ODEFunc(
        dim=input_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        time_dependent=config.get('time_dependent', True)
    ).to(device)

    flow = ContinuousNormalizingFlow(
        func=odefunc,
        t0=config.get('t0', 0.0),
        t1=config.get('t1', 1.0),
        solver=solver,
        rtol=config.get('rtol', 1e-5),
        atol=config.get('atol', 1e-5)
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        flow.load_state_dict(checkpoint['model_state_dict'])
    else:
        flow.load_state_dict(checkpoint)

    flow.eval()
    print("Model loaded successfully!")

    # Create OOD wrapper
    model = NeuralODEOOD(flow, device=device)

    # Load threshold if available
    if isinstance(checkpoint, dict) and 'threshold' in checkpoint:
        model.threshold = checkpoint['threshold']
        print(f"Loaded threshold: {model.threshold}")

    # Test on all distance levels
    results = []
    print(f"\nTesting on {len(distances)} distance levels: {distances}")
    print("=" * 80)

    for distance in distances:
        print(f"\nTesting at OOD distance: {distance}")
        print("-" * 80)
        try:
            result = evaluate_ood_at_distance(
                model=model,
                dataset_name=dataset_name,
                distance=distance,
                base_path=base_path,
                device=device
            )
            results.append(result)
        except FileNotFoundError as e:
            print(f"Skipping distance {distance}: {e}")
            continue

    if not results:
        print("\nNo results to plot. Please check the dataset paths.")
        return

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Distance':<12} {'ROC AUC':<12} {'Separation':<15} {'Accuracy':<12}")
    print("-" * 80)
    for r in results:
        acc_str = f"{r['accuracy']:.4f}" if r['accuracy'] is not None else "N/A"
        print(f"{r['distance']:<12} {r['roc_auc']:<12.4f} {r['separation']:<15.4f} {acc_str:<12}")

    # Plot results
    print("\nGenerating plots...")
    os.makedirs(save_dir, exist_ok=True)
    plot_ood_results(results, save_path=os.path.join(save_dir, 'neuralode_ood_performance.png'))
    plot_roc_curves(results, save_path=os.path.join(save_dir, 'neuralode_roc_curves.png'))

    print("\nDone!")


if __name__ == '__main__':
    main()
