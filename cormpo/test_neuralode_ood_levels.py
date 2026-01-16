"""
Test Neural ODE model on different OOD levels using pre-generated OOD test datasets.
This script evaluates the model's ability to detect OOD samples at different distances.
For Abiomed, the test datasets are loaded from /abiomed/downsampled/ood_test/ood-distance-{distance}.pkl
where the first half is ID (in-distribution) and the second half is OOD (out-of-distribution).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
import json
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

# Add parent directories to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import pickle

# Import Neural ODE model - use relative imports
from neuralode_module.neural_ode_ood import NeuralODEOOD
from neuralode_module.neural_ode_density import ODEFunc, ContinuousNormalizingFlow


def load_ood_test_data(dataset_name, distance, base_path='/abiomed/downsampled/ood_test'):
    """
    Load OOD test data from pickle file.

    Args:
        dataset_name: Name of the dataset (e.g., 'abiomed' or 'halfcheetah-medium-v2')
        distance: OOD distance level (e.g., 0.5, 1, 2, 4, 8 for Abiomed)
        base_path: Base directory containing OOD test datasets

    Returns:
        Numpy array of test data where first half is ID and second half is OOD
    """
    # Format distance value - preserve int/float type
    distance_str = str(int(distance)) if isinstance(distance, int) else str(distance)

    # For Abiomed, files are directly in base_path, for D4RL they're in dataset_name subdirectory
    if 'abiomed' in dataset_name.lower() or base_path == '/abiomed/downsampled/ood_test':
        file_path = os.path.join(base_path, f'ood-distance-{distance_str}.pkl')
    else:
        file_path = os.path.join(base_path, dataset_name, f'ood-distance-{distance_str}.pkl')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test data file not found: {file_path}")

    print(f"Loading test data from: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # If data is a dictionary, extract observations only (Neural ODE was trained on obs only)
    if isinstance(data, dict):
        observations = data['observations']

        # Convert to numpy if needed
        if isinstance(observations, torch.Tensor):
            observations = observations.cpu().numpy()

        # Use only observations (Neural ODE model was trained on obs only, not obs+actions)
        data = observations
    # Convert to numpy if needed
    elif isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    return data


def evaluate_ood_at_distance(model, dataset_name, distance, base_path='/abiomed/downsampled/ood_test', device='cpu', mean=None, std=None, batch_size=16):
    """
    Evaluate Neural ODE model on OOD test data at a specific distance level.

    Args:
        model: Trained NeuralODEOOD model
        dataset_name: Name of the dataset
        distance: OOD distance level
        base_path: Base directory containing OOD test datasets
        device: Device to use
        mean: Mean for normalization (if None, no normalization)
        std: Standard deviation for normalization (if None, no normalization)
        batch_size: Batch size for processing to avoid OOM

    Returns:
        Dictionary with evaluation metrics
    """
    # Load test data
    test_data = load_ood_test_data(dataset_name, distance, base_path)

    # First half is ID (label 0), second half is OOD (label 1)
    n_samples = len(test_data)
    half_point = n_samples // 2

    # Create labels
    labels = np.zeros(n_samples, dtype=int)
    labels[half_point:] = 1  # Second half is OOD

    print(f"  Total samples: {n_samples} (ID: {half_point}, OOD: {n_samples - half_point})")

    # Convert to tensor
    test_data_tensor = torch.FloatTensor(test_data)

    # Get log-likelihood scores in batches to avoid OOM
    model.flow.eval()
    log_probs_list = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = test_data_tensor[i:i+batch_size].to(device)
            batch_log_probs = model.score_samples(batch)
            log_probs_list.append(batch_log_probs)

            # Clear GPU cache after each batch
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

    log_probs = np.concatenate(log_probs_list)

    # Convert to numpy if needed
    if isinstance(log_probs, torch.Tensor):
        log_probs = log_probs.cpu().numpy()

    # Calculate overall metrics
    mean_log_prob = log_probs.mean()
    std_log_prob = log_probs.std()

    # Calculate metrics for ID and OOD separately
    id_mask = labels == 0
    ood_mask = labels == 1

    id_log_probs = log_probs[id_mask]
    ood_log_probs = log_probs[ood_mask]

    mean_id = id_log_probs.mean()
    std_id = id_log_probs.std()
    mean_ood = ood_log_probs.mean()
    std_ood = ood_log_probs.std()

    # Calculate ROC AUC (higher anomaly score for lower log prob)
    anomaly_scores = -log_probs
    roc_auc = roc_auc_score(labels, anomaly_scores)

    # Calculate ROC curve for plotting
    fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)

    # Calculate accuracy if threshold is available
    if model.threshold is not None:
        # Predict OOD if log_prob < threshold
        predictions = (log_probs < model.threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)
    else:
        accuracy = None

    # Print results
    print(f"  Mean log-probability: {mean_log_prob:.4f} ± {std_log_prob:.4f}")
    print(f"  ID mean: {mean_id:.4f} ± {std_id:.4f}")
    print(f"  OOD mean: {mean_ood:.4f} ± {std_ood:.4f}")
    print(f"  Separation (ID - OOD): {mean_id - mean_ood:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    if accuracy is not None:
        print(f"  Accuracy (with threshold={model.threshold:.4f}): {accuracy:.4f}")

    return {
        'distance': distance,
        'mean_log_prob': mean_log_prob,
        'std_log_prob': std_log_prob,
        'mean_id': mean_id,
        'std_id': std_id,
        'mean_ood': mean_ood,
        'std_ood': std_ood,
        'separation': mean_id - mean_ood,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'fpr': fpr,
        'tpr': tpr,
        'threshold': model.threshold,
        'log_probs': log_probs,
        'labels': labels
    }


def plot_results(all_results, save_dir='figures/neuralode_ood_levels', model_name='Neural ODE', threshold=None):
    """
    Plot comprehensive results for different OOD distance levels.

    Args:
        all_results: List of result dictionaries
        save_dir: Directory to save plots
        model_name: Name of the model for plot titles
        threshold: Model threshold for OOD detection (optional)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract data for plotting
    distance_values = [r['distance'] for r in all_results]
    mean_log_probs = [r['mean_log_prob'] for r in all_results]
    mean_id_lps = [r['mean_id'] for r in all_results]
    mean_ood_lps = [r['mean_ood'] for r in all_results]
    roc_aucs = [r['roc_auc'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results if r['accuracy'] is not None]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} Performance on Different OOD Distance Levels',
                 fontsize=16, fontweight='bold')

    # Plot 1: Mean log-probability vs distance
    ax1 = axes[0, 0]
    ax1.plot(distance_values, mean_log_probs, 'o-', linewidth=2, markersize=8,
             color='steelblue', label='Overall')
    ax1.plot(distance_values, mean_id_lps, 's-', linewidth=2, markersize=8,
             color='green', label='ID samples')
    ax1.plot(distance_values, mean_ood_lps, '^-', linewidth=2, markersize=8,
             color='red', label='OOD samples')

    # Add threshold line if available
    if threshold is not None:
        ax1.axhline(y=threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold:.3f}')

    ax1.set_xlabel('OOD Distance', fontsize=12)
    ax1.set_ylabel('Mean Log-Probability', fontsize=12)
    ax1.set_title('Log-Probability vs OOD Distance', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: ROC AUC vs distance
    ax2 = axes[0, 1]
    ax2.plot(distance_values, roc_aucs, 'o-', linewidth=2, markersize=8, color='forestgreen')
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Random Classifier', alpha=0.7)
    ax2.set_xlabel('OOD Distance', fontsize=12)
    ax2.set_ylabel('ROC AUC', fontsize=12)
    ax2.set_title('ROC AUC vs OOD Distance', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Accuracy vs distance (if available)
    ax3 = axes[1, 0]
    if accuracies:
        ax3.plot(distance_values, accuracies, 'o-', linewidth=2, markersize=8, color='purple')
        ax3.set_xlabel('OOD Distance', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Classification Accuracy vs OOD Distance', fontsize=13, fontweight='bold')
        ax3.set_ylim([0, 1.05])
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No accuracy data\n(threshold not set)',
                ha='center', va='center', fontsize=14)
        ax3.set_xticks([])
        ax3.set_yticks([])

    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = []
    headers = ['Distance', 'Mean LP', 'ID LP', 'OOD LP', 'ROC AUC']

    for r in all_results:
        row = [
            f"{r['distance']:.1f}",
            f"{r['mean_log_prob']:.3f}",
            f"{r['mean_id']:.3f}",
            f"{r['mean_ood']:.3f}",
            f"{r['roc_auc']:.3f}"
        ]
        table_data.append(row)

    table = ax4.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'ood_distance_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved summary plot to: {save_path}")
    plt.close()

    # Plot 5: ROC Curves for each distance
    n_distances = len(all_results)
    n_cols = 3
    n_rows = (n_distances + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    fig.suptitle(f'{model_name} ROC Curves for Different OOD Distance Levels',
                 fontsize=16, fontweight='bold')

    if n_distances > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, r in enumerate(all_results):
        ax = axes[idx]

        # Plot ROC curve
        ax.plot(r['fpr'], r['tpr'], linewidth=2, label=f'ROC (AUC = {r["roc_auc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'Distance = {r["distance"]}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

    # Hide unused subplots
    for idx in range(n_distances, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curve plots to: {save_path}")
    plt.close()

    # Plot 6: Log-probability distributions for each distance
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    fig.suptitle(f'{model_name} Log-Probability Distributions for Different OOD Distance Levels',
                 fontsize=16, fontweight='bold')

    if n_distances > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, r in enumerate(all_results):
        ax = axes[idx]

        id_mask = r['labels'] == 0
        ood_mask = r['labels'] == 1

        id_lps = r['log_probs'][id_mask]
        ood_lps = r['log_probs'][ood_mask]

        # Determine appropriate bins
        bins = 50

        # Plot histograms
        sns.histplot(id_lps, bins=bins, color='green', alpha=0.5,
                    label='ID', kde=True, ax=ax, stat='density')

        # Check if OOD values have variance
        if ood_lps.std() < 1e-6:
            # All OOD values are essentially the same - plot as a vertical line
            y_max = ax.get_ylim()[1]
            ax.axvline(x=ood_lps.mean(), color='red', linestyle='-', linewidth=3,
                      alpha=0.7, label=f'OOD (n={len(ood_lps)})')
        else:
            sns.histplot(ood_lps, bins=bins, color='red', alpha=0.5,
                        label='OOD', kde=True, ax=ax, stat='density')

        # Add threshold line if available
        if threshold is not None:
            ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                      label=f'Threshold = {threshold:.3f}')

        ax.set_xlabel('Log-Probability', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Distance = {r["distance"]}, ROC AUC = {r["roc_auc"]:.3f}',
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_distances, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'log_prob_distributions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved distribution plots to: {save_path}")
    plt.close()


def save_results_to_json(results, save_path='cormpo/figures/neuralode_ood_results.json'):
    """
    Save all results data to a JSON file.

    Args:
        results: List of result dictionaries from evaluate_ood_at_distance
        save_path: Path to save the JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    json_results = []
    for r in results:
        json_result = {
            'distance': float(r['distance']),
            'mean_log_prob': float(r['mean_log_prob']),
            'std_log_prob': float(r['std_log_prob']),
            'mean_id': float(r['mean_id']),
            'std_id': float(r['std_id']),
            'mean_ood': float(r['mean_ood']),
            'std_ood': float(r['std_ood']),
            'separation': float(r['separation']),
            'roc_auc': float(r['roc_auc']),
            'accuracy': float(r['accuracy']) if r['accuracy'] is not None else None,
            'threshold': float(r['threshold']) if r['threshold'] is not None else None,
            'fpr': r['fpr'].tolist() if hasattr(r['fpr'], 'tolist') else list(r['fpr']),
            'tpr': r['tpr'].tolist() if hasattr(r['tpr'], 'tolist') else list(r['tpr']),
            'log_probs': r['log_probs'].tolist() if hasattr(r['log_probs'], 'tolist') else list(r['log_probs']),
            'labels': r['labels'].tolist() if hasattr(r['labels'], 'tolist') else list(r['labels'])
        }
        json_results.append(json_result)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Saved results data to: {save_path}")


def main():
    # Stage 1: Parse --config argument
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default='')
    config_args, remaining_argv = config_parser.parse_known_args()

    # Load YAML config if provided
    yaml_defaults = {}
    if config_args.config:
        try:
            import yaml
            with open(config_args.config, 'r') as f:
                yaml_config = yaml.safe_load(f)
                yaml_defaults = {k.replace('-', '_'): v for k, v in yaml_config.items()}
            print(f"Loaded config from: {config_args.config}")
        except Exception as e:
            print(f"Warning: failed to read YAML config: {e}")

    def dget(key, default):
        return yaml_defaults.get(key, default)

    # Stage 2: Build full parser with YAML defaults
    parser = argparse.ArgumentParser(
        description='Test Neural ODE on OOD datasets at different distance levels',
        parents=[config_parser]
    )
    parser.add_argument('--model-path', type=str,
                       required=('model_path' not in yaml_defaults),
                       default=dget('model_path', None),
                       help='Path to the trained Neural ODE model checkpoint')
    parser.add_argument('--dataset', type=str, default=dget('dataset_name', dget('dataset', 'abiomed')),
                       help='Dataset name (default: abiomed)')
    parser.add_argument('--distances', type=float, nargs='+',
                       default=dget('distances', [0.5, 1, 2, 4, 8]),
                       help='List of OOD distance levels to test')
    parser.add_argument('--base-path', type=str, default=dget('base_path', '/abiomed/downsampled/ood_test'),
                       help='Base path to OOD test datasets')
    parser.add_argument('--device', type=str, default=dget('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--devid', type=int, default=dget('devid', 0),
                       help='CUDA device ID')
    parser.add_argument('--input-dim', type=int,
                       required=('input_dim' not in yaml_defaults),
                       default=dget('input_dim', None),
                       help='Input dimension of the model')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=dget('hidden_dims', [512, 512]),
                       help='Hidden dimensions of the model')
    parser.add_argument('--activation', type=str, default=dget('activation', 'silu'),
                       choices=['silu', 'tanh'], help='Activation function')
    parser.add_argument('--solver', type=str, default=dget('solver', 'dopri5'),
                       help='ODE solver method')
    parser.add_argument('--rtol', type=float, default=dget('rtol', 1e-5),
                       help='Relative tolerance for ODE solver')
    parser.add_argument('--atol', type=float, default=dget('atol', 1e-6),
                       help='Absolute tolerance for ODE solver')
    parser.add_argument('--batch-size', type=int, default=dget('batch_size', 256),
                       help='Batch size for evaluation')
    parser.add_argument('--anomaly-fraction', type=float, default=dget('anomaly_fraction', 0.01),
                       help='Anomaly fraction for threshold')
    parser.add_argument('--save-dir', type=str, default=dget('save_dir', 'figures/neuralode_ood_levels'),
                       help='Directory to save plots')
    parser.add_argument('--seed', type=int, default=dget('seed', 42),
                       help='Random seed')

    args = parser.parse_args(remaining_argv)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Handle device with devid
    if args.device.startswith('cuda') and ':' not in args.device:
        device = f"{args.device}:{args.devid}"
    else:
        device = args.device

    print(f"Loading Neural ODE model from: {args.model_path}")
    print(f"Device: {device}")
    print(f"Input dimension: {args.input_dim}")
    print(f"Hidden dimensions: {args.hidden_dims}")
    print(f"ODE solver tolerances: rtol={args.rtol}, atol={args.atol}")

    # Load the trained model
    hidden_dims = tuple(args.hidden_dims)

    # Create ODE function
    odefunc = ODEFunc(
        dim=args.input_dim,
        hidden_dims=hidden_dims,
        activation=args.activation,
        time_dependent=True
    ).to(device)

    # Create flow
    flow = ContinuousNormalizingFlow(
        func=odefunc,
        t0=0.0,
        t1=1.0,
        solver=args.solver,
        rtol=args.rtol,
        atol=args.atol
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        flow.load_state_dict(checkpoint['model_state_dict'])
    else:
        flow.load_state_dict(checkpoint)

    flow.eval()
    print("Model loaded successfully!")

    # Create OOD wrapper
    model = NeuralODEOOD(flow, device=device)

    # Load threshold if available in checkpoint
    if isinstance(checkpoint, dict) and 'threshold' in checkpoint:
        model.threshold = checkpoint['threshold']
        print(f"Loaded threshold from checkpoint: {model.threshold}")
    else:
        # Threshold not in checkpoint, compute it from validation data
        print(f"\nThreshold not found in checkpoint. Computing from validation data...")
        print(f"Loading validation data to set threshold...")

        # Try to load validation data from the first ID test set
        try:
            # Use the first distance's ID samples as validation data
            first_distance = args.distances[0]
            val_test_data = load_ood_test_data(args.dataset, first_distance, args.base_path)
            # Use only the first half (ID samples) as validation data
            val_data = val_test_data[:len(val_test_data)//2]
            val_data_tensor = torch.FloatTensor(val_data).to(device)

            # Set threshold
            model.set_threshold(val_data_tensor, anomaly_fraction=args.anomaly_fraction)
            print(f"✓ Threshold set to: {model.threshold:.4f}")
        except Exception as e:
            print(f"Warning: Could not set threshold from validation data: {e}")
            print("Continuing without threshold (accuracy will be N/A)")

    # Test on all distance levels
    results = []
    print(f"\nTesting on {len(args.distances)} distance levels: {args.distances}")
    print("=" * 80)

    for distance in args.distances:
        print(f"\nTesting at OOD distance: {distance}")
        print("-" * 80)
        try:
            result = evaluate_ood_at_distance(
                model=model,
                dataset_name=args.dataset,
                distance=distance,
                base_path=args.base_path,
                device=device,
                batch_size=args.batch_size
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
    plot_results(results, save_dir=args.save_dir, model_name='Neural ODE', threshold=model.threshold)

    # Save results to JSON
    print("\nSaving results data...")
    save_results_to_json(results, save_path=os.path.join(args.save_dir, 'neuralode_ood_results.json'))

    print("\nDone!")


if __name__ == '__main__':
    main()
