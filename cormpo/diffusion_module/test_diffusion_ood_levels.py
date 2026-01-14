"""
Test Diffusion model on different OOD levels using pre-generated OOD test datasets.
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
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

import pickle

# Import Diffusion model - use relative imports
from diffusion_ood import DiffusionOOD
from diffusion_density import UnconditionalEpsilonMLP, UnconditionalEpsilonTransformer
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


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

    # If data is a dictionary, extract observations only (diffusion was trained on obs only)
    if isinstance(data, dict):
        observations = data['observations']

        # Convert to numpy if needed
        if isinstance(observations, torch.Tensor):
            observations = observations.cpu().numpy()

        # Use only observations (diffusion model was trained on obs only, not obs+actions)
        data = observations
    # Convert to numpy if needed
    elif isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    return data


def evaluate_ood_at_distance(model, dataset_name, distance, base_path='/abiomed/downsampled/ood_test', device='cpu', mean=None, std=None, batch_size=100):
    """
    Evaluate Diffusion model on OOD test data at a specific distance level.

    Args:
        model: Trained DiffusionOOD model
        dataset_name: Name of the dataset
        distance: OOD distance level
        base_path: Base directory containing OOD test datasets
        device: Device to use
        mean: Mean for normalization (if None, no normalization)
        std: Standard deviation for normalization (if None, no normalization)
        batch_size: Batch size for evaluation

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

    # Normalize if mean and std are provided
    if mean is not None and std is not None:
        test_data = (test_data - mean) / (std + 1e-8)

    # Convert to tensor and move to device
    test_data_tensor = torch.FloatTensor(test_data).to(device)

    # Get log-likelihood scores (ELBO)
    model.model.eval()
    log_probs = model.score_samples(test_data_tensor, batch_size=batch_size).numpy()

    # Calculate overall metrics
    mean_log_likelihood = log_probs.mean()
    std_log_likelihood = log_probs.std()

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
    # For OOD detection: ID should have higher log prob (normal), OOD should have lower log prob (anomaly)
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

    results = {
        'distance': distance,
        'mean_log_likelihood': mean_log_likelihood,
        'std_log_likelihood': std_log_likelihood,
        'mean_id_log_likelihood': mean_id,
        'std_id_log_likelihood': std_id,
        'mean_ood_log_likelihood': mean_ood,
        'std_ood_log_likelihood': std_ood,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'log_probs': log_probs,
        'labels': labels,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': thresholds
    }

    return results


def plot_results(all_results, save_dir='figures/diffusion_ood_distance_tests', model_name='Diffusion', threshold=None):
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
    mean_log_likelihoods = [r['mean_log_likelihood'] for r in all_results]
    mean_id_lls = [r['mean_id_log_likelihood'] for r in all_results]
    mean_ood_lls = [r['mean_ood_log_likelihood'] for r in all_results]
    roc_aucs = [r['roc_auc'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results if r['accuracy'] is not None]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} Performance on Different OOD Distance Levels',
                 fontsize=16, fontweight='bold')

    # Plot 1: Mean log-likelihood vs distance
    ax1 = axes[0, 0]
    ax1.plot(distance_values, mean_log_likelihoods, 'o-', linewidth=2, markersize=8,
             color='steelblue', label='Overall')
    ax1.plot(distance_values, mean_id_lls, 's-', linewidth=2, markersize=8,
             color='green', label='ID samples')
    ax1.plot(distance_values, mean_ood_lls, '^-', linewidth=2, markersize=8,
             color='red', label='OOD samples')

    # Add threshold line if available
    if threshold is not None:
        ax1.axhline(y=threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold:.3f}')

    ax1.set_xlabel('OOD Distance', fontsize=12)
    ax1.set_ylabel('Mean Log-Likelihood (ELBO)', fontsize=12)
    ax1.set_title('Log-Likelihood vs OOD Distance', fontsize=13, fontweight='bold')
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
    headers = ['Distance', 'Mean LL', 'ID LL', 'OOD LL', 'ROC AUC']

    for r in all_results:
        row = [
            f"{r['distance']:.1f}",
            f"{r['mean_log_likelihood']:.3f}",
            f"{r['mean_id_log_likelihood']:.3f}",
            f"{r['mean_ood_log_likelihood']:.3f}",
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

    # Plot 6: Log-likelihood distributions for each distance
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    fig.suptitle(f'{model_name} ELBO Log-Likelihood Distributions for Different OOD Distance Levels',
                 fontsize=16, fontweight='bold')

    if n_distances > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, r in enumerate(all_results):
        ax = axes[idx]

        id_mask = r['labels'] == 0
        ood_mask = r['labels'] == 1

        id_lls = r['log_probs'][id_mask]
        ood_lls = r['log_probs'][ood_mask]

        # Determine appropriate bins based on the combined range
        all_lls = np.concatenate([id_lls, ood_lls])
        ll_min, ll_max = all_lls.min(), all_lls.max()

        # Use fixed bins for both distributions
        bins = np.linspace(ll_min, ll_max, 50)

        # Plot ID distribution
        sns.histplot(id_lls, bins=bins, color='green', alpha=0.5,
                    label='ID', kde=True, ax=ax, stat='density')

        # Plot OOD distribution - check if all values are the same
        if ood_lls.std() < 1e-6:
            # All OOD values are essentially the same - plot as a vertical line
            y_max = ax.get_ylim()[1]
            ax.axvline(x=ood_lls.mean(), color='red', linestyle='-', linewidth=3,
                      alpha=0.7, label=f'OOD (n={len(ood_lls)})')
        else:
            sns.histplot(ood_lls, bins=bins, color='red', alpha=0.5,
                        label='OOD', kde=True, ax=ax, stat='density')

        # Add threshold line if available
        if threshold is not None:
            ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                      label=f'Threshold = {threshold:.3f}')

        ax.set_xlabel('Log-Likelihood (ELBO)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Distance = {r["distance"]}, ROC AUC = {r["roc_auc"]:.3f}',
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_distances, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'log_likelihood_distributions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved distribution plots to: {save_path}")
    plt.close()


def save_results_to_json(all_results, save_path='figures/diffusion_ood_results.json'):
    """
    Save all results data to a JSON file.

    Args:
        all_results: List of result dictionaries from evaluate_ood_at_distance
        save_path: Path to save the JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    json_results = []
    for r in all_results:
        json_result = {
            'distance': float(r['distance']),
            'mean_log_likelihood': float(r['mean_log_likelihood']),
            'std_log_likelihood': float(r['std_log_likelihood']),
            'mean_id_log_likelihood': float(r['mean_id_log_likelihood']),
            'std_id_log_likelihood': float(r['std_id_log_likelihood']),
            'mean_ood_log_likelihood': float(r['mean_ood_log_likelihood']),
            'std_ood_log_likelihood': float(r['std_ood_log_likelihood']),
            'separation': float(r['mean_id_log_likelihood'] - r['mean_ood_log_likelihood']),
            'roc_auc': float(r['roc_auc']),
            'accuracy': float(r['accuracy']) if r['accuracy'] is not None else None,
            'fpr': r['fpr'].tolist() if hasattr(r['fpr'], 'tolist') else list(r['fpr']),
            'tpr': r['tpr'].tolist() if hasattr(r['tpr'], 'tolist') else list(r['tpr']),
            'labels': r['labels'].tolist() if hasattr(r['labels'], 'tolist') else list(r['labels']),
            'log_probs': r['log_probs'].tolist() if hasattr(r['log_probs'], 'tolist') else list(r['log_probs'])
        }
        json_results.append(json_result)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Saved results data to: {save_path}")


def load_diffusion_model(model_dir, device='cpu'):
    """
    Load a trained diffusion model from checkpoint.

    Args:
        model_dir: Directory containing checkpoint.pt and scheduler/
        device: Device to load model on

    Returns:
        Tuple of (model, scheduler, metadata)
    """
    ckpt_path = os.path.join(model_dir, 'checkpoint.pt')
    sched_dir = os.path.join(model_dir, 'scheduler')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")
    if not os.path.exists(sched_dir):
        raise FileNotFoundError(f"Missing scheduler directory at {sched_dir}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get('cfg', {})
    target_dim = ckpt.get('target_dim')
    model_type = cfg.get('model_type', 'mlp')
    time_embed_dim = cfg.get('time_embed_dim', 128)

    # Create model
    if model_type == 'mlp':
        model = UnconditionalEpsilonMLP(
            target_dim=target_dim,
            hidden_dim=cfg.get('hidden_dim', 512),
            time_embed_dim=time_embed_dim,
            num_hidden_layers=cfg.get('num_hidden_layers', 3),
            dropout=cfg.get('dropout', 0.0),
        )
    else:
        model = UnconditionalEpsilonTransformer(
            target_dim=target_dim,
            d_model=cfg.get('d_model', 256),
            nhead=cfg.get('nhead', 8),
            num_layers=cfg.get('tf_layers', 4),
            dim_feedforward=cfg.get('ff_dim', 512),
            dropout=cfg.get('dropout', 0.1),
            time_embed_dim=time_embed_dim,
        )

    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device)
    model.eval()

    # Load scheduler
    try:
        scheduler = DDIMScheduler.from_pretrained(sched_dir)
        print("Loaded DDIMScheduler")
    except Exception:
        try:
            scheduler = DDPMScheduler.from_pretrained(sched_dir)
            print("Loaded DDPMScheduler")
        except Exception as e:
            print(f"Warning: Could not load scheduler: {e}")
            print("Creating default DDIMScheduler")
            scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_schedule="linear",
                prediction_type="epsilon",
            )

    # Extract threshold if available
    threshold = ckpt.get('threshold', None)

    metadata = {
        'cfg': cfg,
        'target_dim': target_dim,
        'model_type': model_type,
        'threshold': threshold
    }

    print(f"Model loaded from: {model_dir}")
    print(f"Model type: {model_type}")
    print(f"Target dimension: {target_dim}")
    if threshold is not None:
        print(f"Threshold: {threshold:.4f}")

    return model, scheduler, metadata


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
        description='Test Diffusion on different OOD distance levels',
        parents=[config_parser]
    )
    parser.add_argument('--model_dir', type=str,
                       required=('model_dir' not in yaml_defaults),
                       default=dget('model_dir', None),
                       help='Path to model directory (containing checkpoint.pt and scheduler/)')
    parser.add_argument('--dataset_name', type=str, default=dget('dataset_name', 'abiomed'),
                       help='Dataset name (e.g., abiomed, halfcheetah-medium-v2)')
    parser.add_argument('--distances', type=float, nargs='+', default=dget('distances', [0.5, 1, 2, 4, 8]),
                       help='List of OOD distance values to test (default: 0.5 1 2 4 8 for Abiomed)')
    parser.add_argument('--base_path', type=str, default=dget('base_path', '/abiomed/downsampled/ood_test'),
                       help='Base directory containing OOD test datasets')
    parser.add_argument('--device', type=str, default=dget('device', 'cuda'),
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--devid', type=int, default=dget('devid', 0),
                       help='CUDA device ID')
    parser.add_argument('--save_dir', type=str, default=dget('save_dir', 'figures/diffusion_ood_levels'),
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=dget('batch_size', 100),
                       help='Batch size for evaluation')
    parser.add_argument('--num_inference_steps', type=int, default=dget('num_inference_steps', 20),
                       help='Number of inference steps for ELBO (e.g., 20 for fast, None for all 1000)')
    parser.add_argument('--anomaly_fraction', type=float, default=dget('anomaly_fraction', 0.01),
                       help='Anomaly fraction for threshold setting (default: 0.01 = 1%)')
    parser.add_argument('--seed', type=int, default=dget('seed', 42),
                       help='Random seed')

    args = parser.parse_args(remaining_argv)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    if args.device.startswith('cuda'):
        device_str = f"{args.device}:{args.devid}" if ':' not in args.device else args.device
        if torch.cuda.is_available():
            device = torch.device(device_str)
        else:
            print("CUDA not available, using CPU")
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset_name}")
    print(f"OOD test data path: {args.base_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num inference steps: {args.num_inference_steps}")

    # Load model
    print(f"\nLoading Diffusion model from: {args.model_dir}")
    model, scheduler, metadata = load_diffusion_model(args.model_dir, str(device))

    # Create OOD wrapper
    ood_model = DiffusionOOD(
        model=model,
        scheduler=scheduler,
        device=str(device),
        num_inference_steps=args.num_inference_steps
    )

    # Load threshold from metadata if available
    if metadata.get('threshold') is not None:
        ood_model.threshold = metadata['threshold']

    print(f"\nModel loaded successfully")

    # Set threshold using validation data if not already loaded
    if ood_model.threshold is None:
        print(f"\nThreshold not set. Computing from validation data...")
        print(f"Loading validation data to set threshold...")

        try:
            # Use the first distance's ID samples as validation data
            first_distance = args.distances[0]
            val_test_data = load_ood_test_data(args.dataset_name, first_distance, args.base_path)
            # Use only the first half (ID samples) as validation data
            val_data = val_test_data[:len(val_test_data)//2]
            val_data_tensor = torch.FloatTensor(val_data).to(device)

            # Set threshold
            ood_model.set_threshold(val_data_tensor, anomaly_fraction=args.anomaly_fraction, batch_size=args.batch_size)
            print(f"✓ Threshold set to: {ood_model.threshold:.4f}")
        except Exception as e:
            print(f"Warning: Could not set threshold from validation data: {e}")
            print("Continuing without threshold (accuracy will be N/A)")
    else:
        print(f"Threshold loaded: {ood_model.threshold:.4f}")

    # Test on different distance values
    print("\n" + "="*80)
    print("Testing Diffusion on Different OOD Distance Levels")
    print("="*80)

    all_results = []

    for distance in args.distances:
        print(f"\nTesting distance = {distance}")
        print("-" * 40)

        try:
            results = evaluate_ood_at_distance(
                model=ood_model,
                dataset_name=args.dataset_name,
                distance=distance,
                base_path=args.base_path,
                device=str(device),
                mean=None,  # No normalization for now
                std=None,
                batch_size=args.batch_size
            )

            all_results.append(results)

            print(f"  Mean log-likelihood: {results['mean_log_likelihood']:.4f}")
            print(f"  ID samples mean LL: {results['mean_id_log_likelihood']:.4f}")
            print(f"  OOD samples mean LL: {results['mean_ood_log_likelihood']:.4f}")
            print(f"  ROC AUC: {results['roc_auc']:.4f}")
            if results['accuracy'] is not None:
                print(f"  Accuracy: {results['accuracy']:.4f}")
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            print(f"  Skipping distance {distance}")

    if not all_results:
        print("\nNo test data found! Please check the dataset path and distance values.")
        return

    # Create save directory
    save_dir = os.path.join(args.save_dir, args.dataset_name.replace('-', '_'))

    # Plot results
    print("\n" + "="*80)
    print("Generating Plots")
    print("="*80)

    plot_results(all_results, save_dir=save_dir, model_name='Diffusion', threshold=ood_model.threshold)

    # Save results to JSON
    print("\nSaving results data...")
    save_results_to_json(all_results, save_path=os.path.join(save_dir, 'diffusion_ood_results.json'))

    print("\n" + "="*80)
    print("Testing Complete!")
    print("="*80)
    print(f"\nResults saved to: {save_dir}")

    # Print summary
    print("\nSummary:")
    print("-" * 80)
    print(f"{'Distance':<10} {'Mean LL':<12} {'ID LL':<12} {'OOD LL':<12} {'ROC AUC':<10} {'Accuracy':<10}")
    print("-" * 80)
    for r in all_results:
        acc_str = f"{r['accuracy']:.4f}" if r['accuracy'] is not None else "N/A"
        print(f"{r['distance']:<10.1f} {r['mean_log_likelihood']:<12.4f} "
              f"{r['mean_id_log_likelihood']:<12.4f} "
              f"{r['mean_ood_log_likelihood']:<12.4f} {r['roc_auc']:<10.4f} {acc_str:<10}")
    print("-" * 80)


if __name__ == "__main__":
    main()
