"""
Test Diffusion Model OOD Detection on Abiomed Environment
Adapted from GORMPO's diffusion OOD testing
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import gaussian_kde

import yaml

# Add project root to path (append to avoid conflicts with cormpo/config/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from cormpo.diffusion_module.diffusion_density import (
    UnconditionalEpsilonMLP,
    UnconditionalEpsilonTransformer,
)
from cormpo.diffusion_module.diffusion_ood import DiffusionOOD, plot_likelihood_distributions
from cormpo.diffusion_module.train_diffusion import load_rl_data_for_diffusion

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

try:
    from abiomed_env.rl_env import AbiomedRLEnvFactory
except ImportError:
    print("Warning: Could not import AbiomedRLEnvFactory")
    AbiomedRLEnvFactory = None


def plot_logp_distribution(
    logp_values: np.ndarray,
    percentile_value: float,
    percentile: float,
    save_path: str,
    title: str = "ELBO Log-Likelihood Distribution (Diffusion)",
) -> None:
    """
    Plot histogram with KDE overlay and threshold line for log probability distribution.

    Args:
        logp_values: Array of log probability values (ELBO)
        percentile_value: The threshold value at the specified percentile
        percentile: The percentile used (e.g., 1.0 for 1%)
        save_path: Path to save the figure
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram
    n, bins, patches = ax.hist(
        logp_values,
        bins=50,
        density=False,
        alpha=0.7,
        color='lightblue',
        edgecolor='blue',
        linewidth=1.2,
        label='Test samples'
    )

    # Compute and plot KDE
    try:
        kde = gaussian_kde(logp_values)
        x_kde = np.linspace(logp_values.min(), logp_values.max(), 200)
        y_kde = kde(x_kde)
        # Scale KDE to match histogram scale
        bin_width = bins[1] - bins[0]
        y_kde_scaled = y_kde * len(logp_values) * bin_width
        ax.plot(x_kde, y_kde_scaled, 'b-', linewidth=2, label='KDE')
    except Exception as e:
        print(f"[Plot] Warning: Could not compute KDE: {e}")

    # Draw threshold line
    ax.axvline(
        x=percentile_value,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Threshold ({percentile}% percentile)'
    )

    ax.set_xlabel('Log-likelihood (ELBO)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper center', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Plot] Saved likelihood distribution plot to {save_path}")


def load_model_from_checkpoint(ckpt_path: str, device: str) -> tuple:
    """
    Load diffusion model from checkpoint.

    Args:
        ckpt_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, config_dict)
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    target_dim = ckpt.get("target_dim", 73)
    model_type = cfg.get("model_type", "mlp")
    time_embed_dim = cfg.get("time_embed_dim", 128)

    if model_type == "mlp":
        model = UnconditionalEpsilonMLP(
            target_dim=target_dim,
            hidden_dim=cfg.get("hidden_dim", 512),
            time_embed_dim=time_embed_dim,
            num_hidden_layers=cfg.get("num_hidden_layers", 3),
            dropout=cfg.get("dropout", 0.0),
        )
    else:
        model = UnconditionalEpsilonTransformer(
            target_dim=target_dim,
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 8),
            num_layers=cfg.get("tf_layers", 4),
            dim_feedforward=cfg.get("ff_dim", 512),
            dropout=cfg.get("dropout", 0.1),
            time_embed_dim=time_embed_dim,
        )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def parse_args():
    """Parse command line arguments and YAML config."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="")
    config_args, remaining_argv = config_parser.parse_known_args()

    yaml_defaults = {}
    if config_args.config:
        try:
            with open(config_args.config, "r") as f:
                config = yaml.safe_load(f)
                yaml_defaults = {k.replace("-", "_"): v for k, v in config.items()}
            print(f"Loaded config from: {config_args.config}")
        except Exception as e:
            print(f"Failed to load YAML config: {e}")

    def dget(key, default):
        return yaml_defaults.get(key, default)

    parser = argparse.ArgumentParser(
        description="Test Diffusion Model OOD Detection on Abiomed",
        parents=[config_parser]
    )

    # Model and data paths
    parser.add_argument('--model-dir', type=str,
                        required=('model_dir' not in yaml_defaults and 'out_dir' not in yaml_defaults),
                        default=dget('model_dir', dget('out_dir', None)),
                        help='Directory with checkpoint.pt and scheduler/')
    parser.add_argument('--task', type=str, default=dget('task', 'abiomed'))
    parser.add_argument('--model-name', type=str, default=dget('model_name', '10min_1hr_all_data'))
    parser.add_argument('--model-path-wm', type=str, default=dget('model_path_wm', '/public/gormpo/models/10min_1hr_all_data_model.pth'))
    parser.add_argument('--data-path-wm', type=str, default=dget('data_path_wm', '/public/gormpo/10min_1hr_all_data.pkl'))
    parser.add_argument('--max-steps', type=int, default=dget('max_steps', 6))
    parser.add_argument('--gamma1', type=float, default=dget('gamma1', 0.0))
    parser.add_argument('--gamma2', type=float, default=dget('gamma2', 0.0))
    parser.add_argument('--gamma3', type=float, default=dget('gamma3', 0.0))
    parser.add_argument('--action-space-type', type=str, default=dget('action_space_type', 'continuous'))
    parser.add_argument('--noise-rate', type=float, default=dget('noise_rate', 0.0))
    parser.add_argument('--noise-scale', type=float, default=dget('noise_scale', 0.0))

    # Data settings
    parser.add_argument('--val-ratio', type=float, default=dget('val_ratio', 0.15))
    parser.add_argument('--test-ratio', type=float, default=dget('test_ratio', 0.15))

    # OOD testing parameters
    parser.add_argument('--anomaly-fraction', type=float, default=dget('anomaly_fraction', 0.005),
                        help='Fraction for anomaly threshold (0.5% = 0.005)')
    parser.add_argument('--batch-size', type=int, default=dget('batch_size', 100))
    parser.add_argument('--num-inference-steps', type=int, default=dget('num_inference_steps', None),
                        help='Number of timesteps for ELBO (None=all, e.g. 50 for fast)')

    # Hardware
    parser.add_argument('--device', type=str, default=dget('device', 'cuda'))
    parser.add_argument('--devid', type=int, default=dget('devid', 0))
    parser.add_argument('--seed', type=int, default=dget('seed', 42))

    # Output
    parser.add_argument('--save-dir', type=str, default=dget('save_dir', 'figures/diffusion_ood'))
    parser.add_argument('--save-model-path', type=str, default=dget('save_model_path', ''))
    parser.add_argument('--plot-results', action='store_true', default=dget('plot_results', True))
    parser.add_argument('--verbose', action='store_true', default=dget('verbose', True))

    args = parser.parse_args(remaining_argv)
    return args


def main():
    args = parse_args()

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

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\nLoading Diffusion model from: {args.model_dir}")

    # Load model
    ckpt_path = os.path.join(args.model_dir, "checkpoint.pt")
    sched_dir = os.path.join(args.model_dir, "scheduler")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")
    if not os.path.exists(sched_dir):
        raise FileNotFoundError(f"Missing scheduler directory at {sched_dir}")

    model, train_cfg = load_model_from_checkpoint(ckpt_path, str(device))

    # Load scheduler (try DDIM first, fallback to DDPM)
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

    print("Model loaded successfully")

    # Load data
    print(f"\nLoading data from Abiomed environment...")

    # Create simple args object for data loading
    class DataArgs:
        pass

    data_args = DataArgs()
    for key in ['task', 'model_name', 'model_path_wm', 'data_path_wm', 'max_steps',
                'gamma1', 'gamma2', 'gamma3', 'action_space_type', 'noise_rate',
                'noise_scale', 'seed', 'device']:
        setattr(data_args, key, getattr(args, key))
    setattr(data_args, 'device', str(device))

    train_data, val_data, test_data = load_rl_data_for_diffusion(
        data_args,
        env=None,
        val_split_ratio=args.val_ratio,
        test_split_ratio=args.test_ratio
    )

    # Move data to device
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    # Create OOD wrapper
    print(f"\nUsing num_inference_steps: {args.num_inference_steps if args.num_inference_steps else 'all'}")
    ood_model = DiffusionOOD(
        model,
        scheduler,
        device=str(device),
        num_inference_steps=args.num_inference_steps
    )

    # Set threshold based on validation data
    print(f"\nSetting threshold with {args.anomaly_fraction*100}% anomaly fraction...")
    threshold = ood_model.set_threshold(val_data, args.anomaly_fraction, args.batch_size)

    # Test on in-distribution test data
    print("\nEvaluating on test data (in-distribution)...")
    test_log_probs = ood_model.score_samples(test_data, args.batch_size).numpy()
    test_predictions = ood_model.predict(test_data, args.batch_size)

    print(f"Test set log prob: {test_log_probs.mean():.3f} ± {test_log_probs.std():.3f}")
    print(f"Test set anomalies: {(test_predictions == -1).sum()}/{len(test_data)} "
          f"({(test_predictions == -1).sum()/len(test_data):.1%})")

    # Compute percentile threshold for plotting
    percentile_logp = np.percentile(test_log_probs, args.anomaly_fraction * 100)
    print(f"{args.anomaly_fraction*100}% percentile log-likelihood: {percentile_logp:.6f}")

    # Save metrics
    metrics = {
        "num_samples": int(len(test_log_probs)),
        "elbo_mean": float(test_log_probs.mean()),
        "elbo_std": float(test_log_probs.std()),
        "threshold": float(threshold),
        f"percentile_{args.anomaly_fraction*100}_logp": float(percentile_logp),
        "anomaly_fraction": float(args.anomaly_fraction),
    }

    metrics_path = os.path.join(args.save_dir, "diffusion_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Plot likelihood distribution
    if args.plot_results:
        print("\nPlotting likelihood distribution...")
        likelihood_plot_path = os.path.join(args.save_dir, "diffusion_likelihood_distribution.png")
        plot_logp_distribution(
            logp_values=test_log_probs,
            percentile_value=percentile_logp,
            percentile=args.anomaly_fraction * 100,
            save_path=likelihood_plot_path,
            title="ELBO Log-Likelihood Distribution (Diffusion)"
        )

    # Create OOD test data (10% of normal + noisy versions)
    print("\nCreating OOD test data...")
    predictions_tr = ood_model.predict(train_data, args.batch_size)
    small_train = train_data[predictions_tr == 1][: int(0.1 * len(train_data))].cpu().numpy()
    noisy_train = small_train + np.random.normal(0, 0.1, small_train.shape)
    ood_test_data = torch.FloatTensor(np.concatenate([small_train, noisy_train], axis=0)).to(device)
    print(f"OOD test data created: {len(ood_test_data)} samples")

    # Test on OOD data
    print("\nEvaluating on OOD data...")
    ood_predictions = ood_model.predict(ood_test_data, args.batch_size)
    ood_scores = ood_model.score_samples(ood_test_data, args.batch_size).numpy()

    print(f"OOD scores: {ood_scores.mean():.3f} ± {ood_scores.std():.3f}")
    anomaly_count = (ood_predictions == -1).sum()
    print(f"OOD anomalies detected: {anomaly_count}/{len(ood_test_data)} "
          f"({anomaly_count/len(ood_test_data):.1%})")

    # Evaluate ROC curve
    print("\nEvaluating OOD detection performance (ROC curve)...")
    n_normal = min(len(test_data), len(ood_test_data))
    results = ood_model.evaluate_anomaly_detection(
        normal_data=test_data[:n_normal],
        anomaly_data=ood_test_data[:n_normal],
        plot=args.plot_results,
        save_path=os.path.join(args.save_dir, "diffusion_roc_curve.png"),
        batch_size=args.batch_size
    )

    print(f"\nROC Evaluation Results:")
    print(f"  ROC AUC: {results['roc_auc']:.3f}")
    if results['accuracy'] is not None:
        print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  Normal log prob: {results['normal_log_prob_mean']:.3f} ± {results['normal_log_prob_std']:.3f}")
    print(f"  Anomaly log prob: {results['anomaly_log_prob_mean']:.3f} ± {results['anomaly_log_prob_std']:.3f}")

    # Plot likelihood distributions
    if args.plot_results:
        print("\nPlotting likelihood distributions...")
        plot_likelihood_distributions(
            model=ood_model,
            train_data=train_data,
            val_data=val_data,
            ood_data=ood_test_data,
            save_dir=args.save_dir,
            batch_size=args.batch_size
        )

    # Save model metadata if requested
    if args.save_model_path:
        print(f"\nSaving OOD model metadata to: {args.save_model_path}")
        ood_model.save_model(args.save_model_path, train_data)

    print("\nDiffusion OOD testing completed!")


if __name__ == "__main__":
    main()
