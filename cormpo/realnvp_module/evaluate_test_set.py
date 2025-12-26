"""
Evaluate RealNVP on test set and report log-likelihood
"""

import torch
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realnvp_module.realnvp import RealNVP, load_rl_data_for_kde, parse_args

def main():
    # Parse command line arguments
    args = parse_args()
    args.obs_dim = (args.obs_dim,)

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = args.config

    # Determine device
    device = config.get('device', 'cpu')
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        print(f"{device} not available, falling back to CPU")
        device = "cpu"

    print(f"Using device: {device}")

    # Create Abiomed environment
    env = None
    if args.task.lower() == 'abiomed':
        from abiomed_env.rl_env import AbiomedRLEnvFactory
        print("Creating Abiomed environment...")
        env = AbiomedRLEnvFactory.create_env(
            model_name=getattr(args, 'model_name', '10min_1hr_all_data'),
            model_path=getattr(args, 'model_path_wm', None),
            data_path=getattr(args, 'data_path_wm', None),
            max_steps=getattr(args, 'max_steps', 6),
            gamma1=getattr(args, 'gamma1', 0.0),
            gamma2=getattr(args, 'gamma2', 0.0),
            gamma3=getattr(args, 'gamma3', 0.0),
            action_space_type='continuous',
            reward_type="smooth",
            normalize_rewards=True,
            noise_rate=getattr(args, 'noise_rate', 0.0),
            noise_scale=getattr(args, 'noise_scale', 0.0),
            seed=42,
            device=device
        )
        print("✓ Abiomed environment created successfully")

    # Load RL data (next_observations + actions)
    print("Loading RL dataset...")
    train_data, val_data, test_data, input_dim, norm_stats = load_rl_data_for_kde(
        args=args,
        env=env,
        val_split_ratio=config.get('val_ratio', 0.2)
    )

    print(f"Data shapes - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    # Load trained model
    model_path = config.get('model_save_path', '/public/gormpo/models/abiomed/realnvp/abiomed_realnvp')
    print(f"\nLoading trained RealNVP model from: {model_path}")

    model_dict = RealNVP.load_model(save_path=model_path, hidden_dims=config.get('hidden_dims', [256, 256]))
    model = model_dict['model']
    model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"Model threshold: {model.threshold}")

    # Evaluate on test set
    print("\n" + "="*80)
    print("Evaluating RealNVP on Test Set")
    print("="*80)

    test_data = test_data.to(device)

    with torch.no_grad():
        test_log_probs = model.score_samples(test_data)

    # Compute statistics
    mean_log_likelihood = test_log_probs.mean()
    std_log_likelihood = test_log_probs.std()
    median_log_likelihood = np.median(test_log_probs)
    min_log_likelihood = test_log_probs.min()
    max_log_likelihood = test_log_probs.max()

    # Count anomalies based on threshold
    n_anomalies = (test_log_probs < model.threshold).sum()
    anomaly_rate = n_anomalies / len(test_log_probs) * 100

    print(f"\nTest Set Statistics:")
    print(f"  Number of samples: {len(test_log_probs)}")
    print(f"  Mean log-likelihood: {mean_log_likelihood:.4f}")
    print(f"  Std log-likelihood: {std_log_likelihood:.4f}")
    print(f"  Median log-likelihood: {median_log_likelihood:.4f}")
    print(f"  Min log-likelihood: {min_log_likelihood:.4f}")
    print(f"  Max log-likelihood: {max_log_likelihood:.4f}")
    print(f"\nAnomaly Detection (threshold = {model.threshold:.4f}):")
    print(f"  Number of anomalies: {n_anomalies}/{len(test_log_probs)}")
    print(f"  Anomaly rate: {anomaly_rate:.2f}%")

    # Also evaluate on train and val for comparison
    print("\n" + "="*80)
    print("Comparison with Train and Validation Sets")
    print("="*80)

    train_data = train_data.to(device)
    val_data = val_data.to(device)

    with torch.no_grad():
        train_log_probs = model.score_samples(train_data)
        val_log_probs = model.score_samples(val_data)

    print(f"\nTrain Set:")
    print(f"  Mean log-likelihood: {train_log_probs.mean():.4f}")
    print(f"  Std log-likelihood: {train_log_probs.std():.4f}")

    print(f"\nValidation Set:")
    print(f"  Mean log-likelihood: {val_log_probs.mean():.4f}")
    print(f"  Std log-likelihood: {val_log_probs.std():.4f}")

    print(f"\nTest Set:")
    print(f"  Mean log-likelihood: {mean_log_likelihood:.4f}")
    print(f"  Std log-likelihood: {std_log_likelihood:.4f}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
