"""
Train RealNVP model on Abiomed dataset
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realnvp_module.realnvp import RealNVP, load_rl_data_for_kde, parse_args
import yaml

def main():
    # Parse command line arguments
    args = parse_args()
    args.obs_dim = (args.obs_dim,)

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = args.config

    # Override config with command line arguments if provided
    if args.device is not None:
        config['device'] = args.device
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.verbose:
        config['verbose'] = True

    # Set random seed for reproducibility
    if 'seed' in config:
        torch.manual_seed(config['seed'])
        import numpy as np
        np.random.seed(config['seed'])

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
    print("Loading RL dataset for RealNVP training...")
    train_data, val_data, test_data, input_dim, norm_stats = load_rl_data_for_kde(
        args=args,
        env=env,
        val_split_ratio=config.get('val_ratio', 0.2)
    )

    # Update input dimension in config
    config['input_dim'] = input_dim

    print(f"Data shapes - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    # Create and train model
    print("Creating RealNVP model...")
    model = RealNVP(
        input_dim=config.get('input_dim', 13),
        num_layers=config.get('num_layers', 6),
        hidden_dims=config.get('hidden_dims', [256, 256]),
        device=device
    ).to(device)

    print("Training RealNVP model...")
    history = model.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=config.get('epochs', 100),
        batch_size=config.get('batch_size', 128),
        lr=config.get('lr', 1e-3),
        patience=config.get('patience', 15),
        verbose=config.get('verbose', True)
    )

    # Save model
    save_path = config.get('model_save_path', '/public/gormpo/models/abiomed/realnvp/abiomed_realnvp')
    print(f"\nSaving model to: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_model(save_path, train_data=train_data, norm_stats=norm_stats)

    print("\n✓ RealNVP training completed successfully!")
    print(f"✓ Model saved to: {save_path}_model.pth")
    print(f"✓ Metadata saved to: {save_path}_meta_data.pkl")

if __name__ == "__main__":
    main()
