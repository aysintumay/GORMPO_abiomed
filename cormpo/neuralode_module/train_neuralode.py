"""
Train Neural ODE model on Abiomed dataset with YAML configuration support.

This script supports training Neural ODE models using configuration files,
similar to the mbpo_kde pattern.

Usage:
    python cormpo/neuralode_module/train_neuralode.py --config cormpo/config/neuralode/abiomed.yaml
"""

import torch
import sys
import os
import argparse
import yaml
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cormpo.neuralode_module import (
    ODEFunc,
    ContinuousNormalizingFlow,
    NeuralODEOOD,
    TrainConfig,
)
from torch.utils.data import DataLoader, TensorDataset


def load_rl_data_for_neuralode(args, env=None, val_split_ratio=0.2, test_split_ratio=0.2):
    """
    Load RL dataset and prepare for Neural ODE training.

    Args:
        args: Arguments containing data_path, obs_shape, action_dim
        env: Environment object (for datasets that need it)
        val_split_ratio: Fraction of data for validation
        test_split_ratio: Fraction of data for testing

    Returns:
        Tuple of (train_data, val_data, test_data, input_dim)
    """
    # Load data from environment
    if env is not None and hasattr(env, 'world_model'):
        print("Loading data from environment world model...")
        train_dataset = env.world_model.data_train
        val_dataset = env.world_model.data_val
        test_dataset = env.world_model.data_test

        # Extract concatenated next_observations (labels) + actions (pl)
        # The dataset has .labels (next observations) and .pl (pump levels/actions)
        # Convert to float32 tensors (data may already be tensors)
        train_next_obs = torch.as_tensor(train_dataset.labels, dtype=torch.float32)
        train_actions = torch.as_tensor(train_dataset.pl, dtype=torch.float32)
        train_data = torch.cat([train_next_obs, train_actions], dim=1)

        val_next_obs = torch.as_tensor(val_dataset.labels, dtype=torch.float32)
        val_actions = torch.as_tensor(val_dataset.pl, dtype=torch.float32)
        val_data = torch.cat([val_next_obs, val_actions], dim=1)

        test_next_obs = torch.as_tensor(test_dataset.labels, dtype=torch.float32)
        test_actions = torch.as_tensor(test_dataset.pl, dtype=torch.float32)
        test_data = torch.cat([test_next_obs, test_actions], dim=1)

        input_dim = train_data.shape[1]

        print(f"Loaded from world model - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
        print(f"  Next obs dim: {train_next_obs.shape[1]}, Actions dim: {train_actions.shape[1]}")
    else:
        raise ValueError("Environment with world_model required for Abiomed data loading")

    print(f"Neural ODE input dimension: {input_dim}")
    return train_data, val_data, test_data, input_dim


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Neural ODE on Abiomed dataset')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device override (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs override')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    return parser.parse_args()


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_args_from_config(config):
    """Create args object from config dictionary."""
    class Args:
        pass

    args = Args()
    # Copy all config items to args
    for key, value in config.items():
        setattr(args, key, value)

    # Ensure obs_dim is tuple
    if hasattr(args, 'obs_dim') and isinstance(args.obs_dim, list):
        args.obs_dim = tuple(args.obs_dim)
    if hasattr(args, 'obs_dim') and isinstance(args.obs_dim, int):
        args.obs_dim = (args.obs_dim,)

    return args


def main():
    # Parse command line arguments
    cmd_args = parse_args()

    # Load configuration
    print(f"Loading configuration from: {cmd_args.config}")
    config = load_config(cmd_args.config)

    # Override config with command line arguments if provided
    if cmd_args.device is not None:
        config['device'] = cmd_args.device
    if cmd_args.epochs is not None:
        config['epochs'] = cmd_args.epochs
    if cmd_args.verbose:
        config['verbose'] = True

    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine device
    device = config.get('device', 'cpu')
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        print(f"{device} not available, falling back to CPU")
        device = "cpu"

    print(f"Using device: {device}")

    # Create args object from config
    args = create_args_from_config(config)

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
            action_space_type=getattr(args, 'action_space_type', 'continuous'),
            reward_type="smooth",
            normalize_rewards=True,
            noise_rate=getattr(args, 'noise_rate', 0.0),
            noise_scale=getattr(args, 'noise_scale', 0.0),
            seed=seed,
            device=device
        )
        print("✓ Abiomed environment created successfully")

    # Load RL data
    print("Loading RL dataset for Neural ODE training...")
    train_data, val_data, test_data, input_dim = load_rl_data_for_neuralode(
        args=args,
        env=env,
        val_split_ratio=config.get('val_ratio', 0.2),
        test_split_ratio=config.get('test_ratio', 0.2)
    )

    # Update input dimension in config
    config['input_dim'] = input_dim

    print(f"Data shapes - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    # Create model
    print("Creating Neural ODE model...")
    hidden_dims = tuple(config.get('hidden_dims', [512, 512]))

    odefunc = ODEFunc(
        dim=input_dim,
        hidden_dims=hidden_dims,
        activation=config.get('activation', 'silu'),
        time_dependent=config.get('time_dependent', True)
    ).to(device)

    flow = ContinuousNormalizingFlow(
        func=odefunc,
        t0=config.get('t0', 0.0),
        t1=config.get('t1', 1.0),
        solver=config.get('solver', 'dopri5'),
        rtol=config.get('rtol', 1e-5),
        atol=config.get('atol', 1e-5)
    ).to(device)

    num_params = sum(p.numel() for p in flow.parameters())
    print(f"Model created with {num_params:,} parameters")

    # Training loop
    print("\nTraining Neural ODE model...")
    optimizer = torch.optim.AdamW(
        flow.parameters(),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 0.0)
    )

    batch_size = config.get('batch_size', 512)
    epochs = config.get('epochs', 200)
    log_every = config.get('log_every', 100)
    checkpoint_every = config.get('checkpoint_every', 0)
    out_dir = config.get('out_dir', 'cormpo/checkpoints/neuralode')

    os.makedirs(out_dir, exist_ok=True)

    # Create data loader
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Training loop
    flow.train()
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_tuple in train_loader:
            batch = batch_tuple[0].to(device)

            # Compute loss
            log_px = flow.log_prob(batch)
            loss = -log_px.mean()

            # Optimize
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)
            num_batches += 1

            if (global_step + 1) % log_every == 0:
                print(
                    f"  Step {global_step+1}  Epoch {epoch+1}/{epochs}  "
                    f"Loss {loss.item():.6f}  Mean log-prob {log_px.mean().item():.6f}"
                )

            # Clear CUDA cache periodically
            if (global_step + 1) % 10 == 0 and device.startswith('cuda'):
                torch.cuda.empty_cache()

            global_step += 1

        epoch_loss /= len(train_data)

        # Validation - process in small batches to avoid OOM
        flow.eval()
        val_loss_sum = 0.0
        val_samples = 0
        val_batch_size = 100  # Small batch size for validation

        for i in range(0, len(val_data), val_batch_size):
            val_batch = val_data[i:i+val_batch_size]
            # Note: log_prob needs gradients for divergence computation
            val_log_px = flow.log_prob(val_batch)
            val_loss_sum += -val_log_px.sum().item()
            val_samples += val_batch.size(0)

            # Clear cache after each batch
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

        val_loss = val_loss_sum / val_samples
        flow.train()

        print(f"[Epoch {epoch+1}/{epochs}] Train NLL: {epoch_loss:.6f}, Val NLL: {val_loss:.6f}")

        # Save checkpoint
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'model_state_dict': flow.state_dict(),
                'epoch': epoch,
                'config': config,
                'input_dim': input_dim
            }, ckpt_path)
            print(f"  Checkpoint saved to {ckpt_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(out_dir, "best_model.pt")
            torch.save({
                'model_state_dict': flow.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'config': config,
                'input_dim': input_dim
            }, best_model_path)

    # Save final model
    final_model_path = os.path.join(out_dir, "model.pt")
    torch.save(flow.state_dict(), final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")

    # Create OOD model and set threshold
    print("\nCreating OOD model and setting threshold...")
    flow.eval()
    ood_model = NeuralODEOOD(flow, device=device)

    # Set threshold using validation data
    anomaly_fraction = config.get('anomaly_fraction', 0.01)
    ood_model.set_threshold(val_data, anomaly_fraction=anomaly_fraction)

    # Update best_model.pt to include the threshold
    best_model_path = os.path.join(out_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        print(f"\nUpdating best_model.pt with threshold...")
        best_checkpoint = torch.load(best_model_path)
        best_checkpoint['threshold'] = ood_model.threshold
        best_checkpoint['anomaly_fraction'] = anomaly_fraction
        torch.save(best_checkpoint, best_model_path)
        print(f"✓ Threshold ({ood_model.threshold:.4f}) added to best_model.pt")

    # Save OOD model
    save_path = config.get('model_save_path', os.path.join(out_dir, 'neuralode_ood'))
    print(f"Saving OOD model to: {save_path}")
    ood_model.save_model(save_path, train_data=train_data)

    print("\n✓ Neural ODE training completed successfully!")
    print(f"✓ Model saved to: {final_model_path}")
    print(f"✓ OOD model saved to: {save_path}_model.pt")
    print(f"✓ Metadata saved to: {save_path}_metadata.pkl")
    print(f"✓ Best validation NLL: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
