"""
Train Diffusion Model for Abiomed OOD Detection
Adapted from GORMPO's DDIM training for Abiomed environment
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# Add project root to path (append to avoid conflicts with cormpo/config/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from cormpo.diffusion_module.diffusion_density import (
    UnconditionalEpsilonMLP,
    UnconditionalEpsilonTransformer,
    SinusoidalTimeEmbedding,
)

try:
    from abiomed_env.rl_env import AbiomedRLEnvFactory
except ImportError:
    print("Warning: Could not import AbiomedRLEnvFactory. Make sure abiomed_env is in PYTHONPATH.")
    AbiomedRLEnvFactory = None


def load_rl_data_for_diffusion(args, env=None, val_split_ratio=0.2, test_split_ratio=0.2):
    """
    Load RL data from Abiomed environment for diffusion training.
    Extracts next_observations + actions (same as Neural ODE approach).

    Args:
        args: Arguments containing task/model info
        env: Optional pre-created environment
        val_split_ratio: Validation split ratio
        test_split_ratio: Test split ratio

    Returns:
        train_data, val_data, test_data as torch tensors
    """
    if env is None:
        # Create environment to access world model data
        env = AbiomedRLEnvFactory.create_env(
            model_name=args.model_name,
            model_path=args.model_path_wm,
            data_path=args.data_path_wm,
            max_steps=args.max_steps,
            gamma1=args.gamma1,
            gamma2=args.gamma2,
            gamma3=args.gamma3,
            action_space_type=args.action_space_type,
            normalize_rewards=True,
            noise_rate=args.noise_rate,
            noise_scale=args.noise_scale,
            seed=args.seed,
            device=args.device
        )

    # Load data directly from world model to avoid slow reward computation
    print("Loading data from world model (same format as RealNVP/VAE/KDE)...")

    # Get datasets from world model
    dataset_train = env.world_model.data_train
    dataset_val = env.world_model.data_val
    dataset_test = env.world_model.data_test

    # Concatenate all data
    all_x = torch.cat([dataset_train.data, dataset_val.data, dataset_test.data], axis=0)
    all_pl = torch.cat([dataset_train.pl, dataset_val.pl, dataset_test.pl], axis=0)

    timesteps = 6
    feature_dim = 12

    # Reshape observations to flat format (same as RealNVP/VAE/KDE)
    observation = all_x.reshape(-1, timesteps * feature_dim)  # [N, 72]

    # Process actions: take majority vote and normalize
    action_unnorm = np.array(env.world_model.unnorm_pl(all_pl))
    action_1 = np.array([
        np.bincount(np.rint(a).astype(int)).argmax() for a in action_unnorm
    ]).reshape(-1, 1)
    action = env.world_model.normalize_pl(torch.Tensor(action_1))  # [N, 1]

    # Concatenate observations + actions (same as RealNVP/VAE/KDE)
    X = np.concatenate([observation.numpy(), action.numpy()], axis=1)  # [N, 73]

    n_samples = len(X)
    print(f"Total samples: {n_samples}, Feature dimension: {X.shape[1]}")

    # Split data (random split, same as RealNVP/VAE/KDE)
    np.random.seed(42)
    val_test_size = int(n_samples * (args.val_ratio + args.test_ratio))
    val_size = int(n_samples * args.val_ratio)
    train_size = n_samples - val_test_size

    indices = np.random.permutation(n_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    X_train = X[train_indices]
    X_val = X[val_indices] if len(val_indices) > 0 else None
    X_test = X[test_indices]

    # Convert to tensors
    train_data = torch.FloatTensor(X_train)
    val_data = torch.FloatTensor(X_val) if X_val is not None else None
    test_data = torch.FloatTensor(X_test)

    print(f"Loaded RL data for diffusion:")
    print(f"  Train: {train_data.shape}")
    print(f"  Val: {val_data.shape if val_data is not None else 'None'}")
    print(f"  Test: {test_data.shape}")
    print(f"  Input dimension: {train_data.shape[1]} (current observations + actions)")

    return train_data, val_data, test_data


class TensorDataset(Dataset):
    """Simple dataset wrapper for torch tensors."""

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@dataclass
class TrainConfig:
    # Task settings
    task: str = "abiomed"
    model_name: str = "10min_1hr_all_data"
    model_path_wm: str = "/public/gormpo/models/10min_1hr_all_data_model.pth"
    data_path_wm: str = "/public/gormpo/10min_1hr_all_data.pkl"
    max_steps: int = 6
    gamma1: float = 0.0
    gamma2: float = 0.0
    gamma3: float = 0.0
    action_space_type: str = "continuous"
    noise_rate: float = 0.0
    noise_scale: float = 0.0

    # Data settings
    use_rl_data: bool = True
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    train_ratio: float = 0.7

    # Model architecture
    input_dim: int = 15  # Will be auto-calculated (obs_dim + action_dim)
    hidden_dim: int = 512
    time_embed_dim: int = 128
    num_hidden_layers: int = 3
    dropout: float = 0.0
    model_type: str = "mlp"  # mlp | transformer

    # Transformer hyperparams (only used if model_type == "transformer")
    d_model: int = 256
    nhead: int = 8
    tf_layers: int = 4
    ff_dim: int = 512

    # Diffusion parameters
    num_train_timesteps: int = 1000
    ddim_eta: float = 0.0  # 0 = deterministic DDIM

    # Training parameters
    epochs: int = 50
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 0.0
    log_every: int = 1
    checkpoint_every: int = 5

    # Hardware and output
    device: str = "cuda"
    devid: int = 0
    seed: int = 42
    out_dir: str = "checkpoints/diffusion"
    model_save_path: str = "/public/gormpo/models/abiomed/diffusion/"
    verbose: bool = True
    plot_results: bool = True

    # Early stopping
    patience: int = 10


def train(cfg: TrainConfig) -> None:
    """Train diffusion model for Abiomed OOD detection."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Set device
    if cfg.device.startswith('cuda'):
        device_str = f"{cfg.device}:{cfg.devid}" if ':' not in cfg.device else cfg.device
        if torch.cuda.is_available():
            device = torch.device(device_str)
        else:
            print("CUDA not available, using CPU")
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.model_save_path, exist_ok=True)

    # Load data
    print("\nLoading Abiomed RL data...")
    train_data, val_data, test_data = load_rl_data_for_diffusion(
        cfg,
        env=None,
        val_split_ratio=cfg.val_ratio,
        test_split_ratio=cfg.test_ratio
    )

    # Auto-calculate input dimension
    input_dim = train_data.shape[1]
    print(f"Auto-calculated input dimension: {input_dim}")

    # Create datasets and loaders
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Create model
    print(f"\nBuilding {cfg.model_type.upper()} model...")
    if cfg.model_type == "mlp":
        model = UnconditionalEpsilonMLP(
            target_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            time_embed_dim=cfg.time_embed_dim,
            num_hidden_layers=cfg.num_hidden_layers,
            dropout=cfg.dropout,
        ).to(device)
    elif cfg.model_type == "transformer":
        model = UnconditionalEpsilonTransformer(
            target_dim=input_dim,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.tf_layers,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            time_embed_dim=cfg.time_embed_dim,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Create scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule="linear",
        prediction_type="epsilon",
    )

    # Save scheduler config
    scheduler.save_pretrained(os.path.join(cfg.out_dir, "scheduler"))

    # Training loop
    print(f"\nStarting training for {cfg.epochs} epochs...")
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(cfg.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            clean = batch.to(device)
            bsz = clean.size(0)

            # Sample random timesteps
            t = torch.randint(0, cfg.num_train_timesteps, (bsz,), device=device)

            # Add noise
            noise = torch.randn_like(clean)
            noisy = scheduler.add_noise(clean, noise, t)

            # Predict noise
            pred_noise = model(noisy, t)
            loss = nn.functional.mse_loss(pred_noise, noise)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * bsz
            train_steps += 1

        train_loss /= len(train_data)

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                clean = batch.to(device)
                bsz = clean.size(0)

                # Sample random timesteps
                t = torch.randint(0, cfg.num_train_timesteps, (bsz,), device=device)

                # Add noise
                noise = torch.randn_like(clean)
                noisy = scheduler.add_noise(clean, noise, t)

                # Predict noise
                pred_noise = model(noisy, t)
                loss = nn.functional.mse_loss(pred_noise, noise)

                val_loss += loss.item() * bsz
                val_samples += bsz

                # Clear cache periodically
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        val_loss /= val_samples

        # Logging
        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{cfg.epochs}  "
                  f"Train Loss: {train_loss:.6f}  "
                  f"Val Loss: {val_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % cfg.checkpoint_every == 0:
            ckpt_path = os.path.join(cfg.out_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'cfg': cfg.__dict__,
                'target_dim': input_dim,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Save best checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'cfg': cfg.__dict__,
                'target_dim': input_dim,
            }, os.path.join(cfg.out_dir, "checkpoint.pt"))

            # Also save full checkpoint to model_save_path
            torch.save({
                'model_state_dict': model.state_dict(),
                'cfg': cfg.__dict__,
                'target_dim': input_dim,
            }, os.path.join(cfg.model_save_path, "checkpoint.pt"))
            print(f"New best model saved (val_loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1

        # Early stopping
        if cfg.patience > 0 and epochs_no_improve >= cfg.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs (no improvement for {cfg.patience} epochs)")
            break

    # Final save
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model.pt"))
    print(f"\nTraining completed. Final model saved to {cfg.out_dir}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Create OOD model and set threshold
    print("\nCreating OOD model and setting threshold...")
    from diffusion_ood import DiffusionOOD

    model.eval()
    ood_model = DiffusionOOD(
        model=model,
        scheduler=scheduler,
        device=str(device),
        num_inference_steps=20
    )

    # Set threshold using validation data
    anomaly_fraction = cfg.__dict__.get('anomaly_fraction', 0.01)
    val_data_tensor = torch.FloatTensor(val_data).to(device)
    ood_model.set_threshold(val_data_tensor, anomaly_fraction=anomaly_fraction, batch_size=cfg.batch_size)

    # Update checkpoint.pt to include the threshold (both in out_dir and model_save_path)
    for ckpt_dir in [cfg.out_dir, cfg.model_save_path]:
        checkpoint_path = os.path.join(ckpt_dir, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            print(f"\nUpdating {checkpoint_path} with threshold...")
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            checkpoint['threshold'] = ood_model.threshold
            checkpoint['anomaly_fraction'] = anomaly_fraction
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Threshold ({ood_model.threshold:.4f}) added to {checkpoint_path}")


def parse_args() -> TrainConfig:
    """Parse command line arguments and YAML config."""
    # Stage 1: parse only --config to get YAML path
    config_only = argparse.ArgumentParser(add_help=False)
    config_only.add_argument("--config", type=str, default="")
    known, _ = config_only.parse_known_args()

    # Load YAML if provided
    yaml_defaults = {}
    if known.config:
        try:
            with open(known.config, "r") as f:
                yaml_config = yaml.safe_load(f)
            if isinstance(yaml_config, dict):
                yaml_defaults = yaml_config
            print(f"Loaded config from: {known.config}")
        except Exception as e:
            print(f"Warning: failed to read YAML config: {e}")

    # Stage 2: Build full parser
    parser = argparse.ArgumentParser(
        description="Train Diffusion Model for Abiomed OOD Detection",
        parents=[config_only]
    )

    def dget(key, default):
        return yaml_defaults.get(key, default)

    # Task settings
    parser.add_argument("--task", type=str, default=dget("task", "abiomed"))
    parser.add_argument("--model-name", type=str, default=dget("model_name", "10min_1hr_all_data"))
    parser.add_argument("--model-path-wm", type=str, default=dget("model_path_wm", "/public/gormpo/models/10min_1hr_all_data_model.pth"))
    parser.add_argument("--data-path-wm", type=str, default=dget("data_path_wm", "/public/gormpo/10min_1hr_all_data.pkl"))
    parser.add_argument("--max-steps", type=int, default=dget("max_steps", 6))
    parser.add_argument("--gamma1", type=float, default=dget("gamma1", 0.0))
    parser.add_argument("--gamma2", type=float, default=dget("gamma2", 0.0))
    parser.add_argument("--gamma3", type=float, default=dget("gamma3", 0.0))
    parser.add_argument("--action-space-type", type=str, default=dget("action_space_type", "continuous"))
    parser.add_argument("--noise-rate", type=float, default=dget("noise_rate", 0.0))
    parser.add_argument("--noise-scale", type=float, default=dget("noise_scale", 0.0))

    # Data settings
    parser.add_argument("--use-rl-data", type=bool, default=dget("use_rl_data", True))
    parser.add_argument("--val-ratio", type=float, default=dget("val_ratio", 0.15))
    parser.add_argument("--test-ratio", type=float, default=dget("test_ratio", 0.15))
    parser.add_argument("--train-ratio", type=float, default=dget("train_ratio", 0.7))

    # Model architecture
    parser.add_argument("--input-dim", type=int, default=dget("input_dim", 15))
    parser.add_argument("--hidden-dim", type=int, default=dget("hidden_dim", 512))
    parser.add_argument("--time-embed-dim", type=int, default=dget("time_embed_dim", 128))
    parser.add_argument("--num-hidden-layers", type=int, default=dget("num_hidden_layers", 3))
    parser.add_argument("--dropout", type=float, default=dget("dropout", 0.0))
    parser.add_argument("--model-type", type=str, default=dget("model_type", "mlp"))
    parser.add_argument("--d-model", type=int, default=dget("d_model", 256))
    parser.add_argument("--nhead", type=int, default=dget("nhead", 8))
    parser.add_argument("--tf-layers", type=int, default=dget("tf_layers", 4))
    parser.add_argument("--ff-dim", type=int, default=dget("ff_dim", 512))

    # Diffusion parameters
    parser.add_argument("--num-train-timesteps", type=int, default=dget("num_train_timesteps", 1000))
    parser.add_argument("--ddim-eta", type=float, default=dget("ddim_eta", 0.0))

    # Training parameters
    parser.add_argument("--epochs", type=int, default=dget("epochs", 50))
    parser.add_argument("--batch-size", type=int, default=dget("batch_size", 256))
    parser.add_argument("--lr", type=float, default=dget("lr", 2e-4))
    parser.add_argument("--weight-decay", type=float, default=dget("weight_decay", 0.0))
    parser.add_argument("--log-every", type=int, default=dget("log_every", 1))
    parser.add_argument("--checkpoint-every", type=int, default=dget("checkpoint_every", 5))

    # Hardware and output
    parser.add_argument("--device", type=str, default=dget("device", "cuda"))
    parser.add_argument("--devid", type=int, default=dget("devid", 0))
    parser.add_argument("--seed", type=int, default=dget("seed", 42))
    parser.add_argument("--out-dir", type=str, default=dget("out_dir", "checkpoints/diffusion"))
    parser.add_argument("--model-save-path", type=str, default=dget("model_save_path", "/public/gormpo/models/abiomed/diffusion/"))
    parser.add_argument("--verbose", type=bool, default=dget("verbose", True))
    parser.add_argument("--plot-results", type=bool, default=dget("plot_results", True))

    # Early stopping
    parser.add_argument("--patience", type=int, default=dget("patience", 10))

    args = parser.parse_args()

    # Convert args to TrainConfig
    config = TrainConfig(
        task=args.task,
        model_name=args.model_name,
        model_path_wm=args.model_path_wm,
        data_path_wm=args.data_path_wm,
        max_steps=args.max_steps,
        gamma1=args.gamma1,
        gamma2=args.gamma2,
        gamma3=args.gamma3,
        action_space_type=args.action_space_type,
        noise_rate=args.noise_rate,
        noise_scale=args.noise_scale,
        use_rl_data=args.use_rl_data,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        train_ratio=args.train_ratio,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        time_embed_dim=args.time_embed_dim,
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
        model_type=args.model_type,
        d_model=args.d_model,
        nhead=args.nhead,
        tf_layers=args.tf_layers,
        ff_dim=args.ff_dim,
        num_train_timesteps=args.num_train_timesteps,
        ddim_eta=args.ddim_eta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        device=args.device,
        devid=args.devid,
        seed=args.seed,
        out_dir=args.out_dir,
        model_save_path=args.model_save_path,
        verbose=args.verbose,
        plot_results=args.plot_results,
        patience=args.patience,
    )

    print("\nTraining Configuration:")
    print(f"  Task: {config.task}")
    print(f"  Model: {config.model_type}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Device: {config.device}:{config.devid}")
    print(f"  Output dir: {config.out_dir}")

    return config


if __name__ == "__main__":
    config = parse_args()
    train(config)
