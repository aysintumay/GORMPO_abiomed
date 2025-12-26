import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse
import yaml
import os
import sys
import pickle
import seaborn as sns
import gym
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 2 levels up
from cormpo.common.buffer import ReplayBuffer
from cormpo.mbpo_kde.kde import get_env_data, load_data


class MLP(nn.Module):
    """Multi-layer perceptron for coupling layer transformations."""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CouplingLayer(nn.Module):
    """RealNVP coupling layer with affine transformation."""

    def __init__(self, input_dim: int, hidden_dims: List[int], mask: torch.Tensor):
        super().__init__()

        self.register_buffer('mask', mask)
        masked_dim = int(mask.sum().item())

        # Scale and translation networks
        self.scale_net = MLP(masked_dim, hidden_dims, input_dim - masked_dim).to(self.mask.device)
        self.translate_net = MLP(masked_dim, hidden_dims, input_dim - masked_dim).to(self.mask.device)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through coupling layer.

        Args:
            x: Input tensor
            reverse: If True, compute inverse transformation

        Returns:
            Transformed tensor and log determinant of Jacobian
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.mask.device)
        x_masked = x * self.mask
        x_unmasked = x * (1 - self.mask)

        # Get scale and translation from masked components
        x_masked_input = x_masked[:, self.mask.bool()]
        scale = self.scale_net(x_masked_input)
        translate = self.translate_net(x_masked_input)

        if not reverse:
            # Forward transformation: y = x * exp(s) + t
            log_scale = torch.tanh(scale)  # Stabilize scale
            x_unmasked_vals = x_unmasked[:, ~self.mask.bool()]
            y_unmasked = x_unmasked_vals * torch.exp(log_scale) + translate

            # Reconstruct full tensor
            y = x.clone()
            y[:, ~self.mask.bool()] = y_unmasked

            log_det = log_scale.sum(dim=1)

        else:
            # Inverse transformation: x = (y - t) * exp(-s)
            log_scale = torch.tanh(scale)
            x_unmasked_vals = x_unmasked[:, ~self.mask.bool()]
            y_unmasked = (x_unmasked_vals - translate) * torch.exp(-log_scale)

            # Reconstruct full tensor
            y = x.clone()
            y[:, ~self.mask.bool()] = y_unmasked

            log_det = -log_scale.sum(dim=1)

        return y, log_det


class RealNVP(nn.Module):
    """RealNVP normalizing flow model for density estimation."""

    def __init__(
        self,
        input_dim: int =2,
        num_layers: int = 6,
        hidden_dims: List[int] = [256, 256],
        device: str = 'cpu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.device = device

        # Create alternating masks
        self.masks = []
        for i in range(num_layers):
            mask = torch.zeros(input_dim)
            if i % 2 == 0:
                mask[::2] = 1  # Even indices
            else:
                mask[1::2] = 1  # Odd indices
            self.masks.append(mask)

        # Create coupling layers
        self.coupling_layers = nn.ModuleList([
            CouplingLayer(input_dim, hidden_dims, mask.to(device))
            for mask in self.masks
        ])

        # Prior distribution parameters (standard normal)
        self.register_buffer('prior_mean', torch.zeros(input_dim))
        self.register_buffer('prior_std', torch.ones(input_dim))

        # Threshold for anomaly detection
        self.threshold = None

    def _apply(self, fn):
        """Override _apply to update self.device when model is moved."""
        super()._apply(fn)
        # Update self.device to match the actual device of parameters
        if len(list(self.parameters())) > 0:
            self.device = next(self.parameters()).device
        return self

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the flow.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            reverse: If True, sample from the model

        Returns:
            Transformed tensor and log determinant of Jacobian
        """
        # Use the actual device of model parameters instead of self.device
        model_device = next(self.parameters()).device
        log_det_total = torch.zeros(x.shape[0], device=model_device)

        if not reverse:
            # Forward: data -> latent
            z = x
            for layer in self.coupling_layers:
                z, log_det = layer(z, reverse=False)
                log_det_total += log_det
        else:
            # Reverse: latent -> data
            z = x
            for layer in reversed(self.coupling_layers):
                z, log_det = layer(z, reverse=True)
                log_det_total += log_det

        return z, log_det_total

    def _compute_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability tensor (for training)."""
        z, log_det = self.forward(x, reverse=False)

        # Log probability under prior (standard normal)
        log_prior = -0.5 * (
            z.pow(2).sum(dim=1) +
            self.input_dim * np.log(2 * np.pi)
        )

        return log_prior + log_det

    def score_samples(self, x: torch.Tensor, device='cuda') -> torch.Tensor:
        """Compute log probability of data points (returns numpy array)."""
        z, log_det = self.forward(x, reverse=False)

        # Log probability under prior (standard normal)
        log_prior = -0.5 * (
            z.pow(2).sum(dim=1) +
            self.input_dim * np.log(2 * np.pi)
        )
        # print(log_det.max().item(), np.exp(log_det.max().item()))

        return (log_prior + log_det).detach().cpu().numpy()

    def sample(self, num_samples: int) -> torch.Tensor:
        """Generate samples from the model."""
        with torch.no_grad():
            # Sample from prior
            model_device = next(self.parameters()).device
            z = torch.randn(num_samples, self.input_dim, device=model_device)

            # Transform to data space
            x, _ = self.forward(z, reverse=True)

        return x

    def fit(
        self,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 15,
        verbose: bool = True
    ) -> dict:
        """
        Train the RealNVP model.

        Args:
            train_data: Training dataset
            val_data: Validation dataset for threshold selection
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            patience: Early stopping patience
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=patience//2
        )

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_data),
            batch_size=batch_size,
            shuffle=True
        )

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('-inf')
        patience_counter = 0

        self.train()
        val_data = (val_data + 1e-3)
        for epoch in range(epochs):
            train_loss = 0.0
            num_batches = 0

            for batch_data, in train_loader:
                batch_data = batch_data.to(self.device)
                # noise = torch.rand_like(batch_data)*0.01   # uniform [0,1)
                batch_data = (batch_data + 1e-3)
                optimizer.zero_grad()

                # Compute negative log likelihood
                log_prob = self._compute_log_prob(batch_data)
                loss = -log_prob.mean()

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches

            # Validation
            self.eval()
            with torch.no_grad():
                # noise = torch.rand_like(val_data)*0.01  # uniform [0,1)
                
                val_log_prob = self.score_samples(val_data.to(self.device))
                val_loss = -val_log_prob.mean().item()
                # print("VALIDATION", val_log_prob.max(), np.exp(val_log_prob.max().item()))
            self.train()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            scheduler.step(val_loss)

            if verbose and epoch % 5 == 0:
                print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

            # Early stopping
            if val_loss > best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch}')
                break

        # Set threshold based on validation data
        self.set_threshold(val_data)

        return history

    def set_threshold(
        self,
        val_data: torch.Tensor,
        anomaly_fraction: float = 0.01
    ):
        """
        Set threshold for anomaly detection based on validation data.

        Args:
            val_data: Validation dataset (assumed to be normal data)
            anomaly_fraction: Fraction of validation data to classify as anomalies
        """
        self.eval()
        with torch.no_grad():
            log_probs = self.score_samples(val_data.to(self.device))

        # Convert to tensor if numpy array
        if isinstance(log_probs, np.ndarray):
            log_probs = torch.from_numpy(log_probs)

        # Set threshold as percentile of validation log probabilities
        self.threshold = torch.quantile(log_probs.float(), anomaly_fraction).item()

        print(f'Threshold set to {self.threshold:.4f} '
              f'(marking {anomaly_fraction*100:.1f}% of validation data as anomalies)')

    def predict_anomaly(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict anomalies based on log probability threshold.

        Args:
            x: Input data

        Returns:
            Boolean tensor indicating anomalies (True = anomaly)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")

        self.eval()
        with torch.no_grad():
            log_probs = self.score_samples(x.to(self.device))

        return log_probs < self.threshold
    
    def predict(self, X):
        """
        Predict anomalies based on threshold

        Args:
            X: Test data

        Returns:
            predictions: 1 for normal, -1 for anomaly
        """
        scores = self.score_samples(X)
        return np.where(scores.cpu() >= self.threshold, 1, -1)
    
    def evaluate_anomaly_detection(
    self,
    normal_data: torch.Tensor,
    anomaly_data: torch.Tensor,
    plot: bool = True) -> dict:
        """
        Evaluate anomaly detection performance.
        """
        self.eval()

        # >>> Get the true device of the model <<<
        model_device = next(self.parameters()).device

        # >>> Move data to the same device as the model <<<
        normal_data = normal_data.to(model_device)
        anomaly_data = anomaly_data.to(model_device)

        print("Model device:", model_device)
        print("Normal data device:", normal_data.device)

        with torch.no_grad():
            normal_log_probs = self.score_samples(normal_data).cpu().numpy()
            anomaly_log_probs = self.score_samples(anomaly_data).cpu().numpy()

        # Create labels (0 = normal, 1 = anomaly)
        y_true = np.concatenate([
            np.zeros(len(normal_log_probs)),
            np.ones(len(anomaly_log_probs))
        ])

        # Use negative log prob as anomaly score
        y_scores = np.concatenate([-normal_log_probs, -anomaly_log_probs])

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Compute accuracy with current threshold
        if self.threshold is not None:
            predictions = np.concatenate([
                normal_log_probs < self.threshold,
                anomaly_log_probs < self.threshold
            ])
            accuracy = (predictions == y_true).mean()
        else:
            accuracy = None

        results = {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'normal_log_prob_mean': normal_log_probs.mean(),
            'normal_log_prob_std': normal_log_probs.std(),
            'anomaly_log_prob_mean': anomaly_log_probs.mean(),
            'anomaly_log_prob_std': anomaly_log_probs.std()
        }

        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Anomaly Detection')
            plt.legend()
            plt.grid(True)
            plt.show()

        return results

    def save_model(self, save_path: str, train_data: torch.Tensor = None, norm_stats: dict = None):
        """
        Save the RealNVP model and metadata.

        Args:
            save_path: Base path for saving (without extension)
            train_data: Training data for computing statistics
            norm_stats: Dictionary containing data_mean and data_std for normalization
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save model state dict
        torch.save(self.state_dict(), f"{save_path}_model.pth")

        # Calculate the logprobs in training data if provided
        if train_data is not None:
            self.eval()
            with torch.no_grad():
                train_log_probs = self.score_samples(train_data.to(self.device))
            mean_score = train_log_probs.mean()
            std_score = train_log_probs.std()
        else:
            mean_score = 0.0
            std_score = 0.0

        # Save metadata (threshold and config)
        metadata = {
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'num_layers': self.num_layers,
            'device': self.device,
            "mean": mean_score,
            "std": std_score
        }

        # Add normalization statistics if provided
        if norm_stats is not None:
            metadata['data_mean'] = norm_stats['data_mean']
            metadata['data_std'] = norm_stats['data_std']

        with open(f"{save_path}_meta_data.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Model saved to: {save_path}_model.pth")
        print(f"Metadata saved to: {save_path}_meta_data.pkl")

    @classmethod
    def load_model(cls, save_path: str, hidden_dims: List[int] = [256, 256]):
        """
        Load a saved RealNVP model.

        Args:
            save_path: Base path for loading (without extension)
            hidden_dims: Hidden layer dimensions (must match saved model)

        Returns:
            Loaded RealNVP model
        """
        # Load metadata
        with open(f"{save_path}_meta_data.pkl", 'rb') as f:
            metadata = pickle.load(f)

        # Create model with saved configuration
        model = cls(
            input_dim=metadata['input_dim'],
            num_layers=metadata['num_layers'],
            hidden_dims=hidden_dims,
            device=metadata['device']
        )

        # Load model state dict
        model.load_state_dict(torch.load(f"{save_path}_model.pth", map_location=metadata['device']))
        # model.to(cls.device)

        # Restore threshold
        model.threshold = metadata['threshold']

        print(f"Model loaded from: {save_path}_model.pth")
        print(f"Metadata loaded from: {save_path}_meta_data.pkl")
        print(f"Threshold: {model.threshold}")

        model_dict = {'model': model, 'thr': model.threshold, 'mean': metadata["mean"], 'std': metadata["std"]}

        # Add normalization statistics if available
        if 'data_mean' in metadata and 'data_std' in metadata:
            model_dict['data_mean'] = metadata['data_mean']
            model_dict['data_std'] = metadata['data_std']

        return model_dict


def create_synthetic_data(n_samples=1000, dim=2, anomaly_type="outlier"):
    """
    Generate synthetic normal and anomalous data in arbitrary dimensions.

    Args:
        n_samples (int): number of normal samples
        dim (int): dimensionality of data
        anomaly_type (str): "outlier" or "uniform"

    Returns:
        (torch.FloatTensor, torch.FloatTensor): normal_data, anomaly_data
    """
    normal_data = []
    for _ in range(n_samples):
        if np.random.rand() < 0.7:
            # Main cluster around 0
            mean = np.zeros(dim)
            cov = np.eye(dim)                      # identity covariance
            sample = np.random.multivariate_normal(mean, cov, 1)
        else:
            # Secondary cluster around 3
            mean = np.ones(dim) * 3
            cov = 0.5 * np.eye(dim)                # smaller spread
            sample = np.random.multivariate_normal(mean, cov, 1)
        normal_data.append(sample[0])

    normal_data = np.array(normal_data)

    # Anomalous data
    if anomaly_type == "outlier":
        mean = np.ones(dim) * 10
        cov = 2 * np.eye(dim)
        anomaly_data = np.random.multivariate_normal(mean, cov, n_samples // 5)
    elif anomaly_type == "uniform":
        anomaly_data = np.random.uniform(-5, 8, (n_samples // 5, dim))
    else:
        raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    return torch.FloatTensor(normal_data), torch.FloatTensor(anomaly_data)


def load_rl_data_for_kde(args, env=None, val_split_ratio=0.2):
    """
    Load RL dataset and prepare next_observations + actions for RealNVP training.
    Uses the load_data function from kde.py for consistent data handling.

    Args:
        args: Arguments containing data_path, obs_dim, action_dim, and task
        env: Environment object (required for Abiomed datasets when data_path is None)
        val_split_ratio: Fraction of data for validation (default: 0.2)

    Returns:
        Tuple of (train_kde_input, val_kde_input, test_kde_input, kde_input_dim, norm_stats):
            - train_kde_input: torch.FloatTensor of shape (n_train, obs_dim + action_dim)
            - val_kde_input: torch.FloatTensor of shape (n_val, obs_dim + action_dim)
            - test_kde_input: torch.FloatTensor of shape (n_test, obs_dim + action_dim)
            - kde_input_dim: int, dimension of concatenated input (obs_dim + action_dim)
            - norm_stats: dict with normalization statistics
    """
    # Use load_data from kde.py which handles the train/val/test split
    data_splits = load_data(
        data_path=args.data_path,
        test_size=0.2,  # 20% for test
        validation_size=val_split_ratio,  # user-specified validation size
        args=args
    )

    # Extract the splits (already concatenated obs + actions)
    train_kde_input = torch.FloatTensor(data_splits['X_train'])
    val_kde_input = torch.FloatTensor(data_splits['X_val']) if data_splits['X_val'] is not None else None
    test_kde_input = torch.FloatTensor(data_splits['X_test'])

    # Normalize the data using training set statistics
    print("Normalizing data using training set statistics...")
    data_mean = train_kde_input.mean(dim=0, keepdim=True)
    data_std = train_kde_input.std(dim=0, keepdim=True)
    # Avoid division by zero
    data_std = torch.clamp(data_std, min=1e-6)

    train_kde_input = (train_kde_input - data_mean) / data_std
    if val_kde_input is not None:
        val_kde_input = (val_kde_input - data_mean) / data_std
    test_kde_input = (test_kde_input - data_mean) / data_std

    print(f"✓ Data normalized - Mean: {data_mean.mean().item():.4f}, Std: {data_std.mean().item():.4f}")

    # Calculate input dimension for RealNVP model
    kde_input_dim = train_kde_input.shape[1]

    print(f"✓ RealNVP training data shape: {train_kde_input.shape}")
    if val_kde_input is not None:
        print(f"✓ RealNVP validation data shape: {val_kde_input.shape}")
    print(f"✓ RealNVP test data shape: {test_kde_input.shape}")
    print(f"✓ RealNVP input dimension: {kde_input_dim}")

    # Store normalization statistics as regular Python lists for JSON serialization
    norm_stats = {
        'data_mean': data_mean.squeeze().cpu().numpy().tolist(),
        'data_std': data_std.squeeze().cpu().numpy().tolist()
    }

    return train_kde_input, val_kde_input, test_kde_input, kde_input_dim, norm_stats


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def parse_args():
    """Parse command line arguments."""
    print("Running", __file__)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="configs/realnvp/hopper.yaml")
    config_args, remaining_argv = config_parser.parse_known_args()
    if config_args.config:
        with open(config_args.config, "r") as f:
            config = yaml.safe_load(f)
            config = {k.replace("-", "_"): v for k, v in config.items()}
    else:
        config = {}
    parser = argparse.ArgumentParser(parents=[config_parser])
   
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Overrides config file.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs. Overrides config file.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output. Overrides config file.')

    # RL dataset specific arguments
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to RL dataset file (pickle or npz)')
    parser.add_argument('--task', type=str, default='hopper-medium-v2',
                        help='Task name (e.g., abiomed, halfcheetah-medium-v2)')
    parser.add_argument('--obs_dim', type=int, nargs='+', default=[11],
                        help='Observation shape (default: [17] for HalfCheetah)')
    parser.add_argument('--action_dim', type=int, default=3,
                        help='Action dimension (default: 6 for HalfCheetah)')

    # Abiomed environment specific arguments
    parser.add_argument('--model_name', type=str, default='10min_1hr_all_data',
                        help='Abiomed model name (for Abiomed environment)')
    parser.add_argument('--model_path_wm', type=str, default=None,
                        help='Path to Abiomed world model (for Abiomed environment)')
    parser.add_argument('--data_path_wm', type=str, default=None,
                        help='Path to Abiomed data for world model (for Abiomed environment)')
    parser.add_argument('--max_steps', type=int, default=6,
                        help='Maximum steps per episode (for Abiomed environment)')
    parser.add_argument('--gamma1', type=float, default=0.0,
                        help='Gamma1 parameter (for Abiomed environment)')
    parser.add_argument('--gamma2', type=float, default=0.0,
                        help='Gamma2 parameter (for Abiomed environment)')
    parser.add_argument('--gamma3', type=float, default=0.0,
                        help='Gamma3 parameter (for Abiomed environment)')
    parser.add_argument('--noise_rate', type=float, default=0.0,
                        help='Noise rate (for Abiomed environment)')
    parser.add_argument('--noise_scale', type=float, default=0.0,
                        help='Noise scale (for Abiomed environment)')
    parser.add_argument(
        "--action_space_type",
        type=str,
        default="continuous",
        choices=["continuous", "discrete"],
        help="Type of action space for the environment",
    )
    parser.add_argument(
        "--devid", type=int, default=0, help="GPU device ID (if using GPU)"
    )
    parser.add_argument("--env", type=str, default="abiomed")
    parser.add_argument(
        "--temporal_split",
        action="store_true",
        help="Use temporal split (no shuffle) for time series",
    )
    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)
    args.config = config
   
    return args

def plot_likelihood_distributions(
    model,
    train_data,
    val_data,
    ood_data=None,
    thr= None,
    title="Likelihood Distribution",
    savepath=None,
    bins=50
):
    """
    Visualize log-likelihood distributions for train, val, and OOD data.

    Args:
        model: density model with .score_samples(X) method (returns log probs)
        train_data: np.ndarray or torch.Tensor, in-distribution training set
        val_data:   np.ndarray or torch.Tensor, held-out validation set
        ood_data:   np.ndarray or torch.Tensor, optional OOD dataset
        title: str, title for the plot
        savepath: str, optional path to save figure
        bins: int, number of histogram bins
    """
    # --- Compute log-likelihoods ---
    logp_train = model.score_samples(train_data).detach().cpu().numpy()
    logp_val   = model.score_samples(val_data).detach().cpu().numpy()
    logp_ood   = None
    if ood_data is not None:
        logp_ood = model.score_samples(ood_data).detach().cpu().numpy()


    # --- Plot ---
    plt.figure(figsize=(8, 5))
    sns.histplot(logp_train, bins=bins, color="blue", alpha=0.4, label="Train", kde=True)
    sns.histplot(logp_val, bins=bins, color="green", alpha=0.4, label="Validation", kde=True)
    plt.axvline(x=thr, color='tab:red', linestyle='--', label='Threshold')
    plt.xlabel("Log-likelihood", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.tight_layout(pad=2.0)
    plt.savefig(f"figures/train_distribution_kde.png", dpi=300, bbox_inches="tight")
    print(f"Saved figure at figures/train_distribution_kde.png")
    plt.figure(figsize=(8, 5))


    if logp_ood is not None:
        sns.histplot(logp_ood, bins=bins, color="red", alpha=0.4, label="Test", kde=True)
        plt.axvline(x=thr, color='tab:red', linestyle='--', label='Threshold')

    plt.xlabel("Log-likelihood", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.tight_layout(pad=2.0)
    plt.savefig(f"figures/distribution_kde.png", dpi=300, bbox_inches="tight")
    print(f"Saved figure at figures/distribution_kde.png")


def plot_val_test_id_distribution(
    model,
    val_data,
    test_id_data,
    thr=None,
    title="Validation vs Test ID Log-Likelihood Distribution",
    savepath="figures/val_test_id_distribution.png",
    bins=50
):
    """
    Visualize log-likelihood distributions for validation and test ID data in one plot.

    Args:
        model: density model with .score_samples(X) method (returns log probs)
        val_data: np.ndarray or torch.Tensor, validation set
        test_id_data: np.ndarray or torch.Tensor, in-distribution test set
        thr: float, threshold value to indicate on the plot
        title: str, title for the plot
        savepath: str, path to save figure
        bins: int, number of histogram bins
    """
    # --- Compute log-likelihoods ---
    logp_val = model.score_samples(val_data).detach().cpu().numpy()
    logp_test_id = model.score_samples(test_id_data).detach().cpu().numpy()

    # --- Plot ---
    plt.figure(figsize=(7, 6))
    sns.histplot(logp_val, bins=bins, color="green", alpha=0.5, label="Validation", kde=True, stat="density")
    sns.histplot(logp_test_id, bins=bins, color="blue", alpha=0.5, label="Test ID", kde=True, stat="density")

    if thr is not None:
        plt.axvline(x=thr, color='tab:red', linestyle='--', linewidth=2, label=f'Threshold ({thr:.3f})')

    plt.xlabel("Log-Likelihood", fontsize=16, labelpad=10)
    plt.ylabel("Density", fontsize=16, labelpad=10)
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.legend(fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=2.5)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    print(f"Saved figure at {savepath}")
    plt.close()

    # Print statistics
    print(f"\nValidation Log-Likelihood Statistics:")
    print(f"  Mean: {logp_val.mean():.4f}")
    print(f"  Std:  {logp_val.std():.4f}")
    print(f"  Min:  {logp_val.min():.4f}")
    print(f"  Max:  {logp_val.max():.4f}")

    print(f"\nTest ID Log-Likelihood Statistics:")
    print(f"  Mean: {logp_test_id.mean():.4f}")
    print(f"  Std:  {logp_test_id.std():.4f}")
    print(f"  Min:  {logp_test_id.min():.4f}")
    print(f"  Max:  {logp_test_id.max():.4f}")

    if thr is not None:
        val_below_thr = (logp_val < thr).sum() / len(logp_val) * 100
        test_below_thr = (logp_test_id < thr).sum() / len(logp_test_id) * 100
        print(f"\nPercentage below threshold:")
        print(f"  Validation: {val_below_thr:.2f}%")
        print(f"  Test ID:    {test_below_thr:.2f}%")

def create_ood_test(data, model, percentage=[0.1, 0.3, 0.5, 0.7, 0.9], noise_std=0.1):
    """
    Create OOD test sets with different OOD ratios and compute ROC AUC metrics.

    Args:
        data: torch.Tensor or np.ndarray, assumed to be (mostly) ID data
        model: trained RealNVP model with .predict and .score_samples
        percentage: list of OOD fractions (between 0 and 1)
        noise_std: standard deviation of Gaussian noise to create OOD samples

    Returns:
        data_dict: dict[perc] -> torch.FloatTensor test set on model.device
        res_dict:  dict[perc] -> dict with metrics (mean_score, roc_auc, accuracy, id_accuracy, ood_accuracy)
    """
    data_dict = {}
    res_dict = {}

    # Make a CPU tensor for all indexing / numpy ops
    if isinstance(data, torch.Tensor):
        data_cpu = data.detach().cpu()
    else:
        data_cpu = torch.as_tensor(data, dtype=torch.float32)

    # Get model device from parameters
    model_device = next(model.parameters()).device

    # Get inlier predictions (1 = normal, -1 = anomaly), using model's device internally
    predictions_tr = model.predict(data)  # returns numpy array

    # Indices of in-distribution points
    idx_in = np.where(predictions_tr == 1)[0]
    if len(idx_in) == 0:
        raise RuntimeError("No in-distribution points (predictions == 1) found in data.")

    # Take a 20% random subset of inliers as our base pool
    subset_size = max(1, int(0.2 * len(idx_in)))
    chosen_in = np.random.choice(idx_in, size=subset_size, replace=False)
    small_data = data_cpu[chosen_in].numpy()   # shape: (subset_size, D)

    # For each OOD percentage, build a mixed test set
    for perc in percentage:
        # perc = fraction of OOD samples in the test set
        test_size = subset_size
        n_ood = int(perc * test_size)
        n_id = test_size - n_ood

        # ID part: sample from small_data
        if n_id > 0:
            id_idx = np.random.choice(test_size, size=n_id, replace=False)
            id_part = small_data[id_idx]
        else:
            id_part = np.empty((0, small_data.shape[1]), dtype=small_data.dtype)

        # OOD part: noisy copies of small_data
        if n_ood > 0:
            ood_idx = np.random.choice(test_size, size=n_ood, replace=True)
            ood_base = small_data[ood_idx]
            noisy_part = ood_base + np.random.normal(0, noise_std, ood_base.shape)
        else:
            noisy_part = np.empty((0, small_data.shape[1]), dtype=small_data.dtype)

        mixed = np.concatenate([id_part, noisy_part], axis=0)

        # Move mixed test set onto model device for scoring
        mixed_tensor = torch.FloatTensor(mixed).to(model_device)
        data_dict[perc] = mixed_tensor

        # Create ground truth labels (0 = ID/normal, 1 = OOD/anomaly)
        y_true = np.concatenate([
            np.zeros(n_id),  # ID samples
            np.ones(n_ood)   # OOD samples
        ])

        # Score samples
        with torch.no_grad():
            scores_test = model.score_samples(mixed_tensor.to(model.device)).cpu().numpy()

        mean_score = scores_test.mean()

        # Compute ROC AUC if we have both classes
        if n_id > 0 and n_ood > 0:
            # Use negative log prob as anomaly score (higher = more anomalous)
            y_scores = -scores_test
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = None

        # Compute accuracy metrics using model's threshold
        if model.threshold is not None:
            # Predictions: log_prob < threshold means anomaly
            predictions = scores_test < model.threshold

            # Overall accuracy
            accuracy = (predictions == y_true).mean() if len(y_true) > 0 else None

            # ID accuracy (correctly classified as normal)
            if n_id > 0:
                id_predictions = scores_test[:n_id] >= model.threshold
                id_accuracy = id_predictions.mean()
            else:
                id_accuracy = None

            # OOD accuracy (correctly classified as anomaly)
            if n_ood > 0:
                ood_predictions = scores_test[n_id:] < model.threshold
                ood_accuracy = ood_predictions.mean()
            else:
                ood_accuracy = None
        else:
            accuracy = None
            id_accuracy = None
            ood_accuracy = None

        # Store comprehensive results
        res_dict[perc] = {
            'mean_score': mean_score,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'id_accuracy': id_accuracy,
            'ood_accuracy': ood_accuracy,
            'n_id': n_id,
            'n_ood': n_ood
        }

        print(f"Percentage OOD: {perc:.1%}")
        print(f"  Mean Score: {mean_score:.4f}")
        if roc_auc is not None:
            print(f"  ROC AUC: {roc_auc:.4f}")
        if accuracy is not None:
            print(f"  Overall Accuracy: {accuracy:.4f}")
        if id_accuracy is not None:
            print(f"  ID Accuracy: {id_accuracy:.4f}")
        if ood_accuracy is not None:
            print(f"  OOD Accuracy: {ood_accuracy:.4f}")
        print()

    return data_dict, res_dict


def create_ood_test_multiple_noise_levels(
    data,
    model,
    percentage=[0.1, 0.3, 0.5, 0.7, 0.9],
    noise_levels=[0.05, 0.1, 0.5]
):
    """
    Create multiple OOD test sets with different noise levels.

    Args:
        data: torch.Tensor or np.ndarray, assumed to be (mostly) ID data
        model: trained RealNVP model with .predict and .score_samples
        percentage: list of OOD fractions (between 0 and 1)
        noise_levels: list of noise standard deviations for creating OOD samples

    Returns:
        results_by_noise: dict[noise_std] -> {
            'data_dict': dict[perc] -> torch.FloatTensor test set,
            'res_dict': dict[perc] -> dict with metrics
        }
    """
    results_by_noise = {}

    for noise_std in noise_levels:
        print(f"\n{'='*80}")
        print(f"Creating OOD test sets with noise level: {noise_std}")
        print(f"{'='*80}")

        data_dict, res_dict = create_ood_test(
            data=data,
            model=model,
            percentage=percentage,
            noise_std=noise_std
        )

        results_by_noise[noise_std] = {
            'data_dict': data_dict,
            'res_dict': res_dict
        }

    return results_by_noise


def plot_ood_metrics(res_dict, test_id_score=None, save_dir="figures"):
    """
    Plot comprehensive OOD detection metrics.

    Args:
        res_dict: Dictionary with OOD percentages as keys and metrics as values
        test_id_score: Optional mean score for pure ID test data (0% OOD)
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data including pure ID test (0% OOD) if provided
    percentages = sorted(res_dict.keys())
    if test_id_score is not None:
        percentages = [0.0] + percentages

    mean_scores = []
    roc_aucs = []
    accuracies = []
    id_accuracies = []
    ood_accuracies = []

    for perc in percentages:
        if perc == 0.0 and test_id_score is not None:
            # Pure ID test data
            mean_scores.append(test_id_score)
            roc_aucs.append(None)  # No ROC AUC for pure ID data
            accuracies.append(None)
            id_accuracies.append(None)
            ood_accuracies.append(None)
        else:
            metrics = res_dict[perc]
            mean_scores.append(metrics['mean_score'])
            roc_aucs.append(metrics['roc_auc'])
            accuracies.append(metrics['accuracy'])
            id_accuracies.append(metrics['id_accuracy'])
            ood_accuracies.append(metrics['ood_accuracy'])

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('OOD Detection Performance Metrics', fontsize=18, fontweight='bold', y=0.995)

    # Plot 1: Mean Log-Likelihood vs OOD Ratio
    ax1 = axes[0, 0]
    ax1.plot([p * 100 for p in percentages], mean_scores, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('OOD Percentage (%)', fontsize=14)
    ax1.set_ylabel('Mean Log-Likelihood', fontsize=14)
    ax1.set_title('Log-Likelihood vs OOD Ratio', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([p * 100 for p in percentages])
    ax1.tick_params(labelsize=12)

    # Plot 2: ROC AUC vs OOD Ratio
    ax2 = axes[0, 1]
    valid_percentages = [p for p, auc in zip(percentages, roc_aucs) if auc is not None]
    valid_aucs = [auc for auc in roc_aucs if auc is not None]
    if valid_aucs:
        ax2.plot([p * 100 for p in valid_percentages], valid_aucs, 'o-', linewidth=2, markersize=8, color='forestgreen')
        ax2.axhline(y=0.5, color='r', linestyle='--', label='Random Classifier', alpha=0.7)
        ax2.set_xlabel('OOD Percentage (%)', fontsize=14)
        ax2.set_ylabel('ROC AUC', fontsize=14)
        ax2.set_title('ROC AUC vs OOD Ratio', fontsize=15, fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        ax2.set_xticks([p * 100 for p in valid_percentages])
        ax2.tick_params(labelsize=12)

    # Plot 3: Accuracy Metrics vs OOD Ratio
    ax3 = axes[1, 0]
    valid_percentages_acc = [p for p, acc in zip(percentages, accuracies) if acc is not None]
    valid_accuracies = [acc for acc in accuracies if acc is not None]
    valid_id_acc = [acc for acc in id_accuracies if acc is not None]
    valid_ood_acc = [acc for acc in ood_accuracies if acc is not None]

    if valid_accuracies:
        ax3.plot([p * 100 for p in valid_percentages_acc], valid_accuracies, 'o-', linewidth=2, markersize=8, label='Overall Accuracy', color='purple')
        ax3.plot([p * 100 for p in valid_percentages_acc], valid_id_acc, 's-', linewidth=2, markersize=8, label='ID Accuracy', color='blue')
        ax3.plot([p * 100 for p in valid_percentages_acc], valid_ood_acc, '^-', linewidth=2, markersize=8, label='OOD Accuracy', color='red')
        ax3.set_xlabel('OOD Percentage (%)', fontsize=14)
        ax3.set_ylabel('Accuracy', fontsize=14)
        ax3.set_title('Accuracy Metrics vs OOD Ratio', fontsize=15, fontweight='bold')
        ax3.set_ylim([0, 1.05])
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=12)
        ax3.set_xticks([p * 100 for p in valid_percentages_acc])
        ax3.tick_params(labelsize=12)

    # Plot 4: Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create table data
    table_data = []
    headers = ['OOD %', 'Log-Like', 'ROC AUC', 'Accuracy', 'ID Acc', 'OOD Acc']

    for i, perc in enumerate(percentages):
        row = [f"{perc:.1%}"]
        row.append(f"{mean_scores[i]:.3f}")
        row.append(f"{roc_aucs[i]:.3f}" if roc_aucs[i] is not None else "N/A")
        row.append(f"{accuracies[i]:.3f}" if accuracies[i] is not None else "N/A")
        row.append(f"{id_accuracies[i]:.3f}" if id_accuracies[i] is not None else "N/A")
        row.append(f"{ood_accuracies[i]:.3f}" if ood_accuracies[i] is not None else "N/A")
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

    plt.tight_layout(pad=2.5)
    save_path = os.path.join(save_dir, 'ood_metrics_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics summary plot to: {save_path}")
    plt.close()


def plot_roc_curves(model, test_ood_dict, res_ood_dict, save_dir="figures"):
    """
    Plot ROC curves for different OOD ratios.

    Args:
        model: Trained RealNVP model
        test_ood_dict: Dictionary with OOD percentages as keys and test data as values
        res_ood_dict: Dictionary with metrics including ground truth labels
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_ood_dict)))

    for i, (perc, test_data) in enumerate(sorted(test_ood_dict.items())):
        metrics = res_ood_dict[perc]
        n_id = metrics['n_id']
        n_ood = metrics['n_ood']

        if n_id > 0 and n_ood > 0:
            # Get scores
            with torch.no_grad():
                scores = model.score_samples(test_data.to(model.device)).cpu().numpy()

            # Create labels
            y_true = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
            y_scores = -scores  # Negative log prob as anomaly score

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, linewidth=2, color=colors[i],
                    label=f'{perc:.1%} OOD (AUC = {roc_auc:.3f})')

    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)

    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('ROC Curves for Different OOD Ratios', fontsize=17, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=13)
    plt.tight_layout(pad=2.0)
    save_path = os.path.join(save_dir, 'roc_curves_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curves comparison to: {save_path}")
    plt.close()


def plot_likelihood_histograms(model, test_ood_dict, test_id_data=None, save_dir="figures"):
    """
    Plot likelihood distributions for different OOD ratios.

    Args:
        model: Trained RealNVP model
        test_ood_dict: Dictionary with OOD percentages as keys and test data as values
        test_id_data: Optional pure ID test data
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Log-Likelihood Distributions for Different OOD Ratios', fontsize=18, fontweight='bold', y=0.995)

    axes = axes.flatten()
    percentages = sorted(test_ood_dict.keys())

    # Add pure ID test if available
    if test_id_data is not None:
        with torch.no_grad():
            id_scores = model.score_samples(test_id_data.to(model.device)).cpu().numpy()

        ax = axes[0]
        ax.hist(id_scores, bins=50, color='blue', alpha=0.6, label='ID (0% OOD)', edgecolor='black')
        if model.threshold is not None:
            ax.axvline(x=model.threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Log-Likelihood', fontsize=13)
        ax.set_ylabel('Frequency', fontsize=13)
        ax.set_title(f'0% OOD (Pure ID)\nMean: {id_scores.mean():.3f}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        plot_idx = 1
    else:
        plot_idx = 0

    # Plot for each OOD percentage
    for perc in percentages:
        if plot_idx >= len(axes):
            break

        test_data = test_ood_dict[perc]
        with torch.no_grad():
            scores = model.score_samples(test_data.to(model.device)).cpu().numpy()

        ax = axes[plot_idx]
        ax.hist(scores, bins=50, color='purple', alpha=0.6, edgecolor='black')
        if model.threshold is not None:
            ax.axvline(x=model.threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Log-Likelihood', fontsize=13)
        ax.set_ylabel('Frequency', fontsize=13)
        ax.set_title(f'{perc:.1%} OOD\nMean: {scores.mean():.3f}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout(pad=2.5)
    save_path = os.path.join(save_dir, 'likelihood_histograms.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved likelihood histograms to: {save_path}")
    plt.close()


def plot_tsne(tsne_data1, preds, title):

    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_data1[preds == 1, 0], tsne_data1[preds == 1, 1],
                color='blue', label='ID', alpha=0.5)
    plt.scatter(tsne_data1[preds == -1, 0], tsne_data1[preds == -1, 1],
                color='red', label='OOD', alpha=0.5)
    plt.title(title, fontsize=16, fontweight='bold')

    plt.xlabel('TSNE Dimension 1', fontsize=14)
    plt.ylabel('TSNE Dimension 2', fontsize=14)
    plt.tick_params(labelsize=12)
    #save figure
    plt.legend(fontsize=12)
    plt.tight_layout(pad=2.0)
    plt.savefig(f"figures/{title.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")


def compare_noise_levels(results_by_noise, test_id_score=None, save_dir="figures"):
    """
    Create comprehensive comparison plots for different noise levels.

    Args:
        results_by_noise: dict[noise_std] -> {'data_dict': ..., 'res_dict': ...}
        test_id_score: Optional mean score for pure ID test data (0% OOD)
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)

    noise_levels = sorted(results_by_noise.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(noise_levels)))

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparison of OOD Detection Across Different Noise Levels',
                 fontsize=18, fontweight='bold', y=0.995)

    # Get percentages from first noise level results
    first_noise = noise_levels[0]
    percentages = sorted(results_by_noise[first_noise]['res_dict'].keys())

    # ============ Plot 1: Mean Log-Likelihood vs OOD Ratio for Different Noise Levels ============
    ax1 = axes[0, 0]
    for idx, noise_std in enumerate(noise_levels):
        res_dict = results_by_noise[noise_std]['res_dict']
        mean_scores = [res_dict[perc]['mean_score'] for perc in percentages]
        ax1.plot([p * 100 for p in percentages], mean_scores, 'o-',
                linewidth=2, markersize=8, color=colors[idx],
                label=f'Noise σ={noise_std}')

    ax1.set_xlabel('OOD Percentage (%)', fontsize=14)
    ax1.set_ylabel('Mean Log-Likelihood', fontsize=14)
    ax1.set_title('Log-Likelihood vs OOD Ratio', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=12)
    ax1.set_xticks([p * 100 for p in percentages])
    ax1.tick_params(labelsize=12)

    # ============ Plot 2: Overall Accuracy vs OOD Ratio ============
    ax2 = axes[0, 1]
    for idx, noise_std in enumerate(noise_levels):
        res_dict = results_by_noise[noise_std]['res_dict']
        accuracies = [res_dict[perc]['accuracy'] for perc in percentages]
        ax2.plot([p * 100 for p in percentages], accuracies, 'o-',
                linewidth=2, markersize=8, color=colors[idx],
                label=f'Noise σ={noise_std}')

    ax2.set_xlabel('OOD Percentage (%)', fontsize=14)
    ax2.set_ylabel('Overall Accuracy', fontsize=14)
    ax2.set_title('Overall Accuracy vs OOD Ratio', fontsize=15, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=12)
    ax2.set_xticks([p * 100 for p in percentages])
    ax2.tick_params(labelsize=12)

    # ============ Plot 3: ROC AUC vs OOD Ratio ============
    ax3 = axes[1, 0]
    for idx, noise_std in enumerate(noise_levels):
        res_dict = results_by_noise[noise_std]['res_dict']
        roc_aucs = [res_dict[perc]['roc_auc'] for perc in percentages if res_dict[perc]['roc_auc'] is not None]
        valid_percentages = [perc for perc in percentages if res_dict[perc]['roc_auc'] is not None]

        if roc_aucs:
            ax3.plot([p * 100 for p in valid_percentages], roc_aucs, 'o-',
                    linewidth=2, markersize=8, color=colors[idx],
                    label=f'Noise σ={noise_std}')

    ax3.axhline(y=0.5, color='r', linestyle='--', label='Random Classifier', alpha=0.7)
    ax3.set_xlabel('OOD Percentage (%)', fontsize=14)
    ax3.set_ylabel('ROC AUC', fontsize=14)
    ax3.set_title('ROC AUC vs OOD Ratio', fontsize=15, fontweight='bold')
    ax3.set_ylim([0, 1.05])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=12)
    ax3.tick_params(labelsize=12)

    # ============ Plot 4: ID Accuracy and OOD Accuracy Comparison ============
    ax4 = axes[1, 1]

    # Plot ID accuracy for each noise level
    for idx, noise_std in enumerate(noise_levels):
        res_dict = results_by_noise[noise_std]['res_dict']
        id_accuracies = [res_dict[perc]['id_accuracy'] for perc in percentages]
        ax4.plot([p * 100 for p in percentages], id_accuracies, 's-',
                linewidth=2, markersize=8, color=colors[idx], alpha=0.7,
                label=f'ID Acc (σ={noise_std})')

    # Plot OOD accuracy for each noise level with dashed lines
    for idx, noise_std in enumerate(noise_levels):
        res_dict = results_by_noise[noise_std]['res_dict']
        ood_accuracies = [res_dict[perc]['ood_accuracy'] for perc in percentages]
        ax4.plot([p * 100 for p in percentages], ood_accuracies, 'o--',
                linewidth=2, markersize=8, color=colors[idx], alpha=0.7,
                label=f'OOD Acc (σ={noise_std})')

    ax4.set_xlabel('OOD Percentage (%)', fontsize=14)
    ax4.set_ylabel('Accuracy', fontsize=14)
    ax4.set_title('ID and OOD Accuracy Comparison', fontsize=15, fontweight='bold')
    ax4.set_ylim([0, 1.05])
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=10, ncol=2)
    ax4.set_xticks([p * 100 for p in percentages])
    ax4.tick_params(labelsize=12)

    plt.tight_layout(pad=2.5)
    save_path = os.path.join(save_dir, 'noise_level_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved noise level comparison plot to: {save_path}")
    plt.close()

    # ============ Create a detailed summary table ============
    print("\n" + "="*120)
    print("Detailed Comparison Across Noise Levels:")
    print("="*120)

    for noise_std in noise_levels:
        print(f"\nNoise Level σ = {noise_std}")
        print("-" * 120)
        print(f"{'OOD %':>8} | {'Mean Score':>12} | {'ROC AUC':>10} | {'Accuracy':>10} | {'ID Acc':>10} | {'OOD Acc':>10}")
        print("-" * 120)

        res_dict = results_by_noise[noise_std]['res_dict']
        for perc in percentages:
            metrics = res_dict[perc]
            roc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else "N/A"
            acc_str = f"{metrics['accuracy']:.4f}" if metrics['accuracy'] is not None else "N/A"
            id_acc_str = f"{metrics['id_accuracy']:.4f}" if metrics['id_accuracy'] is not None else "N/A"
            ood_acc_str = f"{metrics['ood_accuracy']:.4f}" if metrics['ood_accuracy'] is not None else "N/A"

            print(f"{perc:>7.1%} | {metrics['mean_score']:>12.4f} | {roc_str:>10} | {acc_str:>10} | {id_acc_str:>10} | {ood_acc_str:>10}")

    print("="*120 + "\n")


if __name__ == "__main__":
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

    # Override config with RL-specific arguments
    if args.data_path is not None:
        args.data_path = args.data_path  # Keep as args attribute for compatibility
    if args.task != 'synthetic':
        args.task = args.task
    args.obs_shape = tuple(args.obs_dim)
    args.action_dim = args.action_dim

    # Set random seed for reproducibility
    if 'seed' in config:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])

    # Determine device: CLI overrides config
    cli_device = args.device
    cfg_device = config.get('device', None)

    device = cli_device if cli_device is not None else (cfg_device or 'cpu')

    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        print(f"{device} not available, falling back to CPU")
        device = "cpu"

    # Make sure everyone agrees on the same device
    args.device = device
    config["device"] = device

    print(f"Using device: {device}")


    # Choose data loading mode
    use_rl_data = config.get('use_rl_data', args.task != 'synthetic')

    # Only create environment if needed (when data_path is not provided)
    env = None
    if use_rl_data and args.data_path is None:
        # Handle Abiomed environment creation
        if args.task.lower() == 'abiomed' or 'abiomed' in args.task.lower():
            try:
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
            except Exception as e:
                print(f"Error: Could not create Abiomed environment: {e}")
                print("Please provide data_path or ensure Abiomed environment is properly set up")
        else:
            # Standard gym environment
            try:
                env = gym.make(args.task)
                print(f"✓ Gym environment '{args.task}' created successfully")
            except Exception as e:
                print(f"Warning: Could not create environment {args.task}: {e}")
                print("Will use data_path instead if provided")

    if use_rl_data:
        print("Loading RL dataset for KDE training...")

        # Load RL data (next_observations + actions)
        train_data, val_data, test_data, kde_input_dim, norm_stats = load_rl_data_for_kde(
            args=args,
            env=env,
            val_split_ratio=config.get('val_ratio', 0.2)
        )

        # For RL data, we don't have separate anomaly data for evaluation
        # We'll use a portion of validation data as "normal" and generate synthetic anomalies
        n_test = len(test_data)
        test_normal = test_data

        # Generate synthetic anomalies in the same dimension as RL data
        anomaly_data = torch.randn(n_test // 2, kde_input_dim) * 3 + 5  # Offset anomalies

        # Update input dimension in config
        config['input_dim'] = kde_input_dim
        config['norm_stats'] = norm_stats

    else:
        print("Creating synthetic data...")
        normal_data, anomaly_data = create_synthetic_data(
            n_samples=config.get('n_samples', 2000),
            dim=config.get('input_dim', 2),
            anomaly_type=config.get('anomaly_type', 'outlier')
        )
        print(anomaly_data.shape)
        # # Split normal data into train/val/test
        # train_ratio = config.get('train_ratio', 0.6)
        # val_ratio = config.get('val_ratio', 0.2)

        # n_train = int(train_ratio * len(normal_data))
        # n_val = int(val_ratio * len(normal_data))

        # train_data = normal_data[:n_train]
        # val_data = normal_data[n_train:n_train+n_val]
        # test_normal = normal_data[n_train+n_val:]

    if config.get('verbose', True):
        print(f"Data shapes - Train: {train_data.shape}, Val: {val_data.shape}, "
              f"Test Normal: {test_normal.shape}, Test Anomaly: {anomaly_data.shape}")

    # Create and train model
    print("Creating RealNVP model...")
    model = RealNVP(
        input_dim=config.get('input_dim', 2),
        num_layers=config.get('num_layers', 6),
        hidden_dims=config.get('hidden_dims', [256, 256]),
        device=device
    ).to(device)

    # print("Training RealNVP model...")
    # history = model.fit(
    #     train_data=train_data,
    #     val_data=val_data,
    #     epochs=config.get('epochs', 100),
    #     batch_size=config.get('batch_size', 128),
    #     lr=config.get('lr', 1e-3),
    #     patience=config.get('patience', 15),
    #     verbose=config.get('verbose', True)
    # )
    # Save model if requested
    # if config.get('model_save_path', False):
    #     save_path = config.get('model_save_path', 'saved_models/realnvp')
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     model.save_model(save_path)
    #     print(f"Model saved to: {save_path}_model.pth")
    #load pretrained model
    model_dict = RealNVP.load_model(save_path=args.model_save_path)
    model = model_dict['model']
    model.to(device)

    print(args.device)
    model.threshold = model_dict['thr']
    print(model.device)
    # Evaluate anomaly detection
    print("\nEvaluating anomaly detection performance...")
    # results = model.evaluate_anomaly_detection(
    #     normal_data=test_normal.to(model.device),
    #     anomaly_data=anomaly_data.to(model.device),
    #     plot=config.get('plot_results', True)
    # )
    print(type(train_data))
    train_data = train_data.to(model.device)
    val_data = val_data.to(model.device)
    test_data = test_data.to(model.device)
    print(train_data.device)
    print(model.device)

    predictions_tr = model.predict(train_data)
    scores_tr = model.score_samples(train_data)
    print('TRAINING SCORE', scores_tr.mean().item())
    scores_test_in_dist = model.score_samples(test_data)
    anomaly_test_res  = model.score_samples(anomaly_data.to(model.device))

    # small_train = train_data[predictions_tr == 1][: int(0.1 * len(train_data))].cpu().numpy()
    # noisy_train = small_train + np.random.normal(0, 0.1, small_train.shape)
    # normal_data = torch.FloatTensor(np.concatenate([small_train, noisy_train], axis=0)).to(model.device)

    print("\n" + "="*80)
    print("OOD Test Results with Different Noise Levels:")
    print("="*80)

    # Test with 3 different noise levels
    noise_levels = [0.05, 0.1, 0.5]  # Low, Medium, High noise
    results_by_noise = create_ood_test_multiple_noise_levels(
        data=train_data,
        model=model,
        percentage=[0.1, 0.3, 0.5, 0.7, 0.9],
        noise_levels=noise_levels
    )

  
    # Plot validation vs test ID distribution
    print("\n" + "="*80)
    print("Creating Validation vs Test ID Distribution Plot...")
    print("="*80 + "\n")
    plot_val_test_id_distribution(
        model=model,
        val_data=val_data,
        test_id_data=test_data,
        thr=model.threshold,
        title="Validation vs Test ID Log-Likelihood Distribution",
        savepath=f"figures/{args.task}/val_test_id_distribution.png",
        bins=50
    )

    # Generate comprehensive plots
    print("\n" + "="*80)
    print("Generating Visualization Plots...")
    print("="*80 + "\n")

    # Plot comparison across all noise levels
    compare_noise_levels(
        results_by_noise=results_by_noise,
        test_id_score=scores_test_in_dist.mean().item(),
        save_dir=f"figures/{args.task}"
    )

    # Generate individual plots for each noise level
    for noise_std in noise_levels:
        print(f"\nGenerating plots for noise level σ={noise_std}...")
        test_ood_dict = results_by_noise[noise_std]['data_dict']
        res_ood_dict = results_by_noise[noise_std]['res_dict']

        # Create subdirectory for this noise level
        noise_dir = f"figures/{args.task}/noise_{noise_std}"
        os.makedirs(noise_dir, exist_ok=True)

        # Plot 1: Comprehensive metrics summary (likelihood, accuracy, AUC)
        plot_ood_metrics(
            res_dict=res_ood_dict,
            test_id_score=scores_test_in_dist.mean().item(),
            save_dir=noise_dir
        )

        # Plot 2: ROC curves comparison
        plot_roc_curves(
            model=model,
            test_ood_dict=test_ood_dict,
            res_ood_dict=res_ood_dict,
            save_dir=noise_dir
        )

        # Plot 3: Likelihood histograms for all OOD ratios
        plot_likelihood_histograms(
            model=model,
            test_ood_dict=test_ood_dict,
            test_id_data=test_data,
            save_dir=noise_dir
        )

    # Original likelihood distribution plot
    # plot_likelihood_distributions(
    #     model,
    #     train_data,
    #     val_data,
    #     ood_data=normal_data,
    #     thr=model.threshold,
    #     title="Likelihood Distribution",
    #     savepath=None,
    #     bins=50
    # )

    print("\n" + "="*80)
    print("All plots saved to 'figures/' directory")
    print("="*80)
    # print(f"ROC AUC: {results['roc_auc']:.3f}")
    # print(f"Accuracy: {results['accuracy']:.3f}")
    # print(f"Normal data log prob: {results['normal_log_prob_mean']:.3f} ± {results['normal_log_prob_std']:.3f}")
    # print(f"Anomaly data log prob: {results['anomaly_log_prob_mean']:.3f} ± {results['anomaly_log_prob_std']:.3f}")


    
    print("Scores test ID (pure test set):", scores_test_in_dist.mean().item())

    print("\nRealNVP training and evaluation completed!")