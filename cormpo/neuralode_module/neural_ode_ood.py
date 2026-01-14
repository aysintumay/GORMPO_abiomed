import argparse
import json
import math
import os
import sys
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, Subset

try:
    import yaml
except Exception:
    yaml = None

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cormpo.neuralode_module.neural_ode_density import (
    ContinuousNormalizingFlow,
    NPZTargetDataset,
    ODEFunc,
)
from cormpo.common.buffer import ReplayBuffer


class NeuralODEOOD:
    """
    Neural ODE-based Out-of-Distribution (OOD) Detection wrapper.

    This class wraps a ContinuousNormalizingFlow model and provides
    OOD detection functionality similar to RealNVP.
    """

    def __init__(
        self,
        flow: ContinuousNormalizingFlow,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the Neural ODE OOD detector.

        Args:
            flow: Trained ContinuousNormalizingFlow model
            device: Device to run computations on
        """
        self.flow = flow
        self.device = device
        self.threshold = None
        self.flow.to(device)

    def score_samples(self, x: torch.Tensor, device: str = 'cuda') -> np.ndarray:
        """
        Compute log probability of data points (matches RealNVP interface).

        Args:
            x: Input tensor of shape (batch_size, dim)
            device: Device to use (ignored, uses model's device)

        Returns:
            Log probabilities as numpy array
        """
        self.flow.eval()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)

        log_probs = self.flow.log_prob(x)
        return log_probs.detach().cpu().numpy()

    def set_threshold(
        self,
        val_data: torch.Tensor,
        anomaly_fraction: float = 0.01,
        batch_size: int = 32
    ):
        """
        Set threshold for anomaly detection based on validation data.

        Args:
            val_data: Validation dataset (assumed to be normal data)
            anomaly_fraction: Fraction of validation data to classify as anomalies
            batch_size: Batch size for processing to avoid OOM
        """
        self.flow.eval()

        # Process in batches to avoid OOM
        n_samples = len(val_data)
        log_probs_list = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = val_data[i:i+batch_size].to(self.device)
                batch_log_probs = self.score_samples(batch)
                log_probs_list.append(batch_log_probs)

                # Clear GPU cache after each batch
                if str(self.device).startswith('cuda'):
                    torch.cuda.empty_cache()

        log_probs = np.concatenate(log_probs_list)

        # Set threshold as percentile of validation log probabilities
        self.threshold = float(np.percentile(log_probs, anomaly_fraction * 100))

        print(f'Threshold set to {self.threshold:.4f} '
              f'(marking {anomaly_fraction*100:.1f}% of validation data as anomalies)')

        return self.threshold

    def predict_anomaly(self, x: torch.Tensor) -> np.ndarray:
        """
        Predict anomalies based on log probability threshold.

        Args:
            x: Input data

        Returns:
            Boolean array indicating anomalies (True = anomaly)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")

        log_probs = self.score_samples(x)
        return log_probs < self.threshold

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Predict anomalies based on threshold (matches RealNVP interface).

        Args:
            x: Test data

        Returns:
            predictions: 1 for normal, -1 for anomaly
        """
        anomalies = self.predict_anomaly(x)
        return np.where(anomalies, -1, 1)

    def evaluate_anomaly_detection(
        self,
        normal_data: torch.Tensor,
        anomaly_data: torch.Tensor,
        plot: bool = True,
        save_path: Optional[str] = None
    ) -> dict:
        """
        Evaluate anomaly detection performance.

        Args:
            normal_data: Normal test data
            anomaly_data: Anomalous test data
            plot: Whether to plot ROC curve
            save_path: Path to save the ROC curve plot

        Returns:
            Dictionary with evaluation metrics
        """
        self.flow.eval()

        # Move data to device
        normal_data = normal_data.to(self.device)
        anomaly_data = anomaly_data.to(self.device)

        # Compute log probabilities for normal data
        normal_log_probs = self.score_samples(normal_data)

        # Compute log probabilities for anomaly data
        anomaly_log_probs = self.score_samples(anomaly_data)

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
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curve for OOD Detection (Neural ODE)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"ROC curve saved to {save_path}")
            else:
                plt.savefig('cormpo/figures/neural_ode_roc_curve.png', dpi=300, bbox_inches='tight')
                print("ROC curve saved to cormpo/figures/neural_ode_roc_curve.png")
            plt.close(fig)

        return results

    def save_model(self, save_path: str, train_data: Optional[torch.Tensor] = None):
        """
        Save the Neural ODE OOD model and metadata (matches RealNVP interface).

        Args:
            save_path: Base path for saving (without extension)
            train_data: Optional training data to compute statistics
        """
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Save model state dict
        torch.save(self.flow.state_dict(), f"{save_path}_model.pt")

        # Compute training log probabilities if provided
        metadata = {
            'threshold': self.threshold,
            'device': str(self.device),
        }

        if train_data is not None:
            self.flow.eval()
            train_data = train_data.to(self.device)
            train_log_probs = self.score_samples(train_data)
            metadata['mean'] = float(np.mean(train_log_probs))
            metadata['std'] = float(np.std(train_log_probs))

        with open(f"{save_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Model saved to: {save_path}_model.pt")
        print(f"Metadata saved to: {save_path}_metadata.pkl")

    @classmethod
    def load_model(
        cls,
        save_path: str,
        target_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 512),
        activation: str = "silu",
        time_dependent: bool = True,
        solver: str = "dopri5",
        t0: float = 0.0,
        t1: float = 1.0,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Load a saved Neural ODE OOD model.

        Args:
            save_path: Base path for loading (without extension)
            target_dim: Dimension of target data
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            time_dependent: Whether to use time-dependent ODE
            solver: ODE solver
            t0, t1: Integration time bounds
            rtol, atol: ODE solver tolerances
            device: Device to load model on

        Returns:
            Dictionary with loaded model, threshold, and statistics
        """
        # Load metadata
        with open(f"{save_path}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        # Create ODE function and flow
        odefunc = ODEFunc(
            dim=target_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            time_dependent=time_dependent,
        ).to(device)

        flow = ContinuousNormalizingFlow(
            func=odefunc,
            t0=t0,
            t1=t1,
            solver=solver,
            rtol=rtol,
            atol=atol,
        ).to(device)

        # Load model state dict
        flow.load_state_dict(torch.load(f"{save_path}_model.pt", map_location=device))
        flow.eval()

        # Create OOD wrapper
        ood_model = cls(flow, device=device)
        ood_model.threshold = metadata.get('threshold')

        print(f"Model loaded from: {save_path}_model.pt")
        print(f"Metadata loaded from: {save_path}_metadata.pkl")
        print(f"Threshold: {ood_model.threshold}")

        model_dict = {
            'model': ood_model,
            'threshold': ood_model.threshold,
            'mean': metadata.get('mean'),
            'std': metadata.get('std')
        }

        return model_dict


def plot_likelihood_distributions(
    model: NeuralODEOOD,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    ood_data: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
    title: str = "Likelihood Distribution (Neural ODE)",
    save_dir: str = "cormpo/figures",
    bins: int = 50
):
    """
    Visualize log-likelihood distributions for train, val, and OOD data.

    Args:
        model: Neural ODE OOD model
        train_data: In-distribution training set
        val_data: Held-out validation set
        ood_data: Optional OOD dataset
        threshold: Threshold value for anomaly detection
        title: Title for the plot
        save_dir: Directory to save figures
        bins: Number of histogram bins
    """
    os.makedirs(save_dir, exist_ok=True)

    # Compute log-likelihoods
    print("Computing log-likelihoods for train data...")
    train_data = train_data.to(model.device)
    logp_train = model.score_samples(train_data)

    print("Computing log-likelihoods for validation data...")
    val_data = val_data.to(model.device)
    logp_val = model.score_samples(val_data)

    logp_ood = None
    if ood_data is not None:
        print("Computing log-likelihoods for OOD data...")
        ood_data = ood_data.to(model.device)
        logp_ood = model.score_samples(ood_data)

    if threshold is None:
        threshold = model.threshold

    # Plot train and validation
    plt.figure(figsize=(10, 6))
    sns.histplot(logp_train, bins=bins, color="blue", alpha=0.4, label="Train", kde=True)
    sns.histplot(logp_val, bins=bins, color="green", alpha=0.4, label="Validation", kde=True)
    if threshold is not None:
        plt.axvline(x=threshold, color='tab:red', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel("Log-likelihood", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"{title} - Train/Val", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    save_path = os.path.join(save_dir, "neural_ode_train_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure at {save_path}")
    plt.close()

    # Plot OOD if available
    if logp_ood is not None:
        plt.figure(figsize=(10, 6))
        sns.histplot(logp_ood, bins=bins, color="red", alpha=0.4, label="OOD", kde=True)
        sns.histplot(logp_val, bins=bins, color="green", alpha=0.3, label="Validation", kde=True)
        if threshold is not None:
            plt.axvline(x=threshold, color='tab:red', linestyle='--', linewidth=2, label='Threshold')
        plt.xlabel("Log-likelihood", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"{title} - OOD vs Validation", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)

        save_path = os.path.join(save_dir, "neural_ode_ood_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure at {save_path}")
        plt.close()


def plot_tsne(tsne_data, preds, title, save_dir="cormpo/figures"):
    """
    Plot t-SNE visualization of OOD predictions.

    Args:
        tsne_data: 2D t-SNE embeddings
        preds: Predictions (1 for ID, -1 for OOD)
        title: Plot title
        save_dir: Directory to save figure
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_data[preds == 1, 0], tsne_data[preds == 1, 1],
                color='blue', label='ID', alpha=0.5)
    plt.scatter(tsne_data[preds == -1, 0], tsne_data[preds == -1, 1],
                color='red', label='OOD', alpha=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved t-SNE plot to {save_path}")
    plt.close()


def load_abiomed_data(data_path: str, val_ratio: float = 0.2, test_ratio: float = 0.2):
    """
    Load Abiomed dataset and prepare for OOD detection.

    Args:
        data_path: Path to the data file
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing

    Returns:
        Tuple of (train_data, val_data, test_data, input_dim)
    """
    # Load the replay buffer
    replay_buffer = ReplayBuffer.load(data_path)

    # Get all data
    all_obs = torch.FloatTensor(replay_buffer.observations)
    all_actions = torch.FloatTensor(replay_buffer.actions)
    all_next_obs = torch.FloatTensor(replay_buffer.next_observations)

    # Concatenate next_observations + actions for OOD detection
    all_data = torch.cat([all_next_obs, all_actions], dim=1)

    # Split data
    n_total = len(all_data)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_test - n_val

    indices = torch.randperm(n_total)
    train_data = all_data[indices[:n_train]]
    val_data = all_data[indices[n_train:n_train+n_val]]
    test_data = all_data[indices[n_train+n_val:]]

    input_dim = all_data.shape[1]

    print(f"Loaded Abiomed data from {data_path}")
    print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    print(f"Input dimension: {input_dim}")

    return train_data, val_data, test_data, input_dim


if __name__ == "__main__":
    print("NeuralODE OOD module for Abiomed data created successfully!")
    print("Use this module for OOD detection with Neural ODE models.")
