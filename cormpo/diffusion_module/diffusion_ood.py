"""
Diffusion Model Out-of-Distribution (OOD) Detection for Abiomed
Adapted from GORMPO's diffusion OOD implementation
"""

import os
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from sklearn.metrics import roc_curve, auc

from cormpo.diffusion_module.diffusion_density import log_prob_elbo


class DiffusionOOD:
    """
    Diffusion Model-based Out-of-Distribution (OOD) Detection wrapper.

    Uses ELBO (Evidence Lower Bound) to compute log-likelihood for OOD detection.
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        num_inference_steps: int = None
    ):
        """
        Initialize the Diffusion OOD detector.

        Args:
            model: Trained diffusion model (epsilon prediction)
            scheduler: Diffusion scheduler (DDPM or DDIM)
            device: Device to run computations on
            num_inference_steps: Number of timesteps for ELBO (None = use all, e.g. 20 for fast)
        """
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.threshold = None
        self.model.to(device)
        self.model.eval()

    def score_samples(self, x: torch.Tensor, batch_size: int = 512, num_inference_steps: int = None) -> torch.Tensor:
        """
        Compute log probability of data points using ELBO.

        Args:
            x: Input tensor of shape (num_samples, dim)
            batch_size: Batch size for processing (default: 512)
            num_inference_steps: Number of timesteps to use for ELBO (default: use self.num_inference_steps)
                If specified, uniformly subsample timesteps for faster approximation

        Returns:
            Log probabilities (ELBO) for each sample
        """
        self.model.eval()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)

        # Use instance default if not specified
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps

        # Process in batches to avoid OOM
        all_log_probs = []
        for i in range(0, len(x), batch_size):
            batch = x[i:min(i+batch_size, len(x))]
            with torch.no_grad():
                log_probs = log_prob_elbo(
                    model=self.model,
                    scheduler=self.scheduler,
                    x0=batch,
                    device=self.device,
                    num_inference_steps=num_inference_steps
                )
            all_log_probs.append(log_probs.cpu())

            # Clear cache periodically
            if self.device.startswith('cuda') and i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

        return torch.cat(all_log_probs)

    def set_threshold(
        self,
        val_data: torch.Tensor,
        anomaly_fraction: float = 0.01,
        batch_size: int = 512
    ) -> float:
        """
        Set threshold for anomaly detection based on validation data.

        Args:
            val_data: Validation dataset (assumed to be normal data)
            anomaly_fraction: Fraction of validation data to classify as anomalies
            batch_size: Batch size for processing validation data

        Returns:
            The computed threshold value
        """
        self.model.eval()
        all_log_probs = []

        # Process in batches
        num_samples = len(val_data)
        for i in range(0, num_samples, batch_size):
            batch = val_data[i:min(i+batch_size, num_samples)]
            batch = batch.to(self.device)
            log_probs = self.score_samples(batch, batch_size=batch_size)
            all_log_probs.append(log_probs)

        log_probs = torch.cat(all_log_probs)

        # Set threshold as percentile of validation log probabilities
        self.threshold = torch.quantile(log_probs, anomaly_fraction).item()

        print(f'Threshold set to {self.threshold:.4f} '
              f'(marking {anomaly_fraction*100:.1f}% of validation data as anomalies)')

        return self.threshold

    def predict_anomaly(self, x: torch.Tensor, batch_size: int = 512) -> np.ndarray:
        """
        Predict anomalies based on log probability threshold.

        Args:
            x: Input data
            batch_size: Batch size for processing

        Returns:
            Boolean array indicating anomalies (True = anomaly)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")

        log_probs = self.score_samples(x, batch_size)
        return log_probs.numpy() < self.threshold

    def predict(self, x: torch.Tensor, batch_size: int = 512) -> np.ndarray:
        """
        Predict anomalies based on threshold.

        Args:
            x: Test data
            batch_size: Batch size for processing

        Returns:
            predictions: 1 for normal, -1 for anomaly
        """
        anomalies = self.predict_anomaly(x, batch_size)
        return np.where(anomalies, -1, 1)

    def evaluate_anomaly_detection(
        self,
        normal_data: torch.Tensor,
        anomaly_data: torch.Tensor,
        plot: bool = True,
        save_path: Optional[str] = None,
        batch_size: int = 512
    ) -> dict:
        """
        Evaluate anomaly detection performance.

        Args:
            normal_data: Normal test data
            anomaly_data: Anomalous test data
            plot: Whether to plot ROC curve
            save_path: Path to save the ROC curve plot
            batch_size: Batch size for processing

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        # Compute log probabilities
        normal_log_probs = self.score_samples(normal_data, batch_size).numpy()
        anomaly_log_probs = self.score_samples(anomaly_data, batch_size).numpy()

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
            ax.set_title('ROC Curve for OOD Detection (Diffusion)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            if save_path:
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"ROC curve saved to {save_path}")
            plt.close(fig)

        return results

    def save_model(self, save_path: str, train_data: Optional[torch.Tensor] = None):
        """
        Save the Diffusion OOD model metadata.

        Args:
            save_path: Base path for saving (without extension)
            train_data: Optional training data to compute statistics
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        # Compute training log probabilities if provided
        metadata = {
            'threshold': self.threshold,
            'device': self.device,
            'num_inference_steps': self.num_inference_steps,
        }

        if train_data is not None:
            self.model.eval()
            train_log_probs = self.score_samples(train_data.to(self.device))
            metadata['mean'] = train_log_probs.mean().item()
            metadata['std'] = train_log_probs.std().item()

        import pickle
        with open(f"{save_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Metadata saved to: {save_path}_metadata.pkl")


def plot_likelihood_distributions(
    model: DiffusionOOD,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    ood_data: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
    title: str = "Likelihood Distribution (Diffusion ELBO)",
    save_dir: str = "figures/diffusion_ood",
    bins: int = 50,
    batch_size: int = 512
):
    """
    Visualize log-likelihood distributions for train, val, and OOD data.

    Args:
        model: Diffusion OOD model
        train_data: In-distribution training set
        val_data: Held-out validation set
        ood_data: Optional OOD dataset
        threshold: Threshold value for anomaly detection
        title: Title for the plot
        save_dir: Directory to save figures
        bins: Number of histogram bins
        batch_size: Batch size for processing
    """
    os.makedirs(save_dir, exist_ok=True)

    # Compute log-likelihoods
    print("Computing log-likelihoods for train data...")
    logp_train = model.score_samples(train_data, batch_size).numpy()

    print("Computing log-likelihoods for validation data...")
    logp_val = model.score_samples(val_data, batch_size).numpy()

    logp_ood = None
    if ood_data is not None:
        print("Computing log-likelihoods for OOD data...")
        logp_ood = model.score_samples(ood_data, batch_size).numpy()

    if threshold is None:
        threshold = model.threshold

    # Plot train and validation
    plt.figure(figsize=(10, 6))
    sns.histplot(logp_train, bins=bins, color="blue", alpha=0.4, label="Train", kde=True)
    sns.histplot(logp_val, bins=bins, color="green", alpha=0.4, label="Validation", kde=True)
    if threshold is not None:
        plt.axvline(x=threshold, color='tab:red', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel("Log-likelihood (ELBO)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"{title} - Train/Val", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    save_path = os.path.join(save_dir, "diffusion_train_distribution.png")
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
        plt.xlabel("Log-likelihood (ELBO)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"{title} - OOD vs Validation", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)

        save_path = os.path.join(save_dir, "diffusion_ood_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure at {save_path}")
        plt.close()
