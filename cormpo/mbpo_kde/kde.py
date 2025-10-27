import os
import sys
import time
import pickle
import argparse

import numpy as np
import torch
import faiss
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.buffer import ReplayBuffer as ReplayBufferAbiomed
from abiomed_env.rl_env import AbiomedRLEnvFactory

def get_env_data(args, val=None):
    """
    Load environment and dataset for Abiomed.

    Args:
        args: Arguments containing environment configuration
        val: Whether to return validation data

    Returns:
        env, dataset, [dataset_val]: Environment and datasets
    """
    env = AbiomedRLEnvFactory.create_env(
        model_name=args.model_name,
        model_path=args.model_path,
        data_path=args.data_path_wm,
        max_steps=args.max_steps,
        gamma1=args.gamma1,
        gamma2=args.gamma2,
        gamma3=args.gamma3,
        action_space_type=args.action_space_type,
        reward_type="smooth",
        normalize_rewards=True,
        noise_rate=args.noise_rate,
        noise_scale=args.noise_scale,
        seed=args.seed,
        device=f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu",
    )

    if args.data_path is None:
        dataset1 = env.world_model.data_train
        dataset2 = env.world_model.data_val
        dataset3 = env.world_model.data_test
        dataset = [dataset1, dataset2, dataset3]
    else:
        try:
            with open(args.data_path, "rb") as f:
                dataset = pickle.load(f)
            print("Opened pickle file for synthetic dataset")
        except Exception:
            dataset = np.load(args.data_path)
            dataset = {k: dataset[k] for k in dataset.files}
            print("Opened npz file for synthetic dataset")

    if val:
        print("Validation data is supported for Abiomed dataset")
        dataset_val = env.world_model.data_test
        return env, dataset, dataset_val
    else:
        print("No validation data for Abiomed dataset")
        return env, dataset
    

class PercentileThresholdKDE:
    """
    KDE-based anomaly detection using percentile thresholds
    """

    def __init__(
        self,
        bandwidth=1.0,
        n_neighbors=100,
        use_gpu=True,
        normalize=True,
        percentile=5.0,
        pca=None,
        devid=0,
    ):
        self.bandwidth = bandwidth
        self.n_neighbors = n_neighbors
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.normalize = normalize
        self.percentile = percentile 

        self.index = None
        self.training_data = None
        self.scaler = None
        self.threshold = None
        self.is_fitted = False
        self.pca = pca
        self.devid = devid

    def fit(self, X, X_val, verbose=True):
        """
        Fit the model and compute percentile-based threshold.

        Args:
            X: Training data (n_samples, n_features)
            X_val: Validation data for threshold computation
            verbose: Print fitting information
        """
        start_time = time.time()

        if torch.is_tensor(X):
            X = X.cpu().numpy()

        X = X.astype(np.float32)

        # Normalize data if requested
        if self.normalize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # Fit PCA and transform data if enabled
        if (X.shape[1] > 30) and (self.pca):
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=7)
            self.pca.fit(X)
            X = self.pca.transform(X)
            if verbose:
                print(f"Reduced to {X.shape[1]} features using PCA")

        self.training_data = X.copy()
        n_samples, n_features = X.shape

        if verbose:
            print(f"Training on {n_samples} samples with {n_features} features")
            print(f"Percentile threshold: {self.percentile}%")
            print(f"Bandwidth: {self.bandwidth}, K-neighbors: {self.n_neighbors}")

        # Build FAISS index
        if n_features <= 64 and n_samples < 1000000:
            self.index = faiss.IndexFlatL2(n_features)
        else:
            nlist = min(int(np.sqrt(n_samples)), 4096)
            quantizer = faiss.IndexFlatL2(n_features)
            self.index = faiss.IndexIVFFlat(quantizer, n_features, nlist)
            if hasattr(self.index, "train"):
                self.index.train(X)

        # Move to GPU if available
        if self.use_gpu:
            try:
                gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(
                    gpu_resources, self.devid, self.index
                )
                if verbose:
                    print("Using GPU acceleration")
            except Exception as e:
                if verbose:
                    print(f"GPU failed, using CPU: {e}")
                self.use_gpu = False

        self.index.add(X)

        # Compute density scores on validation data to find threshold
        if verbose:
            print("Computing density scores for threshold...")

        density_scores = self._score_samples_internal(X_val)

        # Find percentile threshold (lower density = higher anomaly score)
        self.threshold = np.percentile(density_scores, self.percentile)

        self.is_fitted = True
        fit_time = time.time() - start_time

        if verbose:
            print(f"Fitting completed in {fit_time:.2f} seconds")
            print(f"Threshold (log-density): {self.threshold:.4f}")
            print(f"Expected anomaly rate: {self.percentile}%")

        return self

    def _score_samples_internal(self, X):
        """Internal method to compute density scores."""
        k = min(self.n_neighbors, self.training_data.shape[0])
        distances, _ = self.index.search(X, k)

        # Gaussian kernel
        kernel_values = np.exp(-0.5 * distances / (self.bandwidth**2))
        density = np.mean(kernel_values, axis=1)
        log_density = np.log(density + 1e-10)

        return log_density

    def score_samples(self, X):
        """
        Compute log-density estimates for new data

        Args:
            X: Test data (n_samples, n_features)

        Returns:
            log_density: Log-density estimates
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if torch.is_tensor(X):
            X = X.cpu().numpy()

        X = X.astype(np.float32)

        if self.normalize and self.scaler is not None:
            X = self.scaler.transform(X)
        if self.pca:
            X = self.pca.transform(X)

        return self._score_samples_internal(X)

    def predict(self, X):
        """
        Predict anomalies based on threshold

        Args:
            X: Test data

        Returns:
            predictions: 1 for normal, -1 for anomaly
        """
        scores = self.score_samples(X)
        return np.where(scores >= self.threshold, 1, -1)

    def decision_function(self, X):
        """
        Anomaly score (higher = more anomalous)

        Args:
            X: Test data

        Returns:
            anomaly_scores: Higher values indicate more anomalous
        """
        density_scores = self.score_samples(X)
        # Convert to anomaly scores (negative log-density relative to threshold)
        return self.threshold - density_scores

    def get_threshold_stats(self):
        """Get statistics about the threshold"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return {
            "threshold": self.threshold,
            "percentile": self.percentile,
            "bandwidth": self.bandwidth,
            "n_neighbors": self.n_neighbors,
            "n_training_samples": self.training_data.shape[0],
        }

    def save_model(self, base_path):
        """Save FAISS index and metadata separately."""
        print("Transferring from GPU to CPU for saving...")
        index_cpu = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(index_cpu, f"{base_path}.faiss")

        # Save metadata
        metadata = {
            "threshold": self.threshold,
            "bandwidth": self.bandwidth,
            "n_neighbors": self.n_neighbors,
            "percentile": self.percentile,
            "normalize": self.normalize,
            "scaler": self.scaler,
            "training_data": self.training_data,
            "is_fitted": self.is_fitted,
            "model_params": self.get_threshold_stats(),
            "pca": self.pca if hasattr(self, "pca") else None,
        }

        with open(f"{base_path}_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        print(f"Model saved to {base_path}.faiss and {base_path}_metadata.pkl")

    @classmethod
    def load_model(cls, load_path, use_gpu=True, devid=0):
        """
        Load a saved model.

        Args:
            load_path: Base path for loading (without extension)
            use_gpu: Whether to move index to GPU after loading
            devid: GPU device ID

        Returns:
            dict: Dictionary containing model, index, threshold, and scaler
        """
        # Load FAISS index
        index = faiss.read_index(f"{load_path}.faiss")

        # Load metadata
        with open(f"{load_path}_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        # Create model instance
        model = cls(
            bandwidth=metadata["bandwidth"],
            n_neighbors=metadata["n_neighbors"],
            use_gpu=use_gpu,
            normalize=metadata["normalize"],
            percentile=metadata["percentile"],
            devid=devid,
        )

        # Restore state
        model.index = index
        model.threshold = metadata["threshold"]
        model.scaler = metadata["scaler"]
        model.training_data = metadata["training_data"]
        model.is_fitted = metadata["is_fitted"]
        model.pca = metadata.get("pca", None)

        # Move to GPU if requested
        if use_gpu and faiss.get_num_gpus() > 0:
            try:
                gpu_resources = faiss.StandardGpuResources()
                model.index = faiss.index_cpu_to_gpu(gpu_resources, devid, model.index)
                model.use_gpu = True
                print(f"Model loaded and moved to GPU {devid}")
            except Exception as e:
                print(f"Could not move to GPU: {e}, using CPU")
                model.use_gpu = False
        else:
            model.use_gpu = False

        model_dict = {
            "model": model,
            "model_index": model.index,
            "thr": model.threshold,
            "scaler": metadata["scaler"],
        }
        print(f"Model loaded from {load_path}")
        return model_dict


def load_data(data_path, test_size, validation_size, args=None):
    """
    Load and split data for training/validation/testing.

    Args:
        data_path: Path to data file
        test_size: Fraction of data to use for testing
        validation_size: Fraction of data to use for validation
        args: Additional arguments for environment setup

    Returns:
        dict: Dictionary containing train/val/test splits and metadata
    """
    print(f"Loading data from {data_path}")
    if args.env == "abiomed":
        env, data = get_env_data(args)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        replay_buffer = ReplayBufferAbiomed(state_dim, action_dim)
        replay_buffer.convert_abiomed(data, env)
        X = np.concatenate([replay_buffer.state, replay_buffer.action], axis=1)
    else:
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        X = np.concatenate([data["observations"], data["actions"]], axis=1)

    n_samples = len(X)

    if args.temporal_split:
        # Temporal split (no shuffle)
        test_end = int(n_samples * (1 - test_size))
        val_end = int(test_end * (1 - validation_size))

        train_idx = np.arange(0, val_end)
        val_idx = np.arange(val_end, test_end)
        test_idx = np.arange(test_end, n_samples)
        X_train = X[train_idx]
        X_val = X[val_idx] if len(val_idx) > 0 else None
        X_test = X[test_idx]
    else:
        # Random train/test split
        np.random.seed(42)
        total_size = n_samples
        val_test_size = int(total_size * (validation_size + test_size))
        val_size = int(total_size * validation_size)
        train_size = total_size - val_test_size

        # Create random indices for splitting
        indices = np.random.permutation(total_size)
        train_indices = indices[:train_size]
        test_indices = indices[train_size + val_size:]
        val_indices = indices[train_size:val_size + train_size]

        X_train = X[train_indices]
        X_val = X[val_indices] if len(val_indices) > 0 else None
        X_test = X[test_indices]

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "split_info": {
            "total_samples": n_samples,
            "train_samples": len(X_train),
            "val_samples": len(X_val) if X_val is not None else 0,
            "test_samples": len(X_test),
        },
    }

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

def evaluate_anomaly_detection(model, X_test, y_true, verbose=True):
    """
    Evaluate anomaly detection performance

    Args:
        model: Trained anomaly detection model
        X_test: Test data
        y_true: True labels (1=normal, -1=anomaly)
        verbose: Print results
    """
    # Get predictions and scores
    y_pred = model.predict(X_test)
    anomaly_scores = model.decision_function(X_test)

    # Compute metrics
    tp = np.sum((y_true == -1) & (y_pred == -1))  # True anomalies detected
    fp = np.sum((y_true == 1) & (y_pred == -1))  # False alarms
    tn = np.sum((y_true == 1) & (y_pred == 1))  # True normals
    fn = np.sum((y_true == -1) & (y_pred == 1))  # Missed anomalies

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    accuracy = (tp + tn) / len(y_true)

    if verbose:
        print(f"\n=== Anomaly Detection Results ===")
        print(f"True anomalies in test set: {np.sum(y_true == -1)}")
        print(f"Predicted anomalies: {np.sum(y_pred == -1)}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Confusion Matrix:")
        print(f"  TP: {tp:4d} | FP: {fp:4d}")
        print(f"  FN: {fn:4d} | TN: {tn:4d}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }

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
    logp_train = model.score_samples(train_data)
    logp_val   = model.score_samples(val_data)
    logp_ood   = None
    if ood_data is not None:
        logp_ood = model.score_samples(ood_data)
   
    
    # --- Plot ---
    plt.figure(figsize=(8, 5))
    sns.histplot(logp_train, bins=bins, color="blue", alpha=0.4, label="Train", kde=True)
    sns.histplot(logp_val, bins=bins, color="green", alpha=0.4, label="Validation", kde=True)
    plt.axvline(x=thr, color='tab:red', linestyle='--', label='Threshold')
    plt.xlabel("Log-likelihood")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.savefig(f"figures/train_distribution_kde.png", dpi=300, bbox_inches="tight")
    print(f"Saved figure at figures/train_distribution_kde.png")
    plt.figure(figsize=(8, 5))
    

    if logp_ood is not None:
        sns.histplot(logp_ood, bins=bins, color="red", alpha=0.4, label="Test", kde=True)
        plt.axvline(x=thr, color='tab:red', linestyle='--', label='Threshold')

    plt.xlabel("Log-likelihood")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()

    plt.savefig(f"figures/distribution_kde.png", dpi=300, bbox_inches="tight")
    print(f"Saved figure at figures/distribution_kde.png")

def find_optimal_percentile(
    X_train,
    X_val=None,
    percentiles=None,
    bandwidth=1.0,
    n_neighbors=100,
    use_gpu=True,
    metric="density_range",
):
    """
    Find optimal percentile threshold using unsupervised criteria (no labels needed)

    Args:
        X_train: Training data
        X_val: Validation data (if None, uses part of training data)
        percentiles: List of percentiles to test
        bandwidth: KDE bandwidth
        n_neighbors: Number of neighbors for FAISS
        use_gpu: Use GPU acceleration
        metric: 'density_range', 'stability', or 'separation'
    """
    if percentiles is None:
        percentiles = [1, 2, 3, 5, 7, 10, 15, 20]

    print(f"\n=== Finding Optimal Percentile (Unsupervised) ===")
    print(f"Testing percentiles: {percentiles}")
    print(f"Optimization metric: {metric}")

    if X_val is None:
        # Use 20% of training data for validation
        n_val = int(0.2 * len(X_train))
        indices = np.random.permutation(len(X_train))
        X_train_subset = X_train[indices[n_val:]]
        X_val = X_train[indices[:n_val]]
        print(f"Created validation set: {len(X_val)} samples")
    else:
        X_train_subset = X_train

    best_score = -np.inf
    best_percentile = percentiles[0]
    results = []

    for p in percentiles:
        print(f"Testing percentile: {p}%")

        model = PercentileThresholdKDE(
            bandwidth=bandwidth, n_neighbors=n_neighbors, use_gpu=use_gpu, percentile=p
        )

        model.fit(X_train_subset, X_val, verbose=False)

        # Get density scores for validation data
        val_scores = model.score_samples(X_val)
        val_predictions = model.predict(X_val)

        # Compute unsupervised quality metrics
        if metric == "density_range":
            # Maximize the range of density scores (better separation)
            score = np.max(val_scores) - np.min(val_scores)

        elif metric == "stability":
            # Minimize variance in "normal" region (more stable threshold)
            normal_scores = val_scores[val_predictions == 1]
            if len(normal_scores) > 0:
                score = -np.var(normal_scores)  # Negative because we want low variance
            else:
                score = -np.inf

        elif metric == "separation":
            # Maximize separation between normal and anomalous regions
            normal_scores = val_scores[val_predictions == 1]
            anomaly_scores = val_scores[val_predictions == -1]

            if len(normal_scores) > 0 and len(anomaly_scores) > 0:
                normal_mean = np.mean(normal_scores)
                anomaly_mean = np.mean(anomaly_scores)
                pooled_std = np.sqrt(
                    (np.var(normal_scores) + np.var(anomaly_scores)) / 2
                )

                # Cohen's d (effect size)
                score = abs(normal_mean - anomaly_mean) / (pooled_std + 1e-10)
            else:
                score = -np.inf

        anomaly_rate = (val_predictions == -1).mean()

        results.append(
            {
                "percentile": p,
                "score": score,
                "anomaly_rate": anomaly_rate,
                "threshold": model.threshold,
                "val_score_mean": np.mean(val_scores),
                "val_score_std": np.std(val_scores),
            }
        )

        print(
            f"  Score: {score:.4f}, Anomaly rate: {anomaly_rate:.1%}, "
            f"Threshold: {model.threshold:.4f}"
        )

        if score > best_score:
            best_score = score
            best_percentile = p
    #compute negative log likelihood
    loss = -val_scores.mean()
    print(f"  Negative log likelihood: {loss:.4f}")
    print(f"\nBest percentile: {best_percentile}% (Score: {best_score:.4f})")
    return best_percentile, results

def plot_tsne(tsne_data1, preds, title):

    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_data1[preds == 1, 0], tsne_data1[preds == 1, 1], 
                color='blue', label='ID', alpha=0.5)
    plt.scatter(tsne_data1[preds == -1, 0], tsne_data1[preds == -1, 1], 
                color='red', label='OOD', alpha=0.5)
    plt.title(title)
    
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')

    #save figure
    plt.legend()
    plt.savefig(f"figures/{title.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")


def evaluate_and_plot(model, X_val, normal_data, anomaly_data):
        """
        Evaluate the model on validation, normal, and anomalous datasets and plot results.

        Args:
            model: Trained anomaly detection model
            X_val: Validation dataset
            normal_data: Synthetic normal data
            anomaly_data: Synthetic anomalous data
        """
        def evaluate_dataset(data, data_name):
            predictions = model.predict(data)
            scores = model.score_samples(data)
            anomaly_scores = model.decision_function(data)
            anomaly_count = np.sum(predictions == -1)
            anomaly_rate = anomaly_count / len(data)
            print(f"\n=== {data_name} Results ===")
            print(f"{data_name} anomalies detected: {anomaly_count}/{len(data)} ({anomaly_rate:.1%})")
            print(f"Density score range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"Anomaly score range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
            return predictions

        print(f"\nEvaluating on validation set...")
        val_predictions = evaluate_dataset(X_val, "Validation Set")

        print(f"\nEvaluating on normal test set...")
        normal_test_predictions = evaluate_dataset(normal_data, "Train Set")

        print(f"\nEvaluating on anomalous test set...")
        anomaly_test_predictions = evaluate_dataset(anomaly_data, "Half/Half Anomalous Set")

        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
        reduced_val = tsne.fit_transform(X_val)
        reduced_normal = tsne.fit_transform(normal_data)
        reduced_anomaly = tsne.fit_transform(anomaly_data)

        plot_tsne(reduced_val, val_predictions, "Density Results of Validation Set")
        plot_tsne(reduced_normal, normal_test_predictions, "Density Results of Train Set")
        plot_tsne(reduced_anomaly, anomaly_test_predictions, "Density Results of Anomalous Set")

def main():
    print("Running", __file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args, remaining_argv = parser.parse_known_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    # Data loading arguments
    parser.add_argument('--data_path', type=str, default = None,help='Path to data file')

    # Data splitting arguments
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test set fraction"
    )
    parser.add_argument(
        "--val_size", type=float, default=0.1, help="Validation set fraction"
    )
    parser.add_argument(
        "--temporal_split",
        action="store_true",
        help="Use temporal split (no shuffle) for time series",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")

    # Model arguments
    # parser.add_argument(
    #     "--percentile", type=float, default=5.0, help="Percentile threshold"
    # )
    # parser.add_argument('--bandwidth', type=float, default=1.0, help='KDE bandwidth')
    # parser.add_argument('--k_neighbors', type=int, default=100, help='Number of neighbors')
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU")
    parser.add_argument(
        "--no_normalize", action="store_true", help="Disable data normalization"
    )

    # Optimization arguments
    parser.add_argument(
        "--optimize_percentile", action="store_true", help="Find optimal percentile"
    )
    parser.add_argument(
        "--optimization_metric",
        choices=["density_range", "stability", "separation"],
        default="density_range",
        help="Metric for percentile optimization",
    )
    parser.add_argument("--pca", action="store_true")

    # Output arguments
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    # parser.add_argument('--save_model', type=str, default="trained_kde", help='Save model path')
    # parser.add_argument('--save_results', type=str, help='Save results to file')
    parser.add_argument(
        "--devid", type=int, default=0, help="GPU device ID (if using GPU)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/abiomed/models/kde",
        help="Path to save model and results",
    )

    # Synthetic data fallback (if no data_path provided)
    parser.add_argument(
        "--n_samples", type=int, default=10000, help="Number of synthetic samples"
    )
    parser.add_argument(
        "--n_features", type=int, default=2, help="Number of synthetic features"
    )
    parser.add_argument(
        "--anomaly_rate", type=float, default=0.05, help="Synthetic anomaly rate"
    )

    # noisy dataset parameters
    parser.add_argument(
        "--action", action="store_true", help="Create dataset with noisy actions"
    )
    parser.add_argument(
        "--transition",
        action="store_true",
        help="Create dataset with noisy transitions",
    )
    parser.add_argument("--env", type=str, default="")

    # ============ abiomed environment arguments ============
    parser.add_argument("--model_name", type=str, default="10min_1hr_all_data")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path_wm", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--gamma1", type=float, default=0.2)
    parser.add_argument("--gamma2", type=float, default=0.5)
    parser.add_argument("--gamma3", type=float, default=1)
    parser.add_argument(
        "--action_space_type",
        type=str,
        default="continuous",
        choices=["continuous", "discrete"],
        help="Type of action space for the environment",
    )
    parser.add_argument(
        "--noise_rate",
        type=float,
        help="Portion of data to be noisy with probability",
        default=0.0,
    )
    parser.add_argument(
        "--noise_scale", type=float, help="magnitude of noise", default=0.0
    )

    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)

    print("=== Percentile-Based KDE Anomaly Detection ===")

    # Load data
    print(f"\nLoading data from: {args.data_path}")

    print(f"\nSplitting data...")
    data_splits = load_data(
        args.data_path,
        test_size=args.test_size,
        validation_size=args.val_size,
        args=args,
    )

    X_train = data_splits["X_train"]
    X_val = data_splits["X_val"]
    X_test = data_splits["X_test"]

    # No true labels available
    y_test = None

    print(f"\nFinal data splits:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape if X_val is not None else 'None'}")
    print(f"  Test: {X_test.shape}")

    # Find optimal percentile if requested
    if args.optimize_percentile:
        print(f"\nOptimizing percentile using '{args.optimization_metric}' metric...")
        optimal_percentile, search_results = find_optimal_percentile(
            X_train,
            X_val,
            bandwidth=args.bandwidth,
            n_neighbors=args.k_neighbors,
            use_gpu=not args.no_gpu,
            metric=args.optimization_metric,
        )
        percentile_to_use = optimal_percentile
    else:
        percentile_to_use = args.percentile
    #calculate negative log likelihood on validation set

    # Train final model
    print(f"\nTraining final model with {percentile_to_use}% threshold...")
    model = PercentileThresholdKDE(
        bandwidth=args.bandwidth,
        n_neighbors=args.k_neighbors,
        use_gpu=not args.no_gpu,
        normalize=not args.no_normalize,
        percentile=percentile_to_use,
        pca=args.pca,
        devid=args.devid,
    )

    start_time = time.time()
    model.fit(X_train, X_val)
    fit_time = time.time() - start_time

    print(f"Training completed in {fit_time:.2f} seconds")
    print("Creating synthetic data...")
    normal_data, anomaly_data = create_synthetic_data(
        n_samples= X_test.shape[0],
        dim= X_test.shape[1],
        anomaly_type='outlier'
    )

    evaluate_and_plot(model, X_val, X_train, normal_data)
    plot_likelihood_distributions(
                        model,
                        X_train,
                        X_val,
                        ood_data=normal_data,
                        thr = model.threshold,
                        title="Likelihood Distribution",
                        savepath=None,
                        bins=50
                    )
    
    suffix_parts = [args.save_model]
    if args.action:
        suffix_parts.append(f"action")
    if args.transition:
        suffix_parts.append(f"obs")
    if args.noise_scale > 0 or args.noise_rate > 0:
        suffix_parts.append(f"nr{args.noise_rate}")
        suffix_parts.append(f"ns{args.noise_scale}")
    if args.data_path and "5000eps" in args.data_path:
        suffix_parts.append("5000eps")
    elif args.data_path and "200000eps" in args.data_path:
        suffix_parts.append("200000eps")
    suffix_parts.append(str(args.percentile))
    save_path = "_".join(suffix_parts)
    if args.save_model:
        model.save_model(os.path.join(args.save_path, args.env, save_path))

    # Print threshold statistics
    stats = model.get_threshold_stats()
    print(f"\n=== Model Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"\n=== Usage Example ===")
    print("# To use the trained model on new data:")
    print("scores = model.score_samples(new_data)")
    print("predictions = model.predict(new_data)  # 1=normal, -1=anomaly")
    print("anomaly_scores = model.decision_function(new_data)  # higher=more anomalous")
    print(f"# Threshold: {model.threshold:.4f}")


if __name__ == "__main__":
    main()
