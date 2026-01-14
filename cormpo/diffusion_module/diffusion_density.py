"""
Diffusion Model (DDIM/DDPM) for Density Estimation and OOD Detection
Adapted from GORMPO's diffusion implementation for Abiomed environment
"""

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

try:
    import yaml
except Exception:
    yaml = None


# -----------------------------
# Data
# -----------------------------


class NPZTargetDataset(Dataset):
    def __init__(self, npz_path: str, dtype: torch.dtype = torch.float32):
        super().__init__()
        # Robust loader: supports .npz/.npy with object arrays and .pkl pickled dicts
        data = None
        try:
            data = np.load(npz_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object and data.size == 1:
                data = data.item()
        except Exception:
            try:
                import pickle
                with open(npz_path, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                raise ValueError(
                    f"Failed to load dataset from {npz_path}. Tried numpy.load and pickle.load. Error: {e}"
                )
        # Expect common key names; fall back heuristics
        possible_target_keys = [
            "target",
            "y",
            "action",
            "next",
            "outputs",
            "X_target",
            "data",
            "x",
            "samples",
        ]

        def available_keys(obj):
            if isinstance(obj, dict):
                return list(obj.keys())
            try:
                return list(obj.keys())  # NpzFile supports .keys()
            except Exception:
                try:
                    return list(obj.files)
                except Exception:
                    return []

        def pick(keys):
            for k in keys:
                try:
                    if k in data:
                        return data[k]
                except Exception:
                    # For NpzFile mapping interface
                    try:
                        if hasattr(data, "files") and k in data.files:
                            return data[k]
                    except Exception:
                        pass
            raise KeyError(
                f"Could not find any of keys {keys} in {available_keys(data)}. "
                "Please rename your arrays or pass a small shim."
            )

        self.target_np = pick(possible_target_keys)

        self.target = torch.from_numpy(self.target_np).to(dtype)

        # Flatten trailing dims into feature vectors for generic MLP handling
        self.target = self.target.view(self.target.size(0), -1)

        self.num_samples = self.target.size(0)
        self.target_dim = self.target.size(1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.target[idx]


# -----------------------------
# Model: Unconditional noise predictor (epsilon net)
# -----------------------------


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps expected in [0, T), shape [batch]
        half_dim = self.embedding_dim // 2
        device = timesteps.device
        exponent = -math.log(10000.0) / max(1, (half_dim - 1))
        freqs = torch.exp(torch.arange(half_dim, device=device) * exponent)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class UnconditionalEpsilonMLP(nn.Module):
    def __init__(
        self,
        target_dim: int,
        hidden_dim: int = 512,
        time_embed_dim: int = 128,
        num_hidden_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        layers = []
        in_dim = target_dim + time_embed_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, target_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t: [B, target_dim], t: [B]
        t_emb = self.time_embed(t)
        h = torch.cat([x_t, t_emb], dim=1)
        h = self.backbone(h)
        eps = self.head(h)
        return eps


class UnconditionalEpsilonTransformer(nn.Module):
    def __init__(
        self,
        target_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.d_model = d_model

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_to_model = nn.Linear(time_embed_dim, d_model)

        # Represent x_t as a sequence of length L = target_dim with scalar tokens → project to d_model
        self.x_proj = nn.Linear(1, d_model)
        # Positional embedding for L positions
        self.pos_embed = nn.Parameter(torch.zeros(1, target_dim, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Predict per-position noise and then flatten back to vector
        self.out_head = nn.Linear(d_model, 1)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bsz, dim = x_t.shape
        assert dim == self.target_dim, "x_t must have target_dim features"

        # Build token sequence for x_t: shape [B, L, 1] → proj to [B, L, d_model]
        x_tokens = x_t.unsqueeze(-1)
        x_tokens = self.x_proj(x_tokens)

        # Time embedding added to all tokens
        t_emb = self.time_to_model(self.time_embed(t)).unsqueeze(1)  # [B,1,d_model]
        tokens = x_tokens + self.pos_embed + t_emb

        h = self.encoder(tokens)
        eps_tokens = self.out_head(h)  # [B, L, 1]
        eps = eps_tokens.squeeze(-1)  # [B, L]
        return eps


# -----------------------------
# Sampling (unconditional generation) using deterministic DDIM
# -----------------------------


@torch.no_grad()
def sample(
    model: nn.Module,
    scheduler: DDIMScheduler,
    num_samples: int = 1,
    num_inference_steps: int = 50,
    eta: float = 0.0,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()

    scheduler.set_timesteps(num_inference_steps)
    # Get target_dim from model
    if hasattr(model, "target_dim"):
        target_dim = model.target_dim
    elif hasattr(model, "head"):
        target_dim = model.head.out_features
    else:
        raise ValueError("Cannot determine target_dim from model")

    x = torch.randn(num_samples, target_dim, device=device)

    for t in scheduler.timesteps:
        # Predict noise
        eps = model(x, t.expand(num_samples))
        # DDIM step
        out = scheduler.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x = out.prev_sample
    return x


# -----------------------------
# Density Estimation: Computing log p(x) from diffusion model
# -----------------------------


def gaussian_kl(mean1: torch.Tensor, var1: torch.Tensor, mean2: torch.Tensor, var2: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between two diagonal Gaussian distributions.
    KL(N(mean1, var1) || N(mean2, var2)) for diagonal covariances.
    """
    eps = 1e-12
    var1 = torch.clamp(var1, min=eps)
    var2 = torch.clamp(var2, min=eps)
    return 0.5 * (
        ((mean2 - mean1) ** 2) / var2 + (var1 / var2) - 1.0 + torch.log(var2) - torch.log(var1)
    ).sum(dim=1)


@torch.no_grad()
def log_prob_elbo(
    model: nn.Module,
    scheduler,
    x0: torch.Tensor,
    device: str = "cpu",
    num_inference_steps: int = None,
) -> torch.Tensor:
    """
    Compute log p(x0) using the ELBO (Evidence Lower Bound) from the diffusion process.

    This gives a lower bound on log p(x0) by using the variational bound:
    log p(x0) >= ELBO = E_q[log p(x0|x1)] - sum_t KL(q(x_{t-1}|x_t,x0) || p_theta(x_{t-1}|x_t)) - KL(q(x_T|x0) || p(x_T))

    Args:
        model: Unconditional epsilon prediction model
        scheduler: DDIMScheduler (or DDPMScheduler) with diffusion parameters
        x0: Data samples of shape [batch_size, target_dim]
        device: Device to run computation on
        num_inference_steps: Number of timesteps to use (default: None = use all T timesteps)
            If specified, uniformly subsample timesteps for faster approximation

    Returns:
        log_prob: Lower bound on log p(x0) for each sample, shape [batch_size]
    """
    model.eval()
    x0 = x0.to(device)

    # Get diffusion parameters
    betas = scheduler.betas.to(device).to(torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    T = betas.shape[0]
    bsz, dim = x0.shape

    # Determine which timesteps to use
    if num_inference_steps is not None and num_inference_steps < T:
        # Uniformly subsample timesteps for faster approximation
        # Use linspace to get evenly spaced timesteps
        timesteps = torch.linspace(0, T - 1, num_inference_steps, dtype=torch.long, device=device)
        # Ensure we include T-1 (last timestep) and 0 (first timestep)
        if timesteps[-1] != T - 1:
            timesteps[-1] = T - 1
        if timesteps[0] != 0:
            timesteps[0] = 0
        # Convert to list and add 1 (since loop uses 1-indexed)
        timestep_list = [int(t) + 1 for t in reversed(timesteps.cpu().tolist())]
    else:
        # Use all timesteps (original behavior)
        timestep_list = list(reversed(range(1, T + 1)))

    # Sample a single q trajectory for expectation (Monte Carlo)
    eps = torch.randn(bsz, dim, device=device)
    x_t = torch.sqrt(alphas_cumprod[-1]) * x0 + torch.sqrt(1.0 - alphas_cumprod[-1]) * eps

    total = torch.zeros(bsz, device=device)

    # L_T = KL(q(x_T|x0) || N(0, I))
    mean_T = torch.sqrt(alphas_cumprod[-1]) * x0
    var_T = (1.0 - alphas_cumprod[-1]) * torch.ones_like(x0)
    mean_p = torch.zeros_like(x0)
    var_p = torch.ones_like(x0)
    total = total + gaussian_kl(mean_T, var_T, mean_p, var_p)

    # Loop through selected timesteps for KL terms, and handle t=1 as NLL of p_theta(x0|x1)
    for t in timestep_list:
        at = alphas[t - 1]
        a_bar_t = alphas_cumprod[t - 1]
        if t > 1:
            a_bar_prev = alphas_cumprod[t - 2]
        else:
            a_bar_prev = torch.tensor(1.0, device=device)

        beta_t = betas[t - 1]
        # Posterior variance (tilde beta_t) with clamp
        beta_t_tilde = (1.0 - a_bar_prev) / (1.0 - a_bar_t) * beta_t
        beta_t_tilde = torch.clamp(beta_t_tilde, min=1e-20)

        # Predict epsilon at timestep t
        t_batch = torch.full((bsz,), t - 1, device=device, dtype=torch.long)
        eps_theta = model(x_t, t_batch)

        # Predict mean of p_theta(x_{t-1}|x_t) for epsilon parameterization
        coef1 = 1.0 / torch.sqrt(at)
        coef2 = beta_t / torch.sqrt(1.0 - a_bar_t)
        mu_theta = coef1 * (x_t - coef2 * eps_theta)

        # True posterior q(x_{t-1}|x_t, x0)
        mu_q = (
            (torch.sqrt(a_bar_prev) * beta_t / (1.0 - a_bar_t)) * x0
            + (torch.sqrt(at) * (1.0 - a_bar_prev) / (1.0 - a_bar_t)) * x_t
        )
        var_q = beta_t_tilde * torch.ones_like(x0)

        if t > 1:
            # KL term
            total = total + gaussian_kl(mu_q, var_q, mu_theta, var_q)
            # Sample x_{t-1} from q to continue the trajectory
            noise = torch.randn_like(x0)
            x_t = mu_q + torch.sqrt(beta_t_tilde) * noise
        else:
            # Reconstruction term: -log p_theta(x0|x1)
            # At t=1, beta_t_tilde → 0, so use beta_1 instead as in DDPM paper
            # This avoids numerical instability from dividing by near-zero variance
            var = beta_t  # Use beta_1 directly instead of posterior variance
            var = torch.clamp(var, min=1e-20)
            # Proper Gaussian NLL formula: NLL = 0.5 * (log(2πσ²) + (x-μ)²/σ²) summed over dims
            # Since var is scalar (same for all dims), factor out the log term
            nll0 = 0.5 * (
                dim * torch.log(2 * torch.pi * var) +  # Constant log term × dim
                ((x0 - mu_theta) ** 2 / var).sum(dim=1)  # Squared error term summed
            )
            total = total + nll0

    # ELBO is a lower bound, so we return it as log_prob (negative NLL)
    log_prob = -total
    return log_prob  # shape [bsz]


@torch.no_grad()
def score_function(
    model: nn.Module,
    scheduler,
    x_t: torch.Tensor,
    t: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute the score function (gradient of log probability) at timestep t.

    The score function is: ∇_x log p_t(x_t) = -eps_theta(x_t, t) / sqrt(1 - alpha_bar_t)

    Args:
        model: Unconditional epsilon prediction model
        scheduler: Scheduler with diffusion parameters
        x_t: Noisy samples at timestep t, shape [batch_size, target_dim]
        t: Timestep values, shape [batch_size] with values in [0, T-1]
        device: Device to run computation on

    Returns:
        score: Score function values, shape [batch_size, target_dim]
    """
    model.eval()
    x_t = x_t.to(device)
    t = t.to(device)

    # Get diffusion parameters
    betas = scheduler.betas.to(device).to(torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Predict epsilon
    eps_theta = model(x_t, t)

    # Compute alpha_bar_t for each sample
    # t is in [0, T-1], so we need to index into alphas_cumprod
    t_idx = t.long()
    t_idx = torch.clamp(t_idx, 0, len(alphas_cumprod) - 1)
    alpha_bar_t = alphas_cumprod[t_idx].unsqueeze(-1)  # [batch_size, 1]

    # Score = -eps / sqrt(1 - alpha_bar_t)
    score = -eps_theta / torch.sqrt(1.0 - alpha_bar_t + 1e-8)
    return score


@torch.no_grad()
def log_prob(
    model: nn.Module,
    scheduler,
    x0: torch.Tensor,
    device: str = "cpu",
    method: str = "elbo",
    num_inference_steps: int = None,
) -> torch.Tensor:
    """
    Compute log p(x0) from the unconditional diffusion model.

    Args:
        model: Unconditional epsilon prediction model
        scheduler: Scheduler with diffusion parameters
        x0: Data samples of shape [batch_size, target_dim]
        device: Device to run computation on
        method: Method to use. Currently only "elbo" is supported.
        num_inference_steps: Number of timesteps for approximation (None = use all)

    Returns:
        log_prob: Log probability (or lower bound) for each sample, shape [batch_size]
    """
    if method == "elbo":
        return log_prob_elbo(model, scheduler, x0, device, num_inference_steps)
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods: 'elbo'")


def evaluate_log_prob(
    model: nn.Module,
    scheduler,
    data: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 512,
    num_inference_steps: int = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Evaluate log probabilities for a dataset.

    Args:
        model: Unconditional epsilon prediction model
        scheduler: Scheduler with diffusion parameters
        data: Dataset tensor of shape [num_samples, target_dim]
        device: Device to run computation on
        batch_size: Batch size for evaluation
        num_inference_steps: Number of timesteps for approximation (None = use all)

    Returns:
        log_probs: Log probabilities for each sample, shape [num_samples]
        stats: Dictionary with statistics (mean, std, min, max)
    """
    model.eval()
    data = data.to(device)

    log_probs_list = []
    num_samples = data.shape[0]

    for i in range(0, num_samples, batch_size):
        batch = data[i:i + batch_size]
        batch_log_probs = log_prob(model, scheduler, batch, device, num_inference_steps=num_inference_steps)
        log_probs_list.append(batch_log_probs)

    log_probs = torch.cat(log_probs_list, dim=0)

    stats = {
        "mean": log_probs.mean().item(),
        "std": log_probs.std().item(),
        "min": log_probs.min().item(),
        "max": log_probs.max().item(),
    }

    return log_probs, stats
