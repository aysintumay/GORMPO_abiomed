'''
python abiomed_env/evaluate_transformer.py
Evaluates the Transformer world model on the test set with 10 forward passes,
computing MSE, MAE, and CRPS (matching the LLM evaluation protocol).
'''
import sys
import os
import torch
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from model import WorldModel
from config import model_kwargs_10min_1hr_full

DATA_PATH  = "/public/gormpo/10min_1hr_all_data.pkl"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "10min_1hr_all_data_model.pth")
DEVICE     = "cuda:0"
NUM_SAMPLES = 10


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, obs: np.ndarray) -> float:
    sigma = np.maximum(sigma, 1e-8)
    z = (obs - mu) / sigma
    return float((sigma * (z * (2 * stats.norm.cdf(z) - 1)
                           + 2 * stats.norm.pdf(z)
                           - 1 / np.sqrt(np.pi))).mean())


def get_transformer_confidence():
    model_kwargs = dict(model_kwargs_10min_1hr_full)
    model_kwargs["device"] = DEVICE
    model = WorldModel(**model_kwargs)
    model.load_data(DATA_PATH)
    model.load_model(MODEL_PATH)
    model.model.eval()
    # print(f"Test set size: {len(model.data_test)}")

    # ── run 10-sample inference ───────────────────────────────────────────────────

    # print(f"Running test_output_multiple(num_samples={NUM_SAMPLES}) ...")
    outputs, ys, _ = model.test_output_multiple(num_samples=NUM_SAMPLES)

    # outputs : list of (B, num_samples, horizon, 11) tensors
    # ys      : list of (B, horizon, 11) tensors
    all_preds = torch.cat(outputs, dim=0).cpu().numpy()   # (N, 10, horizon, 11)
    all_ys    = torch.cat(ys,      dim=0).cpu().numpy()   # (N, horizon, 11)

    print(f"Predictions shape: {all_preds.shape}")
    print(f"Ground truth shape: {all_ys.shape}")

    mean_pred = all_preds.mean(axis=1)   # (N, horizon, 11)
    std_pred  = all_preds.std(axis=1)    # (N, horizon, 11)

    return mean_pred, std_pred, all_ys

def get_tranformer_metrics(mean_pred, std_pred, all_ys):

    mse_per  = ((mean_pred - all_ys) ** 2).mean(axis=(1, 2))   # (N,)
    mae_per  = np.abs(mean_pred - all_ys).mean(axis=(1, 2))    # (N,)
    crps_per = np.array([
        crps_gaussian(mean_pred[[i]], std_pred[[i]], all_ys[[i]])
        for i in range(len(all_ys))
    ])                                                           # (N,)

    print(f"\n=== Transformer test set metrics (n={len(all_ys)}, normalized space) ===")
    print(f"MSE:  {mse_per.mean():.6f} ± {mse_per.std():.6f}")
    print(f"MAE:  {mae_per.mean():.6f} ± {mae_per.std():.6f}")
    print(f"CRPS: {crps_per.mean():.6f} ± {crps_per.std():.6f}")
