'''
python LLM_world/visualize_transformer.py
Generates forecast figures for the Transformer digital twin on the same
4 test samples used for the LLM visualizations. Uses dropout sampling
(train mode) for uncertainty estimation.
'''
import sys
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abiomed_env.model import WorldModel
from abiomed_env.config import model_kwargs_10min_1hr_full

FEATURE_NAMES = [
    "MAP (PumpPressure)", "PumpSpeed", "PumpFlow", "LVP", "LVEDP",
    "SYSTOLIC", "DIASTOLIC", "PULSAT", "PumpCurrent", "Heart Rate", "ESE_lv",
]

DATA_PATH   = "/public/gormpo/10min_1hr_all_data.pkl"
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "abiomed_env", "data",
                           "10min_1hr_all_data_model.pth")
DEVICE      = "cuda:0"
NUM_SAMPLES = 10
SAMPLE_IDXS = [0, 1, 2, 3]
OUT_DIR     = "LLM_world/figures_transformer"

os.makedirs(OUT_DIR, exist_ok=True)

# ── load model ────────────────────────────────────────────────────────────────

model_kwargs = dict(model_kwargs_10min_1hr_full)
model_kwargs["device"] = DEVICE
model = WorldModel(**model_kwargs)
model.load_data(DATA_PATH)
model.load_model(MODEL_PATH)

# columns used: [0,1,...,10,12] → 12 features; last is Pump Level
cols      = model.columns          # [0,1,...,10,12]
cols_no_pl = model.columns[:-1]    # [0,1,...,10]

t_ctx  = np.arange(-5, 1) * 10    # -50..0 min
t_pred = np.arange(1, 7)  * 10    # 10..60 min

# ── per-sample figure ─────────────────────────────────────────────────────────

for idx in SAMPLE_IDXS:
    x, pl, y = model.data_test[idx]
    x  = x.unsqueeze(0).to(DEVICE).float()    # (1, 6, 12)
    pl = pl.unsqueeze(0).to(DEVICE).float()   # (1, 6)

    pl_int = int(round((pl.mean().item() * model.std[-1] + model.mean[-1]).item()))

    # dropout sampling for uncertainty
    model.model.train()
    with torch.no_grad():
        samples = []
        for _ in range(NUM_SAMPLES):
            out = model.model(x, pl)                          # (1, 66)
            samples.append(out.reshape(model.forecast_horizon, model.num_features - 1))
    model.model.eval()

    samples = torch.stack(samples)          # (S, 6, 11)
    mean_norm = samples.mean(0)             # (6, 11)
    std_norm  = samples.std(0)              # (6, 11)

    # unnormalize
    std_t  = model.std[cols_no_pl].cpu()
    mean_t = model.mean[cols_no_pl].cpu()
    pred_phys = mean_norm.cpu() * std_t + mean_t   # (6, 11)
    std_phys  = std_norm.cpu()  * std_t             # (6, 11)

    ctx_phys = x[0].cpu() * model.std[cols].cpu() + model.mean[cols].cpu()  # (6, 12)

    y_phys = (torch.as_tensor(y).float().reshape(model.forecast_horizon, model.num_features - 1)
              * model.std[cols_no_pl].cpu() + model.mean[cols_no_pl].cpu())  # (6, 11)

    n_feat = model.num_features - 1   # 11
    ncols  = 4
    nrows  = (n_feat + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()
    fig.suptitle(f"Test sample {idx}  |  P-level = {pl_int}  |  Transformer", fontsize=12)

    for j in range(n_feat):
        ax  = axes[j]
        mu  = pred_phys[:, j].numpy()
        sig = std_phys[:, j].numpy()
        label = FEATURE_NAMES[j]
        ax.plot(t_ctx,  ctx_phys[:, j].numpy(),  color="steelblue", marker="o", ms=3, label="context")
        ax.plot(t_pred, y_phys[:, j].numpy(),    color="green",     marker="o", ms=3, ls="--", label="ground truth")
        ax.plot(t_pred, mu,                       color="tomato",    marker="o", ms=3, label="prediction")
        ax.fill_between(t_pred, mu - sig, mu + sig, color="tomato", alpha=0.2)
        ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("time (min)", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)

    for ax in axes[n_feat:]:
        ax.set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"sample_{idx:03d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

print("Done.")
