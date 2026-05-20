'''
python LLM_world/visualize_predictions.py
Plots ground truth vs LLM predictions for a few test samples.
'''
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from LLM_world.world_model import LLMWorldModel, FEATURE_NAMES

DATA_PATH   = "/public/gormpo/10min_1hr_all_data.pkl"
MODEL_ID    = "m42-health/Llama3-Med42-8B"
DEVICE      = "cuda:0"
SAMPLE_IDXS = [0, 1, 2, 3]   # which test samples to visualize
OUT_DIR     = "LLM_world/figures"

os.makedirs(OUT_DIR, exist_ok=True)

model = LLMWorldModel(model_id=MODEL_ID, device=DEVICE)
model.load_data(DATA_PATH)

for idx in SAMPLE_IDXS:
    x, pl, y = model.data_test[idx]
    context   = torch.as_tensor(x)    # (6, 12) normalized
    pl_int    = int(round(model.unnorm_pl(torch.as_tensor(pl).float().mean()).item()))

    mean_norm, std_norm = model.predict(context, p_level_int=pl_int)

    # unnormalize everything to physical units
    ctx_phys  = model.unnorm_output(context.float())                 # (6, 12)
    pred_phys = model.unnorm_output(mean_norm)                       # (6, 12)
    std_phys  = std_norm.float() * model.std[model.columns].cpu()    # (6, 12) — scale only

    # ground truth future: y is (66,) = (6, 11) normalized (no pump level)
    y_norm = torch.as_tensor(y).float().reshape(6, 11)
    # unnormalize the 11 features (all except pump level)
    cols_no_pl = model.columns[:-1]
    y_phys = y_norm.cpu() * model.std[cols_no_pl] + model.mean[cols_no_pl]  # (6, 11)

    # time axes
    t_ctx  = np.arange(-5, 1) * 10    # -50, -40, ..., 0
    t_pred = np.arange(1, 7)  * 10    # +10, +20, ..., +60

    # plot 11 features (skip pump level which is just the action)
    n_features = 11
    ncols = 4
    nrows = (n_features + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()
    fig.suptitle(f"Test sample {idx}  |  P-level action = {pl_int}", fontsize=13)

    for j in range(n_features):
        ax = axes[j]
        # context (past)
        ax.plot(t_ctx, ctx_phys[:, j].numpy(), color="steelblue", marker="o",
                markersize=3, label="context")
        # ground truth future
        ax.plot(t_pred, y_phys[:, j].numpy(), color="green", marker="o",
                markersize=3, linestyle="--", label="ground truth")
        # predicted mean ± 1 std
        mu  = pred_phys[:, j].numpy()
        sig = std_phys[:, j].numpy()
        ax.plot(t_pred, mu, color="tomato", marker="o", markersize=3, label="prediction")
        ax.fill_between(t_pred, mu - sig, mu + sig, color="tomato", alpha=0.2)
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(FEATURE_NAMES[j], fontsize=9)
        ax.set_xlabel("time (min)", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)

    for ax in axes[n_features:]:
        ax.set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"sample_{idx:03d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

print("Done.")
