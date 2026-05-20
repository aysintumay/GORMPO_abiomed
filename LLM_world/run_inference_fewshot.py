'''
Single-shard run (one GPU):
  python LLM_world/run_inference_fewshot.py --shard 0 --num-shards 7 --k-shot 3

Launch all shards:
  bash LLM_world/launch_inference_fewshot.sh
'''
import argparse
import sys
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from LLM_world.world_model_fewshot import FewShotLLMWorldModel
from LLM_world.world_model import FEATURE_NAMES

DATA_PATH   = "/public/gormpo/10min_1hr_all_data.pkl"
MODEL_ID    = "m42-health/Llama3-Med42-8B"
VIZ_SAMPLES = [0, 1, 2, 3]

# ── args ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--shard",      type=int, default=0)
parser.add_argument("--num-shards", type=int, default=1)
parser.add_argument("--k-shot",     type=int, default=3)
args = parser.parse_args()

DEVICE      = f"cuda:{args.shard}"
VIZ_OUT_DIR = f"LLM_world/figures_fewshot_k{args.k_shot}"
RESULTS_DIR = f"LLM_world/results_fewshot_k{args.k_shot}"

os.makedirs(VIZ_OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, obs: np.ndarray) -> float:
    sigma = np.maximum(sigma, 1e-8)
    z = (obs - mu) / sigma
    return float((sigma * (z * (2 * stats.norm.cdf(z) - 1)
                           + 2 * stats.norm.pdf(z)
                           - 1 / np.sqrt(np.pi))).mean())


def plot_sample(model, idx, out_dir):
    x, pl, y = model.data_test[idx]
    ctx    = torch.as_tensor(x)
    pl_int = int(round(model.unnorm_pl(torch.as_tensor(pl).float().mean()).item()))

    mean_norm, std_norm = model.predict(ctx, p_level_int=pl_int)

    ctx_phys  = model.unnorm_output(ctx.float())
    pred_phys = model.unnorm_output(mean_norm)
    std_phys  = std_norm.float() * model.std[model.columns].cpu()

    cols_no_pl = model.columns[:-1]
    y_phys = (torch.as_tensor(y).float().reshape(6, 11)
              * model.std[cols_no_pl] + model.mean[cols_no_pl])

    t_ctx  = np.arange(-5, 1) * 10
    t_pred = np.arange(1, 7)  * 10

    n_feat = 11
    ncols  = 4
    nrows  = (n_feat + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()
    fig.suptitle(f"Test sample {idx}  |  P-level = {pl_int}  |  {args.k_shot}-shot", fontsize=12)

    for j in range(n_feat):
        ax  = axes[j]
        mu  = pred_phys[:, j].numpy()
        sig = std_phys[:, j].numpy()
        ax.plot(t_ctx,  ctx_phys[:, j].numpy(), color="steelblue", marker="o", ms=3, label="context")
        ax.plot(t_pred, y_phys[:, j].numpy(),   color="green",     marker="o", ms=3, ls="--", label="ground truth")
        ax.plot(t_pred, mu,                      color="tomato",    marker="o", ms=3, label="prediction")
        ax.fill_between(t_pred, mu - sig, mu + sig, color="tomato", alpha=0.2)
        ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_title(FEATURE_NAMES[j], fontsize=9)
        ax.set_xlabel("time (min)", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)

    for ax in axes[n_feat:]:
        ax.set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, f"sample_{idx:03d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved {path}", flush=True)


# ── load ───────────────────────────────────────────────────────────────────────

model = FewShotLLMWorldModel(model_id=MODEL_ID, device=DEVICE, k_shot=args.k_shot)
model.load_data(DATA_PATH)

n_test  = len(model.data_test)
indices = list(range(args.shard, n_test, args.num_shards))
print(f"[shard {args.shard}/{args.num_shards}] device={DEVICE} k_shot={args.k_shot} "
      f"samples={len(indices)} ({indices[0]}..{indices[-1]})", flush=True)


# ── visualizations (shard 0 only) ─────────────────────────────────────────────

if args.shard == 0:
    print(f"\nGenerating visualizations for samples {VIZ_SAMPLES} ...")
    for idx in VIZ_SAMPLES:
        plot_sample(model, idx, VIZ_OUT_DIR)


# ── metrics on this shard ─────────────────────────────────────────────────────

print(f"\n[shard {args.shard}] Computing metrics on {len(indices)} samples ...", flush=True)

mse_vals, mae_vals, crps_vals, idx_vals = [], [], [], []
for rank, idx in enumerate(indices):
    xi, pli, yi = model.data_test[idx]
    ctx_i = torch.as_tensor(xi)
    pl_i  = int(round(model.unnorm_pl(torch.as_tensor(pli).float().mean()).item()))

    m_norm, s_norm = model.predict(ctx_i, p_level_int=pl_i)

    y_true = torch.as_tensor(yi).float().reshape(model.forecast_horizon, model.num_features - 1).numpy()
    y_pred = m_norm[:, :-1].float().numpy()
    y_std  = s_norm[:, :-1].float().numpy()

    mse_vals.append(float(((y_pred - y_true) ** 2).mean()))
    mae_vals.append(float(np.abs(y_pred - y_true).mean()))
    crps_vals.append(crps_gaussian(y_pred, y_std, y_true))
    idx_vals.append(idx)
    print(f"  [shard {args.shard} | {rank+1}/{len(indices)}] idx={idx} "
          f"MSE={mse_vals[-1]:.4f} MAE={mae_vals[-1]:.4f} CRPS={crps_vals[-1]:.4f}", flush=True)

out_path = os.path.join(RESULTS_DIR, f"shard_{args.shard:02d}.npz")
np.savez(out_path, idx=idx_vals, mse=mse_vals, mae=mae_vals, crps=crps_vals)
print(f"\n[shard {args.shard}] Results saved to {out_path}", flush=True)
print(f"[shard {args.shard}] MSE={np.mean(mse_vals):.6f} "
      f"MAE={np.mean(mae_vals):.6f} CRPS={np.mean(crps_vals):.6f}", flush=True)
