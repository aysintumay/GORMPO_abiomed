import sys, os, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from LLM_world.world_model_finetune import FinetunedLLMWorldModel
from LLM_world.world_model import FEATURE_NAMES

MODEL_ID     = "m42-health/Llama3-Med42-8B"
ADAPTER_PATH = "LLM_world/checkpoints_finetune/final"
DATA_PATH    = "/public/gormpo/10min_1hr_all_data.pkl"
VIZ_SAMPLES  = [0, 1, 2, 3]
VIZ_OUT_DIR  = "LLM_world/figures_finetune"
DEVICE       = "cuda:1"

os.makedirs(VIZ_OUT_DIR, exist_ok=True)

print("Loading model ...", flush=True)
model = FinetunedLLMWorldModel(model_id=MODEL_ID, adapter_path=ADAPTER_PATH, device=DEVICE)
model.load_data(DATA_PATH)
print(f"Data loaded | test: {len(model.data_test)}", flush=True)


def _sample_arrays(idx):
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
    return ctx_phys, y_phys, pred_phys, std_phys, pl_int


print(f"Computing shared y-limits for samples {VIZ_SAMPLES} ...", flush=True)
n_feat = 11
lo = [float("inf")]  * n_feat
hi = [float("-inf")] * n_feat

for idx in VIZ_SAMPLES:
    print(f"  pre-pass sample {idx}", flush=True)
    ctx_phys, y_phys, pred_phys, std_phys, _ = _sample_arrays(idx)
    mu  = pred_phys.numpy()
    sig = std_phys.numpy()
    for j in range(n_feat):
        vals = np.concatenate([ctx_phys[:, j].numpy(), y_phys[:, j].numpy(),
                               mu[:, j], mu[:, j] - sig[:, j], mu[:, j] + sig[:, j]])
        lo[j] = min(lo[j], float(vals.min()))
        hi[j] = max(hi[j], float(vals.max()))

ylims = []
for j in range(n_feat):
    margin = 0.05 * max(hi[j] - lo[j], 1e-6)
    ylims.append((lo[j] - margin, hi[j] + margin))

t_ctx  = np.arange(-5, 1) * 10
t_pred = np.arange(1, 7)  * 10
ncols  = 4
nrows  = (n_feat + ncols - 1) // ncols

for idx in VIZ_SAMPLES:
    print(f"  plotting sample {idx}", flush=True)
    ctx_phys, y_phys, pred_phys, std_phys, pl_int = _sample_arrays(idx)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()
    fig.suptitle(f"Test sample {idx}  |  P-level = {pl_int}  |  finetuned", fontsize=12)
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
        ax.set_ylim(ylims[j])
        if j == 0:
            ax.legend(fontsize=7)
    for ax in axes[n_feat:]:
        ax.set_visible(False)
    plt.tight_layout()
    path = os.path.join(VIZ_OUT_DIR, f"sample_{idx:03d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved {path}", flush=True)

print("Done.", flush=True)
