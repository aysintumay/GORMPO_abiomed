"""
Regenerate inference figures for fewshot-k3, fewshot-k5, finetune, and transformer
with standardized y-axis limits shared across all models and all 4 viz samples.

Two-pass approach:
  Pass 1: load each model, run inference for 4 samples, save arrays, unload model.
  Pass 2: compute global per-feature ylims from all arrays, then plot everything.
"""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH    = "/public/gormpo/10min_1hr_all_data.pkl"
MODEL_ID     = "m42-health/Llama3-Med42-8B"
ADAPTER_PATH = "LLM_world/checkpoints_finetune/final"
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "..", "abiomed_env", "data",
                            "10min_1hr_all_data_model.pth")
VIZ_SAMPLES  = [0, 1, 2, 3]
DEVICE       = "cuda:1"
N_FEAT       = 11
NUM_DROPOUT  = 10

CONFIGS = [
    dict(key="fewshot_k3",   out_dir="LLM_world/figures_fewshot_k3",  label="3-shot"),
    dict(key="fewshot_k5",   out_dir="LLM_world/figures_fewshot_k5",  label="5-shot"),
    dict(key="finetune",     out_dir="LLM_world/figures_finetune",     label="finetuned"),
    dict(key="transformer",  out_dir="LLM_world/figures_transformer",  label="Transformer"),
]

FEATURE_NAMES = [
    "PumpPressure", "PumpSpeed", "PumpFlow", "LVP", "LVEDP",
    "SYSTOLIC", "DIASTOLIC", "PULSAT", "PumpCurrent", "Heart Rate", "ESE_lv",
]

for cfg in CONFIGS:
    os.makedirs(cfg["out_dir"], exist_ok=True)

# ── Pass 1: collect arrays from each model ─────────────────────────────────────

# arrays[key][idx] = dict(ctx, gt, mu, sig, pl_int)
arrays = {cfg["key"]: {} for cfg in CONFIGS}


def save_llm_arrays(model, key):
    for idx in VIZ_SAMPLES:
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
        arrays[key][idx] = dict(
            ctx=ctx_phys.numpy(), gt=y_phys.numpy(),
            mu=pred_phys.numpy(), sig=std_phys.numpy(), pl_int=pl_int,
        )
        print(f"  [{key}] sample {idx} done", flush=True)


# fewshot k3
print("\n=== Loading fewshot k3 ===", flush=True)
from LLM_world.world_model_fewshot import FewShotLLMWorldModel
m = FewShotLLMWorldModel(model_id=MODEL_ID, device=DEVICE, k_shot=3)
m.load_data(DATA_PATH)
save_llm_arrays(m, "fewshot_k3")
del m; torch.cuda.empty_cache()

# fewshot k5
print("\n=== Loading fewshot k5 ===", flush=True)
m = FewShotLLMWorldModel(model_id=MODEL_ID, device=DEVICE, k_shot=5)
m.load_data(DATA_PATH)
save_llm_arrays(m, "fewshot_k5")
del m; torch.cuda.empty_cache()

# finetune
print("\n=== Loading finetune ===", flush=True)
from LLM_world.world_model_finetune import FinetunedLLMWorldModel
m = FinetunedLLMWorldModel(model_id=MODEL_ID, adapter_path=ADAPTER_PATH, device=DEVICE)
m.load_data(DATA_PATH)
save_llm_arrays(m, "finetune")
del m; torch.cuda.empty_cache()

# transformer
print("\n=== Loading transformer ===", flush=True)
from abiomed_env.model import WorldModel
from abiomed_env.config import model_kwargs_10min_1hr_full
mw = dict(model_kwargs_10min_1hr_full); mw["device"] = DEVICE
tm = WorldModel(**mw)
tm.load_data(DATA_PATH)
tm.load_model(MODEL_PATH)
cols      = tm.columns
cols_no_pl = tm.columns[:-1]
std_t  = tm.std[cols_no_pl].cpu()
mean_t = tm.mean[cols_no_pl].cpu()
for idx in VIZ_SAMPLES:
    x, pl, y = tm.data_test[idx]
    x_d  = x.unsqueeze(0).to(DEVICE).float()
    pl_d = pl.unsqueeze(0).to(DEVICE).float()
    pl_int = int(round((pl_d.mean().item() * tm.std[-1] + tm.mean[-1]).item()))
    tm.model.train()
    with torch.no_grad():
        samples = [tm.model(x_d, pl_d).reshape(tm.forecast_horizon, tm.num_features - 1)
                   for _ in range(NUM_DROPOUT)]
    tm.model.eval()
    samples   = torch.stack(samples)
    mean_norm = samples.mean(0).cpu()
    std_norm  = samples.std(0).cpu()
    pred_phys = mean_norm * std_t + mean_t
    std_phys  = std_norm  * std_t
    ctx_phys  = x.cpu() * tm.std[cols].cpu() + tm.mean[cols].cpu()
    y_phys    = (torch.as_tensor(y).float().reshape(tm.forecast_horizon, tm.num_features - 1)
                 * std_t + mean_t)
    arrays["transformer"][idx] = dict(
        ctx=ctx_phys.numpy(), gt=y_phys.numpy(),
        mu=pred_phys.numpy(), sig=std_phys.numpy(), pl_int=pl_int,
    )
    print(f"  [transformer] sample {idx} done", flush=True)
del tm; torch.cuda.empty_cache()


# ── Pass 2: compute global y-limits ───────────────────────────────────────────

print("\nComputing global y-limits ...", flush=True)
lo = [float("inf")]  * N_FEAT
hi = [float("-inf")] * N_FEAT

for cfg in CONFIGS:
    for idx in VIZ_SAMPLES:
        a = arrays[cfg["key"]][idx]
        for j in range(N_FEAT):
            vals = np.concatenate([
                a["ctx"][:, j], a["gt"][:, j],
                a["mu"][:, j],
                a["mu"][:, j] - a["sig"][:, j],
                a["mu"][:, j] + a["sig"][:, j],
            ])
            lo[j] = min(lo[j], float(vals.min()))
            hi[j] = max(hi[j], float(vals.max()))

ylims = []
for j in range(N_FEAT):
    margin = 0.05 * max(hi[j] - lo[j], 1e-6)
    ylims.append((lo[j] - margin, hi[j] + margin))

print("Per-feature y-limits:")
for j, (l, h) in enumerate(ylims):
    print(f"  {FEATURE_NAMES[j]:<16} [{l:.3f}, {h:.3f}]")


# ── Pass 2: plot ───────────────────────────────────────────────────────────────

t_ctx  = np.arange(-5, 1) * 10
t_pred = np.arange(1, 7)  * 10
ncols  = 4
nrows  = (N_FEAT + ncols - 1) // ncols

for cfg in CONFIGS:
    print(f"\nPlotting {cfg['key']} ...", flush=True)
    for idx in VIZ_SAMPLES:
        a = arrays[cfg["key"]][idx]
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
        axes = axes.flatten()
        fig.suptitle(f"Test sample {idx}  |  P-level = {a['pl_int']}  |  {cfg['label']}",
                     fontsize=12)
        for j in range(N_FEAT):
            ax  = axes[j]
            mu  = a["mu"][:, j]
            sig = a["sig"][:, j]
            ax.plot(t_ctx,  a["ctx"][:, j], color="steelblue", marker="o", ms=3, label="context")
            ax.plot(t_pred, a["gt"][:, j],  color="green",     marker="o", ms=3, ls="--", label="ground truth")
            ax.plot(t_pred, mu,              color="tomato",    marker="o", ms=3, label="prediction")
            ax.fill_between(t_pred, mu - sig, mu + sig, color="tomato", alpha=0.2)
            ax.axvline(0, color="gray", ls=":", lw=0.8)
            ax.set_title(FEATURE_NAMES[j], fontsize=9)
            ax.set_xlabel("time (min)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_ylim(ylims[j])
            if j == 0:
                ax.legend(fontsize=7)
        for ax in axes[N_FEAT:]:
            ax.set_visible(False)
        plt.tight_layout()
        path = os.path.join(cfg["out_dir"], f"sample_{idx:03d}.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"  Saved {path}", flush=True)

print("\nDone.", flush=True)
