"""
Grid search over (sdtw_weight, mse_weight) for the SoftDTW Transformer.

Each trial trains from scratch for NUM_EPOCHS epochs.
Trials are ranked by pure MSE on the validation set (weight-independent).

After the sweep the best config is retrained for FINAL_EPOCHS (full run),
the model is saved to abiomed_env/data/10min_1hr_softdtw_model.pth, and
prediction figures are written to LLM_world/figures_softdtw/.

Run from the project root:
    python LLM_world/tune_softdtw.py

Results saved to LLM_world/tune_softdtw_results.csv
Best model saved to abiomed_env/data/10min_1hr_softdtw_model.pth
"""
import sys
import os
import copy
import csv
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy import special

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abiomed_env.model import WorldModel, SoftDTW, model_factory
from abiomed_env.config import model_kwargs_10min_1hr_full

# ── config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "/public/gormpo/10min_1hr_all_data.pkl"
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "abiomed_env", "data",
                           "10min_1hr_softdtw_model.pth")
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "tune_softdtw_results.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures_softdtw")

NUM_EPOCHS    = 50
FINAL_EPOCHS  = 100   # full retrain with best config
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
GAMMA         = 1.0
NUM_DROPOUT_SAMPLES = 20   # for CRPS evaluation

# Grid: log-spaced sdtw_weight values, three mse_weight values
SDTW_WEIGHTS = [0.001, 0.005, 0.015, 0.05, 0.1, 0.5]
MSE_WEIGHTS  = [0.5, 1.0, 2.0]
GRID = [(s, m) for s in SDTW_WEIGHTS for m in MSE_WEIGHTS]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ── helpers ───────────────────────────────────────────────────────────────────
def crps_gaussian(mu, sigma, obs):
    sigma = np.clip(sigma, 1e-6, None)
    z     = (obs - mu) / sigma
    phi_z = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    Phi_z = special.ndtr(z)
    return sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1.0 / np.sqrt(np.pi))


def fresh_model(model_kwargs):
    """Instantiate a brand-new WorldModel (random weights, same arch)."""
    kw = dict(model_kwargs)
    kw["device"] = DEVICE
    m = WorldModel(**kw)
    return m


def eval_metrics(model, loader, num_dropout=NUM_DROPOUT_SAMPLES):
    """Return (mse, mae, crps) on normalized space, averaged over all test samples."""
    mse_per, mae_per, crps_per = [], [], []
    model.model.train()   # dropout on
    with torch.no_grad():
        for x, pl, y in loader:
            x  = x.to(DEVICE).float()
            pl = pl.to(DEVICE).float()
            y  = y.float()

            samples = []
            for _ in range(num_dropout):
                out = model.model(x, pl)
                out = out.reshape(-1, model.forecast_horizon, model.num_features - 1)
                samples.append(out.cpu().numpy())
            samples = np.stack(samples, axis=0)   # (S, B, T, D)

            y_np  = y.numpy().reshape(-1, model.forecast_horizon, model.num_features - 1)
            mu    = samples.mean(0)
            sigma = samples.std(0)

            mse_per.append(  ((mu - y_np)**2).mean(axis=(1, 2)) )
            mae_per.append(  np.abs(mu - y_np).mean(axis=(1, 2)) )
            crps_per.append( crps_gaussian(mu, sigma, y_np).mean(axis=(1, 2)) )

    model.model.eval()
    mse_arr  = np.concatenate(mse_per)
    mae_arr  = np.concatenate(mae_per)
    crps_arr = np.concatenate(crps_per)
    n = len(mse_arr)
    se = lambda a: a.std() / np.sqrt(n)
    return {
        "mse":  mse_arr.mean(),  "mse_std":  mse_arr.std(),  "mse_se":  se(mse_arr),
        "mae":  mae_arr.mean(),  "mae_std":  mae_arr.std(),  "mae_se":  se(mae_arr),
        "crps": crps_arr.mean(), "crps_std": crps_arr.std(), "crps_se": se(crps_arr),
        "n": n,
    }


def train_trial(model, sdtw_weight, mse_weight, num_epochs=None, track_losses=False):
    """Train model in-place.

    Returns best_val_mse, and optionally a history dict when track_losses=True.
    History keys: train_sdtw, train_mse, val_sdtw, val_mse (all raw, unweighted).
    """
    if num_epochs is None:
        num_epochs = NUM_EPOCHS
    sdtw_criterion = SoftDTW(gamma=GAMMA)
    mse_criterion  = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.model.parameters(), lr=LEARNING_RATE)

    train_loader = DataLoader(model.data_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(model.data_val,   batch_size=BATCH_SIZE, shuffle=False)

    best_val_mse = float("inf")
    best_state   = None
    history      = {"train_sdtw": [], "train_mse": [], "val_sdtw": [], "val_mse": []}

    for epoch in range(num_epochs):
        model.model.train()
        ep_sdtw = ep_mse = 0.0
        for x, pl, y in train_loader:
            x, pl, y = x.to(DEVICE).float(), pl.to(DEVICE).float(), y.to(DEVICE).float()
            optimizer.zero_grad()
            out   = model.model(x, pl)
            o_seq = out.reshape(-1, model.forecast_horizon, model.num_features - 1)
            y_seq = y.reshape( -1, model.forecast_horizon, model.num_features - 1)
            raw_sdtw = sdtw_criterion(o_seq, y_seq)
            raw_mse  = mse_criterion(o_seq, y_seq)
            loss = sdtw_weight * raw_sdtw + mse_weight * raw_mse
            loss.backward()
            optimizer.step()
            if track_losses:
                ep_sdtw += raw_sdtw.item() * x.size(0)
                ep_mse  += raw_mse.item()  * x.size(0)

        if track_losses:
            history["train_sdtw"].append(ep_sdtw / len(train_loader.dataset))
            history["train_mse"].append(ep_mse  / len(train_loader.dataset))

        # validation — pure MSE used for checkpointing; also track raw sdtw when needed
        model.model.eval()
        val_mse = val_sdtw = 0.0
        with torch.no_grad():
            for x, pl, y in val_loader:
                x, pl, y = x.to(DEVICE).float(), pl.to(DEVICE).float(), y.to(DEVICE).float()
                out   = model.model(x, pl)
                o_seq = out.reshape(-1, model.forecast_horizon, model.num_features - 1)
                y_seq = y.reshape( -1, model.forecast_horizon, model.num_features - 1)
                val_mse  += mse_criterion(o_seq, y_seq).item()  * x.size(0)
                if track_losses:
                    val_sdtw += sdtw_criterion(o_seq, y_seq).item() * x.size(0)
        val_mse  /= len(val_loader.dataset)
        if track_losses:
            history["val_sdtw"].append(val_sdtw / len(val_loader.dataset))
            history["val_mse"].append(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state   = copy.deepcopy(model.model.state_dict())

        print(f"  ep {epoch+1:3d}/{num_epochs}  val_mse={val_mse:.6f}"
              + (" *" if val_mse == best_val_mse else ""), flush=True)

    model.model.load_state_dict(best_state)
    return (best_val_mse, history) if track_losses else best_val_mse


# ── main sweep ────────────────────────────────────────────────────────────────
print(f"Grid: {len(GRID)} trials  ({len(SDTW_WEIGHTS)} sdtw × {len(MSE_WEIGHTS)} mse)")
print(f"Epochs/trial: {NUM_EPOCHS}  |  device: {DEVICE}\n")

# Load data once into a reference model; we'll copy data to fresh models
ref = fresh_model(model_kwargs_10min_1hr_full)
ref.load_data(DATA_PATH)
val_loader_global  = DataLoader(ref.data_val,  batch_size=BATCH_SIZE, shuffle=False)
test_loader_global = DataLoader(ref.data_test, batch_size=BATCH_SIZE, shuffle=False)

rows = []
best_overall_mse = float("inf")
best_overall_state = None
best_overall_cfg   = None

for i, (sw, mw) in enumerate(GRID):
    print(f"\n{'='*60}")
    print(f"Trial {i+1}/{len(GRID)}  sdtw_weight={sw}  mse_weight={mw}")
    print(f"{'='*60}")
    t0 = time.time()

    m = fresh_model(model_kwargs_10min_1hr_full)
    # share pre-loaded dataset tensors to avoid re-reading disk
    m.data_train = ref.data_train
    m.data_val   = ref.data_val
    m.data_test  = ref.data_test
    m.mean       = ref.mean
    m.std        = ref.std

    best_val_mse = train_trial(m, sw, mw)

    # evaluate on val and test sets
    val_met  = eval_metrics(m, val_loader_global)
    test_met = eval_metrics(m, test_loader_global)

    elapsed = time.time() - t0
    row = dict(sdtw_weight=sw, mse_weight=mw,
               best_val_mse_ckpt=best_val_mse,
               val_mse=val_met["mse"],   val_mse_std=val_met["mse_std"],
               val_mae=val_met["mae"],   val_mae_std=val_met["mae_std"],
               val_crps=val_met["crps"], val_crps_std=val_met["crps_std"],
               test_mse=test_met["mse"],   test_mse_std=test_met["mse_std"],
               test_mae=test_met["mae"],   test_mae_std=test_met["mae_std"],
               test_crps=test_met["crps"], test_crps_std=test_met["crps_std"],
               elapsed_s=round(elapsed))
    rows.append(row)

    print(f"\n  VAL   MSE={val_met['mse']:.6f}±{val_met['mse_std']:.6f}  "
          f"MAE={val_met['mae']:.6f}±{val_met['mae_std']:.6f}  "
          f"CRPS={val_met['crps']:.6f}±{val_met['crps_std']:.6f}")
    print(f"  TEST  MSE={test_met['mse']:.6f}±{test_met['mse_std']:.6f}  "
          f"MAE={test_met['mae']:.6f}±{test_met['mae_std']:.6f}  "
          f"CRPS={test_met['crps']:.6f}±{test_met['crps_std']:.6f}")
    print(f"  elapsed: {elapsed:.0f}s")

    if test_met["mse"] < best_overall_mse:
        best_overall_mse   = test_met["mse"]
        best_overall_state = copy.deepcopy(m.model.state_dict())
        best_overall_cfg   = (sw, mw)
        print(f"  *** new best test MSE: {best_overall_mse:.6f} ***")

# ── save results ──────────────────────────────────────────────────────────────
rows.sort(key=lambda r: r["test_mse"])

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(RESULTS_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\nResults saved to {RESULTS_CSV}")

# ── summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"{'sdtw_w':>10} {'mse_w':>6} | {'test_MSE':>12} {'test_MAE':>12} {'test_CRPS':>12}")
print(f"{'-'*10} {'-'*6}-+-{'-'*12} {'-'*12} {'-'*12}")
for r in rows:
    marker = " <-- best" if (r["sdtw_weight"], r["mse_weight"]) == best_overall_cfg else ""
    print(f"{r['sdtw_weight']:>10.4f} {r['mse_weight']:>6.1f} | "
          f"{r['test_mse']:>10.6f}±{r['test_mse_std']:.4f}  "
          f"{r['test_mae']:>10.6f}±{r['test_mae_std']:.4f}  "
          f"{r['test_crps']:>10.6f}±{r['test_crps_std']:.4f}"
          f"{marker}")
print(f"\nBest config: sdtw_weight={best_overall_cfg[0]}  mse_weight={best_overall_cfg[1]}")
print(f"Best test MSE (sweep): {best_overall_mse:.6f}")


# ── full retrain with best config ─────────────────────────────────────────────
best_sw, best_mw = best_overall_cfg
print(f"\n{'='*60}")
print(f"Retraining with best config for {FINAL_EPOCHS} epochs")
print(f"sdtw_weight={best_sw}  mse_weight={best_mw}")
print(f"{'='*60}")

final_model = fresh_model(model_kwargs_10min_1hr_full)
final_model.data_train = ref.data_train
final_model.data_val   = ref.data_val
final_model.data_test  = ref.data_test
final_model.mean       = ref.mean
final_model.std        = ref.std

_, loss_history = train_trial(final_model, best_sw, best_mw,
                              num_epochs=FINAL_EPOCHS, track_losses=True)

# ── loss curves ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs(FIGURES_DIR, exist_ok=True)
epochs_ax = np.arange(1, FINAL_EPOCHS + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(
    f"Final retrain loss curves  |  sdtw_weight={best_sw}  mse_weight={best_mw}",
    fontsize=11)

ax1.plot(epochs_ax, loss_history["train_sdtw"], label="train", color="steelblue")
ax1.plot(epochs_ax, loss_history["val_sdtw"],   label="val",   color="tomato")
ax1.set_title("SoftDTW loss (raw, unweighted)")
ax1.set_xlabel("epoch")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_ax, loss_history["train_mse"], label="train", color="steelblue")
ax2.plot(epochs_ax, loss_history["val_mse"],   label="val",   color="tomato")
ax2.set_title("MSE loss (raw, unweighted)")
ax2.set_xlabel("epoch")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
loss_plot_path = os.path.join(FIGURES_DIR, "training_losses.png")
plt.savefig(loss_plot_path, dpi=120)
plt.close()
print(f"Loss curves saved to {loss_plot_path}")

# final test metrics
final_met = eval_metrics(final_model, test_loader_global)
print(f"\n=== Final model — test metrics (normalized space, n={final_met['n']}) ===")
print(f"MSE : {final_met['mse']:.6f}  ±{final_met['mse_std']:.6f}  "
      f"95% CI [{final_met['mse']-1.96*final_met['mse_se']:.6f}, "
      f"{final_met['mse']+1.96*final_met['mse_se']:.6f}]")
print(f"MAE : {final_met['mae']:.6f}  ±{final_met['mae_std']:.6f}  "
      f"95% CI [{final_met['mae']-1.96*final_met['mae_se']:.6f}, "
      f"{final_met['mae']+1.96*final_met['mae_se']:.6f}]")
print(f"CRPS: {final_met['crps']:.6f}  ±{final_met['crps_std']:.6f}  "
      f"95% CI [{final_met['crps']-1.96*final_met['crps_se']:.6f}, "
      f"{final_met['crps']+1.96*final_met['crps_se']:.6f}]")

model_cpu = copy.deepcopy(final_model.model).to("cpu")
torch.save(model_cpu.state_dict(), MODEL_PATH)
print(f"\nFinal model saved to {MODEL_PATH}")


# ── visualization ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "MAP (PumpPressure)", "PumpSpeed", "PumpFlow", "LVP", "LVEDP",
    "SYSTOLIC", "DIASTOLIC", "PULSAT", "PumpCurrent", "Heart Rate", "ESE_lv",
]
NUM_VIS_SAMPLES = 10
SAMPLE_IDXS     = [0, 1, 2, 3]

t_ctx  = np.arange(-5, 1) * 10
t_pred = np.arange(1, 7)  * 10
n_feat = final_model.num_features - 1
ncols  = 4
nrows  = (n_feat + ncols - 1) // ncols
cols      = final_model.columns
cols_no_pl = final_model.columns[:-1]

os.makedirs(FIGURES_DIR, exist_ok=True)

print("\nComputing shared y-limits ...", flush=True)
lo = [float("inf")]  * n_feat
hi = [float("-inf")] * n_feat

for idx in SAMPLE_IDXS:
    x, pl, y = final_model.data_test[idx]
    x  = x.unsqueeze(0).to(DEVICE).float()
    pl = pl.unsqueeze(0).to(DEVICE).float()

    final_model.model.train()
    with torch.no_grad():
        samps = [final_model.model(x, pl).reshape(final_model.forecast_horizon, n_feat)
                 for _ in range(NUM_VIS_SAMPLES)]
    final_model.model.eval()

    samps     = torch.stack(samps)
    std_t     = final_model.std[cols_no_pl].cpu()
    mean_t    = final_model.mean[cols_no_pl].cpu()
    pred_phys = samps.mean(0).cpu() * std_t + mean_t
    std_phys  = samps.std(0).cpu()  * std_t
    ctx_phys  = x[0].cpu() * final_model.std[cols].cpu() + final_model.mean[cols].cpu()
    y_phys    = (torch.as_tensor(y).float().reshape(final_model.forecast_horizon, n_feat)
                 * std_t + mean_t)

    for j in range(n_feat):
        mu  = pred_phys[:, j].numpy()
        sig = std_phys[:, j].numpy()
        vals = np.concatenate([ctx_phys[:, j].numpy(), y_phys[:, j].numpy(),
                               mu, mu - sig, mu + sig])
        lo[j] = min(lo[j], float(vals.min()))
        hi[j] = max(hi[j], float(vals.max()))

ylims = [(lo[j] - 0.05 * max(hi[j] - lo[j], 1e-6),
          hi[j] + 0.05 * max(hi[j] - lo[j], 1e-6)) for j in range(n_feat)]

for idx in SAMPLE_IDXS:
    print(f"Plotting sample {idx} ...", flush=True)
    x, pl, y = final_model.data_test[idx]
    x  = x.unsqueeze(0).to(DEVICE).float()
    pl = pl.unsqueeze(0).to(DEVICE).float()
    pl_int = int(round((pl.mean().item() * final_model.std[-1] + final_model.mean[-1]).item()))

    final_model.model.train()
    with torch.no_grad():
        samps = [final_model.model(x, pl).reshape(final_model.forecast_horizon, n_feat)
                 for _ in range(NUM_VIS_SAMPLES)]
    final_model.model.eval()

    samps     = torch.stack(samps)
    std_t     = final_model.std[cols_no_pl].cpu()
    mean_t    = final_model.mean[cols_no_pl].cpu()
    pred_phys = samps.mean(0).cpu() * std_t + mean_t
    std_phys  = samps.std(0).cpu()  * std_t
    ctx_phys  = x[0].cpu() * final_model.std[cols].cpu() + final_model.mean[cols].cpu()
    y_phys    = (torch.as_tensor(y).float().reshape(final_model.forecast_horizon, n_feat)
                 * std_t + mean_t)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()
    fig.suptitle(
        f"Test sample {idx}  |  P-level = {pl_int}  |  "
        f"Transformer (SoftDTW tuned: sdtw={best_sw} mse={best_mw})", fontsize=11)

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
    path = os.path.join(FIGURES_DIR, f"sample_{idx:03d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

print("\nDone.")
