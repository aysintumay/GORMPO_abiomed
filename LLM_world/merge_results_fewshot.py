'''
python LLM_world/merge_results_fewshot.py [--k 3]
Merges per-shard .npz files and prints a comparison table across all k values.
'''
import os
import sys
import glob
import numpy as np

def load_dir(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "shard_*.npz")))
    if not files:
        return None, None, None, 0
    mse, mae, crps = [], [], []
    for f in files:
        d = np.load(f)
        mse.extend(d["mse"]); mae.extend(d["mae"]); crps.extend(d["crps"])
    return np.array(mse), np.array(mae), np.array(crps), len(mse)

# ── zero-shot ──────────────────────────────────────────────────────────────────
zs_mse, zs_mae, zs_crps, zs_n = load_dir("LLM_world/results")

# ── few-shot runs ──────────────────────────────────────────────────────────────
fewshot_dirs = sorted(glob.glob("LLM_world/results_fewshot_k*"))
results = {}
for d in fewshot_dirs:
    k = int(d.split("_k")[-1])
    mse, mae, crps, n = load_dir(d)
    if n > 0:
        results[k] = (mse, mae, crps, n)
        print(f"\n=== k={k} few-shot (n={n}) ===")
        for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
            dd = np.load(f)
            print(f"  {os.path.basename(f)}: n={len(dd['mse'])} "
                  f"MSE={dd['mse'].mean():.4f} MAE={dd['mae'].mean():.4f} CRPS={dd['crps'].mean():.4f}")

# ── comparison table ───────────────────────────────────────────────────────────
print("\n\n=== Summary (normalized space) ===")
print(f"{'Model':<22} {'n':>5}  {'MSE':>20}  {'MAE':>20}  {'CRPS':>20}")
print("-" * 95)
if zs_n:
    print(f"{'Zero-shot LLM':<22} {zs_n:>5}  "
          f"{zs_mse.mean():>8.4f} ±{zs_mse.std():>8.4f}  "
          f"{zs_mae.mean():>8.4f} ±{zs_mae.std():>8.4f}  "
          f"{zs_crps.mean():>8.4f} ±{zs_crps.std():>8.4f}")
for k in sorted(results):
    mse, mae, crps, n = results[k]
    print(f"  {k}-shot LLM{'':<14} {n:>5}  "
          f"{mse.mean():>8.4f} ±{mse.std():>8.4f}  "
          f"{mae.mean():>8.4f} ±{mae.std():>8.4f}  "
          f"{crps.mean():>8.4f} ±{crps.std():>8.4f}")
print(f"{'Transformer':>22} {'3876':>5}  {'0.1073':>20}  {'0.2041':>20}  {'0.1831':>20}")
