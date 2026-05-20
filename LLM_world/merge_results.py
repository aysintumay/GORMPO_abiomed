'''
python LLM_world/merge_results.py
Merges per-shard .npz files into final metrics.
'''
import os
import glob
import numpy as np

RESULTS_DIR = "LLM_world/results"

files = sorted(glob.glob(os.path.join(RESULTS_DIR, "shard_*.npz")))
if not files:
    print("No shard files found in", RESULTS_DIR)
    exit(1)

mse_all, mae_all, crps_all = [], [], []
for f in files:
    d = np.load(f)
    mse_all.extend(d["mse"])
    mae_all.extend(d["mae"])
    crps_all.extend(d["crps"])
    print(f"{os.path.basename(f)}: n={len(d['mse'])} "
          f"MSE={d['mse'].mean():.4f} MAE={d['mae'].mean():.4f} CRPS={d['crps'].mean():.4f}")

print(f"\n=== Final metrics (n={len(mse_all)}, normalized space) ===")
print(f"MSE:  {np.mean(mse_all):.6f}  ±{np.std(mse_all):.6f}")
print(f"MAE:  {np.mean(mae_all):.6f}  ±{np.std(mae_all):.6f}")
print(f"CRPS: {np.mean(crps_all):.6f}  ±{np.std(crps_all):.6f}")
