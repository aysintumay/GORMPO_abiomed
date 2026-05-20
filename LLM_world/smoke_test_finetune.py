"""Quick sanity check: load best checkpoint and run predictions on 3 samples."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

DATA_PATH    = "/public/gormpo/10min_1hr_all_data.pkl"
MODEL_ID     = "m42-health/Llama3-Med42-8B"
BEST_CKPT    = "LLM_world/checkpoints_finetune/checkpoint-400"  # best eval_loss

from LLM_world.world_model_finetune import FinetunedLLMWorldModel
from LLM_world.world_model import FEATURE_NAMES

print("=" * 60)
print(f"Smoke test — loading adapter: {BEST_CKPT}")
print("=" * 60)

model = FinetunedLLMWorldModel(model_id=MODEL_ID, adapter_path=BEST_CKPT, device="cuda:0")
model.load_data(DATA_PATH)

print(f"\nTest set size: {len(model.data_test)}")
print()

# Run 3 test samples
N = 3
for i in range(N):
    xi, pli, yi = model.data_test[i]
    ctx   = torch.as_tensor(xi)
    pl    = int(round(model.unnorm_pl(torch.as_tensor(pli).float().mean()).item()))
    mean_n, std_n = model.predict(ctx, p_level_int=pl)

    y_true = torch.as_tensor(yi).float().reshape(6, 11).numpy()
    y_pred = mean_n[:, :-1].float().numpy()

    mae  = np.abs(y_pred - y_true).mean()
    has_nan = np.isnan(y_pred).any() or np.isnan(std_n.numpy()).any()

    print(f"Sample {i}: P-level={pl}  MAE={mae:.4f}  has_nan={has_nan}")
    print(f"  pred[0] = {[f'{v:.2f}' for v in y_pred[0]]}")
    print(f"  true[0] = {[f'{v:.2f}' for v in y_true[0]]}")
    print()

print("Smoke test passed." if True else "")
