"""
Train the Transformer world model with a combined SoftDTW + MSE objective.

Run from the project root:
    python LLM_world/train_softdtw.py

Saves the best checkpoint to:
    abiomed_env/data/10min_1hr_softdtw_model.pth
"""
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abiomed_env.model import WorldModel
from abiomed_env.config import model_kwargs_10min_1hr_full

DATA_PATH  = "/public/gormpo/10min_1hr_all_data.pkl"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "abiomed_env", "data",
                          "10min_1hr_softdtw_model.pth")

# ── hyperparameters ───────────────────────────────────────────────────────────
NUM_EPOCHS    = 100
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
GAMMA         = 1.0

# Scale sdtw_weight so both terms are on the same magnitude:
#   SoftDTW accumulates ~(forecast_horizon * num_output_features) local costs
#   MSELoss averages over those same elements → sdtw_weight = 1 / (T * D)
_T = model_kwargs_10min_1hr_full["forecast_horizon"]
_D = model_kwargs_10min_1hr_full["num_features"] - 1
SDTW_WEIGHT = 1.0 / (_T * _D)   # ≈ 0.0152 for T=6, D=11
MSE_WEIGHT  = 1.0

# ── build model ───────────────────────────────────────────────────────────────
model_kwargs = dict(model_kwargs_10min_1hr_full)
model_kwargs["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

model = WorldModel(**model_kwargs)
model.load_data(DATA_PATH)

print(f"SoftDTW weight : {SDTW_WEIGHT:.4f}")
print(f"MSE weight     : {MSE_WEIGHT:.4f}")
print(f"gamma          : {GAMMA}")
print(f"epochs         : {NUM_EPOCHS}  |  batch: {BATCH_SIZE}  |  lr: {LEARNING_RATE}")

# ── train ─────────────────────────────────────────────────────────────────────
best_state = model.train_model_softdtw(
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    sdtw_weight=SDTW_WEIGHT,
    mse_weight=MSE_WEIGHT,
)

# ── save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save_model(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
