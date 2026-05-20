"""
QLoRA fine-tuning of the LLM world model on the Impella forecasting task.

Launch single-GPU:
  conda run -n llm_world python LLM_world/finetune_llm.py

Launch multi-GPU (e.g. 7 GPUs):
  conda run -n llm_world accelerate launch --num_processes 7 LLM_world/finetune_llm.py

The script installs peft automatically if missing.
"""
import subprocess, sys

def _ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("peft")
_ensure("trl")

import os, json, pickle, re
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ── config ─────────────────────────────────────────────────────────────────────

MODEL_ID   = "m42-health/Llama3-Med42-8B"
DATA_PATH  = "/public/gormpo/10min_1hr_all_data.pkl"
OUTPUT_DIR = "LLM_world/checkpoints_finetune"
RUN_NAME   = "llm_world_qlora"

# LoRA
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]

# training
EPOCHS         = 3
BATCH_SIZE     = 1          # per GPU; 1 needed at seq_len=640 on 24GB
GRAD_ACCUM     = 8          # effective batch = BATCH_SIZE * GRAD_ACCUM * n_gpus
LR             = 2e-4
MAX_SEQ_LEN    = 1200       # tokens; prompt=880, response=225, total=1105 → 1200 with headroom
WARMUP_RATIO   = 0.03
SAVE_STEPS     = 200
LOG_STEPS      = 20
VAL_SPLIT      = 200        # number of training samples held out for eval

# data tokenization
NUM_FEATURES      = 12
FORECAST_HORIZON  = 6
NUM_BINS          = 1000
COLUMNS           = [i for i in range(13) if i != 11]  # skip raw col 11

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "PumpPressure", "PumpSpeed", "PumpFlow", "LVP", "LVEDP",
    "SYSTOLIC", "DIASTOLIC", "PULSAT", "PumpCurrent", "Heart Rate",
    "ESE_lv", "Pump Level",
]


def build_bin_boundaries(train_raw: np.ndarray, columns: list):
    """1st/99th percentile bin edges from raw training data."""
    phys = train_raw[:, :, columns].reshape(-1, len(columns)).astype(np.float32)
    return np.percentile(phys, 1, axis=0), np.percentile(phys, 99, axis=0)


def to_bins(x_phys: np.ndarray, bin_low, bin_high) -> np.ndarray:
    ratio = (x_phys - bin_low) / (bin_high - bin_low + 1e-8)
    return np.clip(np.floor(ratio * NUM_BINS).astype(int), 0, NUM_BINS - 1)


def build_system_prompt(bin_low, bin_high) -> str:
    col_order = ", ".join(FEATURE_NAMES)
    bin_table = "\n".join(
        f"  {j:2d}  {FEATURE_NAMES[j]:<14}  "
        f"bin 0 ≈ {bin_low[j]:8.2f}  →  bin 999 ≈ {bin_high[j]:8.2f}"
        for j in range(NUM_FEATURES)
    )
    return (
        "You are a clinical AI assistant expert in Impella LVAD management in the ICU.\n\n"
        "The Impella pump supports the failing heart via P-levels P2–P10. "
        "Higher P-level = more support: higher PumpSpeed/PumpFlow, lower LV preload, lower PULSAT.\n\n"
        "=== VALUE ENCODING ===\n"
        "All hemodynamic values are encoded as INTEGER BIN INDICES 0–999.\n"
        "Each feature is mapped independently: bin 0 = lowest observed, bin 999 = highest observed.\n"
        "Physical ranges per feature (for clinical reasoning):\n"
        f"{bin_table}\n\n"
        "=== FEATURE ORDER (columns 0–11) ===\n"
        f"{col_order}\n\n"
        "=== CLINICAL TARGETS (physical units) ===\n"
        "PumpPressure ≥ 60 mmHg  |  LVEDP < 18 mmHg  |  PULSAT ≥ 20 mmHg  "
        "|  Heart Rate 50–100 bpm  |  SYSTOLIC 90–130 mmHg\n\n"
        "Weaning goal: reduce Pump Level gradually while hemodynamics stay stable.\n\n"
        "=== TASK ===\n"
        f"You receive {FORECAST_HORIZON} rows of context (one row = one 10-min timestep, "
        f"12 bin indices in the column order above), then a new Pump Level action.\n"
        f"Predict the next {FORECAST_HORIZON} timesteps in the same format: "
        f"a JSON array of {FORECAST_HORIZON} rows, each row an array of "
        f"12 bin indices (integers 0–999).\n"
        "Return ONLY the JSON array. No explanations, no markdown."
    )


def build_user_prompt(ctx_bins: np.ndarray, pl_int: int, pl_bin: int) -> str:
    rows = "\n".join(" ".join(str(v) for v in row) for row in ctx_bins)
    return (
        f"## Context ({len(ctx_bins)} timesteps, oldest first):\n"
        f"{rows}\n\n"
        f"## Action: new Pump Level = P{pl_int} (bin {pl_bin})\n\n"
        f"## Predict next {FORECAST_HORIZON} timesteps as a JSON array of arrays:\n"
    )


def build_assistant_response(fc_bins: np.ndarray) -> str:
    inner = ",\n".join(
        "  [" + ", ".join(str(v) for v in row) + "]"
        for row in fc_bins
    )
    return "[\n" + inner + "\n]"


# ── dataset ────────────────────────────────────────────────────────────────────

class ForecastDataset(Dataset):
    """
    Each item: tokenized (input_ids, labels) where labels have -100 on prompt tokens
    so loss is computed only on the assistant response.
    """

    def __init__(self, samples, tokenizer, system_prompt, max_len=MAX_SEQ_LEN):
        self.tokenizer     = tokenizer
        self.system_prompt = system_prompt
        self.max_len       = max_len
        self.items         = []

        for ctx_bins, pl_int, pl_bin, fc_bins in samples:
            user_text      = build_user_prompt(ctx_bins, pl_int, pl_bin)
            assistant_text = build_assistant_response(fc_bins)

            messages = [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]

            # full sequence (prompt + response)
            full_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
                return_tensors="pt",
            )["input_ids"].squeeze(0)

            # prompt-only (to find where response starts)
            prompt_ids = tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            )["input_ids"].squeeze(0)

            if full_ids.shape[0] > max_len:
                full_ids = full_ids[:max_len]

            labels = full_ids.clone()
            prompt_len = min(prompt_ids.shape[0], full_ids.shape[0])
            labels[:prompt_len] = -100   # mask prompt tokens from loss

            self.items.append({"input_ids": full_ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ── data loading ───────────────────────────────────────────────────────────────

print("Loading data ...")
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

mean_t = data["mean"]
std_t  = data["std"]
if not isinstance(mean_t, torch.Tensor):
    mean_t = torch.tensor(mean_t, dtype=torch.float64)
    std_t  = torch.tensor(std_t,  dtype=torch.float64)

train_raw = data["train"]
if isinstance(train_raw, torch.Tensor):
    train_raw = train_raw.numpy()

bin_low, bin_high = build_bin_boundaries(train_raw, COLUMNS)
system_prompt = build_system_prompt(bin_low, bin_high)

mean_np = mean_t[COLUMNS].numpy().astype(np.float32)
std_np  = std_t[COLUMNS].numpy().astype(np.float32)

# P-level bin helper (using only the last column = Pump Level)
pl_low  = bin_low[-1]
pl_high = bin_high[-1]

def pl_int_to_bin(pl_int: int) -> int:
    ratio = (float(pl_int) - pl_low) / (pl_high - pl_low + 1e-8)
    return int(np.clip(np.floor(ratio * NUM_BINS), 0, NUM_BINS - 1))


def build_samples_from_split(raw_split: np.ndarray) -> list:
    """
    raw_split : (N, T_total, 13) raw physical values.
    Returns list of (ctx_bins, pl_int, pl_bin, fc_bins).
    """
    T = FORECAST_HORIZON
    samples = []
    for i in range(len(raw_split)):
        episode = raw_split[i]              # (T_total, 13)
        ctx_phys = episode[:T, COLUMNS]    # (T, 12) physical context
        fc_raw   = episode[T:T*2, COLUMNS] # (T, 12) physical forecast
        pl_col   = episode[T:T*2, -1]      # (T,) P-level during forecast

        pl_int = int(round(float(pl_col.mean())))
        pl_bin = pl_int_to_bin(pl_int)
        ctx_bins = to_bins(ctx_phys.astype(np.float32), bin_low, bin_high)
        fc_bins  = to_bins(fc_raw.astype(np.float32),  bin_low, bin_high)

        samples.append((ctx_bins, pl_int, pl_bin, fc_bins))
    return samples


print("Building training samples ...")
all_samples = build_samples_from_split(train_raw)
val_samples   = all_samples[-VAL_SPLIT:]
train_samples = all_samples[:-VAL_SPLIT]
print(f"  train: {len(train_samples)}  val: {len(val_samples)}")

# ── tokenizer + model ──────────────────────────────────────────────────────────

print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Building tokenized dataset ...")
train_dataset = ForecastDataset(train_samples, tokenizer, system_prompt)
val_dataset   = ForecastDataset(val_samples,   tokenizer, system_prompt)
print(f"  example seq len: {train_dataset[0]['input_ids'].shape[0]}")

print("Loading model in 4-bit ...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
_local_rank = int(os.environ.get("LOCAL_RANK", 0))
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": _local_rank},
    dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGETS,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ── training ───────────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    run_name=RUN_NAME,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    bf16=True,
    gradient_checkpointing=True,       # required at seq_len=1200 to avoid OOM
    optim="paged_adamw_8bit",          # 8-bit optimizer saves ~1.5 GB vs adamw
    logging_steps=LOG_STEPS,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, padding=True,
    pad_to_multiple_of=8, label_pad_token_id=-100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
)

print("Starting training ...")
trainer.train()

print(f"Saving adapter to {OUTPUT_DIR}/final ...")
model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
print("Done.")
