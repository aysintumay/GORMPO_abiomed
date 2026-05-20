"""
QLoRA fine-tuning of LLM world model using DigitTokenWorldModel encoding.

Encoding: z-score normalize, then represent each value as individual digit tokens:
  sign (+/-) | integer digits (omitted when int part is 0) | 3 decimal digits
  e.g. +1.230 -> "+ 1 2 3 0",  -0.456 -> "- 4 5 6"
Rows are separated by ' | ', timesteps by newlines.
Response: FORECAST_HORIZON rows in the same format.

Launch single-GPU:
  conda run -n llm_world python LLM_world/finetune_digit.py

Launch multi-GPU (e.g. 7 GPUs):
  conda run -n llm_world accelerate launch --num_processes 7 LLM_world/finetune_digit.py
"""
import subprocess, sys

def _ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("peft")
_ensure("trl")

import os, pickle
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
OUTPUT_DIR = "LLM_world/checkpoints_finetune_digit"
RUN_NAME   = "llm_world_digit_qlora"

# LoRA
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# training
EPOCHS        = 3
BATCH_SIZE    = 1
GRAD_ACCUM    = 8
LR            = 2e-4
MAX_SEQ_LEN   = 2048   # digit tokens are longer; each value ~5 tokens, 12 vals * 6 rows * 2 = ~720 response tokens
WARMUP_RATIO  = 0.03
SAVE_STEPS    = 200
LOG_STEPS     = 20
VAL_SPLIT     = 200

# data
NUM_FEATURES     = 12
FORECAST_HORIZON = 6
COLUMNS          = [i for i in range(13) if i != 11]   # skip raw col 11

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── feature metadata ───────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "PumpPressure", "PumpSpeed", "PumpFlow", "LVP", "LVEDP",
    "SYSTOLIC", "DIASTOLIC", "PULSAT", "PumpCurrent", "Heart Rate",
    "ESE_lv", "Pump Level",
]

# ── digit-token encoding (mirrors DigitTokenWorldModel exactly) ────────────────

def _encode_value(v: float) -> str:
    sign    = '+' if v >= 0 else '-'
    v_abs   = min(abs(float(v)), 999.999)
    int_part = int(v_abs)
    dec_int  = min(round((v_abs - int_part) * 1000), 999)
    dec_str  = f"{dec_int:03d}"
    if int_part == 0:
        tokens = [sign] + list(dec_str)
    else:
        tokens = [sign] + list(str(int_part)) + list(dec_str)
    return ' '.join(tokens)


def _encode_row(row_norm: np.ndarray) -> str:
    return ' | '.join(_encode_value(v) for v in row_norm)


# ── prompts ────────────────────────────────────────────────────────────────────

def build_system_prompt(mean_np: np.ndarray, std_np: np.ndarray) -> str:
    norm_table = "\n".join(
        f"  {j:2d}  {FEATURE_NAMES[j]:<14}  "
        f"mean={mean_np[j]:8.3f}  std={std_np[j]:8.3f}"
        for j in range(NUM_FEATURES)
    )
    col_order = ", ".join(FEATURE_NAMES)
    return (
        "You are a clinical AI assistant expert in Impella LVAD management in the ICU.\n\n"
        "The Impella pump supports the failing heart via P-levels P2–P10. "
        "Higher P-level = more support: higher PumpSpeed/PumpFlow, lower LV preload, lower PULSAT.\n\n"
        "=== VALUE ENCODING ===\n"
        "All values are Z-SCORE NORMALIZED using training-set mean and std.\n"
        "Each normalized value is encoded as INDIVIDUAL DIGIT TOKENS:\n"
        "  sign (+/-), then integer digits (omitted when integer part is 0), "
        "then exactly 3 decimal digits (zero-padded on the right).\n\n"
        "Encoding examples:\n"
        "  +0.123  ->  + 1 2 3\n"
        "  +1.230  ->  + 1 2 3 0\n"
        "  -0.456  ->  - 4 5 6\n"
        "  -1.230  ->  - 1 2 3 0\n\n"
        "Normalization statistics per feature:\n"
        f"{norm_table}\n\n"
        "=== FEATURE ORDER (columns 0–11) ===\n"
        f"{col_order}\n\n"
        "=== CLINICAL TARGETS (physical units) ===\n"
        "PumpPressure ≥ 60 mmHg  |  LVEDP < 18 mmHg  |  PULSAT ≥ 20 mmHg  "
        "|  Heart Rate 50–100 bpm  |  SYSTOLIC 90–130 mmHg\n\n"
        "Weaning goal: reduce Pump Level gradually while hemodynamics stay stable.\n\n"
        "=== TASK ===\n"
        f"You receive {FORECAST_HORIZON} rows of context "
        f"(one row per 10-min timestep, 12 digit-token values separated by ' | ').\n"
        f"Predict the next {FORECAST_HORIZON} timesteps in the SAME format: "
        f"{FORECAST_HORIZON} rows, one per line, 12 digit-token values per row separated by ' | '.\n"
        "Return ONLY the rows. No explanations, no markdown."
    )


def build_user_prompt(ctx_norm: np.ndarray, pl_int: int, pl_norm: float) -> str:
    rows   = "\n".join(_encode_row(row) for row in ctx_norm)
    pl_enc = _encode_value(pl_norm)
    return (
        f"## Context ({len(ctx_norm)} timesteps, oldest first):\n"
        f"{rows}\n\n"
        f"## Action: new Pump Level = P{pl_int} (normalized: {pl_enc})\n\n"
        f"## Predict next {FORECAST_HORIZON} timesteps "
        f"(12 digit-token values per row, separated by ' | '):"
    )


def build_assistant_response(fc_norm: np.ndarray) -> str:
    return "\n".join(_encode_row(row) for row in fc_norm)


# ── dataset ────────────────────────────────────────────────────────────────────

class ForecastDataset(Dataset):
    def __init__(self, samples, tokenizer, system_prompt, max_len=MAX_SEQ_LEN):
        self.items = []
        for ctx_norm, pl_int, pl_norm, fc_norm in samples:
            user_text      = build_user_prompt(ctx_norm, pl_int, pl_norm)
            assistant_text = build_assistant_response(fc_norm)

            messages = [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]

            full_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
                return_tensors="pt",
            )["input_ids"].squeeze(0)

            prompt_ids = tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            )["input_ids"].squeeze(0)

            if full_ids.shape[0] > max_len:
                full_ids = full_ids[:max_len]

            labels = full_ids.clone()
            prompt_len = min(prompt_ids.shape[0], full_ids.shape[0])
            labels[:prompt_len] = -100

            self.items.append({"input_ids": full_ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ── data loading & normalization ───────────────────────────────────────────────

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

mean_np = mean_t[COLUMNS].numpy().astype(np.float64)
std_np  = std_t[COLUMNS].numpy().astype(np.float64)

# mean/std for Pump Level normalization (column index 12 in raw = last COLUMNS entry)
pl_mean = float(mean_t[-1].item())
pl_std  = float(std_t[-1].item())

system_prompt = build_system_prompt(mean_np.astype(np.float32), std_np.astype(np.float32))


def normalize(x_phys: np.ndarray) -> np.ndarray:
    """x_phys: (..., 12) physical. Returns z-score normalized float32."""
    return ((x_phys.astype(np.float64) - mean_np) / (std_np + 1e-8)).astype(np.float32)


def normalize_pl(pl_int: int) -> float:
    return (float(pl_int) - pl_mean) / (pl_std + 1e-8)


def build_samples(raw_split: np.ndarray) -> list:
    """raw_split: (N, T_total, 13). Returns list of (ctx_norm, pl_int, pl_norm, fc_norm)."""
    T = FORECAST_HORIZON
    samples = []
    for i in range(len(raw_split)):
        episode  = raw_split[i]                         # (T_total, 13)
        ctx_phys = episode[:T, COLUMNS].astype(np.float32)   # (T, 12)
        fc_phys  = episode[T:T*2, COLUMNS].astype(np.float32) # (T, 12)
        pl_col   = episode[T:T*2, -1]                    # (T,)

        pl_int  = int(round(float(pl_col.mean())))
        pl_norm = normalize_pl(pl_int)

        ctx_norm = normalize(ctx_phys)
        fc_norm  = normalize(fc_phys)

        samples.append((ctx_norm, pl_int, pl_norm, fc_norm))
    return samples


print("Building training samples ...")
all_samples   = build_samples(train_raw)
val_samples   = all_samples[-VAL_SPLIT:]
train_samples = all_samples[:-VAL_SPLIT]
print(f"  train: {len(train_samples)}  val: {len(val_samples)}")

# ── tokenizer + model ──────────────────────────────────────────────────────────

print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Building tokenized datasets ...")
train_dataset = ForecastDataset(train_samples, tokenizer, system_prompt)
val_dataset   = ForecastDataset(val_samples,   tokenizer, system_prompt)
example_len = train_dataset[0]['input_ids'].shape[0]
print(f"  example seq len: {example_len}")

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
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
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
