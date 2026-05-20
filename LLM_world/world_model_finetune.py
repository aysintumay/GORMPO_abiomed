"""
Fine-tuned LLM world model — loads base model + QLoRA adapter from disk.

Usage (drop-in replacement for LLMWorldModel):
    from LLM_world.world_model_finetune import FinetunedLLMWorldModel
    model = FinetunedLLMWorldModel(model_id=..., adapter_path=..., device=...)
    model.load_data(DATA_PATH)
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from LLM_world.world_model import LLMWorldModel


class FinetunedLLMWorldModel(LLMWorldModel):
    """
    Extends LLMWorldModel by loading a QLoRA adapter on top of the base model.
    Everything else (tokenization, prompts, predict API) is inherited unchanged.
    """

    def __init__(
        self,
        model_id: str = "m42-health/Llama3-Med42-8B",
        adapter_path: str = "LLM_world/checkpoints_finetune/final",
        device: str = "auto",
        **kwargs,
    ):
        # Bypass parent __init__ model loading — we do it ourselves
        import os, sys
        import numpy as np

        self.num_features     = kwargs.get("num_features", 12)
        self.forecast_horizon = kwargs.get("forecast_horizon", 6)
        self.num_samples      = kwargs.get("num_samples", 10)
        self.temperature      = kwargs.get("temperature", 0.8)
        self.num_bins         = 1000

        self.mean     = None
        self.std      = None
        self.bin_low  = None
        self.bin_high = None
        self.data_train = self.data_val = self.data_test = None

        print(f"Loading tokenizer: {model_id}")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        print(f"Loading base model in 4-bit: {model_id}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )

        print(f"Loading LoRA adapter: {adapter_path}")
        self.llm = PeftModel.from_pretrained(base, adapter_path)
        self.llm.eval()
        print("Finetuned model ready.")
