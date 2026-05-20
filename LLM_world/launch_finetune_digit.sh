#!/bin/bash
# Fine-tune LLM world model with DigitToken encoding on all available GPUs.
#
# Single-GPU:
#   bash LLM_world/launch_finetune_digit.sh 1
#
# Multi-GPU (default 7):
#   bash LLM_world/launch_finetune_digit.sh 7

NUM_GPUS=${1:-7}
mkdir -p LLM_world/checkpoints_finetune_digit

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Single-GPU training ..."
    conda run -n llm_world python LLM_world/finetune_digit.py 2>&1 | tee LLM_world/finetune_digit.log
else
    echo "Multi-GPU training on $NUM_GPUS GPUs ..."
    conda run -n llm_world accelerate launch --num_processes "$NUM_GPUS" \
        LLM_world/finetune_digit.py 2>&1 | tee LLM_world/finetune_digit.log
fi
