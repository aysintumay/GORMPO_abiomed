#!/bin/bash
# Launches 7 parallel inference shards on GPUs 0-6.
mkdir -p LLM_world/logs_finetune_digit LLM_world/results_finetune_digit LLM_world/figures_finetune_digit
for i in $(seq 0 6); do
    nohup python -u LLM_world/run_inference_finetune_digit.py \
        --shard $i --num-shards 7 \
        > LLM_world/logs_finetune_digit/shard_${i}.log 2>&1 &
    echo "Launched shard $i (GPU $i)  PID=$!"
done
echo "All shards launched. Monitor with: tail -f LLM_world/logs_finetune_digit/shard_*.log"
