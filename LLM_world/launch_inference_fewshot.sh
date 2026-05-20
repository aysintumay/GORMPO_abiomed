#!/bin/bash
# Launch 7 shards of few-shot LLM inference across GPUs 0-6.
# Usage: bash LLM_world/launch_inference_fewshot.sh [--k-shot 3]

K_SHOT=${1:-3}
mkdir -p LLM_world/logs_fewshot

for i in $(seq 0 6); do
    nohup python -u LLM_world/run_inference_fewshot.py \
        --shard $i --num-shards 7 --k-shot $K_SHOT \
        > LLM_world/logs_fewshot/shard_${i}.log 2>&1 &
    echo "Launched shard $i on cuda:$i (PID $!)"
done

echo "All 7 shards launched. Monitor with:"
echo "  tail -f LLM_world/logs_fewshot/shard_*.log"
