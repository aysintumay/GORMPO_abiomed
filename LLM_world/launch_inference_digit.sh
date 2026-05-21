#!/bin/bash
# Launch 7 shards of digit-token few-shot LLM inference across GPUs 0-6.
# Usage: bash LLM_world/launch_inference_digit.sh [--k-shot 5]

K_SHOT=${1:-5}
mkdir -p LLM_world/logs_digit_k${K_SHOT}

for i in $(seq 0 6); do
    nohup conda run --no-capture-output -n llm_world python -u LLM_world/run_inference_digit.py \
        --shard $i --num-shards 7 --k-shot $K_SHOT \
        > LLM_world/logs_digit_k${K_SHOT}/shard_${i}.log 2>&1 &
    echo "Launched shard $i on cuda:$i (PID $!)"
done

echo "All 7 shards launched. Monitor with:"
echo "  tail -f LLM_world/logs_digit_k${K_SHOT}/shard_*.log"
