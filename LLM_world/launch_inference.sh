#!/bin/bash
# Launches 7 parallel inference shards on GPUs 0-6.
# Logs go to LLM_world/logs/shard_N.log
mkdir -p LLM_world/logs LLM_world/results
for i in $(seq 0 6); do
    nohup python -u LLM_world/run_inference.py \
        --shard $i --num-shards 7 \
        > LLM_world/logs/shard_${i}.log 2>&1 &
    echo "Launched shard $i (GPU $i)  PID=$!"
done
echo "All shards launched. Monitor with: tail -f LLM_world/logs/shard_*.log"
