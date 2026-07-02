#!/bin/bash
set -e
echo "============================================"
echo "Multi-Seed GORMPO-KDE Training: sparse D4RL (antmaze)"
echo "============================================"

TASK="antmaze-medium-play-v2"
DATA_PATH="data/antmaze.pkl"
seeds=(42 123 456)

timestamp=$(date +"%m%d_%H%M%S")
results_dir="results/${TASK}/mbpo_kde"
results_file="${results_dir}/multiseed_search_${timestamp}.csv"
mkdir -p "$results_dir"

echo "Step 1/2: Dumping D4RL dataset..."
python experiments/leq_guardian/create_d4rl_dataset.py --task "$TASK" --save_path "$DATA_PATH"

echo "Step 2/2: Training KDE guardian on ${TASK}..."
python mbpo_kde/kde.py --config experiments/leq_guardian/kde_antmaze.yaml --devid 0

for seed in "${seeds[@]}"; do
    echo ">>> Training GORMPO-KDE (seed $seed)..."
    python mopo.py \
        --config experiments/leq_guardian/mbpo_kde_antmaze.yaml \
        --seed $seed \
        --devid 0 \
        --classifier_model_name /public/gormpo/models/antmaze/kde/trained_kde_1 \
        --results-path $results_file
done

echo "Results saved to: $results_file"
