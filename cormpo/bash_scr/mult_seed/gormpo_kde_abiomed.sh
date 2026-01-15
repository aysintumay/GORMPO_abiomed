#!/bin/bash

set -e  # Exit on error
echo "============================================"
echo "Multi-Seed GORMPO-KDE Training: Abiomed"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Define shared results file path
timestamp=$(date +"%m%d_%H%M%S")
results_dir="results/abiomed/mbpo_kde"
results_file="${results_dir}/multiseed_search_${timestamp}.csv"

echo "Results will be saved to: $results_file"
echo ""

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train KDE density estimator for this seed
    echo "Step 1/2: Training KDE density estimator (seed $seed)..."
    python mbpo_kde/kde.py \
        --config config/kde/real.yaml \
        --seed $seed \
        --save_path /public/gormpo/models/abiomed/trained_kde_$seed \
        --devid 2
    echo "✓ KDE training complete for seed $seed"
    echo ""

    # Step 2: Train GORMPO policy using the trained KDE model
    echo "Step 2/2: Training GORMPO-KDE policy (seed $seed)..."
    python mopo.py \
        --config config/real/mbpo_kde.yaml \
        --seed $seed \
        --epoch 200 \
        --devid 2 \
        --classifier_model_name /public/gormpo/models/abiomed/trained_kde_$seed/trained_kde_1 \
        --results-path $results_file
    echo "✓ GORMPO-KDE training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-KDE multi-seed experiments completed!"
echo "Results saved to: $results_file"
echo "============================================"
