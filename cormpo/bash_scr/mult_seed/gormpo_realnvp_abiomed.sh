#!/bin/bash

set -e  # Exit on error
echo "============================================"
echo "Multi-Seed GORMPO-RealNVP Training: Abiomed"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Define shared results file path
timestamp=$(date +"%m%d_%H%M%S")
results_dir="results/abiomed/mbpo_realnvp"
results_file="${results_dir}/multiseed_search_${timestamp}.csv"

echo "Results will be saved to: $results_file"
echo ""

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train RealNVP density estimator for this seed
    echo "Step 1/2: Training RealNVP density estimator (seed $seed)..."
    python realnvp_module/realnvp.py \
        --config config/realnvp/real.yaml \
        --seed $seed \
        --save_path /public/gormpo/models/abiomed/trained_realnvp_$seed \
        --devid 6
    echo "✓ RealNVP training complete for seed $seed"
    echo ""

    # Step 2: Train GORMPO policy using the trained RealNVP model
    echo "Step 2/2: Training GORMPO-RealNVP policy (seed $seed)..."
    python mopo.py \
        --config config/real/mbpo_realnvp.yaml \
        --seed $seed \
        --epoch 200 \
        --devid 6 \
        --classifier_model_name /public/gormpo/models/abiomed/trained_realnvp_$seed/trained_realnvp_1 \
        --results-path $results_file
    echo "✓ GORMPO-RealNVP training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-RealNVP multi-seed experiments completed!"
echo "Results saved to: $results_file"
echo "============================================"
