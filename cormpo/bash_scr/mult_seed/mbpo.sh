#!/bin/bash

set -e  # Exit on error
echo "============================================"
echo "Multi-Seed MBPO Training: Abiomed"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Define shared results file path
timestamp=$(date +"%m%d_%H%M%S")
results_dir="results/abiomed/mbpo"
results_file="${results_dir}/multiseed_search_${timestamp}.csv"

echo "Results will be saved to: $results_file"
echo ""

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 2: Train GORMPO policy using the trained VAE model
    echo "Step 2/2: Training MBPO policy (seed $seed)..."
    python mopo.py \
        --config config/real/mbpo.yaml \
        --seed $seed \
        --epoch 200 \
        --devid 7 \
        --classifier_model_name /public/gormpo/models/abiomed/trained_kde_$seed/trained_kde_1 \
        --results-path $results_file
    echo "✓ GORMPO-VAE training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-VAE multi-seed experiments completed!"
echo "Results saved to: $results_file"
echo "============================================"
