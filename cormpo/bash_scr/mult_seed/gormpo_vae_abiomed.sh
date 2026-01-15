#!/bin/bash

set -e  # Exit on error
echo "============================================"
echo "Multi-Seed GORMPO-VAE Training: Abiomed"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Define shared results file path
timestamp=$(date +"%m%d_%H%M%S")
results_dir="results/abiomed/mbpo_vae"
results_file="${results_dir}/multiseed_search_${timestamp}.csv"

echo "Results will be saved to: $results_file"
echo ""

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train VAE density estimator for this seed
    echo "Step 1/2: Training VAE density estimator (seed $seed)..."
    python vae_module/vae.py \
        --config config/vae/real.yaml \
        --seed $seed \
        --save_path /public/gormpo/models/abiomed/trained_vae_$seed/trained_vae_1  \
        --devid 5
    echo "✓ VAE training complete for seed $seed"
    echo ""

    # Step 2: Train GORMPO policy using the trained VAE model
    echo "Step 2/2: Training GORMPO-VAE policy (seed $seed)..."
    python mopo.py \
        --config config/real/mbpo_vae.yaml \
        --seed $seed \
        --epoch 200 \
        --devid 5 \
        --classifier_model_name /public/gormpo/models/abiomed/trained_vae_$seed/trained_vae_1 \
        --results-path $results_file
    echo "✓ GORMPO-VAE training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-VAE multi-seed experiments completed!"
echo "Results saved to: $results_file"
echo "============================================"
