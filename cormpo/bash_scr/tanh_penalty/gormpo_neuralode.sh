#!/bin/bash

set -e  # Exit on error
echo "============================================"
echo "Multi-Seed GORMPO-NeuralODE Training: Abiomed"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42)

# Define shared results file path
timestamp=$(date +"%m%d_%H%M%S")
results_dir="results/abiomed/mbpo_neuralode_penalty"
results_file="${results_dir}/multiseed_search_${timestamp}.csv"

echo "Results will be saved to: $results_file"
echo ""

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="


    # Step 2: Train GORMPO policy using the trained RealNVP model
    echo "Step 2/2: Training GORMPO-NeuralODE policy (seed $seed)..."
    python mopo.py \
        --config config/real/mbpo_neuralode.yaml \
        --devid 2 \
        --seed $seed \
        --epoch 200 \
        --results-path $results_file \
        --penalty_type "tanh_penalty"


    echo "✓ GORMPO-RealNVP training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-NeuralODE multi-seed experiments completed!"
echo "Results saved to: $results_file"
echo "============================================"
