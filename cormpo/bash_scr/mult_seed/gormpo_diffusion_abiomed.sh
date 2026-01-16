#!/bin/bash

set -e  # Exit on error
echo "============================================"
echo "Multi-Seed GORMPO-Diffusion Training: Abiomed"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Define shared results file path
timestamp=$(date +"%m%d_%H%M%S")
results_dir="results/abiomed/mbpo_diffusion"
results_file="${results_dir}/multiseed_search_${timestamp}.csv"

echo "Results will be saved to: $results_file"
echo ""

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train Diffusion density estimator for this seed
    echo "Step 1/2: Training Diffusion density estimator (seed $seed)..."
    python diffusion_module/train_diffusion.py \
        --seed $seed \
        --out-dir checkpoints/diffusion_$seed \
        --model-save-path /public/gormpo/models/abiomed/trained_diffusion_$seed/ \
        --epochs 100 \
        --devid 0
    echo "✓ Diffusion training complete for seed $seed"
    echo ""

    # Step 2: Train GORMPO policy using the trained Diffusion model
    echo "Step 2/2: Training GORMPO-Diffusion policy (seed $seed)..."
    python mopo.py \
        --config config/real/mbpo_diffusion.yaml \
        --seed $seed \
        --epoch 100 \
        --devid 0 \
        --classifier_model_name /public/gormpo/models/abiomed/trained_diffusion_$seed/checkpoint.pt \
        --results-path $results_file
    echo "✓ GORMPO-Diffusion training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-Diffusion multi-seed experiments completed!"
echo "Results saved to: $results_file"
echo "============================================"
