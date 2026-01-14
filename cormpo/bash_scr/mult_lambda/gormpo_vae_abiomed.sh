#!/bin/bash

set -e  # Exit on error
echo "============================================"
echo "Training GORMPO with VAE on Abiomed"
echo "Hyperparameter search for reward penalty coefficients"
echo "============================================"
echo ""

# Array of penalty coefficients to test
penalty_coeffs=(0.05 0.1 0.2 0.3 0.4 0.5)

# Define shared results file path
timestamp=$(date +"%m%d_%H%M%S")
results_dir="results/abiomed/mbpo_vae"
results_file="${results_dir}/hyperparameter_search_${timestamp}.csv"

echo "Results will be saved to: $results_file"
echo ""

# Loop through each penalty coefficient
for coef in "${penalty_coeffs[@]}"; do
    echo "=========================================="
    echo ">>> Training with reward-penalty-coef = $coef"
    echo "=========================================="

    python mopo.py \
        --config config/real/mbpo_vae.yaml \
        --reward-penalty-coef $coef \
        --epoch 200 \
        --devid 6 \
        --results-path $results_file

    echo "✓ Training complete for penalty coefficient $coef"
    echo ""
done

echo "============================================"
echo "All hyperparameter search experiments completed!"
echo "Results saved to: $results_file"
echo "============================================"
