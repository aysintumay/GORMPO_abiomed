#!/bin/bash
set -e
# LEQ x MCS (abiomed), 3 seeds.
# Requires an OfflineRL-Kit checkout (pip install per LEQ/README.md) at $OFFLINERLKIT_DIR,
# and an abiomed offline dataset .npz (see abiomed_env/sample_offline_dataset.py).

OFFLINERLKIT_DIR="${OFFLINERLKIT_DIR:-../OfflineRL-Kit}"
DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to an abiomed .npz}"
TASK="abiomed-v0"
SEEDS=(1 2 3)

# ponytail: bash's answer to pathlib.Path().resolve() — locate the sibling LEQ
# repo relative to this script, regardless of invocation cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEQ_DIR="$(cd "$SCRIPT_DIR/../../../../LEQ" && pwd)"
cd "$LEQ_DIR"

for seed in "${SEEDS[@]}"; do
    echo ">>> [dynamics] seed=$seed"
    cp dynamics/run_dynamics.py "$OFFLINERLKIT_DIR/run_example/run_dynamics.py"
    (cd "$OFFLINERLKIT_DIR" && python run_example/run_dynamics.py \
        --task "$TASK" --dataset_path "$DATASET_PATH" --seed "$seed")

    echo ">>> [LEQ] seed=$seed"
    PYTHONPATH='.' python train/train_LEQ.py \
        --env_name "$TASK" --dataset_path "$DATASET_PATH" --seed "$seed" --expectile 0.5
done
