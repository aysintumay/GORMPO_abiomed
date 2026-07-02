#!/bin/bash
set -e
# LEQ x sparse-D4RL (antmaze, native support), 3 seeds.
# Requires an OfflineRL-Kit checkout (pip install per LEQ/README.md) at $OFFLINERLKIT_DIR.

OFFLINERLKIT_DIR="${OFFLINERLKIT_DIR:-../OfflineRL-Kit}"
TASK="${TASK:-antmaze-medium-play-v2}"
SEEDS=(1 2 3)

# ponytail: bash's answer to pathlib.Path().resolve() — locate the sibling LEQ
# repo relative to this script, regardless of invocation cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEQ_DIR="$(cd "$SCRIPT_DIR/../../../../LEQ" && pwd)"
cd "$LEQ_DIR"

for seed in "${SEEDS[@]}"; do
    echo ">>> [dynamics] seed=$seed"
    cp dynamics/run_dynamics.py "$OFFLINERLKIT_DIR/run_example/run_dynamics.py"
    (cd "$OFFLINERLKIT_DIR" && python run_example/run_dynamics.py --task "$TASK" --seed "$seed")

    echo ">>> [LEQ] seed=$seed"
    PYTHONPATH='.' python train/train_LEQ.py --env_name "$TASK" --seed "$seed" --expectile 0.5
done
