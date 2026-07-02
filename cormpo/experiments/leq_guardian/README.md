# LEQ + density guardian: MCS vs. sparse D4RL

2x2 matrix, 3 seeds each: {LEQ, GORMPO-KDE guardian} x {MCS (abiomed), sparse D4RL (antmaze)}.

Run all commands from `cormpo/` (matches every other script in `bash_scr/`).

| Cell | Where | How |
|---|---|---|
| Guardian x MCS | this repo | `bash_scr/mult_seed/gormpo_kde_abiomed.sh` |
| Guardian x sparse-D4RL | this repo | `experiments/leq_guardian/gormpo_kde_d4rl.sh` |
| LEQ x MCS | sibling `../LEQ` repo | `LEQ/bash_scr/leq_mcs.sh` |
| LEQ x sparse-D4RL | sibling `../LEQ` repo | `LEQ/bash_scr/leq_d4rl_sparse.sh` |

`static_fns/antmaze.py` and `config/antmaze.py` are NOT here — `train.py` finds them via
`importlib.import_module(f"static_fns.{task}")` / `f"config.{task}"`, so they must live in
those exact folders (import path == file path).

Files in this folder:
- `create_d4rl_dataset.py` — dumps `d4rl.qlearning_dataset(task)` to a pickle, feeds both
  `kde_antmaze.yaml` (guardian training) and `mbpo_kde_antmaze.yaml` (policy training).
- `kde_antmaze.yaml` — config for `mbpo_kde/kde.py` (trains the KDE guardian).
- `mbpo_kde_antmaze.yaml` — config for `mopo.py` (trains the policy against that guardian).
- `gormpo_kde_d4rl.sh` — chains all three, 3 seeds.

LEQ lives in a separate sibling repo (`../LEQ`), not vendored here — see its own
`bash_scr/` and `README.md`.
