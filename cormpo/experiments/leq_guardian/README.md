# LEQ + density guardian: MCS vs. sparse D4RL

2x2 matrix, 3 seeds each: {LEQ, GORMPO-KDE guardian} x {MCS (abiomed), sparse D4RL (antmaze)}.

All four launcher scripts live in this folder. The `leq_*.sh` ones run code that lives in
the sibling `LEQ` repo, i.e. `../LEQ` relative to `GORMPO_abiomed/` (LEQ is not vendored here
— see its own `README.md`). They resolve that path from their own file location via
`BASH_SOURCE`, so they work from any invocation cwd.

| Cell | Script (all here) | Code runs in |
|---|---|---|
| Guardian x MCS | `bash_scr/mult_seed/gormpo_kde_abiomed.sh` (pre-existing, not moved) | this repo |
| Guardian x sparse-D4RL | `gormpo_kde_d4rl.sh` | this repo |
| LEQ x MCS | `leq_mcs.sh` | sibling `../LEQ` repo |
| LEQ x sparse-D4RL | `leq_d4rl_sparse.sh` | sibling `../LEQ` repo |

`static_fns/antmaze.py` and `config/antmaze.py` are NOT here — `train.py` finds them via
`importlib.import_module(f"static_fns.{task}")` / `f"config.{task}"`, so they must live in
those exact folders (import path == file path).

Files in this folder:
- `create_d4rl_dataset.py` — dumps `d4rl.qlearning_dataset(task)` to a pickle, feeds both
  `kde_antmaze.yaml` (guardian training) and `mbpo_kde_antmaze.yaml` (policy training).
- `kde_antmaze.yaml` — config for `mbpo_kde/kde.py` (trains the KDE guardian).
- `mbpo_kde_antmaze.yaml` — config for `mopo.py` (trains the policy against that guardian).
- `gormpo_kde_d4rl.sh` — chains all three, 3 seeds.
- `leq_mcs.sh` / `leq_d4rl_sparse.sh` — cd into `../LEQ` (resolved via `BASH_SOURCE`, not cwd)
  and run its `dynamics/run_dynamics.py` + `train/train_LEQ.py`.

## Known caveats

- **Guardian x MCS**: only seed 42's KDE model is pretrained. Seeds 123/456 need training
  first (see step 1 below) — the "Step 1" block in `gormpo_kde_abiomed.sh` is commented out.
- **LEQ x MCS dataset**: the existing `synthetic_data/SAC_5000eps_stochastic.npz` has actions
  ranging `[-2.23, 1.81]`, not the expected `[-1, 1]`. LEQ's `AbiomedDataset` clips to `[-1, 1]`,
  which is lossy at the extremes (destroys some high-P-level signal) but not broken. Regenerate
  via `sample_offline_dataset.py` for a cleaner dataset if this matters for real results.

## Environment setup (two separate envs — different frameworks)

**cormpo env** (torch, for the two guardian experiments):
```bash
cd GORMPO_abiomed
conda create -n gormpo python=3.8 && conda activate gormpo
pip install -r requirements.txt
conda install -c pytorch faiss-gpu   # needed by mbpo_kde/kde.py, not in requirements.txt
```

**LEQ env** (jax, for the two LEQ experiments) — separate terminal/machine, per `LEQ/README.md`:
```bash
cd LEQ   # sibling of GORMPO_abiomed
conda create -n LEQ python=3.9 && conda activate LEQ
pip install -r requirements.txt
pip install jax[cuda]==0.4.8 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
git clone https://github.com/yihaosun1124/OfflineRL-Kit ../OfflineRL-Kit
cd ../OfflineRL-Kit && python setup.py install
```

## Running each cell

**Recommended order** (minimizes wasted compute): 2 -> 4 -> 1 -> 3. Cells 2 and 4 are fully
automatic and the cheapest way to confirm each env/pipeline actually works before spending
time on the two cells with manual prerequisites.

### 1. Guardian x MCS (`gormpo` env)

```bash
cd GORMPO_abiomed/cormpo
```
Only seed 42's KDE model exists. Either train the other two first:
```bash
for seed in 123 456; do
    python mbpo_kde/kde.py --config config/kde/real.yaml --seed $seed \
        --save_path /public/gormpo/models/abiomed/trained_kde_$seed --devid 0
done
```
or uncomment the "Step 1" block in `bash_scr/mult_seed/gormpo_kde_abiomed.sh` so it trains
per-seed automatically. Then:
```bash
bash bash_scr/mult_seed/gormpo_kde_abiomed.sh
```
Results -> `results/abiomed/mbpo_kde/multiseed_search_<timestamp>.csv`.

### 2. Guardian x sparse-D4RL (`gormpo` env)

Self-contained — no pretraining needed, it trains its own dynamics ensemble and KDE guardian
from scratch.
```bash
cd GORMPO_abiomed/cormpo
bash experiments/leq_guardian/gormpo_kde_d4rl.sh
```
This dumps the antmaze dataset, trains one KDE guardian (shared across seeds), then trains 3
policy seeds. Results -> `results/antmaze-medium-play-v2/mbpo_kde/multiseed_search_<timestamp>.csv`.

### 3. LEQ x MCS (`LEQ` env)

Needs an abiomed `.npz`. Reuse the existing one (fast, with the clipping caveat above) or
regenerate cleanly. Run from `GORMPO_abiomed/`:
```bash
# reuse existing:
export DATASET_PATH=$(pwd)/synthetic_data/SAC_5000eps_stochastic.npz

# OR regenerate (slower, from the trained SAC policy already in the repo):
(cd abiomed_env && python sample_offline_dataset.py --model_path data/sac_20250827_0837.zip \
    --num_episodes 5000 --save_path ../synthetic_data/mcs_offline.npz)
export DATASET_PATH=$(pwd)/synthetic_data/mcs_offline.npz
```
Then, from anywhere — the script locates `../LEQ` itself:
```bash
bash cormpo/experiments/leq_guardian/leq_mcs.sh
```
This pretrains a per-seed dynamics ensemble
(`OfflineRL-Kit/models/dynamics-ensemble/{1,2,3}/abiomed-v0/`) then runs LEQ, per seed.
Checkpoints/logs -> `LEQ/tmp/EP/models/abiomed-v0/{seed}/0.5/`.

### 4. LEQ x sparse-D4RL (`LEQ` env)

Native — no abiomed dependency, needs `mujoco`/`d4rl` antmaze assets installed (part of env
setup above). Run from anywhere — the script locates `../LEQ` itself:
```bash
bash cormpo/experiments/leq_guardian/leq_d4rl_sparse.sh
```
Same checkpoint layout, under `antmaze-medium-play-v2/`.
