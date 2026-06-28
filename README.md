# CORMPO: Clinically-aware OOD-regularized Model-based Policy Optimization

## Overview

This repository includes Generative OOD-regularized Model-based Policy Optimization (GORMPO), and a digital twin environment for RL evaluation. See the first paper: [Guardian-regularized Safe Offline Reinforcement Learning for Smart Weaning of Mechanical Circulatory Devices](https://arxiv.org/abs/2511.06111). GORMPO paper is coming out soon.

## Dependencies / Installation

Install all required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Test OOD detection 
```
python test_vae_ood_levels.py   
    --model_path /public/gormpo/models/abiomed/vae/abiomed_vae \
    --dataset_name abiomed  \   
    --distances 1 2 3 4 \   
    --base_path /public/gormpo/ood_test    \
    --device cuda:2     \
    --save_dir figures/vae_ood_distance_tests
```
or basically

```
python test_vae_ood_levels.py   
    --config config/vae/test_ood_abiomed.yaml
```


### MCS Digital Twin and RL Environment

See the README in the `abiomed_env` folder for environment implementation details and example scripts for using the environment.

### GORMPO Training

Train GORMPO-diffusion:
```bash
cd cormpo
python mopo.py --config config/real/mbpo_diffusion.yaml
```

### CORMPO Policy Evaluation

Evaluate a saved policy trained on noisy synthetic dataset:
```bash
python cormpo/helpers/evaluate.py --config cormpo/config/evaluate/noisy/cormpo.yaml --policy_path "checkpoints/policy/noisy_synthetic/policy_abiomed.pth"
```

To evaluate the policy trained on noiseless dataset, change `policy_path` to:
```bash
--policy_path "checkpoints/policy/noiseless_synthetic/policy_abiomed.pth"
```

## Configuration Variables (`cormpo/config/real/`)

Each YAML file in `cormpo/config/real/` configures a different OOD-detection variant of MBPO. The shared variables are:

| Variable | Description |
|---|---|
| `task` | Environment name to run (e.g., `abiomed` for the cardiac assist device digital twin). |
| `algo-name` | Algorithm identifier (e.g., `mbpo`, `mbpo_vae`, `mbpo_realnvp`). Selects the OOD classifier backend. |
| `reward-penalty-coef` | Weight of the OOD penalty term subtracted from the reward. Higher values push the policy away from out-of-distribution states more aggressively. |
| `seed` | Random seed for reproducibility. |
| `noise_rate` | Probability of applying noise to an observation at each step. Set to `0.0` for clean evaluation. |
| `noise_scale` | Magnitude of the Gaussian noise added when `noise_rate > 0`. |
| `gamma1` / `gamma2` / `gamma3` | Coefficients for auxiliary reward shaping terms (e.g., clinical constraint penalties). Set to `0.0` to disable. |
| `epoch` | Number of training epochs. |
| `dynamics-model-dir` | If `true`, loads a pre-trained dynamics (world) model from `model_path` instead of training from scratch. |
| `model_path` | Path to the saved RL/dynamics model weights. |
| `classifier_model_name` | Path to the pre-trained OOD classifier model (VAE, RealNVP, KDE, NeuralODE, or Diffusion checkpoint). |
| `penalty_type` | Shape of the penalty function applied to OOD scores. `tanh` squashes the raw score to `[0, 1]`. |
| `target_dim` | Dimensionality of the state/observation space (73 for the abiomed dataset). |
| `rollout-length` | Planning horizon — number of imagined steps taken inside the world model per real environment step. |
| `max_steps` | Maximum number of steps per episode in the environment. |
| `action_space_type` | Whether the action space is `continuous` or `discrete`. |
| `model_name` | String identifier for the world model checkpoint (used for logging). |
| `model_path_wm` | Path to the world model weights file (`.pth`). |
| `data_path_wm` | Path to the dataset used to train or condition the world model (`.pkl`). |
| `num_inference_steps` | Number of diffusion timesteps used when computing the ELBO-based OOD score. Fewer steps are faster but less accurate. |

## Reference

- The implementation of MOPO and MBPO-KDE is built largely on this implementation of MOPO algorithm: [https://github.com/junming-yang/mopo](https://github.com/junming-yang/mopo)
