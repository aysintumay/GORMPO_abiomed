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

## Reference

- The implementation of MOPO and MBPO-KDE is built largely on this implementation of MOPO algorithm: [https://github.com/junming-yang/mopo](https://github.com/junming-yang/mopo)
