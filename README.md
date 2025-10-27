# CORMPO: Clinically-aware OOD-regularized Model-based Policy Optimization

## Overview

This repository includes an offline RL algorithm, CORMPO, and a medical environment for RL evaluation. CORMPO addresses out-of-distribution (OOD) challenges in offline reinforcement learning by incorporating clinical domain knowledge and regularization techniques for safer policy optimization.

## Installation

Install all required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### MCS Digital Twin and RL Environment

See the README in the `abiomed_env` folder for environment implementation details and example scripts for using the environment.

### CORMPO Training

Train CORMPO with WS penalty on noiseless synthetic dataset:
```bash
python cormpo/mbpo_kde/mopo.py --config cormpo/config/noiseless_synthetic/mbpo_kde_ws.yaml
```
on noiseless synthetic dataset:
```bash
python cormpo/mbpo_kde/mopo.py --config cormpo/config/noisy_synthetic/mbpo_kde.yaml
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

-we borrowed the implementation of MOPO [https://github.com/junming-yang/mopo](https://github.com/junming-yang/mopo)
