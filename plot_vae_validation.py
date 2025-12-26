#!/usr/bin/env python
"""
Plot VAE validation distribution and threshold using existing plotter functions.
"""
import torch
import yaml
import sys
import os

# Add cormpo to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cormpo.vae_module.vae import VAE, load_rl_data_for_vae
from cormpo.helpers.plotter import plot_likelihood_distributions
from abiomed_env.rl_env import AbiomedRLEnvFactory


# Load config
with open('cormpo/config/vae/abiomed.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create args object with necessary attributes
class Args:
    def __init__(self, config):
        self.task = config['task']
        self.model_name = config['model_name']
        self.model_path_wm = config['model_path_wm']
        self.data_path_wm = config['data_path_wm']
        self.max_steps = config['max_steps']
        self.gamma1 = config['gamma1']
        self.gamma2 = config['gamma2']
        self.gamma3 = config['gamma3']
        self.noise_rate = config['noise_rate']
        self.noise_scale = config['noise_scale']
        self.obs_dim = tuple(config['obs_dim'])
        self.obs_shape = self.obs_dim
        self.action_dim = config['action_dim']
        self.data_path = None
        self.device = config['device']

args = Args(config)

# Create environment
print("Creating Abiomed environment...")
env = AbiomedRLEnvFactory.create_env(
    model_name=args.model_name,
    model_path=args.model_path_wm,
    data_path=args.data_path_wm,
    max_steps=args.max_steps,
    gamma1=args.gamma1,
    gamma2=args.gamma2,
    gamma3=args.gamma3,
    action_space_type='continuous',
    reward_type="smooth",
    normalize_rewards=True,
    noise_rate=args.noise_rate,
    noise_scale=args.noise_scale,
    seed=42,
    device=args.device
)

# Load data
print("Loading data...")
train_data, val_data, test_data, vae_input_dim = load_rl_data_for_vae(
    args=args,
    env=env,
    val_split_ratio=config.get('val_ratio', 0.2)
)

# Load trained model
print("Loading trained VAE model...")
model_dict = VAE.load_model(
    'checkpoints/vae/abiomed_vae',
    hidden_dims=config['hidden_dims']
)
model = model_dict['model']
threshold = model_dict['thr']

print(f"\nModel loaded successfully!")
print(f"Threshold: {threshold:.4f}")
print(f"Mean score: {model_dict['mean']:.4f}")
print(f"Std score: {model_dict['std']:.4f}")

# Move data to model's device
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

# Plot using existing plotter function
print("\nPlotting likelihood distributions...")
plot_likelihood_distributions(
    model=model,
    train_data=train_data,
    val_data=val_data,
    ood_data=test_data,
    thr=threshold,
    title="VAE Score Distribution on Abiomed Data",
    savepath="figures/vae_abiomed_distribution.png",
    bins=50,
    use_detach=True
)

print("\nPlots saved successfully!")
print("- figures/train_distribution.png (Train + Val with threshold)")
print("- figures/ood_distribution.png (Test data with threshold)")
