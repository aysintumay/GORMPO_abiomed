"""
Create OOD test datasets for Abiomed environment.
Generates 5 OOD test sets with different noise mean values: 0.5, 1, 2, 4, 8
"""

import numpy as np
import random
import os
import pickle
import sys
import torch

# Add parent directory to path
sys.path.insert(0, '/home/ubuntu/GORMPO_abiomed')
sys.path.insert(0, '/home/ubuntu/GORMPO_abiomed/cormpo')

# Import Abiomed environment
from abiomed_env.rl_env import AbiomedRLEnvFactory


def select_subset_and_add_noise(
    dataset,
    num_trajectories=50,
    noise_std=0.1,
    noise_mean=0.0,
    num_samples=10000,
    clip_actions=True,
    action_low=-3.0,
    action_high=3.0,
    seed=None,
):
    """
    Select a subset of trajectories from the dataset and add noise.

    Args:
        dataset: Dataset object with .data (observations) and .pl (actions) attributes
        num_trajectories: Number of trajectories to select
        noise_std: Standard deviation of noise to add
        noise_mean: Mean of noise to add
        num_samples: Number of samples (not used for Abiomed)
        clip_actions: Whether to clip actions to valid range
        action_low: Minimum action value
        action_high: Maximum action value
        seed: Random seed

    Returns:
        Tuple of (new_dataset, orig_dataset) dictionaries
    """
    rng = np.random.default_rng(seed)

    # Extract data from Abiomed dataset
    # dataset.data shape: [n_episodes, max_steps, n_features]
    # dataset.pl shape: [n_episodes, max_steps]

    if isinstance(dataset.data, torch.Tensor):
        obs = dataset.data.cpu().numpy()
        acts = dataset.pl.cpu().numpy()
    else:
        obs = np.array(dataset.data)
        acts = np.array(dataset.pl)

    n_episodes = obs.shape[0]
    max_steps = obs.shape[1]

    # Sample trajectories
    k = min(num_trajectories, n_episodes)
    selected_indices = random.sample(range(n_episodes), k)

    # Extract selected trajectories
    selected_obs = obs[selected_indices]  # [k, max_steps, n_features]
    selected_acts = acts[selected_indices]  # [k, max_steps]

    # Add noise
    noise_obs = rng.normal(loc=noise_mean, scale=noise_std, size=selected_obs.shape)
    noise_act = rng.normal(loc=noise_mean, scale=noise_std, size=selected_acts.shape)

    noisy_obs = selected_obs + noise_obs
    noisy_acts = selected_acts + noise_act

    if clip_actions:
        noisy_acts = np.clip(noisy_acts, action_low, action_high)

    # Reshape to [n_samples, features] format for compatibility with test script
    # Flatten timesteps dimension into features: [k, max_steps, n_features] -> [k, max_steps * n_features]
    orig_obs_flat = selected_obs.reshape(selected_obs.shape[0], -1)  # [k, 6*12] = [k, 72]
    noisy_obs_flat = noisy_obs.reshape(noisy_obs.shape[0], -1)  # [k, 6*12] = [k, 72]

    # Take the majority vote of actions across timesteps (to match ReplayBuffer processing)
    # For now, just take the mean action across timesteps
    orig_acts_flat = selected_acts.mean(axis=1, keepdims=True)  # [k, 1]
    noisy_acts_flat = noisy_acts.mean(axis=1, keepdims=True)  # [k, 1]

    new_dataset = {
        'observations': noisy_obs_flat,
        'actions': noisy_acts_flat,
    }

    orig_dataset = {
        'observations': orig_obs_flat,
        'actions': orig_acts_flat,
    }

    return new_dataset, orig_dataset


def save_combined_dataset(normal, ood_data, distance):
    """
    Combine normal and OOD datasets and save to file.
    First half is normal (ID), second half is OOD.
    """
    combined_data = {
        'observations': np.concatenate([normal['observations'], ood_data['observations']]),
        'actions': np.concatenate([normal['actions'], ood_data['actions']]),
    }

    output_dir = '/abiomed/downsampled/ood_test/'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f'ood-distance-{str(distance)}.pkl')

    with open(output_path, 'wb') as f:
        pickle.dump(combined_data, f)

    print(f"Saved combined dataset to: {output_path}")
    print(f"  Total samples: {len(combined_data['observations'])} "
          f"(ID: {len(normal['observations'])}, OOD: {len(ood_data['observations'])})")


def main():
    print("="*80)
    print("Creating OOD Test Sets for Abiomed")
    print("="*80)

    # Create Abiomed environment and load data
    print("\nLoading Abiomed environment and data...")
    env = AbiomedRLEnvFactory.create_env(
        model_name='10min_1hr_all_data',
        model_path='/abiomed/downsampled/models/10min_1hr_all_data_model.pth',
        data_path='/abiomed/downsampled/10min_1hr_all_data.pkl',
        max_steps=6,
        gamma1=0.0,
        gamma2=0.0,
        gamma3=0.0,
        noise_rate=0.0,
        noise_scale=0.0,
    )

    # Get the full dataset
    print("Loading dataset...")
    from cormpo.common.util import load_dataset_with_validation_split
    import argparse

    args = argparse.Namespace(
        data_path=None,
        task='abiomed'
    )

    dataset_result = load_dataset_with_validation_split(
        args=args,
        env=env,
        val_split_ratio=0.2
    )

    # Use the train dataset as the base dataset
    abiomed_dataset = dataset_result['train_data']

    print(f"Loaded dataset with {len(abiomed_dataset.data)} episodes")
    print(f"  Observation shape: {abiomed_dataset.data.shape}")
    print(f"  Action shape: {abiomed_dataset.pl.shape}")

    # Generate OOD datasets for different noise mean values
    noise_means = [0.1, 0.5, 1,2,3,4]

    print(f"\nGenerating OOD test sets for noise means: {noise_means}")
    print("-"*80)

    for noise_mean in noise_means:
        print(f"\nGenerating OOD data with noise_mean = {noise_mean}")

        # Generate OOD data
        ood_data, normal = select_subset_and_add_noise(
            abiomed_dataset,
            num_trajectories=200,  # Generate 200 ID and 200 OOD samples
            noise_std=0.1,
            noise_mean=noise_mean,
            clip_actions=True,
            action_low=-3.0,
            action_high=3.0,
            seed=42
        )

        print(f"  Generated {len(ood_data['observations'])} OOD samples")
        print(f"  Generated {len(normal['observations'])} normal samples")

        # Save combined dataset
        save_combined_dataset(normal, ood_data, noise_mean)

    print("\n" + "="*80)
    print("OOD Test Set Generation Complete!")
    print("="*80)
    print(f"\nGenerated {len(noise_means)} OOD test sets in /abiomed/downsampled/ood_test/")


if __name__ == "__main__":
    main()
