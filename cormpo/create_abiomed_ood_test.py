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
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, '/home/ubuntu/GORMPO_abiomed')
sys.path.insert(0, '/home/ubuntu/GORMPO_abiomed/cormpo')

# Import Abiomed environment
from abiomed_env.rl_env import AbiomedRLEnvFactory

def plot_contour_action_norm_vs_reward(dataset, title="Action Norm vs Reward Contour",
                                        bins=20, unsafe_region_reward=None,
                                        unsafe_region_norm=None, cmap='viridis',
                                        ax=None, show=True):
    """
    Create a contour plot showing the density of action norm vs reward samples.

    Args:
        dataset: Dataset dictionary with 'rewards' and 'actions' keys
        title: Title for the plot
        bins: Number of bins for the 2D histogram (default: 50)
        unsafe_region_reward: Tuple (min, max) for unsafe reward range (optional)
        unsafe_region_norm: Tuple (min, max) for unsafe action norm range (optional)
        cmap: Colormap for the contour plot (default: 'viridis')
        ax: Matplotlib axis to plot on (optional, creates new figure if None)
        show: Whether to call plt.show() (default: True)
    """
    # Compute action norms
    actions = dataset['actions']
    action_norms = np.linalg.norm(actions, axis=1)
    state_norms = np.linalg.norm(dataset['observations'], axis=1)
    # rewards = dataset['rewards']

    # Create 2D histogram
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Add padding to the range to avoid cutting off the distribution
    norm_range = action_norms.max() - action_norms.min()
    reward_range = state_norms.max() - state_norms.min()
    norm_padding = norm_range * 0.1  # 10% padding
    reward_padding = reward_range * 0.1  # 10% padding

    # Compute histogram with extended range
    hist, xedges, yedges = np.histogram2d(
        action_norms, state_norms, bins=bins,
        range=[[action_norms.min() - norm_padding, action_norms.max() + norm_padding],
               [state_norms.min() - reward_padding, state_norms.max() + reward_padding]]
    )

    # Create meshgrid for contour plot
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

    # Create filled contour plot
    contour = ax.contourf(X, Y, hist.T, levels=20, cmap=cmap, alpha=0.8)

    # Add contour lines
    contour_lines = ax.contour(X, Y, hist.T, levels=10, colors='black', alpha=0.3, linewidths=0.5)

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, label='Sample Density')

    # Highlight unsafe region if provided
    if unsafe_region_reward is not None and unsafe_region_norm is not None:
        from matplotlib.patches import Rectangle
        rect = Rectangle((unsafe_region_norm[0], unsafe_region_reward[0]),
                         unsafe_region_norm[1] - unsafe_region_norm[0],
                         unsafe_region_reward[1] - unsafe_region_reward[0],
                         linewidth=2, edgecolor='red', facecolor='none',
                         linestyle='--', label='Unsafe Region')
        ax.add_patch(rect)
        ax.legend(fontsize=10)

    ax.set_xlabel('Action L2 Norm', fontsize=12)
    ax.set_ylabel('State L2 Norm', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    if fig is not None:
        plt.tight_layout()
        if show:
            plt.show()

    return fig, ax, contour


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
    noise_obs =  noise_mean*np.ones(selected_obs.shape)
    noise_act = 0*np.ones(selected_acts.shape)

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

    output_dir = '/home/ubuntu/GORMPO_abiomed/figures/ood_test/'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join('/public/gormpo/ood_test', f'ood-distance-{str(distance)}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(combined_data, f)

    print(f"Saved combined dataset to: {output_path}")
    print(f"  Total samples: {len(combined_data['observations'])} "
          f"(ID: {len(normal['observations'])}, OOD: {len(ood_data['observations'])})")

    return combined_data


def main():
    print("="*80)
    print("Creating OOD Test Sets for Abiomed")
    print("="*80)

    # Create Abiomed environment and load data
    print("\nLoading Abiomed environment and data...")
    env = AbiomedRLEnvFactory.create_env(
        model_name='10min_1hr_all_data',
        model_path=None,
        data_path=None,
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

    # Generate OOD datasets for different noise_std values
    noise_mean_levels = [1, 2, 3, 4,]

    print(f"\nGenerating OOD test sets for noise_std levels: {noise_mean_levels}")
    print("-"*80)

    # Store combined datasets for plotting
    combined_datasets = {}

    for noise_mean in noise_mean_levels:
        print(f"\nGenerating OOD data with noise_std = {noise_mean}")

        # Generate OOD data with varying noise_std (noise_mean=0)
        ood_data, normal = select_subset_and_add_noise(
            abiomed_dataset,
            num_trajectories=200,
            noise_std=0.0,
            noise_mean=noise_mean,
            clip_actions=True,
            action_low=-3.0,
            action_high=3.0,
            seed=42
        )

        print(f"  Generated {len(ood_data['observations'])} OOD samples")
        print(f"  Generated {len(normal['observations'])} normal samples")

        # Save combined dataset and store for plotting
        combined_data = save_combined_dataset(normal, ood_data, f"{noise_mean}")
        combined_datasets[noise_mean] = combined_data

    # Create output directory for figures
    figures_dir = '/home/ubuntu/GORMPO_abiomed/figures/ood_test'
    os.makedirs(figures_dir, exist_ok=True)

    # Create combined figure with 4 subplots (2x2 grid)
    print("\nCreating combined contour plot figure...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    axes_flat = axes.flatten()
    
    for idx, noise_mean in enumerate(noise_mean_levels):
        ax = axes_flat[idx]
        dataset = combined_datasets[noise_mean]
        
        if idx == 0:

            ax = axes_flat[idx]
            title = f"Normal"
            plot_contour_action_norm_vs_reward(
            normal,
            title=title,
            bins=20,
            cmap='viridis',
            ax=ax,
            show=False
            )
        idx = idx + 1
        ax = axes_flat[idx]
        title = f"Noise Std = {noise_mean}"
        plot_contour_action_norm_vs_reward(
            dataset,
            title=title,
            bins=20,
            cmap='viridis',
            ax=ax,
            show=False
        )

    fig.suptitle('OOD Test Datasets: Action Norm vs State Norm Contour Plots', fontsize=16, y=1.02)
    plt.tight_layout()

    # Save the combined figure
    output_path = os.path.join(figures_dir, 'noisy_datasets_contour_plots.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved combined figure to: {output_path}")

    plt.close(fig)

    print("\n" + "="*80)
    print("OOD Test Set Generation Complete!")
    print("="*80)
    print(f"\nGenerated {len(noise_mean_levels)} OOD test sets in")
    print(f"Combined figure saved to: {output_path}")


if __name__ == "__main__":
    main()
