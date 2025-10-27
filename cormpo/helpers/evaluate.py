import os
import sys
import random
import argparse
import datetime
import warnings

import numpy as np
import pandas as pd
import torch
import wandb
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy
from helpers.plotter import plot_policy, plot_score_histograms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from abiomed_env.rl_env import AbiomedRLEnvFactory
from abiomed_env.cost_func import (
    unstable_percentage_model_gradient,
    compute_acp_cost_model,
    weaning_score_model_gradient,
    weaning_score_model,
    unstable_percentage_model,
)

warnings.filterwarnings("ignore")


def get_mopo(env, args):
    """
    Initialize and load a MOPO/SAC policy model.

    Args:
        env: The environment instance
        args: Arguments containing model configuration and paths

    Returns:
        SACPolicy: Loaded SAC policy
    """
    # Create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )

    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    # Create policy
    sac_policy = SACPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        dist=dist,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        device=args.device
    )
    policy_state_dict = torch.load(args.policy_path, map_location=args.device)
    sac_policy.load_state_dict(policy_state_dict)
    return sac_policy


def _evaluate(policy, eval_env, episodes, args, plot=None):
    """
    Evaluate a trained policy over multiple episodes.

    Args:
        policy: The policy to evaluate
        eval_env: The evaluation environment
        episodes: Number of episodes to run
        args: Arguments containing evaluation configuration
        plot: Whether to plot the first episode trajectory

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # While evaluating, turn off reward shaping
    eval_env.gamma1 = 0
    eval_env.gamma2 = 0
    eval_env.gamma3 = 0


    # Initialize tracking variables
    eval_ep_info_buffer = []
    num_episodes = 0
    episode_reward, episode_length = 0, 0

    # Metrics accumulators
    total_acp = 0.0
    total_unstable_percentage_sum = 0.0
    total_unstable_percentage_gradient_sum = 0.0
    wean_score = 0.0
    ws_thr = 0.0

    # Episode-level tracking
    actions = []
    states = []
    ep_states = []
    acp_list = []
    ws_list = []
    rwd_list = []

    # Reset environment
    obs, info = eval_env.reset(idx=100)
    all_states = info['all_states']
    all_states = np.concatenate([obs.reshape(1, -1), all_states], axis=0)

    policy.eval()
    while num_episodes < episodes:
        action = policy.sample_action(obs, deterministic=True)
        next_obs, reward, terminal, truncated, _ = eval_env.step(action)
        episode_reward += reward
        episode_length += 1
        ep_states.append(obs)
        obs = next_obs

        if terminal or truncated:
            # Store episode data
            actions.append(eval_env.episode_actions)
            states.append(ep_states)
            ep_states_np = np.array(ep_states)

            # Compute episode metrics
            episode_acp_cost = compute_acp_cost_model(
                eval_env.world_model, eval_env.episode_actions, ep_states_np
            )
            total_acp += episode_acp_cost
            acp_list.append(episode_acp_cost)

            ws, _ = weaning_score_model_gradient(
                eval_env.world_model, ep_states_np, eval_env.episode_actions
            )
            wean_score += ws
            ws_list.append(ws)

            ws_thr += weaning_score_model(
                eval_env.world_model, ep_states_np, eval_env.episode_actions
            )

            unstable_ep = unstable_percentage_model(eval_env.world_model, ep_states_np)
            total_unstable_percentage_sum += unstable_ep
            total_unstable_percentage_gradient_sum += unstable_percentage_model_gradient(
                eval_env.world_model, ep_states_np
            )

            # Store episode info
            eval_ep_info_buffer.append({
                "episode_reward": episode_reward,
                "episode_length": episode_length
            })
            rwd_list.append(episode_reward)

            # Plot first episode if requested
            if (num_episodes == 0) and plot:
                print('WS', ws, 'ACP', episode_acp_cost)
                next_state_l = ep_states.copy()
                next_state_l.append(obs)
                plot_policy(eval_env, next_state_l[1:], all_states, args.algo_name.upper())

            # Reset for next episode
            episode_reward, episode_length = 0, 0
            num_episodes += 1
            obs, info = eval_env.reset()
            ep_states = []
            all_states = info['all_states']
            all_states = np.concatenate([obs.reshape(1, -1), all_states], axis=0)
    # Plot results
    plot_score_histograms(acp_list, ws_list, rwd_list, args.algo_name)

    # Compute statistics
    eval_info = {
        "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
        "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
    }

    ep_reward_mean = np.mean(eval_info["eval/episode_reward"])
    ep_reward_std = np.std(eval_info["eval/episode_reward"])
    ep_length_mean = np.mean(eval_info["eval/episode_length"])
    ep_length_std = np.std(eval_info["eval/episode_length"])

    # Average metrics
    total_acp /= num_episodes
    unsafe_hours = total_unstable_percentage_sum / num_episodes
    unsafe_hours_gradient = total_unstable_percentage_gradient_sum / num_episodes
    final_avg_wean_score = wean_score / num_episodes
    final_wean_thr_score = ws_thr / num_episodes

    # Print results
    print("---------------------------------------")
    print(f"Evaluation over {ep_length_mean} episodes:")
    print(f"  Return: {ep_reward_mean:.3f}")
    print(f"  ACP score: {total_acp:.4f}")
    print(f"  Unstable hours (%): {unsafe_hours:.3f}")
    print(f"  Unstable hours gradient (%): {unsafe_hours_gradient:.3f}")
    print(f"  Weaning score: {final_avg_wean_score:.5f}")
    print(f"  Weaning thr score: {final_wean_thr_score:.5f}")
    print(f"  Maximum ACP: {max(acp_list):.4f}, Minimum ACP: {min(acp_list):.4f}")
    print(f"  Maximum weaning score: {max(ws_list):.5f}, Minimum weaning score: {min(ws_list):.5f}")
    print("---------------------------------------")

    return {
        'mean_return': ep_reward_mean,
        'std_return': ep_reward_std,
        'mean_length': ep_length_mean,
        'std_length': ep_length_std,
        'mean_acp': total_acp,
        'mean_unsafe_hours': unsafe_hours,
        'mean_wean_score': final_avg_wean_score,
    }


def get_env(args):
    """
    Create and configure the Abiomed RL environment.

    Args:
        args: Arguments containing environment configuration

    Returns:
        env: Configured environment instance
    """
    env = AbiomedRLEnvFactory.create_env(
        model_name=args.model_name,
        model_path=args.model_path_wm,
        data_path=args.data_path_wm,
        max_steps=args.max_steps,
        action_space_type="continuous",
        reward_type="smooth",
        normalize_rewards=True,
        seed=args.seed,
        device=args.device,
    )
    args.obs_shape = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    return env


def mopo_args(parser):
    """Add MOPO/MBPO hyperparameters to argument parser."""
    g = parser.add_argument_group("MOPO hyperparameters")

    # SAC hyperparameters
    g.add_argument("--actor-lr", type=float, default=3e-4)
    g.add_argument("--critic-lr", type=float, default=3e-4)
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--tau", type=float, default=0.005)
    g.add_argument("--alpha", type=float, default=0.2)
    g.add_argument('--auto-alpha', default=True)
    g.add_argument('--target-entropy', type=int, default=-1)
    g.add_argument('--alpha-lr', type=float, default=3e-4)

    # Dynamics model arguments
    g.add_argument("--dynamics-lr", type=float, default=0.001)
    g.add_argument("--n-ensembles", type=int, default=7)
    g.add_argument("--n-elites", type=int, default=5)
    g.add_argument("--reward-penalty-coef", type=float, default=5e-3)
    g.add_argument("--rollout-length", type=int, default=5)
    g.add_argument("--rollout-batch-size", type=int, default=5000)
    g.add_argument("--rollout-freq", type=int, default=1000)
    g.add_argument("--model-retain-epochs", type=int, default=5)
    g.add_argument("--real-ratio", type=float, default=0.05)
    g.add_argument("--dynamics-model-dir", type=str, default=None)
    g.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Training arguments
    g.add_argument("--epoch", type=int, default=600)
    g.add_argument("--step-per-epoch", type=int, default=1000)
    g.add_argument("--batch-size", type=int, default=256)
    return parser


if __name__ == "__main__":
    print("Running", __file__)

    # Parse config file
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining_argv = config_parser.parse_known_args()

    if config_args.config:
        with open(config_args.config, "r") as f:
            config = yaml.safe_load(f)
            config = {k.replace("-", "_"): v for k, v in config.items()}
    else:
        config = {}

    # Base parser
    base = argparse.ArgumentParser(parents=[config_parser], add_help=False)
    base.add_argument(
        "--algo-name",
        choices=["mbpo", "mopo", "cormpo"],
        default="mopo",
        help="Which algorithm's flags to load"
    )
    args_partial, remaining_argv = base.parse_known_args()

    # Main parser
    parser = argparse.ArgumentParser(
        parents=[base],
        description="Evaluate your RL method"
    )

    # General arguments
    parser.add_argument("--task", type=str, default="abiomed")
    parser.add_argument("--policy_path", type=str, default="")
    parser.add_argument("--devid", type=int, default=7, help="Which GPU device index to use")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1,2,3,4,5], help="List of seeds for evaluation")
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-freq", type=int, default=1000)

    # Abiomed Environment Arguments
    parser.add_argument("--model_name", type=str, default="10min_1hr_all_data")
    parser.add_argument("--model_path_wm", type=str, default=None)
    parser.add_argument("--data_path_wm", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=6)

    # Add algorithm-specific arguments
    if args_partial.algo_name in ["mopo", "mbpo", "cormpo"]:
        mopo_args(parser)


    # Apply config defaults and parse remaining arguments
    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)
    args.config = config_args.config
    print(f"Config: {args.config}")
    results = []
    for seed in args.seeds:
        args.seed = seed
        # Set random seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Initialize wandb
        t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
        wandb.init(
            project="mopo-eval",
            name=f"eval_{args.task}_{args.algo_name}_{t0}",
            config=vars(args)
        )

        # Set device
        args.device = f'cuda:{args.devid}'

        # Create environment and policy
        env = get_env(args)
        policy = get_mopo(env, args)

        # Evaluate policy
        eval_info = _evaluate(policy, env, args.eval_episodes, args, plot=True)

        # Extract metrics
        mean_return = eval_info["mean_return"]
        std_return = eval_info["std_return"]
        mean_length = eval_info["mean_length"]
        std_length = eval_info["std_length"]
        mean_acp = eval_info["mean_acp"]
        mean_unsafe_hours = eval_info["mean_unsafe_hours"]
        mean_wean_score = eval_info["mean_wean_score"]

        # Compile results
        results.append({
            'mean_return': mean_return,
            'std_return': std_return,
            'mean_length': mean_length,
            'std_length': std_length,
            'mean_acp': mean_acp,
            'mean_unsafe_hours': mean_unsafe_hours,
            'mean_wean_score': mean_wean_score,
        })


    # Save results to CSV
    os.makedirs(os.path.join('results', args.task, args.algo_name), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = os.path.join('results', args.task, args.algo_name, f"{args.task}_results_{t0}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")