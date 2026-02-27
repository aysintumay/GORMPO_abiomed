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
from bc import Actor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from common.buffer import ReplayBuffer
from train import load_data
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from ope.IS.traditional_is import TraditionalIS
from algo.sac import SACPolicy
from helpers.plotter import plot_policy, plot_score_histograms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', '..')))

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


def evaluate_traditional_is(episodes_rewards, episodes_target_log_probs, episodes_base_log_probs, gamma):
    """Compute IS-based OPE metrics using TraditionalIS.

    Parameters
    ----------
    episodes_rewards : list of lists
        Per-step rewards for each episode.
    episodes_target_log_probs : list of lists
        Log probabilities of taken actions under the target (evaluation) policy,
        per step per episode.
    episodes_base_log_probs : list of lists
        Log probabilities of taken actions under the behavior policy, per step
        per episode.  Pass lists of 0.0 as a placeholder when the behavior
        policy is unavailable (equivalent to an unnormalized uniform base).
    gamma : float
        Discount factor.

    Returns
    -------
    dict
        V_naive, V_IS, V_step_IS, V_WIS, V_step_WIS.
    """
    is_estimator = TraditionalIS(gamma=gamma)

    # rho_t = pi_target(a_t | s_t) / pi_base(a_t | s_t)
    episode_rhos = [
        np.exp(np.array(t_lp) - np.array(b_lp))
        for t_lp, b_lp in zip(episodes_target_log_probs, episodes_base_log_probs)
    ]

    return {
        'V_naive':    is_estimator.naive(episode_rhos, episodes_rewards),
        'V_IS':       is_estimator.IS(episode_rhos, episodes_rewards),
        'V_step_IS':  is_estimator.step_IS(episode_rhos, episodes_rewards),
        'V_WIS':      is_estimator.WIS(episode_rhos, episodes_rewards),
        'V_step_WIS': is_estimator.step_WIS(episode_rhos, episodes_rewards),
    }

def buffer_to_episodes(buffer, episode_len=6):
    """Group a flat transition buffer dict into fixed-length episodes.

    Parameters
    ----------
    buffer : dict
        Keys: 'observations', 'actions', 'rewards' (and optionally
        'next_observations', 'terminals').  Each value is an array of
        shape (N, ...).
    episode_len : int
        Number of steps per episode (default 6, matching max_steps).

    Returns
    -------
    list of lists
        Each inner list is one episode: [(obs, action, reward), ...].
        Incomplete trailing steps (N % episode_len != 0) are dropped.
    """
    obs     = buffer['observations']
    actions = buffer['actions']
    rewards = buffer['rewards']
    N = len(obs)

    episodes = []
    for start in range(0, N - episode_len + 1, episode_len):
        end = start + episode_len
        episode = [
            (obs[i], actions[i], float(rewards[i]))
            for i in range(start, end)
        ]
        episodes.append(episode)
    return episodes


def evaluate_is_from_buffer(buffer, target_policy, bc_actor, inv_gauss_coef, gamma, device, episode_len=6):
    """Compute IS-based OPE metrics using the offline logged buffer.

    Unlike _evaluate (which rolls out the target policy in the world model),
    this function operates on trajectories collected by the behavior policy.
    For each logged (obs, action, reward) tuple the importance weight is:
        rho_t = pi_target(a_logged | s) / pi_base(a_logged | s)

    Parameters
    ----------
    buffer : dict
        Flat buffer dict with keys 'observations', 'actions', 'rewards'
        (arrays of shape (N, ...)).  Internally converted to episodes of
        length episode_len via buffer_to_episodes().
    target_policy : SACPolicy
        The policy being evaluated.  Its actor must expose a get_dist(obs)
        method that returns the action distribution so we can evaluate
        log_prob at an arbitrary (logged) action.
    bc_actor : Actor
        Behavior-cloning actor from get_bc().  The behavior policy is modelled
        as a Gaussian centred at bc_actor(obs):
            log pi_base(a|s) = -mean((bc_actor(s) - a)^2, dim) / inv_gauss_coef
    inv_gauss_coef : float
        Bandwidth of the Gaussian kernel for the behavior policy log-prob.
    gamma : float
        Discount factor.
    device : str or torch.device
        Device for tensor computation.
    episode_len : int
        Steps per episode passed to buffer_to_episodes (default 6).

    Returns
    -------
    dict
        V_naive, V_IS, V_step_IS, V_WIS, V_step_WIS.
    """
    buffer_trajectories = buffer_to_episodes(buffer, episode_len)

    episodes_rewards          = []
    episodes_target_log_probs = []
    episodes_base_log_probs   = []

    target_policy.eval()
    bc_actor.eval()

    for episode in buffer_trajectories:
        ep_rewards, ep_target_lp, ep_base_lp = [], [], []

        for obs, action_logged, reward in episode:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            act_t = torch.FloatTensor(action_logged).unsqueeze(0).to(device)

            with torch.no_grad():
                # Target policy: log-prob of the *logged* action under pi_target.
                # We need the distribution, not just the log-prob of a fresh sample,

                # so we call get_dist rather than forward().
                dist      = target_policy.actor.get_dist(obs_t)
                target_lp = dist.log_prob(act_t).sum(-1)

                # Base policy: Gaussian centered at BC prediction (log-space).
                mu_bc   = bc_actor(obs_t)
                base_lp = -torch.mean((mu_bc - act_t) ** 2, dim=1) / inv_gauss_coef

            ep_rewards.append(reward)
            ep_target_lp.append(target_lp.cpu().item())
            ep_base_lp.append(base_lp.cpu().item())

        episodes_rewards.append(ep_rewards)
        episodes_target_log_probs.append(ep_target_lp)
        episodes_base_log_probs.append(ep_base_lp)

    is_metrics = evaluate_traditional_is(
        episodes_rewards, episodes_target_log_probs, episodes_base_log_probs, gamma
    )

    print("---------------------------------------")
    print(f"IS Evaluation from Offline Buffer ({len(buffer_trajectories)} episodes):")
    print(f"  V_naive:    {is_metrics['V_naive']:.5f}")
    print(f"  V_IS:       {is_metrics['V_IS']:.5f}")
    print(f"  V_step_IS:  {is_metrics['V_step_IS']:.5f}")
    print(f"  V_WIS:      {is_metrics['V_WIS']:.5f}")
    print(f"  V_step_WIS: {is_metrics['V_step_WIS']:.5f}")
    print("---------------------------------------")

    return is_metrics


def get_bc(env):


    device = torch.device(f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = (
        env.action_space.shape[0]
        if env.action_space.__class__.__name__ == "Box"
        else env.action_space.n
    )
    max_action = (
        float(env.action_space.high[0])
        if env.action_space.__class__.__name__ == "Box"
        else env.action_space.n + 1
    )
    suffix_parts = ['abiomed']

    suffix_parts.append(f"g1-0.0_g2-0.0_g3-0.0")
    save_path = "_".join(suffix_parts)

    bc_model_path = f"{args.save_path}SVR_bcmodels/bcmodel_" + save_path + ".pt"
    print("BC model path:", bc_model_path)
    behav = Actor(state_dim, action_dim, max_action)
   
    behav.load_state_dict(torch.load(bc_model_path))
    behav.to(device)
    behav.device = device
    behav.eval()
    return behav


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

    # IS tracking: per-step rewards and per-step action log-probs
    ep_rewards = []
    episodes_rewards = []
 


    # Reset environment
    obs, info = eval_env.reset(idx=1590) #100 1591
    all_states = info['all_states']
    all_states = np.concatenate([obs.reshape(1, -1), all_states], axis=0)

    behav = get_bc(eval_env)

    policy.eval()
    while num_episodes < episodes:
       
        # Compute target policy log-prob of the taken action for IS weights
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(args.device)
        with torch.no_grad():
            action = policy.sample_action(obs_tensor, deterministic=False)


        next_obs, reward, terminal, truncated, _ = eval_env.step(action)
        episode_reward += reward
        episode_length += 1
        ep_states.append(obs)
        ep_rewards.append(reward)
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

            # # Plot first episode if requested
            # if plot:
            #     # 
            #     all_state_unnorm = eval_env.world_model.unnorm_output(np.array(all_states).reshape(6+1, 6, -1)) 
            #     doctor_pl = (all_state_unnorm[:,:,-1].reshape(-1)).mean()   
                
            #     if  doctor_pl<4 and reward < -0.75: 
            #         print(doctor_pl, info['init_index'])   
            #         print('reward', reward, 'WS', ws, 'ACP', episode_acp_cost)    
            #         next_state_l = ep_states.copy()
            #         next_state_l.append(obs)
            #         plot_policy(eval_env, next_state_l[1:], all_states, args.algo_name.upper())
            #         # plot =False
            #     else:
            #         pass

            # Store IS data for this episode
            episodes_rewards.append(ep_rewards)

            # Reset for next episode
            episode_reward, episode_length = 0, 0
            num_episodes += 1
            obs, info = eval_env.reset()
            ep_states = []
            ep_rewards = []
            all_states = info['all_states']
            all_states = np.concatenate([obs.reshape(1, -1), all_states], axis=0)
    # Plot results
    # plot_score_histograms(acp_list, ws_list, rwd_list, args.algo_name)

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
        'mean_wean_score_thr': final_wean_thr_score
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
        choices=["mbpo", "mopo", "cormpo", "GORMPO-KDE", "GORMPO-VAE", "GORMPO-RealNVP", "GORMPO-Diffusion", "GORMPO-NeuralODE"],
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
    parser.add_argument("--policy_path", type=str, default="/abiomed/models/policy_models/mbpo_kde_acp_rebuttal/abiomed/seed_1_1006_024330-abiomed_mbpo_kde_acp_rebuttal/policy_abiomed.pth")
    parser.add_argument("--devid", type=int, default=7, help="Which GPU device index to use")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1,2,3,4,5], help="List of seeds for evaluation")
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--data_path", type=str, default=None)


    # Abiomed Environment Arguments
    parser.add_argument("--model_name", type=str, default="10min_1hr_all_data")
    parser.add_argument("--model_path_wm", type=str, default=None)
    parser.add_argument("--data_path_wm", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=6)

    # IS / behavior policy arguments
    parser.add_argument(
        "--save_path",
        type=str,
        default="/abiomed/models/policy_models/",
        help="Path to save model and results",
    )


    mopo_args(parser)


    # Apply config defaults and parse remaining arguments
    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)
    args.config = config_args.config
    print(f"Config: {args.config}")
    # Initialize wandb
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    wandb.init(
        project="mopo-eval",
        name=f"eval_{args.task}_{args.algo_name}_{t0}",
        config=vars(args)
    )
    results = []
    for seed in args.seeds:
        args.seed = seed
        # Set random seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        

        # Set device
        args.device = f'cuda:{args.devid}'

        # Create environment and policy
        env = get_env(args)
        policy = get_mopo(env, args)

        # Evaluate policy (model-based rollout)
        eval_info = _evaluate(policy, env, args.eval_episodes, args, plot=True)

        bc_actor = get_bc(env)
        state_dim = env.observation_space.shape[0]
        action_dim = (
            env.action_space.shape[0]
            if env.action_space.__class__.__name__ == "Box"
            else env.action_space.n
        )
        max_action = (
            float(env.action_space.high[0])
            if env.action_space.__class__.__name__ == "Box"
            else env.action_space.n + 1
        )
        device = args.device  
        obs_shape = env.observation_space.shape
        action_dim = np.prod(env.action_space.shape)     
        length_d, dataset = load_data(args.data_path, env)
        replay_buffer = ReplayBuffer(
                        buffer_size=length_d,
                        obs_shape=obs_shape,
                        obs_dtype=np.float32,
                        action_dim=action_dim,
                        action_dtype=np.float32 )
        replay_buffer.load_dataset(dataset, env)
        buffer_trajectories_dict =  replay_buffer.sample(1000)
        sample_std = 0.2
        is_info = evaluate_is_from_buffer(
            buffer_trajectories_dict, policy, bc_actor,
             2 * (sample_std)**2, args.gamma, args.device
        )

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
            'mean_wean_score_thr': eval_info["mean_wean_score_thr"],
        })


    # Save results to CSV
    os.makedirs(os.path.join('results', args.task, args.algo_name), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = os.path.join('results', args.task, args.algo_name, f"{args.task}_results_{t0}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")