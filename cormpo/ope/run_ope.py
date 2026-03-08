# implement a basic OPE procedure using SCOPE-RL
import os
import sys
import random
import argparse
import datetime
import warnings

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
# import SCOPE-RL modules
from scope_rl.ope import CreateOPEInput
from scope_rl.ope import OffPolicyEvaluation as OPE
from scope_rl.ope.continuous import DirectMethod as DM
from scope_rl.ope.continuous import TrajectoryWiseImportanceSampling as TIS
from scope_rl.ope.continuous import PerDecisionImportanceSampling as PDIS
from scope_rl.ope.continuous import DoublyRobust as DR
import io
from d3rlpy.base import LearnableConfigWithShape
from d3rlpy.ope import FQE as ContinuousFQE

from IS.bc import Actor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.buffer import ReplayBuffer
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from ope.IS.traditional_is import TraditionalIS
from algo.sac import SACPolicy
from ope.sac_scope_rl_wrapper import SACPolicyWrapper
# from helpers.plotter import plot_policy, plot_score_histograms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..',)))

from abiomed_env.rl_env import AbiomedRLEnvFactory


warnings.filterwarnings("ignore")
random_state=42

def load_data(data_path, env):
    """
    Load offline dataset from file or environment's world model.

    Args:
        data_path: Path to pickle/npz file containing dataset, or None to use env data
        env: Environment instance with world_model containing data

    Returns:
        tuple: (dataset_length, data) where data is dict or list of datasets
    """
    if data_path is not None:
        try:
            with open(data_path, "rb") as f:
                data = pickle.load(f)
                print('Opened pickle file for synthetic dataset')
        except Exception:
            dataset = np.load(data_path)
            data = {k: dataset[k] for k in dataset.files}
            print('Opened npz file for synthetic dataset')
        length = len(data['observations'])
    else:
        dataset1 = env.world_model.data_train
        dataset2 = env.world_model.data_val
        dataset3 = env.world_model.data_test
        data = [dataset1, dataset2, dataset3]
        length = len(dataset1.data) + len(dataset2.data) + len(dataset3.data)
    return length, data 

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

def monte_carlo_policy_value(env, policy, n_episodes: int, gamma: float, device: str, seed: int = None) -> float:
    """Roll out the evaluation policy in the environment and return the mean discounted return.

    Parameters
    ----------
    env : gym-compatible environment
    policy : SACPolicy — must implement sample_action(obs_tensor, deterministic=True)
    n_episodes : int — number of independent rollouts
    gamma : float — discount factor
    device : str — torch device string for the policy
    seed : int — seed for reproducible index selection across episodes

    Returns
    -------
    float : mean discounted cumulative reward across n_episodes episodes
    """
    total_size = (len(env.world_model.data_train) +
                  len(env.world_model.data_val) +
                  len(env.world_model.data_test) - env.max_steps)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, total_size, size=n_episodes)
    returns = []
    for id in indices:
        obs, _ = env.reset(idx=int(id))
        done = False
        ep_return = 0.0
        t = 0
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy.sample_action(obs_t, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += (gamma ** t) * reward
            t += 1
            done = terminated or truncated
        returns.append(ep_return)
    return float(np.mean(returns))


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
    parser.add_argument("--fqe_dir", type=str,
                        default='/public/gormpo/ope/fqe_models/',
                        help="Directory to save/load trained FQE models (one per seed). "
                             "If None, FQE is retrained every run.")
    parser.add_argument("--on_policy_only", action="store_true",
                        help="Skip OPE/FQE and only run Monte Carlo on-policy evaluation.")


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

        # --- Monte Carlo on-policy evaluation --------------------------------
        print(f"[seed {seed}] Running Monte Carlo evaluation ({args.eval_episodes} episodes)...")
        mc_value = monte_carlo_policy_value(
            env, policy, n_episodes=args.eval_episodes, gamma=args.gamma, device=args.device, seed=seed
        )
        print(f"[seed {seed}] Monte Carlo value: {mc_value:.6f}")
        wandb.log({f"seed{seed}/mc_value": mc_value})
        results.append({"seed": seed, "mc_value": mc_value})

        if args.on_policy_only:
            continue

        evaluation_policies = [SACPolicyWrapper(policy, name=f"sac_seed{seed}")]
        bc_actor = get_bc(env)
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
        buffer_trajectories_dict = replay_buffer.sample(1200)

        # Convert buffer transitions to scope_rl logged_dataset format.
        # Each sampled transition is treated as a single-step trajectory because
        # replay_buffer.sample() draws random indices without preserving episode order.
        _n = len(buffer_trajectories_dict["observations"])
        _state_dim = buffer_trajectories_dict["observations"].shape[1]
        _action_dim = buffer_trajectories_dict["actions"].shape[1]

        # Compute pscores: model pi_b(a|s) as N(bc_actor(s), _bc_std^2 * I).
        # pscore[i, d] = N(a_d; mu_d, sigma^2) — per-dimension density required by scope_rl.
        _bc_std = 1
        _obs_t = torch.FloatTensor(buffer_trajectories_dict["observations"]).to(args.device)
        _act_t = torch.FloatTensor(buffer_trajectories_dict["actions"]).to(args.device)
        with torch.no_grad():
            _mu_bc = bc_actor(_obs_t)  # (N, action_dim)
            _pscore = (
                (1.0 / (_bc_std * np.sqrt(2 * np.pi))) *
                torch.exp(-0.5 * ((_act_t - _mu_bc) / _bc_std) ** 2)
            ).cpu().numpy().astype(np.float32)  # (N, action_dim)

        _step_per_traj = 6  # matches env max_steps
        _n_traj = _n // _step_per_traj  # number of complete trajectories
        # terminal=True at the last step of each trajectory; done=False (time-limit, not absorption)
        _terminal = np.zeros(_n, dtype=bool)
        _terminal[_step_per_traj - 1::_step_per_traj] = True

        test_logged_dataset = {
            "size": _n_traj * _step_per_traj,
            "n_trajectories": _n_traj,
            "step_per_trajectory": _step_per_traj,
            "action_type": "continuous",
            "n_actions": None,
            "action_dim": _action_dim,
            "action_meaning": None,
            "action_keys": None,
            "state_dim": _state_dim,
            "state_keys": None,
            "state": buffer_trajectories_dict["observations"][:_n_traj * _step_per_traj].astype(np.float32),
            "action": buffer_trajectories_dict["actions"][:_n_traj * _step_per_traj].astype(np.float32),
            "reward": buffer_trajectories_dict["rewards"].flatten()[:_n_traj * _step_per_traj].astype(np.float32),
            "done": _terminal[:_n_traj * _step_per_traj],
            "terminal": _terminal[:_n_traj * _step_per_traj],
            "info": {},
            "pscore": _pscore[:_n_traj * _step_per_traj],
            "behavior_policy": "data_collection_policy",
            "dataset_id": 0,
        }

        # create input for the OPE class
        prep = CreateOPEInput(
            env=env,
        )

        # Load pre-trained FQE if available, so obtain_whole_inputs skips fitting.
        # NOTE: prep.fqe does not exist until _register_logged_dataset is called
        # (which happens inside obtain_whole_inputs) and that call also resets
        # self.fqe = {} unconditionally.  We therefore patch the instance method
        # so the pre-loaded model is injected right after the reset, before
        # build_and_fit_FQE checks whether the policy is already present.
        policy_name = evaluation_policies[0].name
        fqe_path = None
        pre_loaded_fqe = None
        if args.fqe_dir is not None:
            fqe_dir = Path(args.fqe_dir)
            fqe_dir.mkdir(parents=True, exist_ok=True)
            fqe_path = fqe_dir / f"fqe_seed{seed}.d3"
            if fqe_path.exists():
                print(f"[seed {seed}] Loading pre-trained FQE from {fqe_path}")
                with open(str(fqe_path), "rb") as _f:
                    _fqe_obj = pickle.load(_f)
                _cfg = LearnableConfigWithShape.deserialize(_fqe_obj["config"])
                pre_loaded_fqe = ContinuousFQE(
                    algo=evaluation_policies[0],
                    config=_cfg.config,
                    device=args.device,
                )
                pre_loaded_fqe.create_impl(_cfg.observation_shape, _cfg.action_size)
                pre_loaded_fqe._impl.load_model(io.BytesIO(_fqe_obj["torch"]))

        if pre_loaded_fqe is not None:
            _orig_register = prep._register_logged_dataset
            def _patched_register(logged_dataset, _fqe=pre_loaded_fqe, _name=policy_name):
                _orig_register(logged_dataset)
                prep.fqe[_name] = [_fqe]
            prep._register_logged_dataset = _patched_register

        input_dict = prep.obtain_whole_inputs(
            logged_dataset=test_logged_dataset,
            evaluation_policies=evaluation_policies,
            require_value_prediction=True,
            n_trajectories_on_policy_evaluation=10000,
            random_state=random_state,
        )

        # Save the FQE model after training (only if not already loaded from disk).
        if fqe_path is not None and not fqe_path.exists():
            print(f"[seed {seed}] Saving trained FQE to {fqe_path}")
            prep.fqe[policy_name][0].save(str(fqe_path))
        # initialize the OPE class
        ope = OPE(
            logged_dataset=test_logged_dataset,
            ope_estimators=[DM(), TIS(), PDIS(), DR()],
        )

        # --- per-seed figure -------------------------------------------------
        fig_dir = Path("../results/ope_plots")
        fig_dir.mkdir(parents=True, exist_ok=True)
        ope.visualize_off_policy_estimates(
            input_dict,
            random_state=random_state,
            sharey=True,
            fig_dir=fig_dir,
            fig_name=f"estimated_policy_value_seed{seed}.png",
        )

        # --- OPE estimates ---------------------------------------------------
        policy_value_dict = ope.estimate_policy_value(input_dict)
        policy_key = f"sac_seed{seed}"
        ope_estimates = policy_value_dict.get(policy_key, {})

        # mc_value = ope_estimates['on_policy']  # on-policy from SCOPE-RL uses fixed random_state=42, not per-seed
        # --- comparison table ------------------------------------------------
        print(f"\n{'='*55}")
        print(f"  OPE vs Monte Carlo — seed {seed}")
        print(f"{'='*55}")
        print(f"  {'Method':<20}  {'Estimated Value':>15}")
        print(f"  {'-'*20}  {'-'*15}")
        for method, val in ope_estimates.items():
            if val is not None:
                print(f"  {method:<20}  {val:>15.6f}")
        print(f"  {'monte_carlo (true)':<20}  {mc_value:>15.6f}")
        print(f"{'='*55}\n")

        # --- log to wandb ----------------------------------------------------
        seed_log = {f"seed{seed}/mc_value": mc_value}
        for method, val in ope_estimates.items():
            if val is not None:
                seed_log[f"seed{seed}/ope_{method}"] = val
                seed_log[f"seed{seed}/error_{method}"] = abs(val - mc_value)
        wandb.log(seed_log)

        results[-1].update({f"ope_{k}": v for k, v in ope_estimates.items() if v is not None})

    # --- relative MSE averaged over seeds -----------------------------------
    results_df = pd.DataFrame(results)
    ope_cols = [c for c in results_df.columns if c.startswith("ope_")]
    rel_mse = {}

    for col in ope_cols:
        method = col[len("ope_"):]
        mean_true = results_df["mc_value"].mean()
        rel_mse[method] = float(((results_df[col] - mean_true)**2 / mean_true ** 2).mean())

    print(f"\n{'='*55}")
    print("  Relative MSE (averaged over seeds)")
    print(f"{'='*55}")
    print(f"  {'Method':<20}  {'Rel. MSE':>12}")
    print(f"  {'-'*20}  {'-'*12}")
    for method, val in rel_mse.items():
        print(f"  {method:<20}  {val:>12.6f}")
    print(f"{'='*55}\n")

    wandb.log({f"rel_mse/{method}": val for method, val in rel_mse.items()})
    wandb.finish()