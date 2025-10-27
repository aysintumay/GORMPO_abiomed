import os
import sys
import pickle
import importlib

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transition_model import TransitionModel
from kde import PercentileThresholdKDE
from trainer import Trainer
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy
from algo.mopo import MOPO
from common.buffer import ReplayBuffer
from common import util


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


def train(env, run, logger, args):
    """
    Train a MOPO/CORMPO policy using offline data and learned dynamics model.

    Args:
        env: Environment instance for evaluation
        run: Wandb run object for logging (can be None)
        logger: Logger instance for tracking metrics
        args: Training arguments containing hyperparameters and paths

    Returns:
        tuple: (trained_sac_policy, trainer) containing the trained policy and trainer instance
    """
    # Load offline dataset
    buffer_len, dataset = load_data(args.data_path, env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)

    # Import task-specific configurations
    task = args.task.split('-')[0]
    import_path = f"static_fns.{task}"
    static_fns = importlib.import_module(import_path).StaticFns
    config_path = f"config.{task}"
    config = importlib.import_module(config_path).default_config

    # Create policy model components
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )

    actor = ActorProb(actor_backbone, dist, util.device)
    critic1 = Critic(critic1_backbone, util.device)
    critic2 = Critic(critic2_backbone, util.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # Configure automatic entropy tuning if enabled
    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=util.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    # Create SAC policy
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
        device=util.device
    )

    # Load KDE density estimator for uncertainty penalty
    classifier_dict = PercentileThresholdKDE.load_model(
        args.classifier_model_name,
        use_gpu=True,
        devid=args.devid
    )

    # Create dynamics model with uncertainty penalty
    dynamics_model = TransitionModel(
        obs_space=env.observation_space,
        action_space=env.action_space,
        static_fns=static_fns,
        lr=args.dynamics_lr,
        classifier=classifier_dict,
        type=args.penalty_type,
        reward_penalty_coef=args.reward_penalty_coef,
        **config["transition_params"]
    )

    # Create offline data buffer
    offline_buffer = ReplayBuffer(
        buffer_size=buffer_len,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )
    # Load dataset (handles conversion from non-RL format if needed)
    offline_buffer.load_dataset(dataset, env)

    # Create model-generated rollout buffer
    model_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size * args.rollout_length * args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )

    # Create MOPO algorithm
    algo = MOPO(
        sac_policy,
        dynamics_model,
        offline_buffer=offline_buffer,
        model_buffer=model_buffer,
        reward_penalty_coef=args.reward_penalty_coef,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        logger=logger,
        **config["mopo_params"]
    )

    # Create trainer
    trainer = Trainer(
        algo,
        eval_env=env,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        rollout_freq=args.rollout_freq,
        logger=logger,
        log_freq=args.log_freq,
        run_id=run.id if run is not None else 0,
        env_name=args.task,
        eval_episodes=args.eval_episodes,
        terminal_counter=args.terminal_counter if args.task == "Abiomed-v0" else None,
    )

    # Pretrain dynamics model on offline data
    trainer.train_dynamics()

    # Train policy using MOPO
    trainer.train_policy()

    return sac_policy, trainer