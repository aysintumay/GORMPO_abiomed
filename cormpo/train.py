import os
import sys
import pickle
import importlib

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from cormpo.transition_model import TransitionModel
from cormpo.mbpo_kde.kde import PercentileThresholdKDE
from cormpo.trainer import Trainer
from cormpo.models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from cormpo.algo.sac import SACPolicy
from cormpo.algo.mopo import MOPO
from cormpo.common.buffer import ReplayBuffer
from cormpo.common import util
from cormpo.vae_module.vae import VAE
from cormpo.realnvp_module.realnvp import RealNVP


from cormpo.neuralode_module.neural_ode_density import ContinuousNormalizingFlow, ODEFunc
from cormpo.neuralode_module.neural_ode_ood import NeuralODEOOD
from cormpo.diffusion_module.test_diffusion_ood import load_model_from_checkpoint
from cormpo.diffusion_module.diffusion_density import log_prob_elbo
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import d4rl
from typing import Tuple
import torch
from torch import nn
# from cormpo.diffusion_module.diffusion_density import (
#     UnconditionalEpsilonMLP,
#     UnconditionalEpsilonTransformer,
# )


class DiffusionDensityWrapper:
    """Wrapper for diffusion model to provide score_samples interface."""

    def __init__(self, model, scheduler, target_dim, device):
        self.model = model
        self.scheduler = scheduler
        self.target_dim = target_dim
        self.device = device

    @torch.no_grad()
    def score_samples(self, x, device=None):
        """
        Compute log probability using ELBO from unconditional diffusion model.

        Args:
            x: Input samples (numpy array or tensor) of shape (batch_size, target_dim)
            device: Device to use (optional)

        Returns:
            Log probabilities normalized by target dimension
        """
        if device is None:
            device = self.device

        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        x = x.to(device)

        # Compute log probability using ELBO
        log_probs = log_prob_elbo(
            model=self.model,
            scheduler=self.scheduler,
            x0=x,
            num_inference_steps=50,
            device=device,
        )

        # Normalize by target dimension for consistency
        log_probs_per_dim = log_probs

        return log_probs_per_dim

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

    if "vae" in args.classifier_model_name:
        classifier = VAE(
            # hidden_dims= args.vae_hidden_dims,
            device=util.device
        ).to(util.device)
        classifier_dict = classifier.load_model(args.classifier_model_name, hidden_dims=[256,256])
        print("vae laoded")
    elif "realnvp" in args.classifier_model_name:
        classifier = RealNVP(
        device=util.device
        ).to(util.device)
        classifier_dict = classifier.load_model(args.classifier_model_name)
    elif "kde" in args.classifier_model_name:
        classifier_dict = PercentileThresholdKDE.load_model(
            args.classifier_model_name,
            use_gpu=True,
            devid=args.devid
        )
    elif "neuralODE" in args.classifier_model_name:
        print("Loading Neural ODE based classifier... for task:", args.task)
        # Use the new NeuralODEOOD.load_model interface
        device = f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu"

        # Load model using NeuralODEOOD wrapper
        classifier_dict = NeuralODEOOD.load_model(
            save_path=args.classifier_model_name.replace('_model.pt', ''),
            target_dim=args.target_dim,
            hidden_dims=(512, 512),
            activation="silu",
            time_dependent=True,
            solver="dopri5",
            t0=0.0,
            t1=1.0,
            rtol=1e-5,
            atol=1e-5,
            device=device
        )
        # classifier_dict now contains: {'model': ood_model, 'threshold': ..., 'mean': ..., 'std': ...}
        # Rename 'threshold' to 'thr' for compatibility with transition_model
        classifier_dict['thr'] = classifier_dict['threshold']
    elif "diffusion" in args.classifier_model_name:
        print("Loading Diffusion based classifier... for task:", args.task)
        device = f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu"
        ckpt_path = args.classifier_model_name

        # Load checkpoint
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        # Extract metadata safely
        target_dim = ckpt.get("target_dim", args.target_dim)
        threshold = ckpt.get("threshold", None)
        cfg_dict = ckpt.get("cfg", {})

        print(f"Checkpoint metadata:")
        print(f"  - target_dim: {target_dim}")
        print(f"  - threshold: {threshold}")
        print(f"  - config keys: {list(cfg_dict.keys())}")

        # Build model
        model, cfg = load_model_from_checkpoint(ckpt_path, device)

        # Determine scheduler directory
        sched_dir = f"/public/gormpo/models/{args.task.lower().split('_')[0].split('-')[0]}/diffusion/scheduler/scheduler_config.json"

        # Load scheduler with fallback options
        scheduler = None
        if os.path.exists(sched_dir):
            try:
                scheduler = DDIMScheduler.from_pretrained(sched_dir)
                print(f"✓ Loaded DDIMScheduler from {sched_dir}")
            except Exception as e:
                print(f"Warning: Failed to load DDIMScheduler: {e}")
                try:
                    scheduler = DDPMScheduler.from_pretrained(sched_dir)
                    print(f"✓ Loaded DDPMScheduler from {sched_dir}")
                except Exception as e2:
                    print(f"Warning: Failed to load DDPMScheduler: {e2}")
        else:
            print(f"Warning: Scheduler directory not found: {sched_dir}")

        # Final fallback to default scheduler
        if scheduler is None:
            print("Using default DDIMScheduler configuration")
            scheduler = DDIMScheduler(
                num_train_timesteps=cfg_dict.get('num_train_timesteps', 1000),
                beta_schedule="linear",
                prediction_type="epsilon",
            )

        # Wrap in our interface
        diffusion_wrapper = DiffusionDensityWrapper(model, scheduler, target_dim, device)

        # Load threshold with multiple fallback options
        thr = None

        # Option 1: From checkpoint metadata
        if threshold is not None:
            thr = threshold
            print(f"✓ Loaded threshold from checkpoint: {thr}")

        # Option 2: From metrics file
        if thr is None:
            thr_path = f"diffusion/monte_carlo_results/{args.task.lower().split('_')[0].split('-')[0]}_unconditional_ddpm/elbo_metrics.json"
            if os.path.exists(thr_path):
                try:
                    with open(thr_path, 'r') as f:
                        metrics = json.load(f)
                    thr = metrics.get("percentile_1.0_logp", None)
                    if thr is not None:
                        print(f"✓ Loaded threshold from metrics file: {thr}")
                except Exception as e:
                    print(f"Warning: Failed to load threshold from {thr_path}: {e}")

        # Option 3: Default value
        if thr is None:
            print("Warning: No threshold found in checkpoint or metrics file, using default 0.0")
            thr = 0.0

        classifier_dict = {'model': diffusion_wrapper, 'thr': thr}

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

    dynamics_model.load_model(args.task) 

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
    # trainer.train_dynamics()

    # Train policy using MOPO
    trainer.train_policy()

    return sac_policy, trainer