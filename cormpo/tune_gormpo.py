import datetime
import os
import random
import importlib
import sys
import pickle
import ray
from ray import tune
from common.logger_offlinerlkit import Logger as OfflinerlkitLogger, make_log_dirs
import argparse

from mopo import get_args
import gym
import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transition_model import TransitionModel
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy
from algo.mopo import MOPO
from common.buffer import ReplayBuffer
from common.logger import Logger
from trainer import Trainer
from common.util import set_device_and_logger
from common import util
from realnvp_module.realnvp import RealNVP
from vae_module.vae import VAE
from cormpo.mbpo_kde.kde import PercentileThresholdKDE
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from abiomed_env.rl_env import AbiomedRLEnvFactory

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
            num_inference_steps=100,
            device=device,
        )

        # Normalize by target dimension for consistency
        log_probs_per_dim = log_probs

        return log_probs_per_dim


def run_exp(tune_config):
    # Access global args parsed from command line
    global args

    # Merge tune config into args (convert hyphens to underscores)
    args_for_exp = vars(args).copy()
    for k, v in tune_config.items():
        args_for_exp[k.replace("-", "_")] = v
    args_for_exp = argparse.Namespace(**args_for_exp)
    print(args_for_exp.task)

    # Ray Tune manages GPU assignment via CUDA_VISIBLE_DEVICES,
    # so the assigned GPU always appears as device 0 inside the worker.
    if torch.cuda.is_available():
        args_for_exp.devid = 0

    # Set up device and logger
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args_for_exp.seed}_{t0}_{args_for_exp.task.replace("-", "_")}_{args_for_exp.algo_name}'
    log_path = os.path.join(args_for_exp.logdir, args_for_exp.algo_name, args_for_exp.density_model, log_file)
    model_path = os.path.join(args_for_exp.model_path, args_for_exp.task.lower(), args_for_exp.density_model, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args_for_exp))
    logger = Logger(writer=writer, log_path=log_path)
    model_logger = Logger(writer=writer, log_path=model_path)

    Devid = args_for_exp.devid if args_for_exp.device == 'cuda' else -1
    set_device_and_logger(Devid, logger, model_logger)

    args_for_exp.device = util.device

    # Create environment
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
                device=args_for_exp.device  if torch.cuda.is_available() else "cpu"
            )
    # Load offline dataset
    buffer_len, dataset = load_data(args.data_path, env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)

    task = args_for_exp.task.split('-')[0]
    import_path = f"static_fns.{task.lower()}"
    static_fns = importlib.import_module(import_path).StaticFns
    config_path = f"configs.{task.lower()}"
    env_config = importlib.import_module(config_path).default_config

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

    actor = ActorProb(actor_backbone, dist, args_for_exp.device)
    critic1 = Critic(critic1_backbone, args_for_exp.device)
    critic2 = Critic(critic2_backbone, args_for_exp.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args_for_exp.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args_for_exp.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args_for_exp.critic_lr)

    if args_for_exp.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        args_for_exp.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=util.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args_for_exp.alpha_lr)
        args_for_exp.alpha = (target_entropy, log_alpha, alpha_optim)

    # create policy
    sac_policy = SACPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        dist=dist,
        tau=args_for_exp.tau,
        gamma=args_for_exp.gamma,
        alpha=args_for_exp.alpha,
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
    # create dynamics model
    dynamics_model = TransitionModel(obs_space=env.observation_space,
                                     action_space=env.action_space,
                                     static_fns=static_fns,
                                     lr=args_for_exp.dynamics_lr,
                                     classifier = classifier_dict,
                                     type = args_for_exp.penalty_type,
                                     reward_penalty_coef = args_for_exp.reward_penalty_coef,
                                     **env_config["transition_params"]
                                     )


    offline_buffer = ReplayBuffer(
    buffer_size=len(dataset["observations"]),
    obs_shape=args_for_exp.obs_shape,
    obs_dtype=np.float32,
    action_dim=args_for_exp.action_dim,
    action_dtype=np.float32
    )

    offline_buffer.load_dataset(dataset)

    model_buffer = ReplayBuffer(
        buffer_size=args_for_exp.rollout_batch_size * args_for_exp.rollout_length * args_for_exp.model_retain_epochs,
        obs_shape=args_for_exp.obs_shape,
        obs_dtype=np.float32,
        action_dim=args_for_exp.action_dim,
        action_dtype=np.float32
    )

    # create MOPO algo
    algo = MOPO(
        sac_policy,
        dynamics_model,
        offline_buffer=offline_buffer,
        model_buffer=model_buffer,
        reward_penalty_coef=args_for_exp.reward_penalty_coef,
        rollout_length=args_for_exp.rollout_length,
        batch_size=args_for_exp.batch_size,
        real_ratio=args_for_exp.real_ratio,
        logger=logger,
        **env_config["mopo_params"]
    )

    # log
    record_params = list(tune_config.keys())
    if "seed" in record_params:
        record_params.remove("seed")
    log_dirs = make_log_dirs(
        args_for_exp.task,
        args_for_exp.algo_name,
        args_for_exp.seed,
        vars(args_for_exp),
        record_params=record_params
    )
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    offlinerlkit_logger = OfflinerlkitLogger(log_dirs, output_config)
    offlinerlkit_logger.log_hyperparameters(vars(args_for_exp))

    # create trainer
    trainer = Trainer(
        algo,
        eval_env=env,
        epoch=args_for_exp.epoch,
        step_per_epoch=args_for_exp.step_per_epoch,
        rollout_freq=args_for_exp.rollout_freq,
        logger=logger,
        log_freq=args_for_exp.log_freq,
        run_id=0,
        env_name=args_for_exp.task,
        eval_episodes=args_for_exp.eval_episodes,

    )

    if args_for_exp.dynamics_model_dir != None:
        print(f"Loading dynamics model")
        dynamics_model.load_model([args_for_exp.task, args_for_exp.data_path])
        print("Dynamics model loaded.")
    else:
        print("Training dynamics model from scratch.")
        # pretrain dynamics model on the whole dataset
        trainer.train_dynamics()

    # begin train
    result = trainer.train_policy()
    tune.report(**result)



if __name__ == "__main__":

    
    # load default args
    args = get_args()
    # args.device = util.device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.devid},{args.devid+5}" # Let Ray handle GPU assignment, but ensure we have 2 GPUs available
    ray.init(num_gpus=2)
    config = {}
    penalty_coef = [0.1,0.2,0.3, 0.4, 0.5, 0.6,0.7, 0.8]
    # penalty_coef = [0.05, 0.1]
    seeds = list(range(1))
    config["reward_penalty_coef"] = tune.grid_search(penalty_coef)
    config["seed"] = tune.grid_search(seeds)

    analysis = tune.run(
        run_exp,
        name="tune_gormpo",
        config=config,
        resources_per_trial={
            "gpu": 0.25
        }
    )
