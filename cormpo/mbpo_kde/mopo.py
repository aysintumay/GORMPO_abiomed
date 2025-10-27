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
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

from train import train
from cormpo.helpers.evaluate import _evaluate as evaluate
from common.logger import Logger
from common.util import set_device_and_logger
from abiomed_env.rl_env import AbiomedRLEnvFactory

warnings.filterwarnings("ignore")


def get_args():
    """
    Parse command-line arguments and configuration file for MOPO/CORMPO training.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    print("Running", __file__)

    # Parse config file
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="")
    config_args, remaining_argv = config_parser.parse_known_args()

    if config_args.config:
        with open(config_args.config, "r") as f:
            config = yaml.safe_load(f)
            config = {k.replace("-", "_"): v for k, v in config.items()}
    else:
        config = {}

    # Main argument parser
    parser = argparse.ArgumentParser(parents=[config_parser])

    # General arguments
    parser.add_argument("--algo-name", type=str, default="mbpo_kde")
    parser.add_argument("--policy_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="/abiomed/models/policy_models")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--devid", type=int, default=0, help="Which GPU device index to use")

    parser.add_argument("--task", type=str, default="abiomed")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1])

    # SAC hyperparameters
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=True)
    parser.add_argument('--target-entropy', type=int, default=-1)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)

    # Dynamics model arguments
    parser.add_argument("--dynamics-lr", type=float, default=0.001)
    parser.add_argument("--n-ensembles", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--reward-penalty-coef", type=float, default=0.5)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--rollout-batch-size", type=int, default=10000)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--dynamics-model-dir", type=str, default=None)
    parser.add_argument("--penalty_type", type=str, default="linear",
                        choices=["linear", "inverse", "exponential", "softplus"])
    parser.add_argument("--classifier_model_name", type=str, default="abiomed/trained_kde_abiomed")

    # MBPO training arguments
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--terminal_counter", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--root-dir', default='log', help='root dir')

    # Abiomed environment arguments
    parser.add_argument("--model_name", type=str, default="10min_1hr_all_data")
    parser.add_argument("--model_path_wm", type=str, default=None)
    parser.add_argument("--data_path_wm", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--gamma1", type=float, default=0.0)
    parser.add_argument("--gamma2", type=float, default=0.0)
    parser.add_argument("--gamma3", type=float, default=0.0)
    parser.add_argument("--noise_rate", type=float, default=0.0,
                        help="Portion of data to be noisy with probability")
    parser.add_argument("--noise_scale", type=float, default=0.0,
                        help="Magnitude of noise")

    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)
    args.config = config_args.config
    print(args.config)

    return args


def main(args):
    """
    Main training loop for MOPO/CORMPO algorithm.

    Args:
        args: Parsed arguments from get_args()
    """
    # Initialize wandb
    run = wandb.init(
        project=args.task,
        group=args.algo_name,
        config=vars(args),
    )

    results = []
    for seed in args.seeds:
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.device != "cpu":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Adjust save path based on task and data configuration
        taskname = args.task
        if args.task == "abiomed":
            if args.noise_rate > 0:
                taskname += f"_nr{args.noise_rate}_ns{args.noise_scale}"
            if args.data_path and "5000eps" in args.data_path:
                taskname += "_5000eps"
            if args.data_path and "200000eps" in args.data_path:
                taskname += "_200000eps"

        # Create logging and model directories
        t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
        log_file = f'seed_{seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
        log_path = os.path.join(args.logdir, taskname, args.algo_name, log_file)
        model_path = os.path.join(args.model_path, args.algo_name, taskname, log_file)

        # Setup loggers
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        logger = Logger(writer=writer, log_path=log_path)
        model_logger = Logger(writer=writer, log_path=model_path)

        # Set device
        devid = args.devid if args.device == 'cuda' else -1
        set_device_and_logger(devid, logger, model_logger)
        args.model_path = model_path

        # Create environment
        if args.task == 'abiomed':
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
                device=f"cuda:{devid}" if torch.cuda.is_available() else "cpu"
            )

        # Train policy
        policy, trainer = train(env, run, logger, args)
        trainer.algo.save_dynamics_model("dynamics_model")

        # Evaluate policy
        eval_res = evaluate(policy, env, args.eval_episodes, args=args, plot=True)
        eval_res['seed'] = seed
        results.append(eval_res)

    # Save results to CSV
    os.makedirs(os.path.join('results', taskname, args.algo_name), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = os.path.join('results', taskname, args.algo_name, f"{args.task}_results_{t0}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    wandb.finish()


if __name__ == "__main__":
    main(args=get_args())
