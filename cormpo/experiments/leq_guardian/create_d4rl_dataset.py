"""Dump a D4RL task's offline dataset to a pickle, in the flat dict format
common/buffer.py's generic branch and mbpo_kde/kde.py's generic branch both expect.

    python create_d4rl_dataset.py --task antmaze-medium-play-v2 --save_path data/antmaze.pkl
"""
import argparse
import pickle

import gym
import d4rl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    env = gym.make(args.task)
    dataset = d4rl.qlearning_dataset(env)
    with open(args.save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved {len(dataset['observations'])} transitions to {args.save_path}")
