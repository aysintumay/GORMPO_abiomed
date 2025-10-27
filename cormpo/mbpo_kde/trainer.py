
import time
import sys
import os
import wandb
import numpy as np
import torch
from matplotlib import pyplot as plt
import copy 
from tqdm import tqdm
from common import util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers.evaluate import _evaluate as _evaluate

def plot_accuracy(mean_acc, std_acc, name=''):
    epochs = np.arange(mean_acc.shape[0])

    fig, ax = plt.subplots(figsize=(8, 5.8), dpi=300)
    ax.plot(epochs, mean_acc, label=f'{name }')
    ax.fill_between(epochs, mean_acc - std_acc/2, mean_acc + std_acc/2, alpha=0.5, label='± 1/2 Std')
    ax.set_xlabel('time')
    ax.set_ylabel(f'{name}')
    ax.set_title(f'{name} Over Epochs')
    ax.legend()
    wandb.log({f"{name}": wandb.Image(fig)})


def plot_p_loss(critic1,name=''):

    epochs = np.arange(critic1.shape[0])

    mean_c1 = critic1.mean(axis=1)
    std_c1 = critic1.std(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5.8), dpi=300)
    ax.plot(epochs, mean_c1, label=f'{name} Loss')
    ax.fill_between(epochs, mean_c1 - std_c1/2, mean_c1 + std_c1/2, alpha=0.3, label='± 1/2 Std')
    ax.set_xlabel('time')
    ax.set_ylabel('Loss')
    ax.set_title(f'{name} Loss Over Time')
    ax.legend()
    wandb.log({f"{name} Loss": wandb.Image(fig)})


def plot_q_value(q1, name=''):


    epochs = np.arange(q1.shape[0])

    mean_c1 = q1.mean(axis=1)
    std_c1 = q1.std(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5.8), dpi=300)
    ax.plot(epochs, mean_c1, label=f'{name} Value')
    ax.fill_between(epochs, mean_c1 - std_c1/2, mean_c1 + std_c1/2, alpha=0.3, label='± 1/2 Std')
    ax.set_xlabel('time')
    ax.set_ylabel('Loss')
    ax.set_title(f'{name} Value Over Time')
    ax.legend()
    wandb.log({f"{name} Value": wandb.Image(fig)})


class Trainer:
    def __init__(
        self,
        algo,
        eval_env,
        epoch,
        step_per_epoch,
        rollout_freq,
        logger,
        log_freq,
        run_id,
        env_name = '',
        eval_episodes=10,
        terminal_counter=1
        
    ):
        self.algo = algo
        self.eval_env = eval_env

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._rollout_freq = rollout_freq

        self.logger = logger
        self._log_freq = log_freq
        self.run_id = run_id

        self.env_name = env_name

        self._eval_episodes = eval_episodes
        self.terminal_counter = terminal_counter

        if self.run_id !=0 :

            run = wandb.init(project="abiomed",
                    id=self.run_id,
                    resume="allow"
                    )
            
    def train_dynamics(self):
        start_time = time.time()
        self.algo.learn_dynamics()
        self.algo.save_dynamics_model(f"dynamics_model")
        self.logger.print("total time: {:.3f}s".format(time.time() - start_time))

    def train_policy(self):
        start_time = time.time()
        num_timesteps = 0
        # train loop
        q1_l, q2_l,q_l = [], [], []
        critic_loss1,critic_loss2,  actor_loss, entropy,alpha_loss = [], [],[], [], []
        reward_l, acc_l, off_acc = [], [], []
        reward_std_l, acc_std_l, off_acc_std = [], [], []
        for e in range(1, self._epoch + 1):
            self.algo.policy.train()
            with tqdm(total=self._step_per_epoch, desc=f"Epoch #{e}/{self._epoch}") as t:
                while t.n < t.total:
                    if num_timesteps % self._rollout_freq == 0:
                        self.algo.rollout_transitions()
                       
                    # update policy by sac
                    loss,q_values = self.algo.learn_policy()
                    q1_l.append(q_values['q1'])
                    q2_l.append(q_values['q2'])
                    q_l.append(q_values['q_target'])
                    critic_loss1.append(loss["loss/critic1"])
                    critic_loss2.append(loss["loss/critic2"])
                    actor_loss.append(loss["loss/actor"])
                    entropy.append(loss["entropy"])
                    alpha_loss.append(loss["loss/alpha"])
                    t.set_postfix(**loss)
                    # log
                    if num_timesteps % self._log_freq == 0:
                        for k, v in loss.items():
                            self.logger.record(k, v, num_timesteps, printed=False)
                    num_timesteps += 1
                    t.update(1)
            # evaluate current policy
            if e % 10 == 0:
               
                eval_info = self._evaluate()
                ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
                ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
                
           
                reward_l.append(ep_reward_mean)
                reward_std_l.append(ep_reward_std)
               
                self.logger.record("eval/episode_reward", ep_reward_mean, num_timesteps, printed=False)
                self.logger.record("eval/episode_length", ep_length_mean, num_timesteps, printed=False)
            
                self.logger.print(f"Epoch #{e}: episode_reward: {ep_reward_mean:.3f} ± {ep_reward_std:.3f},\
                                episode_length: {ep_length_mean:.3f} ± {ep_length_std:.3f}"
                                )
        
            # save policy
            model_save_dir = util.logger_model.log_path
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            policy_copy = copy.deepcopy(self.algo.policy)
            torch.save(policy_copy.to('cpu').state_dict(), os.path.join(model_save_dir, f"policy_{self.env_name}.pth")) 
        
        if self.run_id != 0:
            #plot q_values for each epoch
            plot_q_value(np.array(q1_l).reshape(-1,1), 'Q1')
            plot_q_value(np.array(q2_l).reshape(-1,1), 'Q2')
            plot_q_value(np.array(q_l).reshape(-1,1), 'Q')

            plot_p_loss(np.array(critic_loss1).reshape(-1,1), 'Critic1')
            plot_p_loss(np.array(critic_loss2).reshape(-1,1), 'Critic2')
            plot_p_loss(np.array(actor_loss).reshape(-1,1), 'Actor')
            plot_p_loss(np.array(entropy).reshape(-1,1), 'Entropy')
            plot_p_loss(np.array(alpha_loss).reshape(-1,1), 'Alpha')

            plot_accuracy(np.array(reward_l), np.array(reward_std_l)/self._eval_episodes, 'Average Return')
          
        self.logger.print("total time: {:.3f}s".format(time.time() - start_time))




    def _evaluate(self):
        self.algo.policy.eval()
        obs, _ = self.eval_env.reset(idx=100)
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = self.algo.policy.sample_action(obs, deterministic=True)
            next_obs, reward, terminal, truncated, _= self.eval_env.step(action) 
            episode_reward += reward
            episode_length += 1

            obs = next_obs  

            if terminal or truncated:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )

                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs, _ = self.eval_env.reset()

        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }