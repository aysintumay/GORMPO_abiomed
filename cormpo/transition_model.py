import os
import sys
from operator import itemgetter
from copy import deepcopy

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cormpo.common import util, functional
from cormpo.common.normalizer import StandardNormalizer
from cormpo.models.ensemble_dynamics import EnsembleModel


class TransitionModel:
    def __init__(self,
                 obs_space,
                 action_space,
                 static_fns,
                 lr,
                 classifier = None,
                 type = "linear",
                 holdout_ratio=0.1,
                 reward_penalty_coef = 1,
                 inc_var_loss=False,
                 use_weight_decay=False,
                 **kwargs):

        obs_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]

        self.device = util.device
        self.model = EnsembleModel(obs_dim=obs_dim, action_dim=action_dim, device=util.device, **kwargs['model'])
        self.static_fns = static_fns
        self.lr = lr
        self.classifier_model = classifier['model']
        self.classifier_thr= classifier['thr']
        self.classifier_name = classifier['name'] if 'name' in classifier else None
        self.classifier_mean = classifier['mean'] if 'mean' in classifier else None
        self.classifier_std = classifier['std'] if 'std' in classifier else None
        print("training log mean and std of classifier: ", self.classifier_mean, self.classifier_std)

        self.reward_penalty_coef = reward_penalty_coef

        self.model_optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.networks = {
            "model": self.model
        }
        self.obs_space = obs_space
        self.holdout_ratio = holdout_ratio
        self.inc_var_loss = inc_var_loss
        self.use_weight_decay = use_weight_decay
        self.obs_normalizer = StandardNormalizer()
        self.act_normalizer = StandardNormalizer()
        self.model_train_timesteps = 0
        self.type = type

    def _return_kde_penalty(self, state, action, reward=None, type= "linear", alpha = 0.1):
        
        """
        Version with optimized IQR filtering.
        """
        
        if self.classifier_name is None:
            input_np = np.concatenate([state, action], axis=1)
        else:
            input_np = torch.cat([action.squeeze(1), reward.view(-1, 1)], dim=1)
        log_probs = self.classifier_model.score_samples(input_np, self.device)
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.detach().cpu().numpy()
        # print('log_probs mean and std: ', log_probs.mean(), log_probs.std())
        if self.classifier_mean is not None and self.classifier_std is not None:
            log_probs = (log_probs - self.classifier_mean) / self.classifier_std
        if type == "linear":
          
            log_weight = self.classifier_thr - log_probs #high means more likely to be OOD
            q1, q3 = np.percentile(log_weight, [25, 75])
            upper_bound = q3 + 1.5 * (q3 - q1)
            weight = np.clip(log_weight, a_min=0, a_max=upper_bound)
        elif type == "inverse":
            weight = np.where(
                log_probs < self.classifier_thr,
                np.exp(self.classifier_thr) / (np.exp(log_probs) + 1e-6),
                0.0
            )
        elif type == "tanh":
            weight = (np.tanh(0.1*(-log_probs + self.classifier_thr)))
            # print(weight.mean(), weight.std())
        elif type == "softplus": #smooth and stable
            weight = np.log(1 + np.exp(-log_probs)).numpy()
        #plot the weights in histogram
        # Plotting moved to algo/mopo.py:rollout_transitions() for better frequency control

        return weight

    @torch.no_grad()
    def eval_data(self, data, update_elite_models=False):
        """
        Evaluate the dynamics model on a dataset.

        Args:
            data: Dictionary containing observations, actions, next_observations, rewards
            update_elite_models: Whether to update elite model indices based on losses

        Returns:
            tuple: (eval_mse_losses, None)
        """
        obs_list, action_list, next_obs_list, reward_list = \
            itemgetter("observations", 'actions', 'next_observations', 'rewards')(data)
        obs_list = torch.Tensor(obs_list)
        action_list = torch.Tensor(action_list)
        next_obs_list = torch.Tensor(next_obs_list)
        reward_list = torch.Tensor(reward_list)
        delta_obs_list = next_obs_list - obs_list
        obs_list, action_list = self.transform_obs_action(obs_list, action_list)
        model_input = torch.cat([obs_list, action_list], dim=-1).to(util.device)
        predictions = functional.minibatch_inference(
            args=[model_input],
            rollout_fn=self.model.predict,
            batch_size=10000,
            cat_dim=1
        )
        groundtruths = torch.cat((delta_obs_list, reward_list), dim=1).to(util.device)
        eval_mse_losses, _ = self.model_loss(predictions, groundtruths, mse_only=True)

        if update_elite_models:
            elite_idx = np.argsort(eval_mse_losses.cpu().numpy())
            self.model.elite_model_idxes = elite_idx[:self.model.num_elite]

        return eval_mse_losses.detach().cpu().numpy(), None

    def reset_normalizers(self):
        """Reset observation and action normalizers."""
        self.obs_normalizer.reset()
        self.act_normalizer.reset()

    def update_normalizer(self, obs, action):
        """Update normalizers with new data."""
        self.obs_normalizer.update(obs)
        self.act_normalizer.update(action)

    def transform_obs_action(self, obs, action):
        """Apply normalization to observations and actions."""
        obs = self.obs_normalizer.transform(obs)
        action = self.act_normalizer.transform(action)
        return obs, action

    def update(self, data_batch):
        """
        Update dynamics model with a batch of data.

        Args:
            data_batch: Dictionary containing observations, actions, next_observations, rewards

        Returns:
            dict: Training metrics including losses
        """
        obs_batch, action_batch, next_obs_batch, reward_batch = \
            itemgetter("observations", 'actions', 'next_observations', 'rewards')(data_batch)
        obs_batch = torch.Tensor(obs_batch)
        action_batch = torch.Tensor(action_batch)
        next_obs_batch = torch.Tensor(next_obs_batch)
        reward_batch = torch.Tensor(reward_batch)

        delta_obs_batch = next_obs_batch - obs_batch
        obs_batch, action_batch = self.transform_obs_action(obs_batch, action_batch)

        # Predict with model
        model_input = torch.cat([obs_batch, action_batch], dim=-1).to(util.device)
        predictions = self.model.predict(model_input)

        # Compute training loss
        groundtruths = torch.cat((delta_obs_batch, reward_batch), dim=-1).to(util.device)
        train_mse_losses, train_var_losses = self.model_loss(predictions, groundtruths)
        train_mse_loss = torch.sum(train_mse_losses)
        train_var_loss = torch.sum(train_var_losses)
        train_transition_loss = train_mse_loss + train_var_loss

        # Add log variance regularization
        train_transition_loss += 0.01 * torch.sum(self.model.max_logvar) - 0.01 * torch.sum(
            self.model.min_logvar
        )

        if self.use_weight_decay:
            decay_loss = self.model.get_decay_loss()
            train_transition_loss += decay_loss
        else:
            decay_loss = None

        # Update transition model
        self.model_optimizer.zero_grad()
        train_transition_loss.backward()
        self.model_optimizer.step()

        return {
            "loss/train_model_loss_mse": train_mse_loss.item(),
            "loss/train_model_loss_var": train_var_loss.item(),
            "loss/train_model_loss": train_var_loss.item(),
            "loss/decay_loss": decay_loss.item() if decay_loss is not None else 0,
            "misc/max_std": self.model.max_logvar.mean().item(),
            "misc/min_std": self.model.min_logvar.mean().item()
        }

    def model_loss(self, predictions, groundtruths, mse_only=False):
        """
        Compute model loss (MSE and variance loss).

        Args:
            predictions: Tuple of (pred_means, pred_logvars)
            groundtruths: Ground truth delta observations and rewards
            mse_only: Whether to compute only MSE loss

        Returns:
            tuple: (mse_losses, var_losses)
        """
        pred_means, pred_logvars = predictions
        if self.inc_var_loss and not mse_only:
            # Average over batch and dim, sum over ensembles
            inv_var = torch.exp(-pred_logvars)
            mse_losses = torch.mean(torch.mean(torch.pow(pred_means - groundtruths, 2) * inv_var, dim=-1), dim=-1)
            var_losses = torch.mean(torch.mean(pred_logvars, dim=-1), dim=-1)
        elif mse_only:
            mse_losses = torch.mean(torch.pow(pred_means - groundtruths, 2), dim=(1, 2))
            var_losses = None
        else:
            raise ValueError("Invalid loss configuration")
        return mse_losses, var_losses

    @torch.no_grad()
    def predict(self, obs, act, deterministic=False):
        """
        Predict next observation and reward with optional KDE penalty.

        Args:
            obs: Current observation (numpy array or torch tensor)
            act: Action (numpy array or torch tensor)
            deterministic: If True, use mean prediction; if False, sample from distribution

        Returns:
            tuple: (next_obs, penalized_rewards, terminals, info)
        """
        # Ensure batch dimension
        if len(obs.shape) == 1:
            obs = obs[None, ]
            act = act[None, ]

        # Convert to tensors if needed
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(util.device)
        if not isinstance(act, torch.Tensor):
            act = torch.FloatTensor(act).to(util.device)

        # Normalize and predict
        scaled_obs, scaled_act = self.transform_obs_action(obs, act)
        model_input = torch.cat([scaled_obs, scaled_act], dim=-1).to(util.device)
        pred_diff_means, pred_diff_logvars = self.model.predict(model_input)
        pred_diff_means = pred_diff_means.detach().cpu().numpy()
        obs = obs.detach().cpu().numpy()
        act = act.detach().cpu().numpy()

        ensemble_model_stds = pred_diff_logvars.exp().sqrt().detach().cpu().numpy()

        # Sample from ensemble if not deterministic
        if not deterministic:
            pred_diff_means = pred_diff_means + np.random.normal(size=pred_diff_means.shape) * ensemble_model_stds

        # Sample from elite models
        _, batch_size, _ = pred_diff_means.shape
        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        batch_idxes = np.arange(0, batch_size)
        pred_diff_samples = pred_diff_means[model_idxes, batch_idxes]

        # Compute next state and reward
        next_obs, rewards = pred_diff_samples[:, :-1] + obs, pred_diff_samples[:, -1]
        terminals = self.static_fns.termination_fn(obs, act, next_obs)

        # Apply KDE penalty to rewards if coefficient is non-zero
        penalty_coeff = self.reward_penalty_coef
        if penalty_coeff != 0:
            penalty = self._return_kde_penalty(next_obs, act, type= self.type)
            penalized_rewards = rewards - penalty_coeff * penalty
            info = {'penalty': penalty, 'penalized_rewards': penalized_rewards}
        else:
            penalized_rewards = rewards
            info = {'penalized_rewards': penalized_rewards}

        assert isinstance(next_obs, np.ndarray)
        penalized_rewards = penalized_rewards[:, None]
        terminals = terminals[:, None]
        return next_obs, penalized_rewards, terminals, info

    def update_best_snapshots(self, val_losses):
        """
        Update best model snapshots if validation losses improve.

        Args:
            val_losses: List of validation losses for each ensemble model

        Returns:
            bool: Whether any snapshot was updated
        """
        updated = False
        for i in range(len(val_losses)):
            current_loss = val_losses[i]
            best_loss = self.best_snapshot_losses[i]
            improvement = (best_loss - current_loss) / best_loss
            if improvement > 0.01:
                self.best_snapshot_losses[i] = current_loss
                self.save_model_snapshot(i)
                updated = True
        return updated

    def reset_best_snapshots(self):
        """Reset best model snapshots to current model state."""
        self.model_best_snapshots = [
            deepcopy(self.model.ensemble_models[idx].state_dict())
            for idx in range(self.model.ensemble_size)
        ]
        self.best_snapshot_losses = [1e10 for _ in range(self.model.ensemble_size)]

    def save_model_snapshot(self, idx):
        """Save snapshot of a specific ensemble model."""
        self.model_best_snapshots[idx] = deepcopy(self.model.ensemble_models[idx].state_dict())

    def load_best_snapshots(self):
        """Load best saved snapshots into the model."""
        self.model.load_state_dicts(self.model_best_snapshots)

    def save_model(self, info='dynamics_model'):
        """
        Save dynamics model to disk.

        Args:
            info: Directory name for saving the model
        """
        model_save_dir = os.path.join(util.logger_model.log_path, info)
        os.makedirs(model_save_dir, exist_ok=True)

        for network_name, network in self.networks.items():
            save_path = os.path.join(model_save_dir, network_name + ".pt")
            torch.save(network.state_dict(), save_path)
         # Save normalizers
        normalizer_path = os.path.join(model_save_dir, "normalizers.pt")
        torch.save({
            'obs_normalizer': {
                'mean': self.obs_normalizer.mean,
                'var': self.obs_normalizer.var,
                'tot_count': self.obs_normalizer.tot_count
            },
            'act_normalizer': {
                'mean': self.act_normalizer.mean,
                'var': self.act_normalizer.var,
                'tot_count': self.act_normalizer.tot_count
            }
        }, normalizer_path)

        
    def load_model(self, model_save_dir):
        """
        Load dynamics model from disk.

        Args:
            model_save_dir: Directory containing the saved model

        Returns:
            Model state dict loading result
        """
        model_save_dir = "/public/gormpo/models/rl/abiomed/vae/seed_456_0114_194521-abiomed_mbpo_vae/dynamics_model"
        print('loaded abiomed transition model from ', model_save_dir)
        for network_name, network in self.networks.items():
            load_path = os.path.join(model_save_dir, network_name + ".pt")
            state_dict = torch.load(load_path, map_location='cuda')
            return network.load_state_dict(state_dict)
        
        normalizer_path = os.path.join(model_save_dir, "normalizers.pt")
        if os.path.exists(normalizer_path):
            normalizer_data = torch.load(normalizer_path, map_location='cuda')

            # Load obs_normalizer
            self.obs_normalizer.mean = normalizer_data['obs_normalizer']['mean']
            self.obs_normalizer.var = normalizer_data['obs_normalizer']['var']
            self.obs_normalizer.tot_count = normalizer_data['obs_normalizer']['tot_count']

            # Load act_normalizer
            self.act_normalizer.mean = normalizer_data['act_normalizer']['mean']
            self.act_normalizer.var = normalizer_data['act_normalizer']['var']
            self.act_normalizer.tot_count = normalizer_data['act_normalizer']['tot_count']

            print(f"Loaded normalizers from {normalizer_path}")
        else:
            print(f"Warning: Normalizer file not found at {normalizer_path}. Normalizers not loaded.")

        return None

