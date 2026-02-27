"""
SACPolicyWrapper: wraps SACPolicy to satisfy scope_rl's evaluation-policy interface.

scope_rl requires evaluation policies to be instances of BaseHead, which itself
inherits from d3rlpy's QLearningAlgoBase.  The FQE estimator (used for DM/DR)
additionally needs evaluation_policy.impl to be a QLearningAlgoImplBase so that
d3rlpy's ContinuousFQE can call impl.predict_best_action(next_obs) when
computing Bellman targets.

Architecture
------------
_SACModules          – empty Modules dataclass (satisfies d3rlpy's eval_api)
SACAlgoImpl          – QLearningAlgoImplBase that wraps SACPolicy at the tensor level
SACPolicyWrapper     – BaseHead (dataclass) that owns an SACPolicy and exposes:
  • predict / sample_action / predict_value  (numpy in/out)
  • impl property  → SACAlgoImpl instance
  • sample_action_and_output_pscore
  • calc_pscore_given_action  (squashed-Gaussian density with tanh correction)
  • calc_action_choice_probability  → None  (not used for continuous IS)
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import torch

from d3rlpy.algos.qlearning.base import QLearningAlgoImplBase
from d3rlpy.torch_utility import Modules, TorchMiniBatch
from scope_rl.policy.head import BaseHead


# ---------------------------------------------------------------------------
# Minimal Modules dataclass (required by d3rlpy's eval_api / train_api)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _SACModules(Modules):
    """Empty modules container — no nn.Module parameters to checkpoint."""


# ---------------------------------------------------------------------------
# QLearningAlgoImplBase wrapper around SACPolicy
# ---------------------------------------------------------------------------

class SACAlgoImpl(QLearningAlgoImplBase):
    """d3rlpy-compatible impl that delegates to a SACPolicy.

    FQEBaseImpl calls ``self._algo.predict_best_action(next_obs_tensor)``
    during Bellman-target computation, so this class must return deterministic
    actions as a torch.Tensor.
    """

    def __init__(self, sac_policy, obs_shape: tuple, action_size: int):
        device = sac_policy._device
        super().__init__(
            observation_shape=obs_shape,
            action_size=action_size,
            modules=_SACModules(),
            device=str(device),
        )
        self._sac = sac_policy

    # --- required abstract methods from QLearningAlgoImplBase ---------------

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic (tanh-squashed) action.

        x may arrive on a different device (e.g., FQE's device). Move it to
        the SAC device, run the forward pass, then return the action on the
        original device so FQE's Q-function can use it without a device error.
        """
        src_device = x.device
        obs = x.to(self._sac._device)
        with torch.no_grad():
            action, _ = self._sac(obs, deterministic=True)
        return action.to(src_device)

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        """Stochastic (tanh-squashed) action."""
        src_device = x.device
        obs = x.to(self._sac._device)
        with torch.no_grad():
            action, _ = self._sac(obs, deterministic=False)
        return action.to(src_device)

    def inner_predict_value(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        obs = x.to(self._sac._device)
        act = action.to(self._sac._device)
        with torch.no_grad():
            q1 = self._sac.critic1(obs, act).flatten()
            q2 = self._sac.critic2(obs, act).flatten()
        return torch.min(q1, q2).to(x.device)

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> dict:
        return {}  # read-only; no training


# ---------------------------------------------------------------------------
# BaseHead wrapper (the object passed to CreateOPEInput)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SACPolicyWrapper(BaseHead):
    """scope_rl-compatible evaluation-policy wrapper for SACPolicy.

    Parameters
    ----------
    sac_policy : SACPolicy
        The trained SAC policy.
    name : str
        Policy identifier used by scope_rl (must be unique across policies).
    random_state : int, optional
        Unused; kept for API symmetry with other BaseHead subclasses.
    """

    sac_policy: object          # SACPolicy (avoid import cycle)
    name: str
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        self.action_type = "continuous"

        device = self.sac_policy._device
        self._eps = self.sac_policy._SACPolicy__eps

        high = torch.tensor(self.sac_policy.action_space.high, dtype=torch.float32)
        low  = torch.tensor(self.sac_policy.action_space.low,  dtype=torch.float32)
        self._action_scale = ((high - low) / 2.0).to(device)

        # Build the impl so FQE's inner_create_impl assertion passes
        obs_size   = self.sac_policy.actor.backbone.model[0].in_features
        action_dim = self.sac_policy.action_space.shape[0]
        self._impl_obj = SACAlgoImpl(
            self.sac_policy, obs_shape=(obs_size,), action_size=action_dim
        )

    # --- impl property expected by d3rlpy / FQE ----------------------------

    @property
    def impl(self):  # type: ignore[override]
        return self._impl_obj

    # --- core action methods ------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Deterministic (mode) action → numpy."""
        obs = torch.as_tensor(x, device=self.sac_policy._device, dtype=torch.float32)
        with torch.no_grad():
            return self.sac_policy.sample_action(obs, deterministic=True)

    def sample_action(self, x: np.ndarray) -> np.ndarray:
        """Stochastic action sample → numpy."""
        obs = torch.as_tensor(x, device=self.sac_policy._device, dtype=torch.float32)
        with torch.no_grad():
            return self.sac_policy.sample_action(obs, deterministic=False)

    # --- value prediction (required for DM / DR) ----------------------------

    def predict_value(
        self,
        x: np.ndarray,
        action: np.ndarray,
        with_std: bool = False,
    ) -> np.ndarray:
        """Conservative Q-value estimate: min(Q1, Q2) → numpy (n,)."""
        obs = torch.as_tensor(x, device=self.sac_policy._device, dtype=torch.float32)
        act = torch.as_tensor(action, device=self.sac_policy._device, dtype=torch.float32)
        with torch.no_grad():
            q1 = self.sac_policy.critic1(obs, act).flatten()
            q2 = self.sac_policy.critic2(obs, act).flatten()
        return torch.min(q1, q2).cpu().numpy()

    # --- IS / probability methods -------------------------------------------

    def sample_action_and_output_pscore(self, x: np.ndarray):
        """Stochastic action + joint density π_e(a|s).

        Returns
        -------
        action : np.ndarray (n, action_dim)
        pscore : np.ndarray (n,)
        """
        obs = torch.as_tensor(x, device=self.sac_policy._device, dtype=torch.float32)
        with torch.no_grad():
            squashed_action, log_prob = self.sac_policy(obs, deterministic=False)
        return (
            squashed_action.cpu().numpy(),
            torch.exp(log_prob).flatten().cpu().numpy(),
        )

    def calc_action_choice_probability(self, x: np.ndarray):
        """Not applicable for continuous actions; returns None."""
        return None

    def calc_pscore_given_action(
        self, x: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        """Evaluate π_e(action | state) for observed (s, a) pairs.

        SAC uses a squashed Gaussian  a = tanh(u), u ~ N(μ(s), σ(s)).
        The change-of-variables density is:

            log π(a|s) = log N(atanh(a); μ, σ)
                       − Σ_d log(scale_d · (1 − a_d²) + ε)

        Parameters
        ----------
        x      : (n, state_dim)
        action : (n, action_dim) – squashed actions from the logged dataset

        Returns
        -------
        pscore : (n,)
        """
        obs = torch.as_tensor(x, device=self.sac_policy._device, dtype=torch.float32)
        act = torch.as_tensor(action, device=self.sac_policy._device, dtype=torch.float32)

        with torch.no_grad():
            dist = self.sac_policy.actor.get_dist(obs)
            # Inverse tanh; clamp boundary to avoid log(0)
            act_c = torch.clamp(act, -1.0 + 1e-6, 1.0 - 1e-6)
            u = torch.atanh(act_c)
            # NormalWrapper.log_prob sums over action dims → (n, 1)
            log_prob = dist.log_prob(u)
            # Tanh Jacobian correction (same formula as SACPolicy.forward)
            log_prob = log_prob - torch.log(
                self._action_scale * (1.0 - act.pow(2)) + self._eps
            ).sum(-1, keepdim=True)
            pscore = torch.exp(log_prob).flatten()

        return pscore.cpu().numpy()

    # --- no-op overrides for BaseHead methods that delegate to base_policy --
    # These are never called during OPE but must not crash if scope_rl tries.

    def build_with_dataset(self, dataset) -> None:
        pass  # impl is already initialized in __post_init__

    def build_with_env(self, env) -> None:
        pass

    def get_action_type(self):
        from d3rlpy.constants import ActionSpace
        return ActionSpace.CONTINUOUS
