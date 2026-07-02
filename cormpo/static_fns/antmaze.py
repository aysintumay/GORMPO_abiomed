import numpy as np

class StaticFns:
    @staticmethod
    def termination_fn(obs, act, next_obs):
        # ponytail: no termination, matches static_fns/abiomed.py; antmaze episodes
        # are timeout-bounded, not physical-failure-bounded. Upgrade to goal-radius
        # termination (see LEQ/dynamics/termination_fns.py::termination_fn_antmaze) if
        # rollout fidelity near the goal matters.
        done = np.array([False]*obs.shape[0])
        return done
