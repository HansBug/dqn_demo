from typing import Optional

import numpy as np
from gym.envs.toy_text import CliffWalkingEnv


class CliffWalkingPlusEnv(CliffWalkingEnv):
    def __init__(self, render_mode: Optional[str] = None):
        CliffWalkingEnv.__init__(self, render_mode)

    def _calculate_transition_prob(self, current, delta):
        f = np.array(np.unravel_index(47, self.shape))
        t = (np.array(current) + np.array(delta))
        dist = ((f - t) ** 2).sum() ** 0.5
        ex_reward = dist / min(*self.shape)

        ops = CliffWalkingEnv._calculate_transition_prob(self, current, delta)
        assert len(ops) == 1
        (prob, new_state, reward, is_terminated), = ops

        return [(prob, new_state, reward - ex_reward, is_terminated)]

    def step(self, a):
        state, reward, terminated, _, info = CliffWalkingEnv.step(self, a)
        return state, reward, terminated, info
