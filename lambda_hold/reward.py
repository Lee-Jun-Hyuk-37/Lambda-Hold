"""Sprint reward: exponential in forward speed, gated by a lateral penalty."""
import gym
import numpy as np

from .model_access import get_inner_gaitgym


class SprintRewardWrapper(gym.Wrapper):
    """Replace the environment reward with the minimal sprint reward.

    ``r = exp(v_x) * exp(-(v_z / sigma_lat)^2)``

    where ``v_x`` and ``v_z`` are the forward and mediolateral components of the
    centre-of-mass velocity. The forward term is uncapped, so the reward keeps
    rising with speed; the lateral term is a narrow Gaussian in ``[0, 1]`` that
    discourages sideways motion. Episode termination (a fall) is left to the
    underlying environment.
    """

    def __init__(self, env, sigma_lat=0.1):
        super().__init__(env)
        self._gait = get_inner_gaitgym(env)
        self.sigma_lat = float(sigma_lat)

    def _sprint_reward(self):
        v = self._gait.model.com_vel()
        return float(np.exp(float(v.x)) * np.exp(-(float(v.z) / self.sigma_lat) ** 2))

    def step(self, action):
        out = self.env.step(action)
        reward = self._sprint_reward()
        # Normalize to the Gymnasium 5-tuple so the stack above sees one API.
        if len(out) == 5:
            obs, _, terminated, truncated, info = out
            return obs, reward, terminated, truncated, info
        obs, _, done, info = out
        return obs, reward, bool(done), False, info

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}
