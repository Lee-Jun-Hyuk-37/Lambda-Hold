"""Assemble the H2190 sprint environment and vectorize it for training.

Wrapper stack, innermost first::

    sconewalk_h2190-v1  (SCONE + Hyfydy)
      -> SprintRewardWrapper      replace the reward with exp(v_x) * lateral
      -> TimeLimitWrapper         truncate after a fixed simulated time
      -> GRFDenseObsWrapper       append per-foot GRF channels
      -> LambdaActionWrapper      threshold-length action, event-triggered holding
"""
import gym
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize)

import sconegym  # noqa: F401  registers the sconewalk_* environments on import

from .obs import GRFDenseObsWrapper
from .reward import SprintRewardWrapper
from .wrapper import LambdaActionWrapper

ENV_ID = "sconewalk_h2190-v1"


class TimeLimitWrapper(gym.Wrapper):
    """Truncate an episode after ``max_seconds`` of simulated time."""

    def __init__(self, env, max_seconds=20.0):
        super().__init__(env)
        self.max_seconds = float(max_seconds)
        self._elapsed = 0.0

    def reset(self, **kwargs):
        self._elapsed = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed += float(getattr(self.env.unwrapped, "step_size", 0.01))
        if not terminated and not truncated and self._elapsed >= self.max_seconds:
            truncated = True
            info = dict(info)
            info["TimeLimit.truncated"] = True
        return obs, reward, terminated, truncated, info


def make_env(sigma_lat=0.1, g_tonic=50.0, g_phasic=0.1,
             lambda_lo=0.6, lambda_hi=1.2, contact_frac=0.05,
             min_hold_sec=0.05, max_hold_sec=0.15, n_subdiv=1, max_seconds=20.0):
    """Build the lambda-hold sprint environment for the H2190 model."""
    env = gym.make(ENV_ID, clip_actions=False)
    env.unwrapped.render_mode = None
    env = SprintRewardWrapper(env, sigma_lat=sigma_lat)
    env = TimeLimitWrapper(env, max_seconds=max_seconds)
    env = GRFDenseObsWrapper(env)
    env = LambdaActionWrapper(
        env, g_tonic=g_tonic, g_phasic=g_phasic,
        lambda_lo=lambda_lo, lambda_hi=lambda_hi, contact_frac=contact_frac,
        min_hold_sec=min_hold_sec, max_hold_sec=max_hold_sec, n_subdiv=n_subdiv)
    return env


def make_vec_env(n_envs=24, subproc=True, norm_obs=True, norm_reward=True, **env_kwargs):
    """Vectorize :func:`make_env` with observation and reward normalization."""
    fns = [(lambda: make_env(**env_kwargs)) for _ in range(n_envs)]
    vec_cls = SubprocVecEnv if (subproc and n_envs > 1) else DummyVecEnv
    vec = VecMonitor(vec_cls(fns))
    if norm_obs or norm_reward:
        vec = VecNormalize(vec, norm_obs=norm_obs, norm_reward=norm_reward)
    return vec
