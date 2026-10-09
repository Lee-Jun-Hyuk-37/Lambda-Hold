"""Per-foot dense ground-reaction-force observation wrapper.

For each foot (left then right) it appends 9 channels, 18 in total:

  contact_force.{x,y,z}  / BW                   (3) net contact force
  contact_moment.{x,y,z} / (BW * REF_LEN)       (3) net contact moment
  (contact_point - body.com).{x,y,z} / REF_LEN  (3) centre of pressure, foot-relative

Forces are normalized by body weight (BW), moments by ``BW * REF_LEN``, and the
centre of pressure is taken relative to the foot body's centre of mass so the
channel stays bounded as the body travels forward.
"""
import gym
import numpy as np

from .model_access import get_inner_gaitgym

REF_LEN = 0.2  # m, typical foot length used to non-dimensionalize moment and CoP


class GRFDenseObsWrapper(gym.Wrapper):
    """Append per-foot 9-channel GRF information (force, moment, CoP) to the obs."""

    def __init__(self, env):
        super().__init__(env)
        self._gait = get_inner_gaitgym(env)
        model = self._gait.model
        self._bw = float(sum(b.mass() for b in model.bodies()) * 9.81)
        # Left foot first, right foot second. Substring match on body names works
        # across sconepy / Hyfydy versions.
        self._feet = [next(b for b in model.bodies() if "calcn_l" in b.name()),
                      next(b for b in model.bodies() if "calcn_r" in b.name())]
        n_add = 18
        base = env.observation_space
        low = np.concatenate([base.low, np.full(n_add, -10.0, dtype=base.dtype)])
        high = np.concatenate([base.high, np.full(n_add, 10.0, dtype=base.dtype)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=base.dtype)

    def _grf_vec(self):
        out = np.empty(18, dtype=np.float32)
        bw = self._bw
        scale_m = bw * REF_LEN
        for i, body in enumerate(self._feet):
            f = body.contact_force()
            mm = body.contact_moment()
            cp = body.contact_point()
            com = body.com_pos()
            off = i * 9
            out[off + 0] = float(f.x) / bw
            out[off + 1] = float(f.y) / bw
            out[off + 2] = float(f.z) / bw
            out[off + 3] = float(mm.x) / scale_m
            out[off + 4] = float(mm.y) / scale_m
            out[off + 5] = float(mm.z) / scale_m
            out[off + 6] = (float(cp.x) - float(com.x)) / REF_LEN
            out[off + 7] = (float(cp.y) - float(com.y)) / REF_LEN
            out[off + 8] = (float(cp.z) - float(com.z)) / REF_LEN
        return out

    def _augment(self, obs):
        return np.concatenate([np.asarray(obs, dtype=np.float32), self._grf_vec()])

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 4:
            obs, rew, done, info = out
            return self._augment(obs), rew, done, info
        obs, rew, term, trunc, info = out
        return self._augment(obs), rew, term, trunc, info

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out
            return self._augment(obs), info
        return self._augment(out)
