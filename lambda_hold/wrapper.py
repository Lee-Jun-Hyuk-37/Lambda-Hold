"""Lambda-hold action wrapper.

The policy emits one per-muscle threshold length ``lambda`` per decision and
holds it across the interval; between decisions the simulation advances one step
at a time while the stretch-reflex law (``recruitment.lambda_to_excitation``)
turns the held ``lambda`` and the evolving muscle length and velocity into
excitation. One hold interval is one reinforcement-learning transition whose
reward is the sum over its inner steps.

Decisions are placed at ground-reaction-force events: either foot's vertical GRF
crossing ``contact_frac`` of body weight, rising (strike) or falling (toe-off),
with one sub-point inserted at half the estimated interval and every hold bounded
to ``[min_hold_sec, max_hold_sec]``.

The observation holds muscle length, velocity and force; foot contact and GRF;
joint angles and angular velocities; a vestibular signal; and an efference copy
of the held ``lambda`` (excitation and activation excluded).
"""
import gym
import numpy as np

from .model_access import get_inner_gaitgym
from .recruitment import lambda_to_excitation


class LambdaActionWrapper(gym.Wrapper):
    """Threshold-length action space with event-triggered holding."""

    def __init__(self, env, g_tonic=50.0, g_phasic=0.1,
                 lambda_lo=0.6, lambda_hi=1.2,
                 contact_frac=0.05, min_hold_sec=0.05, max_hold_sec=0.15,
                 n_subdiv=1):
        super().__init__(env)
        self._gait = get_inner_gaitgym(env)
        model = self._gait.model

        self._n_mus = len(list(model.muscles()))
        self._n_dof = len(list(model.dofs()))

        self.g_tonic = float(g_tonic)
        self.g_phasic = float(g_phasic)
        self.lambda_lo = float(lambda_lo)
        self.lambda_hi = float(lambda_hi)
        self._lam_neutral = np.full(self._n_mus, 0.5 * (self.lambda_lo + self.lambda_hi),
                                    dtype=np.float64)

        self._dt = float(getattr(self._gait, "step_size", 0.01)) or 0.01
        self._min_hold_n = max(0, int(round(float(min_hold_sec) / self._dt)))
        self._max_hold_n = max(1, int(round(float(max_hold_sec) / self._dt)))
        self.n_subdiv = int(n_subdiv)

        # Feet and body weight for the vertical-GRF contact threshold.
        self._feet = [next(b for b in model.bodies() if "calcn_l" in b.name()),
                      next(b for b in model.bodies() if "calcn_r" in b.name())]
        body_weight = float(sum(b.mass() for b in model.bodies()) * 9.81)
        self._contact_thresh = float(contact_frac) * body_weight

        # Vestibular reference body and gravity vector.
        self._torso = model.find_body("torso")
        self._grav = np.asarray(model.gravity().array(), dtype=np.float64)

        # The policy commands one threshold length per muscle.
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._n_mus,), dtype=np.float32)
        # Afferent-style observation (see module docstring).
        obs_dim = 3 * self._n_mus + (6 + 2 * self._n_dof) + 6 + self._n_mus + 18
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._last_lambda = self._lam_neutral.copy()
        self._reset_timing_state()

    # ---- helpers ----

    def _reset_timing_state(self):
        self._t_inner = 0
        self._last_event_t = None
        self._prev_event_t = None
        self._prev_prev_event_t = None
        self._subpoints = []
        self._prev_contact = (False, False)

    def _kinematics(self):
        """Normalized fiber length (L / L_opt) and velocity (L_opt / s)."""
        model = self._gait.model
        length = np.asarray(model.muscle_fiber_length_array(), dtype=np.float64)
        velocity = np.asarray(model.muscle_fiber_velocity_array(), dtype=np.float64)
        return length, velocity

    def _action_to_lambda(self, action):
        """Map the action in [-1, 1] to a held threshold length in [lo, hi]."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        half = 0.5 * (self.lambda_hi - self.lambda_lo)
        lam = self._lam_neutral + a * half
        return np.clip(lam, self.lambda_lo, self.lambda_hi)

    def _foot_contacts(self):
        """(left_in_contact, right_in_contact) by vertical GRF vs the threshold."""
        fy_l = float(self._feet[0].contact_force().y)
        fy_r = float(self._feet[1].contact_force().y)
        return (fy_l > self._contact_thresh, fy_r > self._contact_thresh)

    def _recompute_subpoints(self):
        """Place the sub-point for the current interval.

        The ongoing interval cannot be measured while it runs, so it is estimated
        from the most recent interval of the same gait phase, the gap between the
        third- and second-most-recent events. The sub-point is placed at half that
        estimate after the most recent event.
        """
        self._subpoints = []
        if (self.n_subdiv <= 0 or self._last_event_t is None
                or self._prev_event_t is None or self._prev_prev_event_t is None):
            return
        d = self._prev_event_t - self._prev_prev_event_t
        if d <= 0:
            return
        pts = set()
        for j in range(1, self.n_subdiv + 1):
            off = int(round(j * d / (self.n_subdiv + 1)))
            if off >= 1:
                pts.add(self._last_event_t + off)
        self._subpoints = sorted(p for p in pts if p > self._last_event_t)

    def _fire_event(self):
        """Record a GRF event and rebuild the sub-point schedule."""
        self._prev_prev_event_t = self._prev_event_t
        self._prev_event_t = self._last_event_t
        self._last_event_t = self._t_inner
        self._recompute_subpoints()

    def _event_subdiv_trigger(self, n_inner, new_contact, new_unload):
        """Decide whether to end the hold interval at this inner step.

        A real GRF event (strike or toe-off) ends the interval and updates the
        event history, but only once the minimum-hold floor is reached; an event
        that arrives earlier is ignored entirely, neither triggering a resample
        nor updating the history, so every hold lasts at least the floor. A
        scheduled sub-point likewise ends the interval only at or after the floor;
        one that falls earlier is skipped. The maximum-hold cap forces a resample.
        """
        if (new_contact or new_unload) and n_inner >= self._min_hold_n:
            self._fire_event()
            return True
        while self._subpoints and self._t_inner >= self._subpoints[0]:
            self._subpoints.pop(0)
            if n_inner >= self._min_hold_n:
                return True
        if n_inner >= self._max_hold_n:
            return True
        return False

    def _vestibular(self):
        """Otolith gravity-tilt (gravity in the torso frame) and canal angular velocity."""
        q = self._torso.orientation()
        w, x, y, z = q.w, q.x, q.y, q.z
        rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        g_torso = rot.T @ self._grav
        g_torso = g_torso / (np.linalg.norm(g_torso) + 1e-9)
        ang_vel = np.asarray(self._torso.ang_vel().array())
        return np.concatenate([g_torso, ang_vel])

    def _ep_transform(self, obs):
        """Rebuild the base observation into the afferent-style observation.

        Base layout: L, v, force, excitation (4 * n_mus), orientation (4),
        angular velocity (3), feet (6), joint positions (n_dof), joint velocities
        (n_dof), activations (n_mus), GRF (18). Excitation and activation are
        dropped, orientation and angular velocity are replaced by the vestibular
        signal, and the held lambda is appended.
        """
        obs = np.asarray(obs, dtype=np.float32)
        nm, nd = self._n_mus, self._n_dof
        parts = [
            obs[0:3 * nm],                          # L, v, force
            obs[4 * nm + 7:4 * nm + 13 + 2 * nd],   # feet(6), dofpos, dofvel
            self._vestibular().astype(np.float32),  # vestibular(6)
            self._last_lambda.astype(np.float32),   # held lambda (efference copy)
            obs[-18:],                              # GRF(18)
        ]
        return np.concatenate(parts).astype(np.float32)

    # ---- gym hooks ----

    def step(self, action):
        lam = self._action_to_lambda(np.asarray(action, dtype=np.float64)[:self._n_mus])
        total_reward = 0.0
        done = False
        out = None
        n_inner = 0
        while True:
            length, velocity = self._kinematics()
            e = lambda_to_excitation(lam, length, velocity, self.g_tonic, self.g_phasic)
            out = self.env.step(e)
            total_reward += float(out[1])
            n_inner += 1
            d = out[2] if len(out) == 4 else (out[2] or out[3])
            cur = self._foot_contacts()
            new_contact = (cur[0] and not self._prev_contact[0]) or \
                          (cur[1] and not self._prev_contact[1])
            new_unload = (self._prev_contact[0] and not cur[0]) or \
                         (self._prev_contact[1] and not cur[1])
            self._prev_contact = cur
            self._t_inner += 1
            if d:
                done = True
                break
            if self._event_subdiv_trigger(n_inner, new_contact, new_unload):
                break

        obs = out[0]
        self._last_lambda = np.asarray(lam, dtype=np.float64)
        obs = self._ep_transform(obs)
        info = dict(out[-1]) if isinstance(out[-1], dict) else {}
        # Number of simulation steps this decision spanned (one decision covers
        # many simulation steps); used to track the simulation-step budget.
        info["macro_inner_steps"] = n_inner
        if len(out) == 4:
            trunc = bool(info.get("TimeLimit.truncated", False))
            return obs, total_reward, bool(done) and not trunc, trunc, info
        return obs, total_reward, bool(out[2]), bool(out[3]), info

    def reset(self, **kwargs):
        self._reset_timing_state()
        out = self.env.reset(**kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        self._prev_contact = self._foot_contacts()
        self._last_lambda = self._lam_neutral.copy()
        obs = self._ep_transform(obs)
        if isinstance(out, tuple):
            return (obs,) + tuple(out[1:])
        return obs
