"""Train the lambda-hold sprint policy with SAC + gSDE.

Example::

    python train.py --out runs/lambda_hold --seed 1

Training stops when the cumulative simulation-step budget is reached. Because one
decision spans many simulation steps, the simulation-step count is accumulated
from each environment's ``macro_inner_steps`` rather than from the policy-query
count. Checkpoints, the final model, the VecNormalize statistics, and a
learning-curve ``result.json`` are written to the output directory.
"""
import argparse
import json
import os
import time

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from lambda_hold.env import make_vec_env


class SimStepReport(BaseCallback):
    """Log the learning curve, checkpoint periodically, and stop at the budget."""

    def __init__(self, save_dir, log_interval, ckpt_interval, sim_step_budget):
        super().__init__()
        self.save_dir = save_dir
        self.log_interval = int(log_interval)
        self.ckpt_interval = int(ckpt_interval)
        self.sim_step_budget = int(sim_step_budget)
        self.sim_steps = 0
        self.curve = []
        self.t0 = time.perf_counter()
        self._last_log = 0
        self._last_ckpt = 0

    def _dump(self):
        with open(os.path.join(self.save_dir, "result.json"), "w") as f:
            json.dump({"reward_curve": self.curve}, f, indent=2)

    def save_checkpoint(self, name):
        self.model.save(os.path.join(self.save_dir, name))
        vecnorm = self.model.get_vec_normalize_env()
        if vecnorm is not None:
            vecnorm.save(os.path.join(self.save_dir, "vecnormalize.pkl"))

    def _on_step(self):
        for info in self.locals.get("infos", []):
            self.sim_steps += int(info.get("macro_inner_steps", 1))
        n = self.num_timesteps
        if n - self._last_log >= self.log_interval:
            self._last_log = n
            ep_info = self.model.ep_info_buffer
            if ep_info:
                mean_reward = float(np.mean([e["r"] for e in ep_info]))
                elapsed = time.perf_counter() - self.t0
                self.curve.append({"step": int(n), "sim_step": int(self.sim_steps),
                                   "mean_reward": mean_reward, "elapsed": float(elapsed)})
                print(f"  dec={n:>9d}  sim={self.sim_steps:>10d}  "
                      f"rew={mean_reward:>9.2f}  {elapsed:>6.0f}s", flush=True)
                self._dump()
        if n - self._last_ckpt >= self.ckpt_interval:
            self._last_ckpt = n
            self.save_checkpoint(f"model_step{n}")
        if self.sim_step_budget > 0 and self.sim_steps >= self.sim_step_budget:
            print(f"  [simulation-step budget {self.sim_step_budget} reached] stopping.",
                  flush=True)
            return False
        return True


def main():
    parser = argparse.ArgumentParser(description="Train the lambda-hold sprint policy.")
    parser.add_argument("--out", required=True,
                        help="output directory for checkpoints and result.json")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sim-step-budget", type=int, default=150_000_000)
    parser.add_argument("--n-envs", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=10_000)
    parser.add_argument("--ckpt-interval", type=int, default=1_000_000)
    parser.add_argument("--g-tonic", type=float, default=50.0)
    parser.add_argument("--g-phasic", type=float, default=0.1)
    parser.add_argument("--lambda-lo", type=float, default=0.6)
    parser.add_argument("--lambda-hi", type=float, default=1.2)
    parser.add_argument("--sigma-lat", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    vec = make_vec_env(n_envs=args.n_envs, g_tonic=args.g_tonic, g_phasic=args.g_phasic,
                       lambda_lo=args.lambda_lo, lambda_hi=args.lambda_hi,
                       sigma_lat=args.sigma_lat)
    model = SAC(
        policy="MlpPolicy", env=vec, seed=args.seed, device=args.device, verbose=0,
        gamma=0.99, learning_rate=3e-4, batch_size=256, buffer_size=500_000,
        tau=0.005, train_freq=1, gradient_steps=2, learning_starts=5000,
        use_sde=True, sde_sample_freq=4,
    )
    report = SimStepReport(args.out, args.log_interval, args.ckpt_interval,
                           args.sim_step_budget)
    # total_timesteps bounds the policy-query count; the simulation-step budget
    # (enforced by the callback) stops training first.
    model.learn(total_timesteps=10 ** 12, callback=report)
    report.save_checkpoint("model_final")


if __name__ == "__main__":
    main()
