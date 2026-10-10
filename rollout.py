"""Roll out a trained lambda-hold policy and write a replayable .sto.

Example::

    python rollout.py --model runs/lambda_hold/model_final.zip \
        --vecnormalize runs/lambda_hold/vecnormalize.pkl --episodes 1

The policy runs deterministically with the saved observation normalization. Each
episode is written by SCONE to its results directory in the native format
(``DATE_TIME.<model>``), which opens directly in SCONE Studio. The controller and
reward settings must match those used during training.
"""
import argparse
import pickle

import numpy as np
from stable_baselines3 import SAC

from lambda_hold.env import make_env
from lambda_hold.model_access import get_inner_gaitgym


def main():
    parser = argparse.ArgumentParser(description="Roll out a trained lambda-hold policy.")
    parser.add_argument("--model", required=True, help="path to a saved model .zip")
    parser.add_argument("--vecnormalize", required=True, help="path to vecnormalize.pkl")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="run the policy deterministically (default); "
                             "use --no-deterministic to sample stochastically")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--g-tonic", type=float, default=50.0)
    parser.add_argument("--g-phasic", type=float, default=0.1)
    parser.add_argument("--lambda-lo", type=float, default=0.6)
    parser.add_argument("--lambda-hi", type=float, default=1.2)
    parser.add_argument("--sigma-lat", type=float, default=0.1)
    args = parser.parse_args()

    env = make_env(g_tonic=args.g_tonic, g_phasic=args.g_phasic,
                   lambda_lo=args.lambda_lo, lambda_hi=args.lambda_hi,
                   sigma_lat=args.sigma_lat)
    gait = get_inner_gaitgym(env)

    # Saved observation-normalization statistics (same normalization as training).
    with open(args.vecnormalize, "rb") as f:
        vecnorm = pickle.load(f)
    mean, var = vecnorm.obs_rms.mean, vecnorm.obs_rms.var
    eps, clip = vecnorm.epsilon, vecnorm.clip_obs

    def normalize(obs):
        return np.clip((np.asarray(obs) - mean) / np.sqrt(var + eps), -clip, clip)

    model = SAC.load(args.model, device=args.device)

    for ep in range(args.episodes):
        gait.store_next = True  # arm SCONE to record this episode
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        done = False
        ep_reward = 0.0
        decisions = 0
        while not done:
            action, _ = model.predict(normalize(obs), deterministic=args.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
            decisions += 1
        gait.write_now()  # flush the .sto if the episode did not end in a fall
        print(f"  episode {ep + 1}/{args.episodes}  return={ep_reward:.1f}  "
              f"decisions={decisions}", flush=True)
    env.close()
    print("done. Each episode is written under SCONE's results directory "
          "(open the .sto in SCONE Studio).", flush=True)


if __name__ == "__main__":
    main()
