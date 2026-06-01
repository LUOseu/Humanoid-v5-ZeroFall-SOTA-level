import argparse
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+",
                        default=[
                            "runs/sac_humanoid_v5/sac_humanoid_v5.zip",
                            "runs/sac_humanoid_v5_s2028/sac_humanoid_v5.zip",
                            "runs/sac_humanoid_v5_s2029/sac_humanoid_v5.zip",
                        ])
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()

    models = []
    for p in args.models:
        if not Path(p).exists():
            raise FileNotFoundError(p)
        models.append(SAC.load(p, device=args.device))
    print(f"Loaded {len(models)} SAC models for ensemble")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("Humanoid-v5")
    scores = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        score = 0.0
        steps = 0
        while not done:
            actions = []
            for m in models:
                action, _ = m.predict(obs, deterministic=True)
                actions.append(action)
            avg_action = np.mean(actions, axis=0)
            obs, reward, terminated, truncated, info = env.step(avg_action)
            score += float(reward)
            done = bool(terminated or truncated)
            steps += 1
        scores.append(score)
        print(f"episode={ep + 1} seed={args.seed + ep} raw_return={score:.3f} steps={steps}")

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    print(f"mean_raw_return={mean:.3f} std={std:.3f} episodes={args.episodes}")
    env.close()


if __name__ == "__main__":
    main()
