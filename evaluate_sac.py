import argparse
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runs/sac_humanoid_v5/sac_humanoid_v5.zip")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    env = gym.make("Humanoid-v5")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    model = SAC.load(str(model_path), device=args.device)
    scores = []

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        done = False
        score = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            score += float(reward)
            done = bool(terminated or truncated)
            steps += 1
        scores.append(score)
        print(f"episode={episode + 1} seed={args.seed + episode} raw_return={score:.3f} steps={steps}")

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    print(f"mean_raw_return={mean:.3f} std={std:.3f} episodes={args.episodes}")
    env.close()


if __name__ == "__main__":
    main()
