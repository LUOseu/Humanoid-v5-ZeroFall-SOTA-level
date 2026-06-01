import argparse
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runs/ppo_humanoid_v5/ppo_humanoid_v5.zip")
    parser.add_argument("--vecnormalize", type=str, default="runs/ppo_humanoid_v5/vecnormalize.pkl")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def make_eval_env(seed: int):
    def _init():
        env = gym.make("Humanoid-v5")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return _init


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_path = Path(args.model)
    vecnorm_path = Path(args.vecnormalize)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not vecnorm_path.exists():
        raise FileNotFoundError(vecnorm_path)

    env = DummyVecEnv([make_eval_env(args.seed)])
    env = VecNormalize.load(str(vecnorm_path), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), env=env, device=args.device)
    scores = []

    for episode in range(args.episodes):
        env.seed(args.seed + episode)
        obs = env.reset()
        done = False
        score = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, infos = env.step(action)
            score += float(rewards[0])
            done = bool(dones[0])
            steps += 1
        scores.append(score)
        print(f"episode={episode + 1} seed={args.seed + episode} raw_return={score:.3f} steps={steps}")

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    print(f"mean_raw_return={mean:.3f} std={std:.3f} episodes={args.episodes}")
    env.close()


if __name__ == "__main__":
    main()
