"""Evaluate all trained models: PPO 2028, SAC+VecNorm 2027/2028/2029."""
import argparse
import sys
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def evaluate(model, env, episodes=10, seed=2040, deterministic=True):
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        score = 0.0
        steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            score += float(reward)
            done = terminated or truncated
            steps += 1
            if done:
                break
        scores.append(score)
        print(f"  ep={ep+1} return={score:.1f} steps={steps}")
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    print(f"  MEAN={mean:.1f} STD={std:.1f}")
    return mean, std, scores

def make_env():
    return gym.make("Humanoid-v5")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=["ppo2028", "sacvn2027", "sacvn2028", "sacvn2029", "ppo2027"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    tasks = {
        "ppo2028": {
            "type": "ppo",
            "model": "runs/ppo_humanoid_v5_s2028/ppo_humanoid_v5.zip",
            "vecnorm": "runs/ppo_humanoid_v5_s2028/vecnormalize.pkl",
        },
        "ppo2027": {
            "type": "ppo",
            "model": "runs/ppo_humanoid_v5_s2027/ppo_humanoid_v5.zip",
            "vecnorm": "runs/ppo_humanoid_v5_s2027/vecnormalize.pkl",
        },
        "sacvn2027": {
            "type": "sac",
            "model": "runs/sac_vecnorm_humanoid_v5_s2027/sac_vecnorm_humanoid_v5.zip",
            "vecnorm": "runs/sac_vecnorm_humanoid_v5_s2027/vecnormalize.pkl",
        },
        "sacvn2028": {
            "type": "sac",
            "model": "runs/sac_vecnorm_humanoid_v5_s2028/sac_vecnorm_humanoid_v5.zip",
            "vecnorm": "runs/sac_vecnorm_humanoid_v5_s2028/vecnormalize.pkl",
        },
        "sacvn2029": {
            "type": "sac",
            "model": "runs/sac_vecnorm_humanoid_v5_s2029/sac_vecnorm_humanoid_v5.zip",
            "vecnorm": "runs/sac_vecnorm_humanoid_v5_s2029/vecnormalize.pkl",
        },
    }

    cfg = tasks[args.task]
    if not Path(cfg["model"]).exists():
        print(f"ERROR: model not found: {cfg['model']}")
        sys.exit(1)
    if not Path(cfg["vecnorm"]).exists():
        print(f"ERROR: vecnorm not found: {cfg['vecnorm']}")
        sys.exit(1)

    print(f"=== {args.task} ===")
    print(f"Model: {cfg['model']}")
    print(f"VecNorm: {cfg['vecnorm']}")

    base_env = DummyVecEnv([make_env])
    env = VecNormalize.load(cfg["vecnorm"], base_env)
    env.training = False

    if cfg["type"] == "ppo":
        model = PPO.load(cfg["model"], device=args.device)
    else:
        model = SAC.load(cfg["model"], device=args.device)

    mean, std, scores = evaluate(model, env, episodes=args.episodes, seed=args.seed)

    env.close()
    # Print final line in machine-parseable format
    print(f"FINAL: task={args.task} mean={mean:.2f} std={std:.2f} scores={[round(s,1) for s in scores]}")


if __name__ == "__main__":
    main()
