"""Evaluate CrossQ model (no VecNormalize needed)."""
import argparse
import sys
from pathlib import Path
import gymnasium as gym
import numpy as np
from sb3_contrib import CrossQ

def evaluate(model, env, episodes=10, seed=2040):
    scores = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed+ep)
        score = 0.0
        steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            score += float(reward)
            steps += 1
            if terminated or truncated:
                break
        scores.append(score)
        print(f"  ep_{ep+1}={score:.1f} steps={steps}")
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    print(f"RESULT: mean={mean:.2f} std={std:.2f}")
    return mean, std

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)

    print(f"=== CrossQ Eval: {args.model} ===")
    env = gym.make("Humanoid-v5")
    model = CrossQ.load(args.model, device=args.device)
    mean, std = evaluate(model, env, episodes=args.episodes, seed=args.seed)
    env.close()

if __name__ == "__main__":
    main()
