import argparse
import json
import os
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import set_random_seed


MAX_ASSIGNMENT_STEPS = 5_000_000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--save-dir", type=str, default="runs/ppo_humanoid_v5")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--net-width", type=int, default=256)
    return parser.parse_args()


def make_env(seed: int):
    def _init():
        env = gym.make("Humanoid-v5")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return Monitor(env)

    return _init


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, vecnorm_path: Path, save_freq: int = 50_000):
        super().__init__()
        self.vecnorm_path = vecnorm_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.training_env.save(str(self.vecnorm_path))
        return True


def build_base_env(seed: int, n_envs: int):
    set_random_seed(seed)
    env = make_vec_env(
        "Humanoid-v5",
        n_envs=n_envs,
        seed=seed,
        monitor_dir=None,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else None,
    )
    env = VecMonitor(env)
    return env


def main():
    args = parse_args()
    if args.total_timesteps > MAX_ASSIGNMENT_STEPS:
        raise ValueError(f"total timesteps must be <= {MAX_ASSIGNMENT_STEPS}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "ppo_humanoid_v5.zip"
    vecnorm_path = save_dir / "vecnormalize.pkl"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_random_seed(args.seed)

    base_env = build_base_env(args.seed, args.n_envs)
    steps_per_update = args.n_steps

    if args.resume and model_path.exists() and vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), base_env)
        env.training = True
        model = PPO.load(str(model_path), env=env, device=args.device, seed=args.seed)
    else:
        env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            device=args.device,
            verbose=1,
            n_steps=steps_per_update,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=args.learning_rate,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=str(save_dir / "tb"),
            policy_kwargs=dict(net_arch=dict(pi=[args.net_width, args.net_width], vf=[args.net_width, args.net_width])),
        )

    config = vars(args).copy()
    config.update(
        env_id="Humanoid-v5",
        algorithm="PPO",
        max_assignment_steps=MAX_ASSIGNMENT_STEPS,
        gymnasium="1.2.3",
        mujoco="3.8.1",
        notes="VecNormalize statistics are saved to vecnormalize.pkl and must be loaded for evaluation.",
    )
    (save_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, 100_000 // args.n_envs),
        save_path=str(save_dir / "checkpoints"),
        name_prefix="ppo_humanoid_v5",
        save_vecnormalize=True,
    )
    vecnorm_cb = SaveVecNormalizeCallback(vecnorm_path, save_freq=max(1, 20_000 // args.n_envs))

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_cb, vecnorm_cb],
            tb_log_name="ppo",
            reset_num_timesteps=not args.resume,
            progress_bar=True,
        )
    finally:
        model.save(str(model_path))
        env.save(str(vecnorm_path))
        env.close()


if __name__ == "__main__":
    main()
