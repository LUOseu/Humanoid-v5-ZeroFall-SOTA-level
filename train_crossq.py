"""Train CrossQ (BatchNorm SAC) on Humanoid-v5.

CrossQ (Bhatt et al., ICLR 2024) removes target networks and uses
BatchNorm in the critic to stabilize training. This should avoid the
VecNormalize replay-buffer mismatch that kills standard SAC.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import CrossQ
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

MAX_ASSIGNMENT_STEPS = 5_000_000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="runs/crossq_humanoid_v5")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--net-width", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


class SaveReplayBufferCallback(BaseCallback):
    def __init__(self, path: Path, save_freq: int = 1_000_000):
        super().__init__()
        self.path = path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save_replay_buffer(str(self.path))
        return True


def main():
    args = parse_args()
    if args.total_timesteps > MAX_ASSIGNMENT_STEPS:
        raise ValueError(f"total timesteps must be <= {MAX_ASSIGNMENT_STEPS}")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "crossq_humanoid_v5.zip"
    replay_buffer_path = save_dir / "crossq_humanoid_v5_replay_buffer.pkl"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_random_seed(args.seed)

    base_env = make_vec_env(
        "Humanoid-v5",
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=None,
        wrapper_class=Monitor,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
    )
    env = VecMonitor(base_env)

    model = CrossQ(
        "MlpPolicy",
        env,
        seed=args.seed,
        device=args.device,
        verbose=1,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=10_000,
        batch_size=args.batch_size,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        tensorboard_log=str(save_dir / "tb"),
        policy_kwargs=dict(
            net_arch=[args.net_width, args.net_width],
        ),
    )

    config = vars(args).copy()
    config.update(
        env_id="Humanoid-v5",
        algorithm="CrossQ+BatchNorm",
        max_assignment_steps=MAX_ASSIGNMENT_STEPS,
        gymnasium="1.2.3",
        mujoco="3.8.1",
        sb3_contrib="2.8.0",
        notes="CrossQ (ICLR 2024) with BatchNorm critic. No target network, no VecNormalize.",
    )
    (save_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, 1_000_000 // args.n_envs),
        save_path=str(save_dir / "checkpoints"),
        name_prefix="crossq_humanoid_v5",
        save_replay_buffer=False,
    )
    replay_cb = SaveReplayBufferCallback(replay_buffer_path, save_freq=max(1, 1_000_000 // args.n_envs))

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_cb, replay_cb],
            tb_log_name="crossq",
            progress_bar=True,
        )
    finally:
        model.save(str(model_path))
        model.save_replay_buffer(str(replay_buffer_path))
        env.close()


if __name__ == "__main__":
    main()
