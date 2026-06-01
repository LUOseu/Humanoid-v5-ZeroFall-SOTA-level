"""Train PPO on Humanoid-v5 with configurable hyperparameters for optimization experiments."""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

MAX_ASSIGNMENT_STEPS = 5_000_000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--save-dir", type=str, default="runs/ppo_humanoid_v5")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", action="store_true")
    # PPO hyperparams
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    # Network
    parser.add_argument("--net-arch", type=str, default="256,256",
                        help="Comma-separated: '512,256,128' or '256,256'")
    parser.add_argument("--ortho-init", action="store_true")
    parser.add_argument("--use-sde", action="store_true")
    # LR schedule
    parser.add_argument("--lr-schedule", type=str, default="constant",
                        choices=["constant", "linear"])
    parser.add_argument("--lr-end", type=float, default=1e-5,
                        help="Final LR for linear schedule")
    # GPU
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, vecnorm_path: Path, save_freq: int = 50_000):
        super().__init__()
        self.vecnorm_path = vecnorm_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.training_env.save(str(self.vecnorm_path))
        return True


def make_lr_schedule(start_lr: float, end_lr: float, total_steps: int):
    """Linear learning rate schedule."""
    def schedule(progress_remaining: float) -> float:
        return end_lr + (start_lr - end_lr) * progress_remaining
    return schedule


def main():
    args = parse_args()
    if args.total_timesteps > MAX_ASSIGNMENT_STEPS:
        raise ValueError(f"total timesteps must be <= {MAX_ASSIGNMENT_STEPS}")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    net_arch = [int(x) for x in args.net_arch.split(",")]
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

    base_env = make_vec_env(
        "Humanoid-v5",
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=None,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
    )
    base_env = VecMonitor(base_env)

    if args.resume and model_path.exists():
        env = VecNormalize.load(str(vecnorm_path), base_env)
        env.training = True
        model = PPO.load(str(model_path), env=env, device=args.device, seed=args.seed)
    else:
        env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

        lr = args.learning_rate
        if args.lr_schedule == "linear":
            lr = make_lr_schedule(args.learning_rate, args.lr_end, args.total_timesteps)

        policy_kwargs = dict(
            net_arch=dict(pi=net_arch, vf=net_arch),
            ortho_init=args.ortho_init,
        )

        model = PPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            device=args.device,
            verbose=1,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            learning_rate=lr,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            use_sde=args.use_sde,
            tensorboard_log=str(save_dir / "tb"),
            policy_kwargs=policy_kwargs,
        )

    config = vars(args).copy()
    config.update(
        env_id="Humanoid-v5",
        algorithm="PPO",
        max_assignment_steps=MAX_ASSIGNMENT_STEPS,
        net_arch=net_arch,
        gymnasium="1.2.3",
        mujoco="3.8.1",
    )
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))

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
