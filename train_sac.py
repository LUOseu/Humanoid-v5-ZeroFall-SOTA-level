import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
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
    parser.add_argument("--save-dir", type=str, default="runs/sac_humanoid_v5")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--net-width", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID (sets CUDA_VISIBLE_DEVICES)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.total_timesteps > MAX_ASSIGNMENT_STEPS:
        raise ValueError(f"total timesteps must be <= {MAX_ASSIGNMENT_STEPS}")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "sac_humanoid_v5.zip"
    replay_buffer_path = save_dir / "sac_humanoid_v5_replay_buffer.pkl"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_random_seed(args.seed)

    env = make_vec_env(
        "Humanoid-v5",
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=None,
        wrapper_class=Monitor,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
    )
    env = VecMonitor(env)

    if args.resume and model_path.exists():
        model = SAC.load(str(model_path), env=env, device=args.device, seed=args.seed)
        if replay_buffer_path.exists():
            model.load_replay_buffer(str(replay_buffer_path))
    else:
        model = SAC(
            "MlpPolicy",
            env,
            seed=args.seed,
            device=args.device,
            verbose=1,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=10_000,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=-1,
            ent_coef="auto",
            target_update_interval=1,
            tensorboard_log=str(save_dir / "tb"),
            policy_kwargs=dict(net_arch=[args.net_width, args.net_width]),
        )

    config = vars(args).copy()
    config.update(
        env_id="Humanoid-v5",
        algorithm="SAC",
        max_assignment_steps=MAX_ASSIGNMENT_STEPS,
        gymnasium="1.2.3",
        mujoco="3.8.1",
        notes="SAC run does not use reward/observation normalization; evaluation reports raw environment return.",
    )
    (save_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, 1_000_000 // args.n_envs),
        save_path=str(save_dir / "checkpoints"),
        name_prefix="sac_humanoid_v5",
        save_replay_buffer=False,
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint_cb,
            tb_log_name="sac",
            reset_num_timesteps=not args.resume,
            progress_bar=True,
        )
    finally:
        model.save(str(model_path))
        model.save_replay_buffer(str(replay_buffer_path))
        env.close()


if __name__ == "__main__":
    main()
