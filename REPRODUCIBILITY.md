# Reproducibility

This document describes the reproducibility protocol for the `Humanoid-v5` experiments in this repository.

## Environment

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Recommended Python version:

```text
Python 3.10
```

For headless GPU machines, set MuJoCo rendering before recording videos:

```bash
export MUJOCO_GL=egl
```

## Dependencies

The main dependencies are listed in [requirements.txt](requirements.txt).

Core packages:

- Gymnasium
- MuJoCo
- Stable-Baselines3
- sb3-contrib
- PyTorch
- NumPy
- TensorBoard
- imageio
- Pillow

## Training Setup

All main experiments use Gymnasium `Humanoid-v5`.

Default training budget:

```text
5,000,000 environment steps
```

The repository currently reports results for:

- PPO + VecNormalize
- SAC
- SAC + VecNormalize
- CrossQ
- CrossQ R024 stability-focused run

## Evaluation Protocol

Default evaluation setting:

```text
Environment: Humanoid-v5
Policy: deterministic
Episodes: 10
Metric: raw cumulative environment return
Episode horizon: 1000 steps
```

For PPO and SAC+VecNormalize, reward normalization should be disabled during evaluation.

## Seeding

Training scripts should set seeds for:

- Python
- NumPy
- PyTorch
- Gymnasium environment
- action space
- observation space
- vectorized environments, when used

Evaluation should report:

- evaluation seed
- number of episodes
- per-episode return
- per-episode length

## Recommended Commands

CrossQ baseline:

```bash
python train_crossq.py \
  --seed 2027 \
  --total-timesteps 5000000 \
  --n-envs 4 \
  --save-dir runs/crossq_humanoid_v5_s2027 \
  --device cuda
```

PPO baseline:

```bash
python train.py \
  --seed 2026 \
  --total-timesteps 5000000 \
  --n-envs 32 \
  --save-dir runs/ppo_humanoid_v5_s2026
```

SAC baseline:

```bash
python train_sac.py \
  --seed 2027 \
  --total-timesteps 5000000 \
  --n-envs 4 \
  --save-dir runs/sac_humanoid_v5_s2027
```

Evaluation example:

```bash
python evaluate_crossq.py \
  --model runs/crossq_humanoid_v5_s2027/crossq_humanoid_v5.zip \
  --episodes 10 \
  --seed 2040 \
  --device cuda
```

## Artifacts to Keep

For each reported run, keep:

- training command
- evaluation command
- model checkpoint
- `vecnormalize.pkl`, if the run uses VecNormalize
- evaluation stdout
- per-episode returns
- per-episode lengths
- TensorBoard logs or exported CSV files

## Reporting Notes

Use the format:

```text
mean ± standard deviation
```

Always state whether the standard deviation is computed across training seeds or evaluation episodes.

The current `CrossQ baseline` result is a 3-seed result. The current `CrossQ R024` result is a single-seed evaluation result and should be treated as a stability-focused run rather than a complete multi-seed benchmark.
