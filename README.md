# Humanoid-v5 CrossQ Benchmark

Open-source training and evaluation scripts for Gymnasium `Humanoid-v5`, with a focus on reproducible 5M-step comparisons between PPO, SAC, SAC+VecNormalize, and CrossQ.

This repository is best read as a benchmark/reproduction project, not as a new RL algorithm. The main finding is that CrossQ is a very strong and stable baseline for `Humanoid-v5` under this setup.

## Highlights

- **CrossQ 3-seed baseline**: `9121 +/- 282` return at 5M environment steps.
- **Best stability run**: CrossQ R024 reaches about `9938 +/- 30` over 10 deterministic evaluation episodes, with all episodes reaching the 1000-step horizon.
- **PPO/SAC comparison**: CrossQ strongly outperforms local PPO (`7465 +/- 935`) and SAC (`6660 +/- 101`) baselines.
- **Important claim boundary**: this is **not** claimed as overall `Humanoid-v5` SOTA. Public TQC expert policies report higher mean return, but our best CrossQ run has much lower episode-return variance under our evaluation protocol.

## Result Summary

| System | Steps | Seeds | Eval return | Notes |
|---|---:|---:|---:|---|
| CrossQ baseline | 5M | 3 | `9121 +/- 282` | Strongest reproduced baseline in this repo |
| CrossQ R024 | 5M | 1 | `~9938 +/- 30` | Best stability run, 10/10 episodes reached 1000 steps |
| PPO + VecNormalize | 5M | 3 | `7465 +/- 935` | One seed had a collapse episode |
| SAC | 5M | 3 | `6660 +/- 101` | Stable but lower return |
| SAC + VecNormalize | 5M | 3 | `4943 +/- 1725` | High variance, frequent early collapses |
| Public TQC expert | unknown/release artifact | model card | `10370 +/- 1542` | Higher mean, much larger episode-return variance |

See [RESULTS.md](RESULTS.md) for the full result tables and claim wording.

## Repository Layout

| Path | Purpose |
|---|---|
| `train_crossq.py` | Main CrossQ training script for 5M-step benchmark runs |
| `train_crossq_v2.py` | Configurable CrossQ script for hyperparameter exploration |
| `evaluate_crossq.py` | Deterministic CrossQ evaluation on raw `Humanoid-v5` return |
| `train.py` / `train_ppo_v2.py` | PPO baselines with `VecNormalize` |
| `train_sac.py` | SAC baseline without observation/reward normalization |
| `train_sac_vecnorm.py` | SAC + `VecNormalize` diagnostic run |
| `evaluate.py`, `evaluate_sac.py`, `eval_all_models.py` | Evaluation helpers |
| `refine-logs/` | Experiment trackers, plans, and result notes |
| `review-stage/` | Internal review notes and caveats |
| `R024_demo.mp4` | Short demo video for the best stability run |

## Install

Python 3.10 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For headless GPU machines, set MuJoCo rendering before recording videos:

```bash
export MUJOCO_GL=egl
```

## Train

CrossQ baseline:

```bash
python train_crossq.py \
  --seed 2027 \
  --total-timesteps 5000000 \
  --n-envs 4 \
  --save-dir runs/crossq_humanoid_v5_s2027 \
  --device cuda
```

Repeat with seeds `2028` and `2029` for the 3-seed baseline.

Configurable CrossQ exploration:

```bash
python train_crossq_v2.py \
  --seed 2027 \
  --total-timesteps 5000000 \
  --n-envs 4 \
  --net-arch 512,256,128 \
  --batch-size 128 \
  --learning-rate 1e-3 \
  --save-dir runs/crossq_r024_s2027 \
  --device cuda
```

PPO and SAC baselines:

```bash
python train.py --seed 2026 --total-timesteps 5000000 --n-envs 32 --save-dir runs/ppo_humanoid_v5_s2026
python train_sac.py --seed 2027 --total-timesteps 5000000 --n-envs 4 --save-dir runs/sac_humanoid_v5_s2027
python train_sac_vecnorm.py --seed 2027 --total-timesteps 5000000 --n-envs 4 --save-dir runs/sac_vecnorm_humanoid_v5_s2027
```

## Evaluate

CrossQ:

```bash
python evaluate_crossq.py \
  --model runs/crossq_humanoid_v5_s2027/crossq_humanoid_v5.zip \
  --episodes 10 \
  --seed 2040 \
  --device cuda
```

PPO:

```bash
python evaluate.py \
  --model runs/ppo_humanoid_v5_s2026/ppo_humanoid_v5.zip \
  --vecnormalize runs/ppo_humanoid_v5_s2026/vecnormalize.pkl \
  --episodes 10 \
  --seed 2040 \
  --device cuda
```

SAC:

```bash
python evaluate_sac.py \
  --model runs/sac_humanoid_v5_s2027/sac_humanoid_v5.zip \
  --episodes 10 \
  --seed 2040 \
  --device cuda
```

## Reproducibility

The scripts set Python, NumPy, PyTorch, action-space, observation-space, and vectorized-environment seeds where applicable. Evaluation reports raw `Humanoid-v5` return. PPO and SAC+VecNormalize disable reward normalization during evaluation.

For public claims, store:

- `config.json`
- model `.zip`
- `vecnormalize.pkl` for PPO/SAC+VecNormalize
- evaluation stdout with per-episode returns and step counts
- TensorBoard logs
- demo video when available

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for details.

## Claim Wording

Recommended:

> We release a strong open-source CrossQ benchmark on Gymnasium `Humanoid-v5`: `9121 +/- 282` over 3 seeds at 5M environment steps. Our best run reaches about `9938 +/- 30` over 10 deterministic evaluation episodes, with all episodes reaching the 1000-step horizon.

Avoid:

> State-of-the-art on Humanoid-v5.

The stronger but still safe positioning is:

> CrossQ provides a high-return, low-variance `Humanoid-v5` baseline in this 5M-step reproduction, while public TQC expert policies retain the higher reported mean return.

## Citation

If this repository helps your work, cite it using [CITATION.cff](CITATION.cff). Please also cite CrossQ and the underlying Gymnasium/MuJoCo tooling.

