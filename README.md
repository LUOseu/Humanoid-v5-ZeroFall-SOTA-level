# Humanoid-v5 CrossQ Benchmark

This repository provides training and evaluation scripts for reproducible 5M-step experiments on Gymnasium `Humanoid-v5`.

The focus is not to introduce a new reinforcement-learning algorithm. Instead, the repository compares local PPO, SAC, SAC+VecNormalize, and CrossQ baselines under a shared evaluation protocol. The main observation is that CrossQ is a strong and relatively stable baseline in this setup.

## Results

| System | Steps | Seeds | Evaluation return | Notes |
|---|---:|---:|---:|---|
| CrossQ baseline | 5M | 3 | `9121 ± 282` | Best reproduced baseline in this repository |
| CrossQ R024 | 5M | 1 | `9938 ± 30` | Stability-focused run; 10/10 episodes reached 1000 steps |
| PPO + VecNormalize | 5M | 3 | `7465 ± 935` | One seed had a collapse episode |
| SAC | 5M | 3 | `6660 ± 101` | Stable but lower return |
| SAC + VecNormalize | 5M | 3 | `4943 ± 1725` | High variance and frequent early collapses |

The R024 result is a single-seed result evaluated over 10 deterministic episodes. It should not be interpreted as a statistically complete stability study.

This repository does not claim overall `Humanoid-v5` state of the art. Public TQC expert policies report a higher mean return, while the R024 run here shows lower variance under this repository's evaluation protocol.

See [RESULTS.md](RESULTS.md) for detailed tables and caveats.

## Evaluation protocol

- Environment: Gymnasium `Humanoid-v5`
- MuJoCo: `3.8.1`
- Training budget: 5,000,000 environment steps per run
- Evaluation: deterministic policy rollout
- Default evaluation episodes: 10
- Metric: raw cumulative environment return
- Episode horizon: 1000 steps

## Installation

Python 3.10 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
