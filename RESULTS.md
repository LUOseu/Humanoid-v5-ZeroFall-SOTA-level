# Results

This document reports the currently reproduced `Humanoid-v5` results in this repository.

The numbers below should be interpreted as repository-level benchmark results under the stated 5M-step training setup. They are not presented as a new algorithmic state-of-the-art claim.

## Evaluation Protocol

- Environment: Gymnasium `Humanoid-v5`
- MuJoCo: `3.8.1`
- Training budget: 5,000,000 environment steps per run
- Evaluation policy: deterministic rollout
- Default evaluation episodes: 10
- Reported metric: raw cumulative environment return
- Episode horizon: 1000 steps
- Reward normalization during evaluation: disabled for normalized baselines

## Main Result Summary

| System | Steps | Seeds | Evaluation return | Notes |
|---|---:|---:|---:|---|
| CrossQ baseline | 5M | 3 | `9121 ± 282` | Best reproduced 3-seed baseline in this repository |
| CrossQ R024 | 5M | 1 | `~9938 ± 30` | Stability-focused run; 10/10 episodes reached 1000 steps |
| PPO + VecNormalize | 5M | 3 | `7465 ± 935` | One seed had a collapse episode |
| SAC | 5M | 3 | `6660 ± 101` | Stable but lower return |
| SAC + VecNormalize | 5M | 3 | `4943 ± 1725` | High variance and frequent early collapses |

## CrossQ Results

CrossQ uses the `sb3-contrib` implementation.

The baseline configuration uses BatchNorm in the critic, no target network, no `VecNormalize`, `batch_size=128`, `net_arch=[256, 256]`, `learning_rate=1e-3`, and `gradient_steps=1`.

| Run | Steps | Seed setting | Evaluation return | Notes |
|---|---:|---:|---:|---|
| CrossQ baseline | 5M | 3 seeds | `9121 ± 282` | Main reproduced CrossQ baseline |
| CrossQ R024 | 5M | 1 seed | `~9938 ± 30` | Stability-focused checkpoint |

The R024 run is a single-seed result evaluated over 10 deterministic episodes. It is useful as a stability example, but it should not be interpreted as a statistically complete multi-seed result.

## PPO and SAC Baselines

| Algorithm | Steps | Seeds | Evaluation return | Notes |
|---|---:|---:|---:|---|
| PPO + VecNormalize | 5M | 3 | `7465 ± 935` | One seed had a collapse episode |
| SAC | 5M | 3 | `6660 ± 101` | Stable but lower return |
| SAC + VecNormalize | 5M | 3 | `4943 ± 1725` | High variance and frequent early collapses |

Under this repository's 5M-step setup, CrossQ obtains the strongest reproduced result among the local PPO, SAC, SAC+VecNormalize, and CrossQ baselines.

## External Reference

Some public expert policies for `Humanoid-v5`, such as TQC-based expert policies, report higher mean returns than the results reproduced in this repository. Those policies are external references and are not trained or reproduced here.

Therefore, this repository should not claim overall `Humanoid-v5` state of the art.

A safe comparison is:

> CrossQ provides a strong `Humanoid-v5` baseline under this repository's 5M-step evaluation protocol, while some public expert policies may retain higher reported mean return.

## Recommended Claim Boundary

Recommended wording:

> We release a reproducible CrossQ benchmark for Gymnasium `Humanoid-v5`. In our 5M-step setup, CrossQ reaches `9121 ± 282` over three seeds. A stability-focused single-seed run reaches approximately `9938 ± 30` over 10 deterministic evaluation episodes, with all episodes reaching the 1000-step horizon.

Avoid:

> State-of-the-art on Humanoid-v5.

Also avoid:

> Solves Humanoid-v5.

## How to Strengthen the Evidence

For stronger public claims, add the following:

- 100 or 1000 deterministic evaluation episodes for the R024 checkpoint
- at least 3 training seeds for the R024 configuration
- per-episode return list
- per-episode length list
- exact training commands
- exact evaluation commands
- TensorBoard logs or exported CSV learning curves
- model checkpoints or external artifact links
- demo video
