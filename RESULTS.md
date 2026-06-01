# Results

This page collects the benchmark results used by the README. The numbers come from the local experiment trackers under `refine-logs/`.

## Evaluation Protocol

- Environment: Gymnasium `Humanoid-v5`
- MuJoCo: `3.8.1`
- Main training budget: 5,000,000 environment steps per run
- Evaluation: deterministic policy rollout
- Default evaluation episodes: 10
- Reported metric: raw environment cumulative return
- Episode horizon: 1000 steps, using the environment's default truncation

## CrossQ Results

CrossQ uses the `sb3-contrib` implementation. The baseline configuration uses BatchNorm in the critic, no target network, no `VecNormalize`, `batch_size=128`, `net_arch=[256, 256]`, `learning_rate=1e-3`, and `gradient_steps=1`.

| Run | Config | Seed | Eval return | Stability note |
|---|---|---:|---:|---|
| CrossQ baseline | batch=128, `[256, 256]`, lr=1e-3 | 2027 | `8836 +/- 2832` | 1 collapse |
| CrossQ baseline | batch=128, `[256, 256]`, lr=1e-3 | 2028 | `9399 +/- 60` | Perfect stability in 10 episodes |
| CrossQ baseline | batch=128, `[256, 256]`, lr=1e-3 | 2029 | `9129 +/- 491` | 1 partial low-return episode |
| CrossQ baseline mean | same | 3 seeds | `9121 +/- 282` | Across-seed sample std |
| CrossQ R024 | batch=128, `[512, 256, 128]`, lr=1e-3 | 2027 | `~9938 +/- 30` | 10/10 episodes reached 1000 steps |

The R024 result is the best stability-focused run observed so far. It should be treated as a single-seed result until additional seeds and longer evaluation are logged.

## PPO / SAC Baselines

| System | Seed | Eval return | Notes |
|---|---:|---:|---|
| PPO + VecNormalize | 2026 | `8107 +/- 54` | All 10 episodes reached 1000 steps |
| PPO + VecNormalize | 2027 | `7900 +/- 50` | All 10 episodes reached 1000 steps |
| PPO + VecNormalize | 2028 | `6387 +/- 1747` | 1/10 episodes collapsed at step 189 |
| SAC | 2027 | `6787 +/- 25` | All 10 episodes reached 1000 steps |
| SAC | 2028 | `6610 +/- 20` | All 10 episodes reached 1000 steps |
| SAC | 2029 | `6584 +/- 11` | All 10 episodes reached 1000 steps |
| SAC + VecNormalize | 2027 | `6883 +/- 2372` | About half of episodes collapsed early |
| SAC + VecNormalize | 2028 | `4521 +/- 3113` | Frequent early collapse |
| SAC + VecNormalize | 2029 | `3425 +/- 2450` | Frequent early collapse |

Summary:

| System | Seeds | Mean return | Across-seed std | Stability |
|---|---:|---:|---:|---|
| CrossQ baseline | 3 | `9121` | `282` | High mean, two seeds had at least one unstable episode |
| PPO + VecNormalize | 3 | `7465` | `935` | 29/30 episodes reached 1000 steps |
| SAC | 3 | `6660` | `101` | 30/30 episodes reached 1000 steps |
| SAC + VecNormalize | 3 | `4943` | `1725` | Frequent early collapses |

## Public Comparison

A public Farama/Minari `Humanoid-v5` TQC expert policy reports about `10370 +/- 1542`, which is higher than the CrossQ mean here. For that reason, this repository does not claim overall `Humanoid-v5` SOTA.

The stability comparison is still meaningful:

| Model | Mean | Std | Stability framing |
|---|---:|---:|---|
| Public TQC expert | `~10370` | `~1542` | Higher mean, large episode-return spread |
| CrossQ R024 | `~9938` | `~30` | Lower mean, much tighter 10-episode evaluation |

Use this comparison carefully: TQC's reported result used many more evaluation episodes, while R024 currently has a 10-episode evaluation. A matched 100 or 1000 episode evaluation would make the stability claim much stronger.

## Suggested Claims

Safe:

- CrossQ is the strongest reproduced baseline in this repository.
- CrossQ reaches `9121 +/- 282` over 3 seeds at 5M steps.
- R024 shows very low episode-return variance in a 10-episode deterministic evaluation.
- SAC + VecNormalize was unstable in this off-policy setup, supporting the replay-buffer/running-statistics mismatch concern.

Avoid:

- Overall `Humanoid-v5` SOTA.
- Method novelty claims for CrossQ itself.
- Stability superiority over TQC without noting protocol mismatch.

