# Reproducibility Notes

This repository is designed for reproducible `Humanoid-v5` training runs, but MuJoCo locomotion experiments can still vary across hardware, drivers, PyTorch/CUDA builds, and evaluation episode counts.

## Environment

Recommended:

- Python 3.10
- `gymnasium==1.2.3`
- `mujoco==3.8.1`
- `stable-baselines3[extra]==2.8.0`
- `sb3-contrib==2.8.0`
- PyTorch with CUDA support on Linux for full-speed training

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Headless rendering:

```bash
export MUJOCO_GL=egl
```

## Seeds

The training scripts set:

- Python `random`
- NumPy
- PyTorch CPU/CUDA seeds
- Stable-Baselines3 random seed
- vectorized environment seed
- action and observation space seeds where evaluation scripts create raw Gymnasium environments

The main benchmark seeds are:

- PPO: `2026`, `2027`, `2028`
- SAC: `2027`, `2028`, `2029`
- SAC + VecNormalize: `2027`, `2028`, `2029`
- CrossQ: `2027`, `2028`, `2029`

## Evaluation

All evaluation scripts report raw environment return. For normalized environments, evaluation disables reward normalization:

- PPO: `VecNormalize.training = False`, `VecNormalize.norm_reward = False`
- SAC + VecNormalize: use the matching saved `vecnormalize.pkl`
- CrossQ and SAC: no normalization wrapper is needed

Default evaluation:

```bash
python evaluate_crossq.py --model runs/crossq_humanoid_v5_s2027/crossq_humanoid_v5.zip --episodes 10 --seed 2040
```

For public stability claims, prefer a longer evaluation:

```bash
python evaluate_crossq.py --model runs/crossq_r024_s2027/crossq_humanoid_v5.zip --episodes 100 --seed 2040
```

or:

```bash
python evaluate_crossq.py --model runs/crossq_r024_s2027/crossq_humanoid_v5.zip --episodes 1000 --seed 2040
```

## Artifact Checklist

For each published run, keep:

- `config.json`
- final model `.zip`
- replay buffer `.pkl`, if needed for resuming
- `vecnormalize.pkl`, if the algorithm used `VecNormalize`
- TensorBoard events under `tb/`
- checkpoint directory, if available
- evaluation stdout with each episode's return and step count
- demo video for qualitative inspection

Recommended run directory layout:

```text
runs/
  crossq_humanoid_v5_s2027/
    config.json
    crossq_humanoid_v5.zip
    crossq_humanoid_v5_replay_buffer.pkl
    tb/
    checkpoints/
    eval_seed2040_episodes10.txt
```

## Known Caveats

- The best R024 stability number is currently a single-seed 10-episode result. It should not be presented as a statistically complete stability study until rerun with more episodes and additional seeds.
- Public TQC expert results have a higher reported mean return, so this repository does not claim overall `Humanoid-v5` SOTA.
- SAC + VecNormalize is included as a diagnostic baseline. Its instability is expected to be related to off-policy replay interacting with evolving normalization statistics.

