
---

```markdown
# Artifacts

This repository should keep lightweight code and documentation in GitHub. Large generated files should be stored through GitHub Releases, Hugging Face, Google Drive, or another persistent artifact host.

## Current Local Artifact

| Artifact | Description |
|---|---|
| `R024_demo.mp4` | Demo video for the stability-focused CrossQ run |

## Recommended Public Artifacts

For each published run, provide:

| Artifact | Required | Notes |
|---|---:|---|
| `config.json` | Yes | Hyperparameters, seed, environment, total steps |
| `train_command.txt` | Yes | Exact command used for training |
| `eval_command.txt` | Yes | Exact command used for evaluation |
| `eval_stdout.txt` | Yes | Must include per-episode returns and lengths |
| model checkpoint | Recommended | Use GitHub Releases or Hugging Face if large |
| `vecnormalize.pkl` | Required for normalized baselines | Needed to reproduce PPO/SAC+VecNormalize evaluation |
| TensorBoard logs | Recommended | Useful for learning curves |
| exported scalar CSV files | Recommended | Easier to inspect than TensorBoard logs alone |
| demo video | Optional | Useful for qualitative inspection |

## Suggested Release Structure

```text
release-v0.1/
  crossq_baseline_s2027/
    config.json
    train_command.txt
    eval_command.txt
    eval_stdout.txt
    model.zip
    tensorboard/
  crossq_baseline_s2028/
    config.json
    train_command.txt
    eval_command.txt
    eval_stdout.txt
    model.zip
    tensorboard/
  crossq_baseline_s2029/
    config.json
    train_command.txt
    eval_command.txt
    eval_stdout.txt
    model.zip
    tensorboard/
  crossq_r024_s2027/
    config.json
    train_command.txt
    eval_command.txt
    eval_stdout.txt
    R024_demo.mp4
