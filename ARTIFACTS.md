# Artifact Publishing Guide

Large training artifacts are intentionally ignored by git. Use GitHub Releases, Hugging Face, Google Drive, or another artifact host for model files and logs.

## Recommended Release Assets

For each run, publish a directory or archive with:

- `config.json`
- final model `.zip`
- `vecnormalize.pkl` for PPO and SAC+VecNormalize
- evaluation output, for example `eval_seed2040_episodes10.txt`
- TensorBoard events under `tb/`
- optional checkpoints
- optional demo video

## Suggested Names

```text
crossq_humanoid_v5_s2027_5m.zip
crossq_humanoid_v5_s2028_5m.zip
crossq_humanoid_v5_s2029_5m.zip
crossq_r024_s2027_5m.zip
ppo_humanoid_v5_s2026_5m.zip
sac_humanoid_v5_s2027_5m.zip
```

## README Links

After uploading artifacts, add links in `README.md` under a new `Artifacts` section:

```markdown
## Artifacts

| Run | Model | Eval log | Demo |
|---|---|---|---|
| CrossQ R024 | [model](...) | [eval log](...) | [video](...) |
```

Do not commit model `.zip`, replay buffer `.pkl`, or TensorBoard event files directly unless the repository is intentionally using Git LFS.

