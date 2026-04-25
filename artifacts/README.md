# Local Model Artifacts

This directory is the local landing zone for trained model artifacts downloaded from Kaggle or the professor GPU server.

Large files are intentionally ignored by git. Keep downloaded archives and extracted checkpoints under:

```text
artifacts/models/
```

## Kaggle Easy GRPO Export

In the Kaggle-backed notebook, run:

```bash
!git pull
!bash scripts/run_kaggle_qwen3b_grpo.sh export-easy-artifacts
```

Download this file from Kaggle:

```text
/kaggle/working/easy_grpo_qwen3b_artifacts.tar.gz
```

Place it locally at:

```text
artifacts/models/easy_grpo_qwen3b_artifacts.tar.gz
```

Then extract locally from the repo root:

```bash
mkdir -p artifacts/models/easy_grpo_qwen3b
tar -xzf artifacts/models/easy_grpo_qwen3b_artifacts.tar.gz -C artifacts/models/easy_grpo_qwen3b
```

Expected contents include:

- `training_results/unsloth_grpo_qwen3b_easy/checkpoint-*`
- `training_results/unsloth_grpo_qwen3b_easy/baseline_generations.json`
- `training_results/unsloth_grpo_qwen3b_easy/checkpoint_report.json`
- `training_results/unsloth_grpo_qwen3b_easy/checkpoint_eval_summary.json` if checkpoint eval was run
- `training_results/unsloth_grpo_qwen3b_easy/checkpoint_generations.json` if checkpoint eval was run
- `docs/plots/easy_grpo_qwen3b_*.png`

The trained model artifact is a LoRA/PEFT checkpoint, not a full merged Qwen base model. The base model remains:

```text
unsloth/Qwen2.5-3B-Instruct-bnb-4bit
```
