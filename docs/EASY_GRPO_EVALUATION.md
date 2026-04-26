# Easy GRPO Evaluation Suite

This evaluation pass focuses on in-domain held-out behavior, where the Qwen2.5-3B easy GRPO checkpoint is actually trained to operate.

## Fast Analysis

Run without loading the model:

```bash
PYTHONPATH=src:scripts python scripts/analyze_easy_grpo_generations.py \
  --artifact-archive artifacts/easy_grpo_qwen3b_artifacts.tar.gz
```

Kaggle:

```bash
bash scripts/run_kaggle_qwen3b_grpo.sh easy-analysis
```

Outputs:

```text
eval_results/easy_grpo_analysis/summary.json
eval_results/easy_grpo_analysis/easy_grpo_reward_by_step.png
eval_results/easy_grpo_analysis/easy_grpo_process_metrics.png
eval_results/easy_grpo_analysis/easy_grpo_reward_by_fault.png
eval_results/easy_grpo_analysis/easy_grpo_reward_by_service.png
eval_results/easy_grpo_analysis/easy_grpo_task_reward_distribution.png
```

Current artifact result:

| Metric | Value |
| --- | ---: |
| Mean reward | 0.9051 |
| Valid XML action block rate | 1.0000 |
| Root service mention rate | 1.0000 |
| Required tool-family hit rate | 1.0000 |
| Exact required remediation coverage | 0.9833 |
| All required remediations exact | 0.9667 |
| Diagnostic action rate | 0.7333 |
| Diagnostic before mutation rate | 0.6333 |
| Early `declare_resolved` rate | 0.4000 |
| Unsafe stateful restart/undo rate | 0.3333 |

Interpretation:

- The checkpoint is strong at following the easy action format and hitting the required remediation commands.
- The main weakness is process discipline: it often declares resolution too early and sometimes restarts/rolls back stateful services.
- This is a better story than RCAEval zero-shot because it explains what the trained SRE policy has learned and where it still needs safety/process reward shaping.

## GPU Checkpoint Curve

Run on Kaggle/GPU when time permits:

```bash
bash scripts/run_kaggle_qwen3b_grpo.sh easy-checkpoint-curve
```

This loads each saved checkpoint and evaluates it on the same held-out rows:

```text
checkpoint-50
checkpoint-100
checkpoint-150
checkpoint-200
```

Outputs:

```text
eval_results/easy_grpo_checkpoint_curve/summary.json
eval_results/easy_grpo_checkpoint_curve/easy_grpo_checkpoint_reward_curve.png
eval_results/easy_grpo_checkpoint_curve/checkpoint_50_generations.json
eval_results/easy_grpo_checkpoint_curve/checkpoint_100_generations.json
eval_results/easy_grpo_checkpoint_curve/checkpoint_150_generations.json
eval_results/easy_grpo_checkpoint_curve/checkpoint_200_generations.json
```
