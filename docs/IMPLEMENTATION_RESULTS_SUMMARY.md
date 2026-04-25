# Implementation and Results Summary

This document summarizes the Round 2 work completed so far, the training runs completed or in progress, and the current result status.

## Executive Summary

OnCallEnv Red Shift has been upgraded into an OpenEnv-compatible SRE incident-response training environment with:

- A pure-Python microservice incident simulator.
- Realistic SRE actions and telemetry.
- Composable OpenEnv reward rubrics.
- Regret/autocurriculum scenario generation.
- A trained symbolic neural defender policy.
- A real Unsloth + TRL GRPO LLM training path.
- Kaggle 2xT4 support for a complementary Qwen2.5 3B GRPO run.

The strongest completed quantitative result remains the curriculum-trained symbolic policy:

| Run | Baseline | Trained | Status |
| --- | ---: | ---: | --- |
| Seed GPU symbolic policy | 0.5589 | 0.9789 | Complete |
| Curriculum symbolic policy | 0.5538-0.5626 | 0.9073 | Complete |
| Qwen2.5 3B GRPO smoke | 0.2615 | 0.3145 | Complete |
| Qwen2.5 3B GRPO main | partial logs around 0.35-0.37 reward | final eval pending | Interrupted around step 317/600 |

## Implementation Completed

### OpenEnv Environment

Implemented `OnCallRedShiftEnv`, an OpenEnv-compatible incident-response environment.

Key behavior:

- `reset()` loads seed or evolved scenarios.
- `step()` executes realistic SRE commands.
- Observations include alerts, available tools, services, elapsed time, and reward breakdown.
- `task_id="curriculum"` samples from the autocurriculum buffer.
- Round 1 code is preserved in `v1_legacy/`.

### Simulator and Faults

Built a fast pure-Python simulator for microservice incidents.

Implemented fault families include:

- OOM kill / memory leak.
- DNS misconfiguration.
- Certificate expiry.
- Cache stampede.
- Replica lag.
- HTTP 503 loop.
- CPU hog.
- Clock skew.
- Deadlock.
- Disk full.
- GC pause.
- Network partition.

This allows high-throughput rollouts without Kubernetes or Chaos Mesh.

### SRE Tool Surface

Implemented realistic tools, including:

- `kubectl_logs`
- `kubectl_top`
- `promql_query`
- `jaeger_search`
- `istioctl_routes`
- `kubectl_rollout_restart`
- `kubectl_rollout_undo`
- `kubectl_scale`
- `feature_flag_toggle`
- `traffic_split_update`
- `kubectl_apply_config`
- `declare_resolved`
- `submit_rca`

### Telemetry

Added OpenTelemetry-style signals:

- Metrics.
- JSON logs.
- Jaeger-style traces.
- Realistic incident strings such as `OOMKilled`, `exit code 137`, `x509: certificate has expired`, and `context deadline exceeded`.

### Reward Rubrics

Implemented composable OpenEnv reward rubrics:

| Rubric | Weight | Purpose |
| --- | ---: | --- |
| Recovery | 0.35 | Rewards actual recovery after declaring resolution. |
| RCA quality | 0.30 | Rewards grounded RCA content, timeline, five-whys, and action items. |
| Blast radius | 0.25 | Penalizes slow recovery and customer-facing impact. |
| Safety | 0.10 | Penalizes risky or destructive actions. |

### Regret Autocurriculum

Implemented an ACCEL-style regret/autocurriculum pipeline.

Artifacts:

- `curriculum_results/buffer.json`
- `curriculum_results/summary.json`
- `curriculum_results/scenarios/*.yaml`
- `docs/plots/autocurriculum_diversity.png`
- `docs/plots/autocurriculum_solve_rate_hist.png`

Results:

| Metric | Value |
| --- | ---: |
| Evolved scenarios | 200 |
| Final buffer size | 206 |
| Solve-rate mean | 0.3150 |
| Regret mean | 0.5917 |

### FastAPI, Docker, Notebooks, Scripts

Implemented or updated:

- FastAPI app and Docker surface.
- OpenEnv validation path.
- Smoke, training, and evaluation notebooks.
- Remote server training launcher.
- Kaggle Qwen2.5 3B GRPO notebook.
- Interrupted-run summarizer and checkpoint resume support.

Key scripts:

- `scripts/train_redshift_policy.py`
- `scripts/evaluate_redshift_policy.py`
- `scripts/run_autocurriculum.py`
- `scripts/train_unsloth_grpo.py`
- `scripts/run_kaggle_qwen3b_grpo.sh`
- `scripts/run_unsloth_grpo_screen.sh`
- `scripts/summarize_unsloth_grpo.py`

## Training Runs and Results

### 1. Remote Simulator Sanity Run

Purpose:

- Verify the environment rewards correct incident-response behavior.
- Compare a weak baseline to a scripted responder.

Result:

| Policy | Mean reward |
| --- | ---: |
| Weak baseline | 0.2681 |
| Scripted defender | 0.9211 |

Artifact:

- `remote_results/summary.json`
- `docs/plots/remote_baseline_vs_scripted.png`

### 2. Seed GPU Symbolic Policy

Model:

- Small PyTorch MLP over symbolic SRE action choices.
- Trained on six seed incidents.
- Device: Tesla V100-SXM2-32GB.

Result:

| Metric | Value |
| --- | ---: |
| Steps | 800 |
| Batch size | 96 |
| Duration | 26.32 sec |
| Baseline mean | 0.5589 |
| Trained mean | 0.9789 |

Artifacts:

- `training_results/gpu_policy/policy.pt`
- `training_results/gpu_policy/summary.json`
- `docs/plots/gpu_policy_baseline_vs_trained.png`
- `docs/plots/gpu_policy_training_curve.png`

### 3. Curriculum GPU Symbolic Policy

Model:

- Small PyTorch MLP over symbolic SRE actions.
- Trained on the autocurriculum buffer.
- Device: Tesla V100-SXM2-32GB.

Training result:

| Metric | Value |
| --- | ---: |
| Steps | 2500 |
| Batch size | 256 |
| Duration | 321.81 sec |
| Train split | 164 scenarios |
| Eval split | 42 scenarios |
| Baseline mean | 0.5626 |
| Trained mean | 0.9073 |

Independent checkpoint evaluation:

| Metric | Value |
| --- | ---: |
| Baseline mean | 0.5538 |
| Trained mean | 0.9073 |

Artifacts:

- `training_results/curriculum_policy/policy.pt`
- `training_results/curriculum_policy/summary.json`
- `eval_results/curriculum_policy/summary.json`
- `docs/plots/curriculum_policy_baseline_vs_trained.png`
- `docs/plots/curriculum_policy_training_curve.png`

### 4. Unsloth Qwen2.5 3B GRPO Smoke Run

Purpose:

- Validate real LLM GRPO training on Kaggle 2xT4.
- Confirm LoRA parameters are actually trainable.
- Prove the text-to-action reward loop works end-to-end.

Model:

- `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
- LoRA GRPO with Unsloth + TRL.
- Kaggle GPU: 2x Tesla T4 available, training configured conservatively.

Completed smoke result:

| Metric | Value |
| --- | ---: |
| Steps | 50 |
| Train tasks | 40 |
| Eval tasks | 8 |
| Duration | 1472.07 sec |
| Baseline mean reward | 0.2615 |
| Trained mean reward | 0.3145 |
| Trainable parameters | 29,933,568 |
| LoRA tensors | 504 |

Interpretation:

- The LLM GRPO path works.
- LoRA is training correctly.
- Reward improved from 0.2615 to 0.3145 in the smoke run.

Artifacts on Kaggle:

- `training_results/unsloth_grpo_qwen3b_smoke/summary.json`
- `training_results/unsloth_grpo_qwen3b_smoke/baseline_generations.json`
- `training_results/unsloth_grpo_qwen3b_smoke/trained_generations.json`
- `training_results/unsloth_grpo_qwen3b_smoke/adapter/`

### 5. Unsloth Qwen2.5 3B GRPO Main Run

Purpose:

- Run a larger complementary LLM experiment while the campus server handles the default 1.5B GRPO run.

Configuration:

| Setting | Value |
| --- | ---: |
| Model | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` |
| Planned steps | 600 |
| Train tasks | 160 |
| Eval tasks | 32 |
| Save steps | 100 |
| Batch size per device | 2 |
| Num generations | 2 |
| Gradient accumulation | 8 |

Current status:

- Run was interrupted around step 317/600.
- Checkpoints should exist at `checkpoint-100`, `checkpoint-200`, and `checkpoint-300`.
- Final `summary.json` is not expected unless training reaches the post-training evaluation phase.
- Logged reward around step 300 was approximately 0.35-0.37.

Important distinction:

- The smoke run has a completed final summary.
- The 600-step main run is currently a partial checkpointed run.
- Use `scripts/summarize_unsloth_grpo.py` to summarize interrupted checkpoints.

Kaggle commands:

```bash
bash scripts/run_kaggle_qwen3b_grpo.sh summary
bash scripts/run_kaggle_qwen3b_grpo.sh resume-main
bash scripts/run_kaggle_qwen3b_grpo.sh archive
```

## Validation Status

Local validation has passed:

| Check | Status |
| --- | --- |
| Unit tests | `21 passed` |
| OpenEnv validation | Passed |
| Training script compile checks | Passed |
| Kaggle helper shell syntax | Passed |

## Current Best Submission Numbers

Use these as the clean headline numbers right now:

| Method | Baseline | Trained | Notes |
| --- | ---: | ---: | --- |
| Scripted defender sanity check | 0.2681 | 0.9211 | Not learned, validates environment. |
| Seed symbolic neural policy | 0.5589 | 0.9789 | Completed GPU training. |
| Curriculum symbolic neural policy | 0.5538 | 0.9073 | Strongest completed learned result. |
| Qwen2.5 3B GRPO smoke | 0.2615 | 0.3145 | Completed LLM smoke run. |
| Qwen2.5 3B GRPO main | partial | approx. 0.35-0.37 logged reward | Interrupted around step 317/600. |

## What Still Needs To Be Done

Highest priority:

1. Resume or archive the interrupted 600-step Qwen2.5 3B GRPO main run.
2. Pull Kaggle result artifacts locally.
3. If the campus 1.5B GRPO run finishes, add its summary beside the Kaggle 3B result.
4. Update README result tables and plots with final LLM numbers.
5. Add final submission links: Hugging Face Space, Colab/Kaggle notebook, video, blog, and deck.

Optional but valuable:

1. Run a 1.5B vs 3B LLM comparison table.
2. Add plot for GRPO logged reward over training.
3. Add a short demo episode script/video using one seed and one evolved scenario.
4. Add a final Docker smoke test before submission packaging.

## Honest Current Status

The project now has two kinds of learning evidence:

- A strong, completed symbolic policy result on the curriculum environment.
- A real LLM GRPO pipeline that has completed a smoke run and improved reward, with a larger 600-step run partially complete and checkpointed.

For the submission narrative, the safest claim is:

> Red Shift is a complete OpenEnv SRE training environment with composable rubrics, regret autocurriculum, and GPU-trained defenders. The curriculum symbolic policy improves from roughly 0.55 to 0.91 reward. A real Unsloth Qwen2.5 3B GRPO text-policy run has been validated on Kaggle, improving from 0.2615 to 0.3145 in a smoke run, with a longer checkpointed run reaching roughly 0.35-0.37 logged reward before interruption.

