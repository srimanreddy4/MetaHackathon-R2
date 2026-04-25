# Round 2 Implementation and Training Summary

This document summarizes what was implemented for the Round 2 submission, what was trained, the exact results obtained, and what still remains before final submission.

## Current Branch and Commit State

- Active branch: `round2-redshift`
- Round 1 preserved at tag: `round1-submission`
- Latest Round 2 commit at the time of this report: `71fcd97 training: add curriculum policy results`
- Round 1 implementation was moved into `v1_legacy/` so the Round 2 environment can be presented cleanly without losing the old submission.

## High-Level Goal

The project was upgraded from the original Round 1 submission into **OnCallEnv Red Shift**, an OpenEnv-compatible SRE incident-response training environment.

The core idea is:

- A simulated production system generates realistic incidents.
- A defender agent investigates and applies SRE actions.
- A reviewer/rubric layer scores the outcome using recovery, RCA quality, blast radius, and safety.
- Scenarios can be evolved through a regret/autocurriculum loop.
- A lightweight neural policy can be trained and evaluated on seed and evolved scenarios.

## What Was Implemented

### 1. OpenEnv-Compatible Environment

Implemented `OnCallRedShiftEnv` in `src/oncallenv/core/env.py`.

Key capabilities:

- Subclasses the installed `openenv.core.Environment`.
- Supports `reset()` and `step()` with realistic incident-response observations and actions.
- Supports seed tasks and evolved curriculum tasks.
- Supports `task_id="curriculum"` to sample from the regret buffer.
- Exposes deterministic, fast simulation suitable for many rollouts.

### 2. Pure-Python SRE Incident Simulator

Implemented simulator modules under `src/oncallenv/simulation/`.

Major components:

- Service dependency graph.
- Scenario compiler.
- Fault primitives.
- Deterministic incident state updates.

Implemented or represented fault types include:

- Memory leak / OOM kill.
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

This avoids needing a real Kubernetes or Chaos Mesh cluster for every rollout, which keeps training fast and cheap.

### 3. Realistic SRE Tool Surface

Implemented SRE-like defender tools in `src/oncallenv/core/tools.py`.

Examples:

- `kubectl_logs`
- `promql_query`
- `jaeger_search`
- `istioctl_routes`
- `kubectl_rollout_restart`
- `kubectl_rollout_undo`
- `kubectl_scale`
- `kubectl_apply_config`
- `feature_flag_toggle`
- `traffic_split_update`
- `declare_resolved`
- `submit_rca`

The action space used by the trained symbolic policy is built from these command families and services.

### 4. OpenTelemetry-Style Telemetry

Implemented telemetry generation in `src/oncallenv/telemetry/otlp.py`.

The environment produces production-style signals:

- Metrics.
- JSON logs.
- Jaeger-style traces.
- Incident strings such as `OOMKilled`, `exit code 137`, `x509: certificate has expired`, `context deadline exceeded`, and Envoy failure flags.

### 5. Composable Reward Rubrics

Implemented reward modules under `src/oncallenv/rewards/`.

The final reward is composed using OpenEnv `WeightedSum`:

| Component | Weight | Purpose |
| --- | ---: | --- |
| Recovery rubric | 0.35 | Rewards actual recovery after `declare_resolved`. |
| RCA quality rubric | 0.30 | Scores timeline, root cause, five-whys quality, and action items. |
| Blast radius rubric | 0.25 | Penalizes slow recovery and user-facing error exposure. |
| Safety rubric | 0.10 | Penalizes unsafe or destructive actions. |

This is important for Round 2 because the reward is not a single brittle heuristic; it is decomposed into interpretable judging criteria.

### 6. Reviewer Fallback

Implemented deterministic reviewer logic in `src/oncallenv/agents/reviewer.py`.

Purpose:

- Provides stable RCA/reviewer scoring without requiring an external LLM call.
- Makes tests, local validation, Docker runs, and remote server runs reproducible.
- Leaves room to plug in LLM-as-a-judge later if desired.

### 7. Regret Autocurriculum

Implemented curriculum modules under `src/oncallenv/curriculum/`.

Components:

- `RegretBuffer`
- `ScenarioMutator`
- `AutocurriculumRunner`

What it does:

- Starts from seed incidents.
- Mutates scenarios.
- Scores candidate scenarios using solve-rate/regret-style signals.
- Keeps a curriculum buffer of harder and diverse tasks.
- Writes evolved YAML scenarios under `curriculum_results/scenarios/`.

Generated artifacts:

- `curriculum_results/buffer.json`
- `curriculum_results/summary.json`
- `curriculum_results/scenarios/*.yaml`
- `docs/plots/autocurriculum_diversity.png`
- `docs/plots/autocurriculum_solve_rate_hist.png`

Autocurriculum results:

| Metric | Value |
| --- | ---: |
| Requested iterations | 200 |
| Evolved scenarios | 200 |
| Final buffer size | 206 |
| Solve-rate min | 0.0845 |
| Solve-rate max | 0.6705 |
| Solve-rate mean | 0.3150 |
| Regret mean | 0.5917 |

Fault distribution among evolved scenarios:

| Fault type | Count |
| --- | ---: |
| replica_lag | 40 |
| cert_expiry | 36 |
| deadlock | 26 |
| dns_misconfig | 20 |
| http_503_loop | 17 |
| cache_stampede | 16 |
| clock_skew | 16 |
| cpu_hog | 15 |
| oom_kill | 9 |
| disk_full | 4 |
| gc_pause | 4 |
| network_partition | 3 |

### 8. FastAPI / Docker / Submission Surface

Implemented:

- FastAPI app in `src/oncallenv/server/app.py`.
- Root wrappers: `app.py` and `server/app.py`.
- Dockerfile updated for the Round 2 app layout.
- `openenv.yaml` updated for validation.
- `scripts/validate.sh` for local validation.

The Docker image was previously built successfully during the implementation pass.

### 9. Notebooks

Added/updated notebooks:

- `notebooks/01_smoke_test.ipynb`
- `notebooks/02_train_grpo_unsloth.ipynb`
- `notebooks/03_eval_baseline_vs_trained.ipynb`
- `notebooks/04_kaggle_qwen3b_grpo.ipynb`
- `notebooks/05_easy_grpo_kaggle.ipynb`

Important clarification:

- The original completed trainer is a lightweight neural policy over the symbolic SRE action space.
- The later Kaggle work added real Unsloth Qwen2.5-3B LoRA checkpoints for both easy all-at-once GRPO and interactive ReAct-style SFT.

## What Models Were Used

### Model 1: Deterministic Scripted Defender

Used for:

- Sanity checking the environment.
- Establishing that the environment rewards correct incident resolution.
- Remote simulator comparison against a weak baseline.

This is not a learned model. It is a scripted policy that chooses known corrective actions for seed incidents.

Remote simulator result:

| Policy | Mean reward |
| --- | ---: |
| Weak baseline | 0.2681 |
| Scripted defender | 0.9211 |

Artifact:

- `remote_results/summary.json`
- `docs/plots/remote_baseline_vs_scripted.png`

### Model 2: Seed GPU Neural Defender Policy

Used for:

- Proving that the environment can train a neural policy on GPU.
- Training over the original seed incidents.

Implementation:

- Script: `scripts/train_redshift_policy.py`
- Checkpoint: `training_results/gpu_policy/policy.pt`
- Model type: small PyTorch MLP classifier/policy over symbolic SRE actions.
- Device: CUDA
- GPU: Tesla V100-SXM2-32GB
- Steps: 800
- Batch size: 96
- Duration: 26.32 seconds

Training result:

| Policy | Mean reward |
| --- | ---: |
| Random/weak baseline | 0.5589 |
| Trained seed policy | 0.9789 |

Evaluation artifact:

- `eval_results/gpu_policy/summary.json`
- `docs/plots/gpu_policy_baseline_vs_trained.png`
- `docs/plots/gpu_policy_training_curve.png`

### Model 3: Curriculum-Aware GPU Neural Defender Policy

Used for:

- Training over the regret/autocurriculum buffer.
- Demonstrating improvement on evolved scenarios, not only the six seed incidents.
- Producing the strongest Round 2 training artifact.

Implementation:

- Script: `scripts/train_redshift_policy.py`
- Checkpoint: `training_results/curriculum_policy/policy.pt`
- Model type: small PyTorch MLP classifier/policy over symbolic SRE actions.
- Feature mode: `spec`
- Optimizer: AdamW
- Input dimension: 91
- Curriculum buffer: `curriculum_results/buffer.json`
- Train split size: 164
- Eval split size: 42
- Device: CUDA
- GPU: Tesla V100-SXM2-32GB
- Steps: 2500
- Batch size: 256
- Duration: 321.81 seconds, about 5.4 minutes

Training command:

```bash
PYTHONPATH=src:scripts python scripts/train_redshift_policy.py \
  --steps 2500 \
  --batch-size 256 \
  --lr 0.003 \
  --curriculum-buffer curriculum_results/buffer.json \
  --feature-mode spec \
  --out-dir training_results/curriculum_policy \
  --plot-prefix curriculum_policy
```

Training result:

| Policy | Mean reward |
| --- | ---: |
| Random/weak baseline | 0.5626 |
| Curriculum-trained policy | 0.9073 |

Independent checkpoint evaluation:

| Policy | Mean reward |
| --- | ---: |
| Random/weak baseline | 0.5538 |
| Curriculum-trained policy | 0.9073 |

Artifacts:

- `training_results/curriculum_policy/policy.pt`
- `training_results/curriculum_policy/summary.json`
- `training_results/curriculum_policy/train.log`
- `eval_results/curriculum_policy/summary.json`
- `eval_results/curriculum_policy_remote.log`
- `docs/plots/curriculum_policy_baseline_vs_trained.png`
- `docs/plots/curriculum_policy_training_curve.png`

### Model 4: Kaggle Qwen2.5-3B Easy GRPO Checkpoint

Used for:

- Running an actual Unsloth/LoRA text-to-actions training lane.
- Testing easier prompts and shaped reward credit so the LLM receives a smoother learning signal.
- Producing a high-score checkpoint comparison against the earlier hard all-at-once GRPO run.

Implementation:

- Script: `scripts/train_unsloth_grpo.py`
- Notebook: `notebooks/05_easy_grpo_kaggle.ipynb`
- Model: `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
- Mode: all-at-once `<actions>` generation
- Latest evaluated checkpoint: `training_results/unsloth_grpo_qwen3b_easy/checkpoint-200`

Checkpoint result:

| Policy | Mean reward |
| --- | ---: |
| Easy baseline | 0.7464 |
| Easy GRPO checkpoint | 0.9051 |

Artifacts:

- `docs/easy_grpo_interrupted_report.json`
- `docs/plots/easy_grpo_qwen3b_reward_curve.png`
- `docs/plots/easy_grpo_qwen3b_checkpoint_results.png`
- `docs/plots/easy_grpo_qwen3b_completion_health.png`

### Model 5: Interactive ReAct Qwen2.5-3B SFT Checkpoint

Used for:

- Training a defender that acts one command at a time after each observation.
- Demonstrating an SRE-style investigation process rather than a single all-at-once action block.
- Creating trajectory artifacts that can power an interactive UI/demo.

Implementation:

- Scripts: `scripts/generate_react_trajectories.py`, `scripts/train_react_sft.py`, `scripts/evaluate_react_sft_checkpoint.py`
- Notebook: `notebooks/05_easy_grpo_kaggle.ipynb`
- Model: `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
- Dataset: 120 scripted expert scenarios, 927 turn-level rows, deterministic 100/20 train/eval task split
- Latest evaluated checkpoint: `training_results/react_sft_qwen3b/checkpoint-100`

Checkpoint result:

| Metric | Value |
| --- | ---: |
| Held-out next-action accuracy | 0.9583 |
| Interactive mean reward, 5 rollout fast eval | 0.8068 |
| Scripted expert mean reward on generated trajectories | 0.9715 |

Artifacts:

- `docs/react_sft_checkpoint_eval_report.json`
- `docs/react_sft_interrupted_report.json`
- `docs/plots/react_sft_qwen3b_fast_checkpoint_eval.png`
- `docs/plots/react_sft_qwen3b_fast_rollout_rewards.png`

## Server Usage

The professor server was used for remote validation and GPU training.

Safety notes:

- Existing GPU processes were not killed.
- The main training runs used GPU 1 when it was free.
- The final curriculum run used `CUDA_VISIBLE_DEVICES=1`.
- Remote working directories were created under `/mnt/himanshu/oncallenv-redshift-runs/`.

Important remote run directories:

- Initial simulator run: `/mnt/himanshu/oncallenv-redshift-runs/run-20260424-160959`
- Curriculum policy run: `/mnt/himanshu/oncallenv-redshift-runs/run-e128a14-20260424-164718`

## Validation Completed

Validation checks completed successfully:

```bash
PYTHONPATH=src openenv validate .
PYTHONPATH=src:scripts python -m pytest tests -q
python -m py_compile scripts/train_redshift_policy.py scripts/evaluate_redshift_policy.py scripts/run_autocurriculum.py
```

Results:

- OpenEnv validation: passed.
- Test suite: `21 passed`.
- Training/evaluation scripts compile successfully.

## Result Plots Added

Committed plots under `docs/plots/`:

- `baseline_vs_trained.png`
- `training_curve.png`
- `gpu_policy_baseline_vs_trained.png`
- `gpu_policy_training_curve.png`
- `curriculum_policy_baseline_vs_trained.png`
- `curriculum_policy_training_curve.png`
- `remote_baseline_vs_scripted.png`
- `autocurriculum_diversity.png`
- `autocurriculum_solve_rate_hist.png`
- `schema_drift_ablation.png`

## What Is Done

Completed:

- Round 2 codebase structure.
- OpenEnv-compatible environment.
- Pure-Python SRE simulator.
- Seed incident scenarios.
- Realistic telemetry.
- SRE tool/action surface.
- Composable reward rubrics.
- Deterministic reviewer.
- Regret/autocurriculum pipeline.
- 200 evolved scenarios.
- GPU seed-policy training.
- GPU curriculum-policy training.
- Evaluation scripts and artifacts.
- Result plots.
- Notebooks for smoke, training, and evaluation.
- README updates.
- Docker/OpenEnv validation path.

## What Still Needs To Be Done

Highest-priority submission tasks:

1. Fill final public links in `README.md`:
   - Hugging Face Space URL.
   - Colab notebook URL.
   - Unlisted YouTube demo URL.
   - Blog URL.
   - Pitch deck URL.

2. Run one final Docker smoke test before packaging:

```bash
docker build -t oncallenv-redshift .
docker run --rm -p 7860:7860 oncallenv-redshift
```

3. Publish/deploy the Hugging Face Space.

4. Create the short demo video:
   - Show OpenEnv validation.
   - Show one incident rollout.
   - Show telemetry/tools.
   - Show reward breakdown.
   - Show training plots and curriculum results.

5. Write final blog/deck narrative around:
   - SRE incident-response as an RL environment.
   - Why pure-Python simulation matters.
   - Why composable rubrics are better than one opaque score.
   - Why regret/autocurriculum creates harder incidents.
   - Training improvement from baseline to curriculum policy.
   - Why the ReAct defender is more realistic than all-at-once action generation.

Optional stretch work:

1. Extend actual LLM training runs.
   - Easy Qwen2.5-3B GRPO has produced a checkpoint reward result.
   - Interactive ReAct SFT has produced a checkpoint action-accuracy and rollout-reward result.
   - The next useful extension is a longer ReAct eval or a short interactive GRPO continuation from the SFT adapter.

2. Add external LLM reviewer mode.
   - Current reviewer is deterministic and reproducible.
   - An LLM reviewer could be added as an optional non-default scoring/audit layer.

3. Add more scenario families.
   - Security incidents.
   - Multi-region failover.
   - Queue backpressure.
   - Bad feature-flag rollouts.

4. Add richer ablations.
   - Reward weights.
   - Curriculum vs no-curriculum.
   - RCA rubric on/off.
   - Safety rubric on/off.

## Honest Status of GRPO/Unsloth

What is ready:

- Environment rollouts.
- Rewards.
- Action parser target.
- Fast simulator.
- Unsloth Qwen2.5-3B easy GRPO checkpoint evaluation.
- Unsloth Qwen2.5-3B interactive ReAct SFT checkpoint evaluation.

What is still not final:

- The easy GRPO run was interrupted and should be reported as a checkpoint result, not a final converged run.
- The ReAct rollout result is a fast 5-task checkpoint eval; a larger held-out rollout eval should be run if GPU time permits.
- The Kaggle adapter artifacts should be exported into `artifacts/models/` for final packaging.

For the current submission, the strongest defensible results are:

- Curriculum symbolic policy: 0.5538 baseline to 0.9073 trained on evolved tasks.
- Easy Qwen2.5-3B GRPO checkpoint: 0.7464 baseline to 0.9051 checkpoint reward.
- Interactive ReAct Qwen2.5-3B SFT checkpoint: 0.9583 next-action accuracy and 0.8068 interactive rollout reward on fast eval.

## Recommended Submission Story

Use this framing:

> OnCallEnv Red Shift turns production incident response into a fast, trainable OpenEnv environment. Instead of relying on slow real infrastructure, it uses a deterministic microservice simulator with realistic telemetry, SRE actions, composable rubrics, and regret-based autocurriculum. A lightweight GPU-trained defender improves from roughly 0.55 baseline reward to 0.91 on evolved incidents, while the seed-policy run reaches 0.98 on the original tasks.

This is truthful, strong, and aligned with the artifacts in the repository.
