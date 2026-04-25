# Easy GRPO Eval Knowledge Transfer

## Short Answer

There is no separate `easy_eval_scenarios/` folder.

The easy GRPO run uses the same Red Shift scenarios as the hard run:

- Source scenario index: `curriculum_results/buffer.json`
- Materialized evolved YAMLs: `curriculum_results/scenarios/*.yaml`
- Seed fallback scenarios: `scenarios_seed/*.yaml`

What is new in easy mode is not the scenario set. The new pieces are:

- Easier prompts with root-cause/runbook hints.
- Dense shaped rewards with partial credit.
- Prompt variants per scenario.
- A generated training/eval dataset written at run time.

## Where The Easy Eval Rows Are

After running:

```bash
bash scripts/run_kaggle_qwen3b_grpo.sh easy-main
```

the exact prompts used for training and eval are written here:

```text
training_results/unsloth_grpo_qwen3b_easy/dataset.jsonl
```

The eval rows are the first `--eval-tasks` rows from that file after deterministic shuffle.

For easy main:

```text
max_tasks = 120
prompt_variants = 3
train/eval dataset rows = 120 * 3 = 360
eval_tasks = 32
eval rows = first 32 rows in dataset.jsonl
```

For easy smoke:

```text
max_tasks = 40
prompt_variants = 2
train/eval dataset rows = 40 * 2 = 80
eval_tasks = 8
eval rows = first 8 rows in dataset.jsonl
```

## Code Path

The task list is built in:

```text
scripts/train_unsloth_grpo.py
```

Important functions:

- `load_task_ids(...)`: loads task IDs from `curriculum_results/buffer.json`; if the buffer is missing, it falls back to seed tasks.
- `inspect_task(...)`: resets the environment for each task and extracts root service, fault category, required remediations, and prompt text.
- `build_prompt(...)`: creates either hard or easy prompts.
- `shaped_easy_reward(...)`: gives partial-credit easy reward.
- `evaluate_model(...)`: evaluates baseline and trained generations on the selected eval rows.

The Kaggle run mode is defined in:

```text
scripts/run_kaggle_qwen3b_grpo.sh
```

Easy main expands to:

```bash
python scripts/train_unsloth_grpo.py \
  --curriculum-buffer curriculum_results/buffer.json \
  --out-dir training_results/unsloth_grpo_qwen3b_easy \
  --max-tasks 120 \
  --max-steps 300 \
  --eval-tasks 32 \
  --reward-mode easy \
  --prompt-mode easy \
  --prompt-variants 3
```

## Current Easy-Main Eval Rows

With the default seed `20260424`, `easy-main` evaluates these 32 rows:

| # | Task | Prompt | Root Service | Fault | Required Remediation |
| ---: | --- | --- | --- | --- | --- |
| 1 | `evolved_0170_7d5cd0cef1` | standard | `redis-cache` | `clock_skew` | `kubectl_apply_config:inventory-service`, `kubectl_rollout_restart:redis-cache` |
| 2 | `evolved_0010_8aa9da8e06` | standard | `redis-cache` | `cert_expiry` | `kubectl_apply_config:inventory-service`, `kubectl_apply_config:redis-cache` |
| 3 | `evolved_0137_404e32e1f8` | standard | `redis-cache` | `cert_expiry` | `kubectl_apply_config:api-gateway`, `kubectl_apply_config:redis-cache` |
| 4 | `evolved_0059_15bd1795ef` | runbook | `redis-cache` | `dns_misconfig` | `feature_flag_toggle:checkout-service`, `kubectl_apply_config:redis-cache` |
| 5 | `evolved_0191_412ace2f63` | runbook | `user-service` | `dns_misconfig` | `kubectl_apply_config:user-service` |
| 6 | `evolved_0089_3526439777` | triage | `redis-cache` | `cert_expiry` | `feature_flag_toggle:inventory-service`, `kubectl_apply_config:redis-cache` |
| 7 | `evolved_0092_4eb403c031` | standard | `postgres-primary` | `cert_expiry` | `kubectl_apply_config:postgres-primary` |
| 8 | `evolved_0033_d61c0d23d5` | runbook | `redis-cache` | `network_partition` | `kubectl_rollout_undo:checkout-service`, `traffic_split_update:redis-cache` |
| 9 | `evolved_0164_a2f80c213b` | standard | `postgres-primary` | `http_503_loop` | `feature_flag_toggle:redis-cache`, `kubectl_rollout_undo:postgres-primary` |
| 10 | `evolved_0140_14df7713f6` | standard | `postgres-primary` | `cert_expiry` | `kubectl_apply_config:postgres-primary` |
| 11 | `evolved_0083_2b0acf4515` | triage | `checkout-service` | `cpu_hog` | `kubectl_scale:checkout-service` |
| 12 | `evolved_0004_63fa0016a2` | standard | `api-gateway` | `cert_expiry` | `kubectl_apply_config:api-gateway` |
| 13 | `evolved_0007_9b2e7570b5` | standard | `redis-cache` | `cert_expiry` | `kubectl_apply_config:postgres-primary`, `kubectl_apply_config:redis-cache` |
| 14 | `evolved_0072_696576af3b` | triage | `checkout-service` | `replica_lag` | `feature_flag_toggle:checkout-service` |
| 15 | `evolved_0004_63fa0016a2` | triage | `api-gateway` | `cert_expiry` | `kubectl_apply_config:api-gateway` |
| 16 | `evolved_0186_6c98c960ed` | triage | `postgres-primary` | `oom_kill` | `kubectl_rollout_restart:postgres-primary` |
| 17 | `evolved_0047_ba047400b4` | runbook | `redis-cache` | `disk_full` | `kubectl_apply_config:redis-cache`, `kubectl_rollout_restart:checkout-service` |
| 18 | `evolved_0094_c3ec7cf9d4` | triage | `redis-cache` | `disk_full` | `kubectl_apply_config:redis-cache`, `kubectl_rollout_restart:checkout-service` |
| 19 | `evolved_0030_dcb3db647a` | triage | `postgres-primary` | `disk_full` | `kubectl_apply_config:postgres-primary`, `kubectl_rollout_restart:redis-cache` |
| 20 | `evolved_0197_88a931f292` | runbook | `postgres-primary` | `deadlock` | `kubectl_rollout_restart:postgres-primary` |
| 21 | `evolved_0113_c2918e3b1d` | runbook | `redis-cache` | `network_partition` | `feature_flag_toggle:inventory-service`, `traffic_split_update:redis-cache` |
| 22 | `evolved_0188_02bdafaf1e` | standard | `redis-cache` | `disk_full` | `kubectl_apply_config:redis-cache`, `traffic_split_update:checkout-service` |
| 23 | `evolved_0193_2244c6d8e2` | standard | `redis-cache` | `oom_kill` | `kubectl_rollout_restart:checkout-service`, `kubectl_rollout_restart:redis-cache` |
| 24 | `evolved_0113_c2918e3b1d` | standard | `redis-cache` | `network_partition` | `feature_flag_toggle:inventory-service`, `traffic_split_update:redis-cache` |
| 25 | `evolved_0199_6b9c73c7b4` | standard | `redis-cache` | `cert_expiry` | `kubectl_apply_config:redis-cache`, `kubectl_scale:postgres-primary` |
| 26 | `evolved_0159_330292c509` | runbook | `redis-cache` | `cache_stampede` | `feature_flag_toggle:redis-cache`, `kubectl_rollout_restart:postgres-primary` |
| 27 | `evolved_0108_2b1094d421` | standard | `redis-cache` | `network_partition` | `feature_flag_toggle:checkout-service`, `traffic_split_update:redis-cache` |
| 28 | `evolved_0014_b1b3d6ee85` | standard | `redis-cache` | `cert_expiry` | `kubectl_apply_config:checkout-service`, `kubectl_apply_config:redis-cache` |
| 29 | `evolved_0002_516316e0e5` | triage | `redis-cache` | `cert_expiry` | `kubectl_apply_config:redis-cache`, `kubectl_apply_config:user-service` |
| 30 | `evolved_0134_301844eb45` | triage | `redis-cache` | `dns_misconfig` | `kubectl_apply_config:redis-cache`, `kubectl_rollout_restart:checkout-service` |
| 31 | `evolved_0192_49b93a5c23` | runbook | `redis-cache` | `deadlock` | `kubectl_rollout_restart:redis-cache`, `kubectl_rollout_undo:checkout-service` |
| 32 | `evolved_0046_96b6b18e41` | standard | `checkout-service` | `cert_expiry` | `kubectl_apply_config:checkout-service` |

## How To Reproduce The Eval List

Run this from the repo root:

```bash
PYTHONPATH=src:scripts python - <<'PY'
from pathlib import Path
from train_unsloth_grpo import load_task_ids, inspect_task, PROMPT_TEMPLATES
import random

task_ids = load_task_ids(Path("curriculum_results/buffer.json"), 120, 20260424)
rows = []
for task_id in task_ids:
    for idx in range(3):
        rows.append(inspect_task(task_id, prompt_mode="easy", template=PROMPT_TEMPLATES[idx % len(PROMPT_TEMPLATES)]))
random.Random(20260424).shuffle(rows)

for i, row in enumerate(rows[:32], 1):
    print(i, row["task_id"], row["template"], row["root_service"], row["root_category"], row["required"])
PY
```

## KT Talking Points

- The simulator remains the real Red Shift environment.
- Easy mode does not create artificial easy incidents.
- Easy mode exposes more guidance to the model and makes reward less binary.
- This is useful as a curriculum stage: teach action format and remediation mapping first, then move back toward hard prompts/rewards.
- The exact generated prompts can always be audited from `dataset.jsonl`.
