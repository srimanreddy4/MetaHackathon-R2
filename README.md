# OnCallEnv Red Shift — SRE GRPO

Pure-Python SRE incident-response environment used to GRPO-tune a Qwen-2.5-3B
defender on hard, partially-observable on-call scenarios. The runnable artifact
is `notebooks/07_sre_three_way_grpo.ipynb`.

## Layout

```
src/oncallenv/
  core/          OpenEnv API, Pydantic types, defender tools
  simulation/    microservice graph, scenario compiler, fault primitives
  telemetry/     OTLP-shaped metrics, logs, traces
  rewards/       composable rubrics + sre_shaped reward (format / investigation / remediation)
  curriculum/    regret buffer, ACCEL-style mutator, autocurriculum runner
scripts/
  train_unsloth_grpo.py    Unsloth + TRL GRPO trainer with three-way reward logging
  train_redshift_policy.py reference text/spec MLP policy trainer
notebooks/
  07_sre_three_way_grpo.ipynb  Kaggle T4 GRPO run + baseline-vs-trained eval
curriculum_results/
  buffer.json              precomputed regret buffer (~206 scenarios)
```

## What the environment does

- `OnCallRedShiftEnv` exposes realistic SRE tools: `kubectl_logs`, `promql_query`,
  `jaeger_search`, `istioctl_routes`, `kubectl_rollout_restart`,
  `traffic_split_update`, `submit_rca`, `declare_resolved`.
- Alerts are obfuscated: every incident surfaces as a generic `api-gateway`
  symptom so the policy must trace through the dependency graph instead of
  pattern-matching on a leaked service.
- The simulator emits OpenTelemetry-shaped metrics, JSON logs, and Jaeger-style
  traces with realistic strings (OOMKilled, exit code 137, x509 expired,
  context deadline exceeded, Envoy flags).

## Reward (`oncallenv.rewards.sre_shaped`)

The `sre_shaped_reward` total in `[-0.6, +1.0]` is split into three GRPO reward
functions so TRL/WandB log them as separate columns:

| Function                  | Range            | Measures                                                  |
| ------------------------- | ---------------- | --------------------------------------------------------- |
| `sre_format_score`        | `[-0.50, +0.30]` | think/actions block well-formed                           |
| `sre_investigation_score` | `[-0.35, +0.25]` | Read-only tools touch the right services / required pairs |
| `sre_remediation_score`   | `[-0.15, +0.50]` | Correct fix tool on correct service + RCA category match  |

Sub-scores sum exactly to the pre-clip total. A `functools.lru_cache` on the
parsed completion ensures the simulator runs once per generation, not three
times.

## Curriculum filter

`scripts/train_unsloth_grpo.py` accepts `--max-solve-rate 0.25` to load only
hard tasks (~73 / 206 scenarios in the shipped buffer).

## Run

On Kaggle (T4) the notebook does it all. Locally:

```
pip install -r requirements.txt
pip install -r requirements-llm.txt
PYTHONPATH=src python scripts/train_unsloth_grpo.py \
  --buffer curriculum_results/buffer.json \
  --max-solve-rate 0.25 \
  --reward-mode sre \
  --steps 200
```

Smoke-test the env:

```
from oncallenv import OnCallRedShiftEnv
from oncallenv.core.types import Action

env = OnCallRedShiftEnv()
obs = env.reset(task_id="evolved_0001_823796c06b")
print(obs.alerts[0].service, obs.alerts[0].message)
obs = env.step(Action(command="jaeger_search api-gateway"))
obs = env.step(Action(command="kubectl_logs payment-service"))
obs = env.step(Action(command="kubectl_rollout_restart payment-service"))
obs = env.step(Action(command="declare_resolved"))
```
