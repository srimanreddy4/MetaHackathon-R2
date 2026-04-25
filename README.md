# OnCallEnv Red Shift

A pure-Python, partially-observable SRE training environment for tuning small
language models with **GRPO** (Group Relative Policy Optimization) on hard
on-call incidents. A simulator injects one of 12 fault primitives into one of
6 microservice topologies, surfaces a deliberately generic frontend alert, and
forces the model to investigate, remediate, and submit a grounded RCA without
ever being told the true root cause.

The headline run trains **Qwen-2.5-3B-Instruct** on a Kaggle T4 with TRL +
Unsloth and three split reward channels (format / investigation / remediation)
that show up as separate WandB / TRL log columns.

The runnable artifact is `notebooks/OnCall_RedShift_GRPO_Training.ipynb`.

---

## Repository layout

```
.
├── README.md                          this file
├── LICENSE
├── pyproject.toml                     core package metadata + minimal deps
├── requirements.txt                   runtime deps for the env (openenv-core, pydantic, pyyaml)
├── requirements-llm.txt               extra deps for GRPO training (torch, trl, unsloth, peft, ...)
├── .env.example                       optional REWARD_CAP / CURRICULUM_BUFFER overrides
│
├── notebooks/
│   └── OnCall_RedShift_GRPO_Training.ipynb
│       end-to-end Kaggle T4 notebook: install deps, train an MLP baseline,
│       GRPO-train Qwen-3B, then evaluate MLP vs Qwen-base vs Qwen-GRPO under
│       the SRE-shaped reward and plot the breakdown.
│
├── scripts/
│   ├── train_unsloth_grpo.py          main GRPO trainer (TRL + Unsloth, 4-bit LoRA).
│   │                                   Loads the regret buffer, builds prompts,
│   │                                   generates rollouts, scores them with the
│   │                                   3 split reward funcs, runs LoRA training,
│   │                                   evaluates baseline vs trained adapter.
│   └── train_redshift_policy.py       lightweight reference MLP policy trainer
│                                       used as a non-LLM baseline in the notebook.
│
├── curriculum_results/
│   └── buffer.json                    precomputed regret buffer of 206 scenarios
│                                       (6 hand-authored seeds + 200 mutated children).
│
└── src/oncallenv/
    ├── __init__.py                    re-exports OnCallRedShiftEnv
    │
    ├── core/
    │   ├── env.py                     OnCallRedShiftEnv (OpenEnv-compatible).
    │   │                               reset / step / list_tasks / observation builder.
    │   │                               Obfuscates the alert so the agent never sees
    │   │                               the real root-cause service.
    │   ├── tools.py                   ToolRuntime + the full SRE command surface
    │   │                               (12 read-only + 6 mutating + declare_resolved + submit_rca).
    │   └── types.py                   Pydantic models: ScenarioSpec, Alert, Observation,
    │                                   Action, State, RCA, plus the FaultName /
    │                                   topology / red-herring literal types.
    │
    ├── simulation/
    │   ├── graph.py                   MicroserviceGraph + Service dataclass.
    │   │                               6 topologies + tick / mark_remediated /
    │   │                               is_recovered / blast_radius_score.
    │   ├── faults.py                  the 12 FaultPrimitive definitions
    │   │                               (log lines, CPU/mem/latency/error deltas,
    │   │                               correct remediation tool).
    │   └── scenario_compiler.py       ScenarioSpec -> live MicroserviceGraph
    │                                   (applies primary + optional secondary fault,
    │                                   red herrings, deploy windows).
    │
    ├── telemetry/
    │   └── otlp.py                    OpenTelemetry-shaped exports used by tools:
    │                                   prometheus_metrics, json_logs, jaeger_traces.
    │
    ├── rewards/
    │   ├── composed.py                build_default_rubric(): WeightedSum of the
    │   │                               environment-side rubrics (recovery 0.35,
    │   │                               rca_quality 0.30, blast_radius 0.25, safety 0.10).
    │   ├── recovery.py                +reward when the cluster is actually recovered.
    │   ├── rca_quality.py             grades the structured RCA (timeline, category,
    │   │                               5-whys, action items).
    │   ├── blast_radius.py            penalizes how much customer traffic was hit.
    │   ├── safety.py                  penalizes destructive ops on stateful services.
    │   └── sre_shaped.py              the *behavioral* reward used for GRPO. Parses
    │                                   <thought>+<actions>+submit_rca, scores it,
    │                                   and exposes 3 disjoint sub-scores
    │                                   (sre_format / sre_investigation / sre_remediation)
    │                                   that TRL logs as separate columns.
    │
    └── curriculum/
        ├── buffer.py                  RegretBuffer: regret-prioritized scenario sampler
        │                               with eps-greedy uniform fallback.
        ├── mutator.py                 ScenarioMutator: structured mutation of one
        │                               ScenarioSpec field at a time + novelty key.
        └── autocurriculum.py          AutocurriculumRunner: ACCEL-style evolution loop
                                        that grows the regret buffer from seed scenarios.
```

---

## The task

The agent is paged with one critical alert that always reads:

```
service: api-gateway
severity: critical
message: "High latency and elevated error rate detected in customer telemetry"
```

The true root cause is hidden somewhere in the dependency graph
(`api-gateway -> checkout-service / user-service -> payment-service /
inventory-service -> postgres-primary / redis-cache`). The agent must:

1. **Investigate** — call read-only tools (`kubectl_logs`, `kubectl_top`,
   `promql_query`, `jaeger_search`, `dns_lookup`, `curl_service`,
   `check_deploy_history`, ...) on candidate downstream services to find the
   one emitting fault-specific symptoms (OOMKilled, x509 expired, NXDOMAIN,
   Envoy `UH` flags, replication lag, etc.).
2. **Remediate** — apply exactly one mutating fix on the suspected service.
   The correct (tool, service) pair is fault-specific and listed below.
3. **Resolve** — emit `declare_resolved`.
4. **Report** — emit `submit_rca <service> <category>` naming the root cause.

The expected output format is two XML blocks:

```xml
<thought>
short paragraph reasoning from the symptom back through the dependency graph
</thought>
<actions>
kubectl_logs payment-service
kubectl_top  payment-service
kubectl_rollout_restart payment-service
declare_resolved
submit_rca payment-service oom_kill
</actions>
```

---

## Topologies (`src/oncallenv/simulation/graph.py`)

All six topologies share the same seven services but rewire the dependency
edges, which changes which downstream services the agent has to inspect:

| name            | shape                                                                    |
| --------------- | ------------------------------------------------------------------------ |
| `simple_fanout` | gateway → {checkout, user}; checkout → {payment, inventory}; payment/inventory/user → {postgres, redis} |
| `deep_chain`    | gateway → checkout → payment → inventory → postgres (longest trace)      |
| `mesh`          | extra cross-edges (checkout↔user, payment↔inventory, inventory→redis)    |
| `star`          | every non-gateway service depends on `api-gateway`                       |
| `bipartite`     | gateway fans out to checkout/payment/inventory; each fans into postgres+redis |
| `diamond`       | gateway → {checkout, user} → payment → postgres (two paths converge)     |

---

## Fault primitives (`src/oncallenv/simulation/faults.py`)

12 distinct incident types. Each one stamps fault-specific log lines, bumps
CPU/mem/latency/error rates, and registers a single canonical remediation
that the agent must match.

| fault              | telltale log line(s)                                    | correct remediation       |
| ------------------ | ------------------------------------------------------- | ------------------------- |
| `oom_kill`         | `java.lang.OutOfMemoryError`, `OOMKilled / exit code 137` | `kubectl_rollout_restart` |
| `cpu_hog`          | `run queue saturated`, `CPU throttling at 98 percent`   | `kubectl_scale`           |
| `network_partition`| `context deadline exceeded`, `Envoy response flag UF`   | `traffic_split_update`    |
| `dns_misconfig`    | `lookup ...: no such host`, `Envoy response flag NR`    | `kubectl_apply_config`    |
| `replica_lag`      | `replication_lag_seconds above threshold`, `stale read` | `feature_flag_toggle`     |
| `cache_stampede`   | `redis miss rate 96 percent`, `thundering herd`         | `feature_flag_toggle`     |
| `http_503_loop`    | `upstream returned HTTP 503`, `Envoy flag UH`           | `kubectl_rollout_undo`    |
| `deadlock`         | `deadlock detected ...`, `PSQLException: Cannot get connection` | `kubectl_rollout_restart` |
| `disk_full`        | `no space left on device`, `etcdserver: data corruption`| `kubectl_apply_config`    |
| `cert_expiry`      | `x509: certificate has expired`, `TLS handshake failed` | `kubectl_apply_config`    |
| `clock_skew`       | `JWT not valid yet due to clock skew`, `signature timestamp outside tolerance` | `kubectl_rollout_restart` |
| `gc_pause`         | `GC overhead limit exceeded`, `Result::unwrap() on Err` | `kubectl_rollout_restart` |


A scenario can also stack a `fault_secondary` (compounded incident), one of
seven `red_herring` distractors (`stale_deploy_notice`, `flapping_canary`,
`unrelated_alert`, `false_correlation`, `old_anomaly`, `unused_service_spike`,
`innocent_config_change`), a `deploy_window` annotation
(`recent_deploy` / `flag_flip` / `config_change`), and a `schema_drift`
(rename_metric / swap_units / rotate_creds / new_required_field /
field_type_change / endpoint_version_bump).

---

## Tool surface (`src/oncallenv/core/tools.py`)

The agent picks from a shell-style command surface. Each command takes one
service name as its argument.

**Read-only (12)** — investigation tools, no side effects:
`kubectl_get_pods`, `kubectl_describe_pod`, `kubectl_logs`, `kubectl_top`,
`promql_query`, `logql_query`, `jaeger_search`, `istioctl_proxy_status`,
`istioctl_routes`, `curl_service`, `dns_lookup`, `check_deploy_history`.

**Mutating (6)** — remediation tools, change cluster state:
`kubectl_rollout_undo`, `kubectl_rollout_restart`, `kubectl_scale`,
`feature_flag_toggle`, `traffic_split_update`, `kubectl_apply_config`.

**Communication (1)**: `post_status_update`.

**Terminal (2)**: `declare_resolved`, `submit_rca <json>` — the simulator
accepts the structured `RCA` Pydantic model defined in `core/types.py`.

Read-only tools return realistic, OTLP-shaped payloads (Prometheus exposition
format, JSONL logs with trace/span IDs and severity, Jaeger trace JSON).
Mutating tools verify the (tool, service) pair against the simulator's
required-remediation set and return either `OK: ...` or `WARN: ... but
incident symptoms persist`. Restart-style ops on `postgres-primary` or
`redis-cache` count as **unsafe actions** and feed the safety rubric.

---

## Reward functions

Two reward systems live in the codebase. They serve different purposes:

### 1. Environment rubric (`oncallenv.rewards.composed.build_default_rubric`)

A `WeightedSum` of four rubrics — used by `OnCallRedShiftEnv.step` to populate
`obs.reward` and `obs.reward_breakdown`. Each component returns a value in
`[0, 1]` and is then weighted:

| rubric              | weight | what it measures                                                      |
| ------------------- | ------ | --------------------------------------------------------------------- |
| `RecoveryRubric`    | 0.35   | resolution declared **and** cluster-wide success rate ≥ 0.99          |
| `RCAQualityRubric`  | 0.30   | structured RCA: timeline mentions root service, category matches, ≥3 substantive 5-whys, action items reference real services |
| `BlastRadiusRubric` | 0.25   | `1 - (root_service.error_rate * 0.65 + min(elapsed/900s, 1) * 0.35)` |
| `SafetyRubric`      | 0.10   | `max(0, 1 - 0.5 * unsafe_actions)` — destructive ops on stateful services |

This rubric is generous (it auto-credits trivially-correct RCAs) and is kept
mainly for env-side diagnostics and the OpenEnv API contract. **It is not used
as the GRPO training signal.**

### 2. SRE-shaped behavioral reward (`oncallenv.rewards.sre_shaped`)

This is the actual GRPO training signal. It scores **only the model's emitted
text** — not the legacy environment reward — so every point has to be earned
by structurally and semantically correct output. The total is clipped to
`[-0.6, +1.0]`.

#### Bonuses

| component                       | value  | trigger                                                          |
| ------------------------------- | ------ | ---------------------------------------------------------------- |
| `thought_format_bonus`          | +0.05  | `<thought>...</thought>` tag present                             |
| `thought_substance_bonus`       | +0.05  | thought ≥ 8 words                                                |
| `thought_semantic_bonus`        | +0.10  | thought mentions an alias of the true root-cause category        |
| `actions_format_bonus`          | +0.10  | `<actions>...</actions>` tag present and parseable               |
| `first_cmd_bonus`               | +0.05  | at least one command parsed inside the actions block             |
| `read_only_first_bonus`         | +0.05  | first parsed command is a read-only investigation tool           |
| `first_targets_root_bonus`      | +0.10  | first command targets a service in the fault set                 |
| `read_before_mutate_bonus`      | +0.05  | a read-only probe precedes any mutating command                  |
| `correct_remediation_bonus`     | +0.20  | a required `(tool, service)` pair appears verbatim               |
| `declare_resolved_bonus`        | +0.05  | `declare_resolved` is emitted                                    |
| `rca_emitted_bonus`             | +0.05  | `submit_rca <service> <category>` is parseable                   |
| `rca_service_bonus`             | +0.15  | predicted service equals the true root-cause service             |
| `rca_category_bonus`            | +0.20  | predicted category equals the true category exactly              |
| `rca_category_bonus` (alias)    | +0.08  | predicted category is a known alias (e.g. "memory_leak" → oom_kill) |

#### Penalties

| component                       | value          | trigger                                                              |
| ------------------------------- | -------------- | -------------------------------------------------------------------- |
| `no_actions_penalty`            | −0.30          | no `<actions>` block at all                                          |
| `no_commands_penalty`           | −0.25 / −0.10  | actions tag present but nothing parses / no actions tag and nothing parses |
| `truncation_penalty`            | −0.10          | output cut off (no closing `</actions>`)                             |
| `verbose_penalty`               | −0.10          | more than 10 commands emitted                                        |
| `shotgun_penalty`               | −0.15 per hit, capped at −0.45 | each mutating command issued on a non-fault service       |
| `early_mutate_penalty`          | −0.10          | a mutating command runs before any read-only probe                   |
| `no_rca_penalty`                | −0.15          | no `submit_rca` emitted                                              |
| `wrong_category_penalty`        | −0.10          | thought mentions an alias of the *wrong* fault family                |

#### Three-way reward split for GRPO logging

`sre_shaped_reward` partitions the components into three disjoint sub-scores
that are exposed as separate reward functions. TRL accepts a list of reward
functions and logs each one as its own `rewards/<func>/mean` column, and the
policy gradient is taken over their sum (which equals the original pre-clip
total exactly):

| reward func                | sums                                                                                                  | rough range       |
| -------------------------- | ----------------------------------------------------------------------------------------------------- | ----------------- |
| `sre_format_score`         | thought_format + thought_substance + thought_semantic + actions_format + no_actions + no_commands + truncation + verbose + wrong_category | `[-0.50, +0.30]` |
| `sre_investigation_score`  | first_cmd + read_only_first + first_targets_root + read_before_mutate + shotgun + early_mutate        | `[-0.55, +0.25]` |
| `sre_remediation_score`    | correct_remediation + declare_resolved + rca_emitted + rca_service + rca_category + no_rca           | `[-0.15, +0.65]` |

To avoid running the simulator three times per generation, the breakdown is
memoized with `functools.lru_cache(maxsize=512)` keyed on `(text, task_id)`.

---

## Curriculum (`src/oncallenv/curriculum/`)

- **`RegretBuffer`** — list of `BufferedScenario(spec, regret, solve_rate, seen_count)`.
  `sample()` is regret-weighted with an `epsilon=0.08` uniform fallback.
- **`ScenarioMutator`** — picks one field of a parent `ScenarioSpec`
  (topology / fault_primary / fault_secondary / inject_service / latency_ms /
  blast_radius / metric_noise / red_herring / deploy_window / schema_drift)
  and resamples it.
- **`AutocurriculumRunner`** — ACCEL-style loop. Each iteration samples a
  parent, mutates one field, computes a heuristic solve rate (penalizing
  secondary faults, noise, blast radius, drift, stateful root causes), drops
  duplicates and trivial/impossible scenarios, then writes the survivor with
  `regret = 1 - 2|0.5 - solve_rate|` (peaking at solve_rate ≈ 0.5).

The shipped `curriculum_results/buffer.json` contains 6 hand-authored seeds
plus 200 evolved children.

The trainer accepts `--max-solve-rate 0.25` to filter the buffer down to
genuinely hard scenarios (~73 / 206 in the shipped buffer).


---

## Running

### On Kaggle (T4)

Open `notebooks/OnCall_RedShift_GRPO_Training.ipynb` and run top-to-bottom.
The notebook clones this repo, installs deps, trains the MLP baseline, runs
GRPO on Qwen-2.5-3B with the three-way reward split, evaluates all three
models (MLP / Qwen-base / Qwen-GRPO) under the SRE-shaped reward, and renders
a stacked bar chart of the per-component breakdown.

### Locally

```bash
pip install -r requirements.txt
pip install -r requirements-llm.txt   # only needed for GRPO

# Three-way GRPO with hard-task filter
PYTHONPATH=src python scripts/train_unsloth_grpo.py \
  --model-name unsloth/Qwen2.5-3B-Instruct-bnb-4bit \
  --out-dir training_results/sre_qwen_grpo \
  --curriculum-buffer curriculum_results/buffer.json \
  --reward-mode sre \
  --prompt-mode sre \
  --max-solve-rate 0.25 \
  --max-tasks 120 \
  --max-steps 200 \
  --num-generations 4 \
  --max-completion-length 256 \
  --logging-steps 1
```

WandB / TRL will surface three columns:
`rewards/sre_format_reward/mean`, `rewards/sre_investigation_reward/mean`,
`rewards/sre_remediation_reward/mean`.

### Smoke-test the environment

```python
from oncallenv import OnCallRedShiftEnv
from oncallenv.core.types import Action

env = OnCallRedShiftEnv()
obs = env.reset(task_id="seed_easy_memory_leak")
print(obs.alerts[0].service, "->", obs.alerts[0].message)
# api-gateway -> High latency and elevated error rate detected in customer telemetry

obs = env.step(Action(command="kubectl_logs payment-service"))   # investigate
obs = env.step(Action(command="kubectl_top  payment-service"))    # confirm
obs = env.step(Action(command="kubectl_rollout_restart payment-service"))  # remediate
obs = env.step(Action(command="declare_resolved"))
print("reward:", obs.reward, "breakdown:", obs.reward_breakdown)
```

---

## Environment variables (`.env.example`)

| variable            | purpose                                                                  |
| ------------------- | ------------------------------------------------------------------------ |
| `REWARD_CAP`        | optional float ceiling applied to `obs.reward` after the rubric fires    |
| `CURRICULUM_BUFFER` | path to a `RegretBuffer` JSON, defaults to `./curriculum_results/buffer.json` |
| `WANDB_DISABLED`    | set to `true` to suppress W&B (the trainer already defaults to this)     |
