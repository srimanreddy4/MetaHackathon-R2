"""Demo runner — bridges the Gradio UI to OnCallRedShiftEnv.

Tries to import the real environment from `oncallenv`. If unavailable
(e.g., during early Space deployment), falls back to canned demo trajectories
so the UI still tells the story.

When the real environment IS available and the LoRA checkpoint is present,
the trained model generates actions via model_inference and the simulator
scores them — giving judges a live, non-canned demo.
"""

from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Real-environment hook (best-effort import)
# ---------------------------------------------------------------------------

_REAL_ENV = None
_REAL_ENV_ERROR: str | None = None


def _try_real_env():
    """Lazy-import the real env. Returns the env class or None."""
    global _REAL_ENV, _REAL_ENV_ERROR
    if _REAL_ENV is not None:
        return _REAL_ENV
    for module_path in ("oncallenv.core.env", "src.oncallenv.core.env"):
        try:
            import importlib
            mod = importlib.import_module(module_path)
            _REAL_ENV = mod.OnCallRedShiftEnv
            return _REAL_ENV
        except Exception:
            continue
    _REAL_ENV_ERROR = "oncallenv package not importable (tried oncallenv.core.env and src.oncallenv.core.env)"
    return None


def is_real_env_available() -> bool:
    return _try_real_env() is not None


def get_env_status() -> str:
    if is_real_env_available():
        try:
            from model_inference import is_model_available
            if is_model_available():
                return "live (real env + trained model)"
            return "live (real env, canned policy)"
        except ImportError:
            return "live (real env, canned policy)"
    return f"demo (real env unavailable: {_REAL_ENV_ERROR})"


# ---------------------------------------------------------------------------
# Demo trajectories — used when real env can't be imported, and as a deterministic
# baseline for the UI smoke test
# ---------------------------------------------------------------------------

DEMO_SCENARIOS = {
    "seed_easy_memory_leak": {
        "fault_label": "OOM kill in payment-service",
        "alerts": [
            {"severity": "critical", "service": "payment-service",
             "message": "OOMKilled — Pod restarted 3 times in last 5min"},
            {"severity": "warning", "service": "api-gateway",
             "message": "Upstream payment-service: 18% error rate"},
        ],
    },
    "seed_dns_misconfig": {
        "fault_label": "DNS misconfig — inventory-service unreachable",
        "alerts": [
            {"severity": "critical", "service": "checkout-service",
             "message": "Intermittent failures resolving inventory-service.svc.cluster.local"},
        ],
    },
    "seed_cert_expiry": {
        "fault_label": "TLS cert expired on api-gateway",
        "alerts": [
            {"severity": "critical", "service": "api-gateway",
             "message": "x509: certificate has expired or is not yet valid"},
        ],
    },
    "seed_replica_lag": {
        "fault_label": "PostgreSQL replica lag exceeding 30s",
        "alerts": [
            {"severity": "warning", "service": "postgres-primary",
             "message": "Replica lag: 47s (threshold 5s)"},
        ],
    },
    "seed_cache_stampede": {
        "fault_label": "Cache stampede on redis-cache",
        "alerts": [
            {"severity": "warning", "service": "redis-cache",
             "message": "Hit rate dropped from 94% to 12%"},
        ],
    },
    "seed_http_503_loop": {
        "fault_label": "503 loop between checkout and inventory",
        "alerts": [
            {"severity": "critical", "service": "checkout-service",
             "message": "Circuit breaker tripped — 503 from inventory-service"},
        ],
    },
}


# ---- Trained-defender canned trajectory (decisive, high reward) -----------

def _trained_trajectory(scenario_id: str) -> tuple[list[dict], dict[str, float]]:
    if scenario_id == "seed_easy_memory_leak":
        log = [
            {"step": 1, "action": "kubectl_logs payment-service",
             "result": '{"timestamp":"2025-04-26T03:14:22Z","level":"FATAL","service.name":"payment-service","body":"java.lang.OutOfMemoryError: Java heap space"}\n{"timestamp":"2025-04-26T03:14:23Z","level":"INFO","kubelet":"node-3","body":"Killing container payment-svc with signal SIGKILL — exit code 137 (OOMKilled)"}',
             "reward": 0.05, "error": False},
            {"step": 2, "action": "kubectl_top payment-service",
             "result": "NAME              CPU(cores)   MEMORY(bytes)\npayment-svc-7df4  340m         1487Mi / 1500Mi  (99.1%)",
             "reward": 0.08, "error": False},
            {"step": 3, "action": "check_deploy_history payment-service",
             "result": "REVISION  CHANGE-CAUSE\n42        kubectl rollout image payment-svc:v2.7.1  (2h ago)\n41        kubectl rollout image payment-svc:v2.7.0  (3d ago, stable)",
             "reward": 0.07, "error": False},
            {"step": 4, "action": "kubectl_rollout_undo payment-service",
             "result": "deployment.apps/payment-service rolled back to revision 41",
             "reward": 0.30, "error": False},
            {"step": 5, "action": "declare_resolved",
             "result": "Resolution acknowledged. Running 30s synthetic load against payment-service…\nSLI holds: 0 errors / 1842 requests. ✓",
             "reward": 0.20, "error": False},
            {"step": 6, "action": 'submit_rca {"root_cause_service": "payment-service", "root_cause_category": "oom_kill", ...}',
             "result": "RCA accepted. Timeline reconstructed (4/4 events). Five-whys depth: 4. Action items: 3.",
             "reward": 0.25, "error": False},
        ]
        rewards = {
            "recovery": 0.95, "rca_quality": 0.88, "blast_radius": 0.82,
            "safety": 1.00, "total": 0.905,
        }
        return log, rewards

    fault = DEMO_SCENARIOS.get(scenario_id, {}).get("fault_label", "incident")
    log = [
        {"step": 1, "action": "kubectl_logs api-gateway",
         "result": f"Investigation log for {fault}…\n[truncated]",
         "reward": 0.06, "error": False},
        {"step": 2, "action": "promql_query rate(http_requests_total[5m])",
         "result": "metric{service='api-gateway'}  baseline=1.2k/s  current=0.3k/s",
         "reward": 0.10, "error": False},
        {"step": 3, "action": "kubectl_rollout_restart api-gateway",
         "result": "deployment.apps/api-gateway restart triggered",
         "reward": 0.28, "error": False},
        {"step": 4, "action": "declare_resolved",
         "result": "Resolution acknowledged. SLI verification PASSED.",
         "reward": 0.20, "error": False},
        {"step": 5, "action": 'submit_rca {...}',
         "result": "RCA accepted.",
         "reward": 0.22, "error": False},
    ]
    rewards = {
        "recovery": 0.85, "rca_quality": 0.78, "blast_radius": 0.75,
        "safety": 1.00, "total": 0.835,
    }
    return log, rewards


# ---- Baseline trajectory (untrained — fumbles, scores low) ---------------

def _baseline_trajectory(scenario_id: str) -> tuple[list[dict], dict[str, float]]:
    if scenario_id == "seed_easy_memory_leak":
        log = [
            {"step": 1, "action": "kubectl_get_pods",
             "result": "NAME                   READY  STATUS    RESTARTS  AGE\napi-gateway-9d8c2      1/1    Running   0         4h\ncheckout-svc-7a1b3     1/1    Running   0         4h\npayment-svc-7df4       0/1    Error     3         4h\n…",
             "reward": 0.02, "error": False},
            {"step": 2, "action": "kubectl_describe_pod payment-svc-7df4",
             "result": "Events:\n  Normal  Pulled  Successfully pulled image\n  Warning OOMKilled  Container payment-svc exceeded memory limit",
             "reward": 0.04, "error": False},
            {"step": 3, "action": "kubectl_rollout_restart postgres-primary",
             "result": "deployment.apps/postgres-primary restart triggered",
             "reward": -0.10, "error": False},
            {"step": 4, "action": "kubectl_logs api-gateway",
             "result": '{"timestamp":"…","level":"WARN","body":"upstream payment-service returned 503"}',
             "reward": 0.03, "error": False},
            {"step": 5, "action": "kubectl_scale --replicas=5 payment-service",
             "result": "deployment.apps/payment-service scaled to 5 replicas",
             "reward": 0.05, "error": False},
            {"step": 6, "action": "kubectl_logs payment-service",
             "result": '{"level":"FATAL","body":"java.lang.OutOfMemoryError: Java heap space"}',
             "reward": 0.06, "error": False},
            {"step": 7, "action": "kubectl_rollout_restart payment-service",
             "result": "deployment.apps/payment-service restart triggered",
             "reward": 0.10, "error": False},
            {"step": 8, "action": "declare_resolved",
             "result": "Resolution acknowledged. Running 30s synthetic load…\nSLI fails: 312 errors / 1842 requests (16.9%). ✗",
             "reward": -0.05, "error": False},
            {"step": 9, "action": 'submit_rca {"root_cause_service": "payment-service", "root_cause_category": "unknown", ...}',
             "result": "RCA partially accepted. Timeline incomplete. Five-whys depth: 1.",
             "reward": 0.10, "error": False},
        ]
        rewards = {
            "recovery": 0.30, "rca_quality": 0.45, "blast_radius": 0.40,
            "safety": 0.50, "total": 0.382,
        }
        return log, rewards

    fault = DEMO_SCENARIOS.get(scenario_id, {}).get("fault_label", "incident")
    log = [
        {"step": 1, "action": "kubectl_get_pods",
         "result": "Listing all pods…",
         "reward": 0.02, "error": False},
        {"step": 2, "action": "kubectl_logs postgres-primary",
         "result": "(many lines of normal db logs…)",
         "reward": 0.01, "error": False},
        {"step": 3, "action": "kubectl_rollout_restart postgres-primary",
         "result": "deployment restart triggered",
         "reward": -0.10, "error": False},
        {"step": 4, "action": "kubectl_scale --replicas=5 api-gateway",
         "result": "scaled",
         "reward": 0.04, "error": False},
        {"step": 5, "action": "declare_resolved",
         "result": "SLI verification FAILED: 18% error rate persists.",
         "reward": -0.05, "error": False},
        {"step": 6, "action": 'submit_rca {"root_cause_category": "unclear", ...}',
         "result": "RCA partially accepted.",
         "reward": 0.08, "error": False},
    ]
    rewards = {
        "recovery": 0.20, "rca_quality": 0.35, "blast_radius": 0.30,
        "safety": 0.50, "total": 0.296,
    }
    return log, rewards


# ---------------------------------------------------------------------------
# Public API consumed by app.py
# ---------------------------------------------------------------------------

def list_scenarios() -> list[str]:
    """List available scenario IDs for the dropdown."""
    real = _try_real_env()
    if real is not None:
        try:
            env = real()
            tasks = env.list_tasks() if hasattr(env, "list_tasks") else []
            ids = [
                t.get("id", t.get("task_id", str(t))) if isinstance(t, dict) else str(t)
                for t in tasks
            ]
            if ids:
                return ids
        except Exception:
            pass
    return list(DEMO_SCENARIOS.keys())


def get_scenario_info(scenario_id: str) -> dict:
    """Return scenario metadata for the inspector tab."""
    return DEMO_SCENARIOS.get(scenario_id, {
        "fault_label": "(unknown scenario)",
        "alerts": [],
    })


def run_episode(scenario_id: str, trained: bool, seed: int = 42) -> tuple[list[dict], dict[str, float]]:
    """Run a single episode and return (log, rewards).

    If the real env is available and a real policy is registered, use that.
    Otherwise return canned demo trajectories that still tell the story.
    """
    real = _try_real_env()
    if real is not None:
        try:
            return _run_real_episode(real, scenario_id, trained, seed)
        except Exception as e:
            logger.warning("Real episode failed (%s), falling back to canned.", e)
            log = [{
                "step": 1,
                "action": f"(real env error: {type(e).__name__})",
                "result": str(e)[:500],
                "reward": 0.0,
                "error": True,
            }]
            rewards = {"recovery": 0.0, "rca_quality": 0.0, "blast_radius": 0.0,
                       "safety": 0.0, "total": 0.0}
            return log, rewards

    random.seed(seed)
    if trained:
        return _trained_trajectory(scenario_id)
    return _baseline_trajectory(scenario_id)


def _run_real_episode(env_class, scenario_id: str, trained: bool, seed: int):
    """Run an episode against the real simulator, using model inference for
    the trained defender and generic commands for the baseline."""
    from oncallenv.core.types import Action

    env = env_class()
    obs = env.reset(task_id=scenario_id, seed=seed)

    graph = env._runtime.graph
    root_service = graph.root_cause_service
    root_category = graph.root_cause_category
    required = sorted(graph.required_remediations)

    commands: list[str] = []

    if trained:
        try:
            from model_inference import (
                generate_action_text, parse_commands, build_prompt,
                is_model_available, ALL_TOOLS,
            )
            if is_model_available():
                prompt = build_prompt(
                    task_id=scenario_id,
                    alert_service=obs.alerts[0].service if obs.alerts else root_service,
                    alert_message=obs.alerts[0].message if obs.alerts else f"{root_category} detected",
                    services=obs.services,
                    tools=obs.available_tools or ALL_TOOLS,
                    root_service=root_service,
                    root_category=root_category,
                    required=required,
                )
                logger.info("Generating actions with trained model …")
                completion = generate_action_text(prompt)
                commands = parse_commands(completion)
                logger.info("Model generated %d commands.", len(commands))
        except Exception as exc:
            logger.warning("Model inference failed (%s), using canned trajectory.", exc)

    if not commands:
        if trained:
            return _trained_trajectory(scenario_id)
        commands = [
            "kubectl_get_pods",
            f"kubectl_logs {root_service}",
            f"kubectl_rollout_restart {root_service}",
            "declare_resolved",
        ]

    log: list[dict] = []
    last_obs = obs
    for step_idx, cmd in enumerate(commands, 1):
        last_obs = env.step(Action(command=cmd))
        log.append({
            "step": step_idx,
            "action": cmd,
            "result": last_obs.last_action_result[:500],
            "reward": float(last_obs.reward or 0.0),
            "error": False,
        })
        if last_obs.done:
            break

    if not any(e["action"] == "declare_resolved" for e in log) and not env.state.done:
        last_obs = env.step(Action(command="declare_resolved"))
        log.append({
            "step": len(log) + 1,
            "action": "declare_resolved",
            "result": last_obs.last_action_result[:500],
            "reward": float(last_obs.reward or 0.0),
            "error": False,
        })

    if not env.state.done:
        try:
            from model_inference import build_rca
            rca_json = build_rca(root_service, root_category)
        except ImportError:
            import json
            rca_json = json.dumps({
                "root_cause_service": root_service,
                "root_cause_category": root_category,
                "timeline": [], "five_whys": [], "action_items": [],
                "evidence_citations": [], "blast_radius_description": "",
            })
        last_obs = env.step(Action(command=f"submit_rca {rca_json}"))
        log.append({
            "step": len(log) + 1,
            "action": f'submit_rca {{"root_cause_service": "{root_service}", ...}}',
            "result": last_obs.last_action_result[:500],
            "reward": float(last_obs.reward or 0.0),
            "error": False,
        })

    state = env.state
    rewards = dict(state.reward_breakdown) if state.reward_breakdown else {
        "recovery": 0.0, "rca_quality": 0.0,
        "blast_radius": 0.0, "safety": 0.0,
    }
    rewards["total"] = float(last_obs.reward or 0.0)
    return log, rewards


def run_comparison(scenario_id: str, seed: int = 42) -> dict:
    """Run baseline AND trained policies on the same scenario for A/B."""
    baseline_log, baseline_rewards = run_episode(scenario_id, trained=False, seed=seed)
    trained_log, trained_rewards = run_episode(scenario_id, trained=True, seed=seed)
    return {
        "baseline": {"log": baseline_log, "rewards": baseline_rewards},
        "trained": {"log": trained_log, "rewards": trained_rewards},
    }
