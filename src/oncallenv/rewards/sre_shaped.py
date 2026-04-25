"""Dense, SRE-favoring shaped reward for the Red Shift defender.

The default environment reward is sparse: the agent only sees a single scalar
once the episode ends. That signal is too coarse for a step-by-step LLM agent
trained with GRPO -- it cannot tell which command in a sequence helped, hurt,
or was useless. This module decomposes the reward into dense, behavior-specific
shaping terms that explicitly reward proper SRE practice:

  +0.10  chain-of-thought:   <thought>...</thought> precedes <actions> and is
                             not a stub
  +0.05  thought semantics:  the thought mentions a keyword aliased to the true
                             root-cause category (NLP-grounded credit)
  +0.05  investigative lead: the first emitted command is a read-only probe
                             targeting the alerting / root-cause service
  -0.20  shotgun mutation:   per mutating command issued against a service that
                             is NOT in the fault graph (capped at -0.60)
  +0.10  RCA service:        submit_rca names the correct root-cause service
  +0.10  RCA category exact: submit_rca names the exact root-cause category
  +0.05  RCA category alias: submit_rca category is in the alias map of truth
  -0.10  format break:       no parsable <actions> block / no executable cmds

These bonuses are added to the underlying environment reward (recovery / RCA /
blast / safety) so a model that simply restarts everything cannot dominate a
model that diagnoses, investigates, and surgically remediates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from oncallenv.core.env import OnCallRedShiftEnv
from oncallenv.core.tools import MUTATING_TOOLS, READ_ONLY_TOOLS
from oncallenv.core.types import Action


SERVICES = (
    "api-gateway",
    "checkout-service",
    "payment-service",
    "inventory-service",
    "user-service",
    "postgres-primary",
    "redis-cache",
)

READ_ONLY_SET = set(READ_ONLY_TOOLS)
MUTATING_SET = set(MUTATING_TOOLS)

# NLP alias map: words that semantically match each fault category. Used both
# for the <thought> semantic bonus and for partial-credit RCA grading when the
# model's predicted category is "close" to the true one (e.g. "memory_leak"
# vs "oom_kill").
RCA_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "oom_kill": ("oom", "memory", "memory_leak", "heap", "out of memory", "exit code 137", "oomkilled"),
    "cpu_hog": ("cpu", "cpu_saturation", "throttle", "throttling", "saturated", "run queue", "high cpu"),
    "network_partition": ("network", "partition", "connectivity", "context deadline", "upstream failure", "envoy uf"),
    "dns_misconfig": ("dns", "lookup", "no such host", "nxdomain", "dns_misconfiguration", "no route"),
    "replica_lag": ("replica", "replication", "lag", "stale read", "replication_lag"),
    "cache_stampede": ("cache", "stampede", "thundering herd", "redis miss", "miss rate"),
    "http_503_loop": ("503", "http_503", "no healthy upstream", "envoy uh", "http503"),
    "deadlock": ("deadlock", "lock contention", "psql deadlock", "cannot get connection"),
    "disk_full": ("disk", "no space", "etcd", "data corruption", "disk_full"),
    "cert_expiry": ("cert", "x509", "certificate", "tls", "expired", "handshake"),
    "clock_skew": ("clock", "skew", "jwt", "timestamp", "clock_skew"),
    "gc_pause": ("gc", "garbage collect", "overhead limit", "gc_pause"),
}

THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.IGNORECASE | re.DOTALL)
ACTIONS_RE = re.compile(r"<actions>(.*?)</actions>", re.IGNORECASE | re.DOTALL)
SUBMIT_RCA_RE = re.compile(r"submit_rca\s+([a-z0-9_-]+)\s+([a-z0-9_]+)", re.IGNORECASE)
TOOL_NAMES = sorted(set(READ_ONLY_TOOLS) | set(MUTATING_TOOLS) | {"declare_resolved"}, key=len, reverse=True)
TOOL_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in TOOL_NAMES) + r")\b(?:\s+([a-z0-9_.:/={}\"'-]+))?", re.IGNORECASE)


@dataclass
class SREScore:
    total: float
    env_reward: float
    thought_bonus: float = 0.0
    thought_semantic_bonus: float = 0.0
    investigative_bonus: float = 0.0
    shotgun_penalty: float = 0.0
    rca_service_bonus: float = 0.0
    rca_category_bonus: float = 0.0
    format_penalty: float = 0.0
    parsed_commands: list[str] = field(default_factory=list)
    rca_prediction: Optional[tuple[str, str]] = None
    truth: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "env_reward": self.env_reward,
            "thought_bonus": self.thought_bonus,
            "thought_semantic_bonus": self.thought_semantic_bonus,
            "investigative_bonus": self.investigative_bonus,
            "shotgun_penalty": self.shotgun_penalty,
            "rca_service_bonus": self.rca_service_bonus,
            "rca_category_bonus": self.rca_category_bonus,
            "format_penalty": self.format_penalty,
            "parsed_commands": list(self.parsed_commands),
            "rca_prediction": list(self.rca_prediction) if self.rca_prediction else None,
            "truth": dict(self.truth),
        }


def _build_rca_json(service: str, category: str) -> str:
    return json.dumps(
        {
            "root_cause_service": service,
            "root_cause_category": category,
            "timeline": [
                {
                    "timestamp": "2026-04-24T09:00:00Z",
                    "service": service,
                    "description": f"{category} identified from Red Shift telemetry",
                }
            ],
            "five_whys": [
                f"{service} emitted direct {category} symptoms.",
                "The failure propagated through dependent customer-facing services.",
                "The first mitigation needed to target the true faulty component.",
            ],
            "action_items": [f"Add regression alerting and runbook coverage for {service} {category}."],
            "evidence_citations": [{"source": "log", "ref": f"kubectl_logs {service}", "excerpt": category}],
            "blast_radius_description": "Customer-facing requests saw elevated latency or errors before remediation.",
        }
    )


def parse_actions_block(text: str) -> tuple[Optional[str], Optional[str], list[str], Optional[tuple[str, str]], bool]:
    """Pull out <thought>, <actions>, runnable commands, and submit_rca prediction.

    Returns: (thought_text, actions_body, runnable_commands, rca_prediction, has_actions_tag)
    runnable_commands excludes submit_rca; declare_resolved is preserved.
    """
    thought_match = THOUGHT_RE.search(text)
    actions_match = ACTIONS_RE.search(text)
    thought_text = thought_match.group(1).strip() if thought_match else None
    has_actions_tag = actions_match is not None
    body = actions_match.group(1) if actions_match else text

    rca_pred: Optional[tuple[str, str]] = None
    rca_match = SUBMIT_RCA_RE.search(body)
    if rca_match:
        svc = rca_match.group(1).strip().lower()
        cat = rca_match.group(2).strip().lower()
        rca_pred = (svc, cat)

    runnable: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip().strip("-*` ")
        if not line or line.lower().startswith("submit_rca"):
            continue
        m = TOOL_RE.search(line)
        if not m:
            continue
        tool = m.group(1).lower()
        arg = (m.group(2) or "").strip().strip("'\"`")
        if tool == "declare_resolved":
            runnable.append("declare_resolved")
        else:
            service = next((svc for svc in SERVICES if svc in arg or svc in line), "")
            if not service:
                continue
            runnable.append(f"{tool} {service}")
        if len(runnable) >= 12:
            break

    # Order check: if thought exists but appears AFTER actions, drop it.
    if thought_match and actions_match and thought_match.start() > actions_match.start():
        thought_text = None
    return thought_text, body if has_actions_tag else None, runnable, rca_pred, has_actions_tag


def _is_mutating(cmd: str) -> bool:
    return cmd.split(" ", 1)[0].lower() in MUTATING_SET


def _is_read_only(cmd: str) -> bool:
    return cmd.split(" ", 1)[0].lower() in READ_ONLY_SET


def _service_of(cmd: str) -> str:
    parts = cmd.split(" ", 1)
    return parts[1].strip() if len(parts) == 2 else ""


SRE_SYSTEM_PROMPT = """You are the on-call SRE for a production microservice cluster. You are paged with a critical alert. Diagnose the root cause, investigate before mutating, apply ONE targeted fix, then submit your RCA.

OUTPUT FORMAT (follow exactly, nothing else outside the tags):
<thought>
One short paragraph: which service is the likely root cause and why, based on the alert.
</thought>
<actions>
kubectl_logs SERVICE              # 1) investigate FIRST with read-only probes on the alerting service
kubectl_top SERVICE               # 2) optional second read-only probe
kubectl_rollout_restart SERVICE   # 3) ONE targeted mutating fix on the root-cause service
declare_resolved
submit_rca SERVICE CATEGORY       # e.g. submit_rca payment-service oom_kill
</actions>

RULES:
- Lead with read-only commands (kubectl_logs, kubectl_top, promql_query, jaeger_search). Do NOT start with a mutating command.
- Do NOT carpet-bomb the cluster. Mutating commands (kubectl_rollout_restart, kubectl_scale, kubectl_apply_config, traffic_split_update, feature_flag_toggle, kubectl_rollout_undo) MUST target only the affected service. Restarting unrelated services is penalized.
- Always end with declare_resolved followed by submit_rca SERVICE CATEGORY.
- Pick CATEGORY from: oom_kill, cpu_hog, network_partition, dns_misconfig, replica_lag, cache_stampede, http_503_loop, deadlock, disk_full, cert_expiry, clock_skew, gc_pause.
"""


def build_sre_prompt(task_id: str, alert_service: str, alert_message: str, services: list[str]) -> str:
    return (
        SRE_SYSTEM_PROMPT
        + f"\nTask id: {task_id}\n"
        + f"Critical alert: {alert_service} reports \"{alert_message}\"\n"
        + f"Available services: {', '.join(services)}\n"
        + "Begin now.\n"
    )


def get_ground_truth(task_id: str) -> dict[str, Any]:
    """Snapshot the simulator's ground truth for a task without leaking it to the agent."""
    env = OnCallRedShiftEnv()
    env.reset(task_id=task_id)
    graph = env._runtime.graph
    fault_services = {svc for req in graph.required_remediations for svc in [req.split(":", 1)[1]]}
    return {
        "root_service": graph.root_cause_service,
        "root_category": graph.root_cause_category,
        "fault_services": fault_services,
        "required": set(graph.required_remediations),
        "alert_service": graph.root_cause_service,
        "alert_message": f"{graph.root_cause_category} symptoms detected with elevated p99/error rate",
    }


def sre_shaped_reward(
    text: str,
    task_id: str,
    truth: Optional[dict[str, Any]] = None,
    *,
    thought_min_words: int = 8,
    shotgun_per_hit: float = 0.20,
    shotgun_floor: float = 0.60,
    return_breakdown: bool = False,
) -> Any:
    """Compute the dense SRE-shaped reward for a model's free-form text output.

    The function executes the parsed commands inside a fresh OnCallRedShiftEnv,
    grants the environment's underlying weighted reward, then layers in the
    behavior-specific shaping bonuses described at the top of this module.
    """
    if truth is None:
        truth = get_ground_truth(task_id)

    thought_text, _actions_body, runnable, rca_pred, has_actions_tag = parse_actions_block(text)

    env = OnCallRedShiftEnv()
    env.reset(task_id=task_id)
    obs = None
    for cmd in runnable:
        obs = env.step(Action(command=cmd))
        if obs.done:
            break
    if obs is None or not env.state.resolved_declared:
        obs = env.step(Action(command="declare_resolved"))
    if rca_pred is not None:
        rca_service, rca_category = rca_pred
        obs = env.step(Action(command=f"submit_rca {_build_rca_json(rca_service, rca_category)}"))

    env_reward = float(obs.reward or 0.0) if obs is not None else 0.0

    score = SREScore(total=env_reward, env_reward=env_reward, parsed_commands=list(runnable), rca_prediction=rca_pred, truth=dict(truth))

    if not has_actions_tag or not runnable:
        score.format_penalty = -0.10

    if thought_text and len(thought_text.split()) >= thought_min_words:
        score.thought_bonus = 0.10
        category = truth["root_category"]
        aliases = RCA_ALIAS_MAP.get(category, ())
        thought_lower = thought_text.lower()
        if any(alias in thought_lower for alias in aliases):
            score.thought_semantic_bonus = 0.05

    if runnable:
        first = runnable[0]
        if _is_read_only(first) and _service_of(first) in truth["fault_services"]:
            score.investigative_bonus = 0.05

    fault_services: set[str] = truth["fault_services"]
    shotgun_hits = 0
    for cmd in runnable:
        if _is_mutating(cmd):
            svc = _service_of(cmd)
            if svc and svc not in fault_services:
                shotgun_hits += 1
    score.shotgun_penalty = -min(shotgun_floor, shotgun_per_hit * shotgun_hits)

    if rca_pred is not None:
        pred_service, pred_category = rca_pred
        if pred_service == truth["root_service"]:
            score.rca_service_bonus = 0.10
        true_category = truth["root_category"]
        if pred_category == true_category:
            score.rca_category_bonus = 0.10
        elif pred_category in RCA_ALIAS_MAP.get(true_category, ()):
            score.rca_category_bonus = 0.05

    total = (
        env_reward
        + score.thought_bonus
        + score.thought_semantic_bonus
        + score.investigative_bonus
        + score.shotgun_penalty
        + score.rca_service_bonus
        + score.rca_category_bonus
        + score.format_penalty
    )
    score.total = max(-0.5, min(1.5, total))
    if return_breakdown:
        return score
    return score.total
