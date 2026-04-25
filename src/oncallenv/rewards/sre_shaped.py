"""Dense, SRE-favoring shaped reward for the Red Shift defender.

This score is *purely behavioral*: it does NOT add the legacy environment
reward into the total. The legacy rubric is generous (it auto-credits a
forced declare_resolved + injected RCA), which saturated baseline scores
near ~0.9 and left no gradient for GRPO to climb. Here every point must be
earned by the model's own emitted text.

Bonuses (must be earned):
  +0.05  format-thought:    <thought>...</thought> tag present
  +0.05  substantive thought: thought >= 8 words
  +0.10  format-actions:    <actions>...</actions> tag present and parseable
  +0.05  first command parsed
  +0.05  read-only first move
  +0.10  first move targets the alerting / root-cause service
  +0.05  read-only probe precedes any mutating command
  +0.20  correct remediation: a required (tool, service) pair appears verbatim
  +0.05  declare_resolved emitted
  +0.05  submit_rca emitted
  +0.15  submit_rca names the correct root-cause service
  +0.20  submit_rca names the exact root-cause category
  +0.08  submit_rca category is an alias of the truth (partial credit)

Penalties (cap baseline behavior):
  -0.30  no <actions> block at all
  -0.25  <actions> present but no parseable command
  -0.15  no submit_rca emitted
  -0.10  truncated output (no closing </actions>)
  -0.10  verbose output (>10 commands)
  -0.15  per mutating command on a non-fault service (cap -0.45)
  -0.10  mutating command issued before any read-only probe
  -0.10  thought mentions the WRONG category alias

Final total is clipped to [-0.6, +1.0]. A perfectly behaved trace caps near
+1.0; a baseline LLM that just emits boilerplate sits around 0.0-0.3.
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
    env_reward: float  # tracked for diagnostics only; NOT added into total
    thought_format_bonus: float = 0.0
    thought_substance_bonus: float = 0.0
    thought_semantic_bonus: float = 0.0
    actions_format_bonus: float = 0.0
    first_cmd_bonus: float = 0.0
    read_only_first_bonus: float = 0.0
    first_targets_root_bonus: float = 0.0
    read_before_mutate_bonus: float = 0.0
    correct_remediation_bonus: float = 0.0
    declare_resolved_bonus: float = 0.0
    rca_emitted_bonus: float = 0.0
    rca_service_bonus: float = 0.0
    rca_category_bonus: float = 0.0
    no_actions_penalty: float = 0.0
    no_commands_penalty: float = 0.0
    no_rca_penalty: float = 0.0
    truncation_penalty: float = 0.0
    verbose_penalty: float = 0.0
    shotgun_penalty: float = 0.0
    early_mutate_penalty: float = 0.0
    wrong_category_penalty: float = 0.0
    parsed_commands: list[str] = field(default_factory=list)
    rca_prediction: Optional[tuple[str, str]] = None
    truth: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "env_reward": self.env_reward,
            "thought_format_bonus": self.thought_format_bonus,
            "thought_substance_bonus": self.thought_substance_bonus,
            "thought_semantic_bonus": self.thought_semantic_bonus,
            "actions_format_bonus": self.actions_format_bonus,
            "first_cmd_bonus": self.first_cmd_bonus,
            "read_only_first_bonus": self.read_only_first_bonus,
            "first_targets_root_bonus": self.first_targets_root_bonus,
            "read_before_mutate_bonus": self.read_before_mutate_bonus,
            "correct_remediation_bonus": self.correct_remediation_bonus,
            "declare_resolved_bonus": self.declare_resolved_bonus,
            "rca_emitted_bonus": self.rca_emitted_bonus,
            "rca_service_bonus": self.rca_service_bonus,
            "rca_category_bonus": self.rca_category_bonus,
            "no_actions_penalty": self.no_actions_penalty,
            "no_commands_penalty": self.no_commands_penalty,
            "no_rca_penalty": self.no_rca_penalty,
            "truncation_penalty": self.truncation_penalty,
            "verbose_penalty": self.verbose_penalty,
            "shotgun_penalty": self.shotgun_penalty,
            "early_mutate_penalty": self.early_mutate_penalty,
            "wrong_category_penalty": self.wrong_category_penalty,
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


SRE_SYSTEM_PROMPT = """You are the on-call SRE for a production microservice cluster. You are paged with a critical alert at the customer-facing edge (api-gateway). The alert is generic -- it only tells you that customers are seeing high latency and errors. The true root-cause service is HIDDEN somewhere in the dependency graph (api-gateway -> checkout-service / user-service -> payment-service / inventory-service -> postgres-primary / redis-cache). You must INVESTIGATE to find it, then apply ONE targeted fix, then submit your RCA.

OUTPUT FORMAT (follow exactly, nothing else outside the tags):
<thought>
One short paragraph reasoning from the symptom back through the dependency graph: which downstream service is the likely root cause and why, based on what you would expect to see in logs / metrics / traces.
</thought>
<actions>
kubectl_logs SERVICE              # 1) investigate FIRST with read-only probes (try downstream services)
kubectl_top SERVICE               # 2) optional second read-only probe (e.g. promql_query / jaeger_search)
kubectl_rollout_restart SERVICE   # 3) ONE targeted mutating fix on the suspected root-cause service
declare_resolved
submit_rca SERVICE CATEGORY       # fill in the suspected root-cause service and its fault category
</actions>

RULES:
- Do NOT assume the alerting service (api-gateway) is the root cause. It is the symptom, not the source. Investigate downstream first.
- Lead with read-only commands (kubectl_logs, kubectl_top, promql_query, jaeger_search, dns_lookup, curl_service). Do NOT start with a mutating command.
- Do NOT carpet-bomb the cluster. Mutating commands (kubectl_rollout_restart, kubectl_scale, kubectl_apply_config, traffic_split_update, feature_flag_toggle, kubectl_rollout_undo) MUST target only the actually-affected service. Restarting unrelated services is penalized.
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


GENERIC_ALERT_SERVICE = "api-gateway"
GENERIC_ALERT_MESSAGE = "High latency and elevated error rate detected in customer telemetry"


def get_ground_truth(task_id: str) -> dict[str, Any]:
    """Snapshot the simulator's ground truth for a task without leaking it to the agent.

    The advertised alert (service + message) is intentionally generic -- the
    agent only sees a frontend page and must trace the fault back to the true
    root-cause service. The hidden truth fields (root_service, root_category,
    fault_services, required) are used by the reward function to grade
    correctness AFTER the rollout, never shown to the model.
    """
    env = OnCallRedShiftEnv()
    env.reset(task_id=task_id)
    graph = env._runtime.graph
    fault_services = {svc for req in graph.required_remediations for svc in [req.split(":", 1)[1]]}
    return {
        "root_service": graph.root_cause_service,
        "root_category": graph.root_cause_category,
        "fault_services": fault_services,
        "required": set(graph.required_remediations),
        "alert_service": GENERIC_ALERT_SERVICE,
        "alert_message": GENERIC_ALERT_MESSAGE,
    }


def sre_shaped_reward(
    text: str,
    task_id: str,
    truth: Optional[dict[str, Any]] = None,
    *,
    thought_min_words: int = 8,
    shotgun_per_hit: float = 0.15,
    shotgun_floor: float = 0.45,
    reward_floor: float = -0.6,
    reward_cap: float = 1.0,
    return_breakdown: bool = False,
) -> Any:
    """Compute the purely behavioral SRE-shaped reward for a model output.

    The legacy environment reward is computed for diagnostics only -- it is
    NOT added into the total. Every point in the total comes from the model's
    own emitted text being structurally and semantically correct. See module
    docstring for the full rubric.
    """
    if truth is None:
        truth = get_ground_truth(task_id)

    thought_text, _actions_body, runnable, rca_pred, has_actions_tag = parse_actions_block(text)
    fault_services: set[str] = truth["fault_services"]
    required_pairs: set[str] = truth["required"]

    # Run the env purely to expose env_reward in the breakdown for debugging.
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

    score = SREScore(total=0.0, env_reward=env_reward, parsed_commands=list(runnable), rca_prediction=rca_pred, truth=dict(truth))

    # ---- thought formatting ----
    if thought_text:
        score.thought_format_bonus = 0.05
        if len(thought_text.split()) >= thought_min_words:
            score.thought_substance_bonus = 0.05
        thought_lower = thought_text.lower()
        true_category = truth["root_category"]
        true_aliases = RCA_ALIAS_MAP.get(true_category, ())
        if any(alias in thought_lower for alias in true_aliases):
            score.thought_semantic_bonus = 0.10
        else:
            for other_cat, other_aliases in RCA_ALIAS_MAP.items():
                if other_cat == true_category:
                    continue
                if any(alias in thought_lower for alias in other_aliases):
                    score.wrong_category_penalty = -0.10
                    break

    # ---- actions block formatting / structural penalties ----
    if has_actions_tag:
        score.actions_format_bonus = 0.10
    else:
        score.no_actions_penalty = -0.30
    if has_actions_tag and not runnable:
        score.no_commands_penalty = -0.25
    elif not has_actions_tag and not runnable:
        score.no_commands_penalty = -0.10  # double-down lightly when nothing parses
    if "</actions>" not in text.lower():
        score.truncation_penalty = -0.10
    if len(runnable) > 10:
        score.verbose_penalty = -0.10

    # ---- investigation behavior ----
    if runnable:
        first = runnable[0]
        score.first_cmd_bonus = 0.05
        if _is_read_only(first):
            score.read_only_first_bonus = 0.05
        if _service_of(first) in fault_services:
            score.first_targets_root_bonus = 0.10
        first_mutating_idx = next((i for i, c in enumerate(runnable) if _is_mutating(c)), None)
        first_read_idx = next((i for i, c in enumerate(runnable) if _is_read_only(c)), None)
        if first_mutating_idx is not None and first_read_idx is not None and first_read_idx < first_mutating_idx:
            score.read_before_mutate_bonus = 0.05
        if first_mutating_idx is not None and (first_read_idx is None or first_read_idx > first_mutating_idx):
            score.early_mutate_penalty = -0.10

    # ---- correct remediation ----
    if required_pairs:
        normalized = {f"{cmd.split(' ', 1)[0].lower()}:{_service_of(cmd)}" for cmd in runnable if _is_mutating(cmd)}
        if normalized & required_pairs:
            score.correct_remediation_bonus = 0.20

    # ---- shotgun mutations on non-fault services ----
    shotgun_hits = sum(1 for cmd in runnable if _is_mutating(cmd) and _service_of(cmd) and _service_of(cmd) not in fault_services)
    score.shotgun_penalty = -min(shotgun_floor, shotgun_per_hit * shotgun_hits)

    # ---- declare_resolved + RCA emission ----
    if any(cmd == "declare_resolved" for cmd in runnable):
        score.declare_resolved_bonus = 0.05
    if rca_pred is None:
        score.no_rca_penalty = -0.15
    else:
        score.rca_emitted_bonus = 0.05
        pred_service, pred_category = rca_pred
        if pred_service == truth["root_service"]:
            score.rca_service_bonus = 0.15
        true_category = truth["root_category"]
        if pred_category == true_category:
            score.rca_category_bonus = 0.20
        elif pred_category in RCA_ALIAS_MAP.get(true_category, ()):
            score.rca_category_bonus = 0.08

    total = (
        score.thought_format_bonus
        + score.thought_substance_bonus
        + score.thought_semantic_bonus
        + score.actions_format_bonus
        + score.first_cmd_bonus
        + score.read_only_first_bonus
        + score.first_targets_root_bonus
        + score.read_before_mutate_bonus
        + score.correct_remediation_bonus
        + score.declare_resolved_bonus
        + score.rca_emitted_bonus
        + score.rca_service_bonus
        + score.rca_category_bonus
        + score.no_actions_penalty
        + score.no_commands_penalty
        + score.no_rca_penalty
        + score.truncation_penalty
        + score.verbose_penalty
        + score.shotgun_penalty
        + score.early_mutate_penalty
        + score.wrong_category_penalty
    )
    score.total = max(reward_floor, min(reward_cap, total))
    if return_breakdown:
        return score
    return score.total


# ---------------------------------------------------------------------------
# Granular reward split for TRL/GRPO logging.
#
# TRL accepts a list of reward functions and logs each as its own column
# (rewards/<func_name>/mean). The total reward used for the policy gradient
# is the SUM across the list. We partition the SRE breakdown into 3 disjoint
# logical buckets so each shows up as its own training-log column:
#
#   sre_format_reward       -- did the model produce well-formed output?
#   sre_investigation_reward-- did the model investigate before mutating?
#   sre_remediation_reward  -- did the model remediate the right service and
#                              produce a correct RCA?
#
# Each call to the reward funcs would otherwise re-parse and re-step the env
# three times for the same completion, so we cache the breakdown by
# (text, task_id) with an LRU. The cached value is the SREScore dataclass.
# ---------------------------------------------------------------------------

import functools as _functools


@_functools.lru_cache(maxsize=512)
def _cached_breakdown(text: str, task_id: str) -> SREScore:
    return sre_shaped_reward(text, task_id, return_breakdown=True)


def _format_subscore(score: SREScore) -> float:
    return (
        score.thought_format_bonus
        + score.thought_substance_bonus
        + score.thought_semantic_bonus
        + score.actions_format_bonus
        + score.no_actions_penalty
        + score.no_commands_penalty
        + score.truncation_penalty
        + score.verbose_penalty
        + score.wrong_category_penalty
    )


def _investigation_subscore(score: SREScore) -> float:
    return (
        score.first_cmd_bonus
        + score.read_only_first_bonus
        + score.first_targets_root_bonus
        + score.read_before_mutate_bonus
        + score.shotgun_penalty
        + score.early_mutate_penalty
    )


def _remediation_subscore(score: SREScore) -> float:
    return (
        score.correct_remediation_bonus
        + score.declare_resolved_bonus
        + score.rca_emitted_bonus
        + score.rca_service_bonus
        + score.rca_category_bonus
        + score.no_rca_penalty
    )


def sre_format_score(text: str, task_id: str) -> float:
    """Format / structure / chain-of-thought sub-reward."""
    return _format_subscore(_cached_breakdown(text, task_id))


def sre_investigation_score(text: str, task_id: str) -> float:
    """Investigative-behavior sub-reward (read-before-mutate, no shotgun)."""
    return _investigation_subscore(_cached_breakdown(text, task_id))


def sre_remediation_score(text: str, task_id: str) -> float:
    """Remediation-correctness + RCA-accuracy sub-reward."""
    return _remediation_subscore(_cached_breakdown(text, task_id))
