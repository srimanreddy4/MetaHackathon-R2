"""Centralized prompt building logic for OnCallEnv Red Shift.

Includes both 'hard' (production-style) and 'easy' (runbook-style hints) modes.
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from models import ScenarioSpec
    from oncallenv.simulation.microservice_graph import MicroserviceGraph

FAULT_RUNBOOK_HINTS = {
    "oom_kill": "memory pressure or OOMKilled usually needs kubectl_rollout_restart on the faulty service",
    "cpu_hog": "CPU saturation usually needs kubectl_scale on the faulty service",
    "network_partition": "network partition symptoms usually need traffic_split_update on the faulty service",
    "dns_misconfig": "DNS or no-route symptoms usually need kubectl_apply_config on the faulty service",
    "replica_lag": "replica lag usually needs feature_flag_toggle on the faulty service",
    "cache_stampede": "cache stampede usually needs feature_flag_toggle on the faulty service",
    "http_503_loop": "HTTP 503 loops usually need kubectl_rollout_undo on the faulty service",
    "deadlock": "deadlocks usually need kubectl_rollout_restart on the faulty service",
    "disk_full": "disk-full configuration incidents usually need kubectl_apply_config on the faulty service",
    "cert_expiry": "certificate expiry usually needs kubectl_apply_config on the faulty service",
    "clock_skew": "clock skew usually needs kubectl_rollout_restart on the faulty service",
    "gc_pause": "GC pause incidents usually need kubectl_rollout_restart on the faulty service",
}

DEFENDER_SYSTEM = (
    "You are the on-call SRE for OnCallEnv Red Shift. "
    "Investigate alerts, diagnose root cause, remediate safely, "
    "then declare_resolved."
)


def required_pairs(required: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for item in required:
        if ":" in item:
            tool, service = item.split(":", 1)
            pairs.append((tool, service))
    return pairs


def build_defender_prompt(
    spec: ScenarioSpec,
    graph: MicroserviceGraph,
    alert_message: str,
    prompt_mode: str = "hard",
    template: str = "standard",
) -> str:
    """Build the SRE prompt.
    
    If prompt_mode is 'easy', appends runbook hints based on the ground truth root cause.
    """
    service = graph.root_cause_service
    category = graph.root_cause_category
    services = sorted(graph.services)
    tools = "kubectl_logs, promql_query, jaeger_search, kubectl_describe_pod, kubectl_top, dns_lookup, check_deploy_history, curl_service, istioctl_routes, kubectl_rollout_restart, kubectl_rollout_undo, kubectl_scale, feature_flag_toggle, traffic_split_update, kubectl_apply_config, declare_resolved"

    base = f"""{DEFENDER_SYSTEM}

Task id: {spec.task_id}
Critical alert: {service} reports {alert_message}
Available services: {", ".join(services)}
Available tools: {tools}

STRICT RULES:
1. You MUST first think step-by-step inside <reasoning> and </reasoning> tags.
2. After reasoning, you MUST output ONLY the executable commands inside <actions> and </actions> tags.
3. Put exactly ONE simulator command per line.
4. Use the provided tools and services to diagnose and remediate, ending with declare_resolved.

Example Output:
<reasoning>
The alert states the issue is with the api-gateway service experiencing elevated error rates. I will first query its metrics, then fetch its logs, and if a bad deploy is found, I will roll it back.
</reasoning>
<actions>
promql_query api-gateway
kubectl_logs api-gateway
kubectl_rollout_undo api-gateway
declare_resolved
</actions>

Use real commands such as kubectl_logs SERVICE, promql_query SERVICE,
jaeger_search SERVICE, kubectl_rollout_restart SERVICE,
kubectl_rollout_undo SERVICE, kubectl_scale SERVICE,
feature_flag_toggle SERVICE, traffic_split_update SERVICE,
kubectl_apply_config SERVICE, and declare_resolved.
Do not include explanations, markdown, bullets, JSON, RCA text, or prose outside the tags.
"""

    if prompt_mode == "hard":
        return base

    required = sorted(graph.required_remediations)
    accepted = ", ".join(f"{tool} {svc}" for tool, svc in required_pairs(required))
    hint = FAULT_RUNBOOK_HINTS.get(category, f"{category} symptoms should be remediated on {service}")
    
    if template == "runbook":
        return (
            base
            + f"\nRunbook hint: suspected faulty service is {service}. "
            + f"Fault family is {category}. {hint}. "
            + f"Accepted remediation command: {accepted}. "
            + "A good answer is exactly 3-5 command lines and ends with </actions>.\n"
        )
    if template == "triage":
        return (
            base
            + f"\nEasy triage hints: first inspect {service}; then apply the remediation matching {category}; "
            + f"then declare_resolved. Gold remediation: {accepted}. "
            + "A good answer is exactly 3-5 command lines and ends with </actions>.\n"
        )
    return (
        base
        + f"\nEasy-mode hints: root service = {service}; fault = {category}; "
        + f"best remediation = {accepted}. Include declare_resolved after the fix. "
        + "A good answer is exactly 3-5 command lines and ends with </actions>.\n"
    )
