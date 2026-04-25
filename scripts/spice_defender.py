"""Defender (Reasoner) module for SPICE-style self-play.

Wraps the existing OnCallEnv rollout logic so the self-play trainer can
evaluate any ScenarioSpec produced by the LLM Attacker.
"""

from __future__ import annotations

import json
import re
from typing import Any

from oncallenv import OnCallRedShiftEnv
from oncallenv.core.tools import MUTATING_TOOLS, READ_ONLY_TOOLS
from oncallenv.core.types import Action, ScenarioSpec
from oncallenv.simulation.scenario_compiler import compile_scenario


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVICES = [
    "api-gateway", "checkout-service", "payment-service",
    "inventory-service", "user-service", "postgres-primary", "redis-cache",
]

COMMAND_RE = re.compile(
    r"\b("
    + "|".join(re.escape(t) for t in [*READ_ONLY_TOOLS, *MUTATING_TOOLS, "declare_resolved"])
    + r")\b(?:\s+([a-z0-9_.:/={}\"\'-]+))?",
    re.IGNORECASE,
)

DEFENDER_SYSTEM = (
    "You are the on-call SRE for OnCallEnv Red Shift. "
    "Investigate alerts, diagnose root cause, remediate safely, "
    "then declare_resolved."
)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

# 1. You MUST first think step-by-step inside <reasoning> and </reasoning> tags. Explain your logic based purely on the Alert details.
# 2. After reasoning, you MUST output ONLY the executable commands inside <actions> and </actions> tags.
# 3. Put exactly ONE simulator command per line inside the actions block.

def build_defender_prompt(spec: ScenarioSpec) -> str:
    """Build the SRE prompt from a ScenarioSpec (compiles it first)."""
    graph = compile_scenario(spec)
    service = graph.root_cause_service
    category = graph.root_cause_category

    # Grab the initial alert text
    alert_msg = f"{category} symptoms detected with elevated p99/error rate"

    return f"""{DEFENDER_SYSTEM}

Task id: {spec.task_id}
Critical alert: {service} reports {alert_msg}
Available services: {", ".join(sorted(graph.services))}
Available tools: kubectl_logs, promql_query, jaeger_search, kubectl_describe_pod, kubectl_top, dns_lookup, check_deploy_history, curl_service, istioctl_routes, kubectl_rollout_restart, kubectl_rollout_undo, kubectl_scale, feature_flag_toggle, traffic_split_update, kubectl_apply_config, declare_resolved

You are a strict code execution agent. You must output a sequence of commands to solve the incident.

STRICT RULES:
1. You MUST output ONLY the commands inside <actions> and </actions> tags.
2. DO NOT write ANY conversational text or markdown blocks outside the tags.
3. Put exactly ONE simulator command per line.
4. Use the provided tools and services to diagnose and remediate, ending with declare_resolved.

Example Output:
<actions>
promql_query {service}
kubectl_logs {service}
kubectl_rollout_undo {service}
declare_resolved
</actions>

Use real commands such as kubectl_logs SERVICE, promql_query SERVICE,
jaeger_search SERVICE, kubectl_rollout_restart SERVICE,
kubectl_rollout_undo SERVICE, kubectl_scale SERVICE,
feature_flag_toggle SERVICE, traffic_split_update SERVICE,
kubectl_apply_config SERVICE, and declare_resolved.
"""


def build_rca(service: str, category: str) -> str:
    """Build a minimal valid RCA JSON for auto-submission."""
    return json.dumps({
        "root_cause_service": service,
        "root_cause_category": category,
        "timeline": [{
            "timestamp": "2026-04-24T09:00:00Z",
            "service": service,
            "description": f"{category} identified from Red Shift telemetry",
        }],
        "five_whys": [
            f"{service} emitted direct {category} symptoms.",
            "The failure propagated through dependent customer-facing services.",
            "The first mitigation needed to target the true faulty component.",
        ],
        "action_items": [
            f"Add regression alerting and runbook coverage for {service} {category}.",
        ],
        "evidence_citations": [
            {"source": "telemetry", "ref": f"kubectl_logs {service}", "excerpt": category},
        ],
        "blast_radius_description":
            "Customer-facing requests saw elevated latency or errors before remediation.",
    })


# ---------------------------------------------------------------------------
# Command parser (reused from train_unsloth_grpo)
# ---------------------------------------------------------------------------


def parse_commands(text: str, max_commands: int = 10) -> list[str]:
    """Extract simulator commands from LLM output."""
    match = re.search(r"<actions>(.*?)</actions>", text, flags=re.IGNORECASE | re.DOTALL)
    body = match.group(1) if match else text
    commands: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip().strip("-*` ")
        if not line:
            continue
        found = COMMAND_RE.search(line)
        if not found:
            continue
        tool = found.group(1).lower()
        arg = (found.group(2) or "").strip().strip("'\"`")
        if tool in MUTATING_TOOLS or tool in {
            "kubectl_logs", "kubectl_top", "kubectl_describe_pod",
            "jaeger_search", "dns_lookup", "check_deploy_history",
            "curl_service", "promql_query", "logql_query", "istioctl_routes",
        }:
            service = next((svc for svc in SERVICES if svc in arg or svc in line), "")
            if not service:
                continue
            commands.append(f"{tool} {service}")
        elif tool == "declare_resolved":
            commands.append("declare_resolved")
        else:
            commands.append(tool)
        if len(commands) >= max_commands:
            break
    return commands


# ---------------------------------------------------------------------------
# Rollout & reward
# ---------------------------------------------------------------------------


def extract_completion_text(completion: Any) -> str:
    """Normalise various completion formats to plain text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def defender_rollout_reward(
    spec: ScenarioSpec,
    completion: Any,
) -> float:
    """Run a single defender completion through the simulator and return reward.

    This compiles the spec into a live env, steps through the parsed commands,
    auto-submits declare_resolved + RCA, and returns the raw reward.
    """
    text = extract_completion_text(completion)
    commands = parse_commands(text)
    if not commands:
        return -0.25

    graph = compile_scenario(spec)
    root_service = graph.root_cause_service
    root_category = graph.root_cause_category

    env = OnCallRedShiftEnv()
    env.reset(task_id=spec.task_id)
    # Override the runtime graph with the one compiled from this spec
    env._runtime = __import__(
        "oncallenv.core.tools", fromlist=["ToolRuntime"]
    ).ToolRuntime(graph)
    env._state.task_id = spec.task_id

    obs = None
    for command in commands:
        obs = env.step(Action(command=command))
        if obs.done:
            break
    if not any(c == "declare_resolved" for c in commands):
        obs = env.step(Action(command="declare_resolved"))
    obs = env.step(Action(command=f"submit_rca {build_rca(root_service, root_category)}"))

    reward = float(obs.reward or 0.0)
    format_bonus = 0.05 if "<actions>" in text.lower() and "</actions>" in text.lower() else 0.0
    concise_bonus = 0.03 if 2 <= len(commands) <= 8 else 0.0
    return max(-0.25, min(1.1, reward + format_bonus + concise_bonus))
