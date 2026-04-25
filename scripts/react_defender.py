"""Interactive ReAct-style defender utilities.

This module turns Red Shift episodes into turn-level examples:

    observation history -> exactly one next SRE command

The same prompt builder and parser are used for SFT data generation and
interactive model evaluation, so the training target matches the demo loop.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from oncallenv import OnCallRedShiftEnv
from oncallenv.core.tools import AVAILABLE_TOOLS, MUTATING_TOOLS, READ_ONLY_TOOLS
from oncallenv.core.types import Action, Observation
from train_unsloth_grpo import load_task_ids


COMMAND_RE = re.compile(
    r"\b("
    + "|".join(re.escape(tool) for tool in [*AVAILABLE_TOOLS])
    + r")\b(?:\s+([a-z0-9_.:/={}\"'-]+))?",
    re.IGNORECASE,
)

SERVICES = [
    "api-gateway",
    "checkout-service",
    "payment-service",
    "inventory-service",
    "user-service",
    "postgres-primary",
    "redis-cache",
]


def truncate(text: str, limit: int = 900) -> str:
    compact = "\n".join(line.rstrip() for line in text.strip().splitlines() if line.strip())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "\n... [truncated]"


def observation_brief(obs: Observation) -> str:
    alert = obs.alerts[0] if obs.alerts else None
    alert_text = f"{alert.severity} {alert.service}: {alert.message}" if alert else "none"
    return "\n".join(
        [
            f"Task: {obs.task_id}",
            f"Goal: {obs.goal}",
            f"Alert: {alert_text}",
            f"Services: {', '.join(obs.services)}",
            f"Elapsed seconds: {obs.time_elapsed_sec}",
            "Latest observation:",
            truncate(obs.last_action_result),
        ]
    )


def build_react_prompt(obs: Observation, history: list[dict[str, str]], *, max_history: int = 5) -> str:
    turns = history[-max_history:]
    rendered_turns: list[str] = []
    for idx, turn in enumerate(turns, 1):
        rendered_turns.append(
            "\n".join(
                [
                    f"Turn {idx} command: {turn['command']}",
                    "Observation:",
                    truncate(turn["observation"]),
                ]
            )
        )
    history_text = "\n\n".join(rendered_turns) if rendered_turns else "(no previous commands)"
    return f"""You are the interactive on-call SRE for OnCallEnv Red Shift.

Choose exactly one next simulator command based on the current observation and command history.
Return only:
<command>COMMAND</command>

Allowed commands include read-only diagnostics:
{", ".join(READ_ONLY_TOOLS)}

Allowed remediation commands:
{", ".join(MUTATING_TOOLS)}

Terminal command:
declare_resolved

Do not include explanations, markdown, JSON, RCA text, or multiple commands.

Current state:
{observation_brief(obs)}

Command history:
{history_text}

Next command:"""


def format_completion(command: str) -> str:
    return f"<command>{command}</command>"


def parse_command(text: str) -> str:
    match = re.search(r"<command>(.*?)</command>", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = match.group(1).strip() if match else text.strip().splitlines()[0].strip() if text.strip() else ""
    candidate = candidate.strip("-*` ")
    found = COMMAND_RE.search(candidate)
    if not found:
        return ""
    tool = found.group(1).lower()
    arg = (found.group(2) or "").strip().strip("'\"`")
    if tool == "declare_resolved":
        return "declare_resolved"
    if tool == "kubectl_get_pods":
        return "kubectl_get_pods"
    service = next((service for service in SERVICES if service in arg or service in candidate), "")
    if service:
        return f"{tool} {service}"
    return tool if tool in {"istioctl_proxy_status"} else ""


def required_pairs(required: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in required:
        if ":" in item:
            tool, service = item.split(":", 1)
            pairs.append((tool, service))
    return pairs


def build_react_rca(service: str, category: str) -> str:
    return json.dumps(
        {
            "root_cause_service": service,
            "root_cause_category": category,
            "timeline": [
                {
                    "timestamp": "2026-04-24T09:00:00Z",
                    "service": service,
                    "description": f"{category} identified from command observations",
                }
            ],
            "five_whys": [
                f"{service} emitted direct {category} symptoms in logs, metrics, or traces.",
                "The failure propagated through dependent customer-facing services.",
                "A targeted remediation on the faulty component was required.",
            ],
            "action_items": [f"Add runbook coverage and regression alerting for {service} {category}."],
            "evidence_citations": [{"source": "log", "ref": f"kubectl_logs {service}", "excerpt": category}],
            "blast_radius_description": "Customer-facing requests saw elevated latency or errors before remediation.",
        }
    )


def expert_commands_for_task(task_id: str) -> tuple[list[str], dict[str, Any]]:
    env = OnCallRedShiftEnv()
    obs = env.reset(task_id=task_id)
    graph = env._runtime.graph
    root_service = graph.root_cause_service
    root_category = graph.root_cause_category
    required = sorted(graph.required_remediations)
    commands = [
        "kubectl_get_pods",
        f"kubectl_logs {root_service}",
        f"kubectl_top {root_service}",
        f"promql_query {root_service}",
        f"check_deploy_history {root_service}",
    ]
    commands.extend(f"{tool} {service}" for tool, service in required_pairs(required))
    commands.append("declare_resolved")
    return commands, {
        "task_id": task_id,
        "root_service": root_service,
        "root_category": root_category,
        "required": required,
        "alert_service": obs.alerts[0].service if obs.alerts else root_service,
        "alert_message": obs.alerts[0].message if obs.alerts else "",
    }


def generate_task_rows(task_id: str, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = OnCallRedShiftEnv()
    obs = env.reset(task_id=task_id)
    commands, info = expert_commands_for_task(task_id)
    history: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    final_obs: Observation | None = None
    for turn_index, command in enumerate(commands):
        prompt = build_react_prompt(obs, history)
        rows.append(
            {
                "task_id": task_id,
                "split": split,
                "turn_index": turn_index,
                "prompt": prompt,
                "target_command": command,
                "completion": format_completion(command),
                "text": prompt + "\n" + format_completion(command),
                **info,
            }
        )
        final_obs = env.step(Action(command=command))
        history.append({"command": command, "observation": final_obs.last_action_result})
        obs = final_obs
    if final_obs is None or not env.state.resolved_declared:
        final_obs = env.step(Action(command="declare_resolved"))
    final_obs = env.step(Action(command=f"submit_rca {build_react_rca(info['root_service'], info['root_category'])}"))
    info["expert_reward"] = float(final_obs.reward or 0.0)
    info["num_turns"] = len(commands)
    return rows, info


def generate_react_dataset(
    *,
    curriculum_buffer: Path | None,
    max_tasks: int,
    train_tasks: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_ids = load_task_ids(curriculum_buffer, max_tasks, seed)
    rng = random.Random(seed)
    shuffled = list(task_ids)
    rng.shuffle(shuffled)
    train_set = set(shuffled[: min(train_tasks, max(1, len(shuffled) - 1))])
    rows: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    for task_id in shuffled:
        split = "train" if task_id in train_set else "eval"
        task_rows, summary = generate_task_rows(task_id, split)
        rows.extend(task_rows)
        task_summaries.append({"split": split, **summary})
    metadata = {
        "num_tasks": len(shuffled),
        "train_tasks": sum(1 for item in task_summaries if item["split"] == "train"),
        "eval_tasks": sum(1 for item in task_summaries if item["split"] == "eval"),
        "num_rows": len(rows),
        "train_rows": sum(1 for row in rows if row["split"] == "train"),
        "eval_rows": sum(1 for row in rows if row["split"] == "eval"),
        "task_summaries": task_summaries,
    }
    return rows, metadata


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_interactive_rollout(
    *,
    task_id: str,
    command_fn,
    max_turns: int,
) -> dict[str, Any]:
    env = OnCallRedShiftEnv()
    obs = env.reset(task_id=task_id)
    graph = env._runtime.graph
    history: list[dict[str, str]] = []
    trajectory: list[dict[str, str]] = []
    for turn_index in range(max_turns):
        prompt = build_react_prompt(obs, history)
        raw = command_fn(prompt, turn_index)
        command = parse_command(raw)
        if not command:
            command = raw.strip().splitlines()[0].strip() if raw.strip() else "invalid_command"
        obs = env.step(Action(command=command))
        trajectory.append(
            {
                "turn_index": str(turn_index),
                "prompt": prompt,
                "raw_completion": raw,
                "command": command,
                "observation": obs.last_action_result,
            }
        )
        history.append({"command": command, "observation": obs.last_action_result})
        if command == "declare_resolved":
            break
    if not env.state.resolved_declared:
        obs = env.step(Action(command="declare_resolved"))
        trajectory.append(
            {
                "turn_index": str(len(trajectory)),
                "prompt": "(auto)",
                "raw_completion": "declare_resolved",
                "command": "declare_resolved",
                "observation": obs.last_action_result,
            }
        )
    obs = env.step(Action(command=f"submit_rca {build_react_rca(graph.root_cause_service, graph.root_cause_category)}"))
    return {
        "task_id": task_id,
        "reward": float(obs.reward or 0.0),
        "done": obs.done,
        "root_service": graph.root_cause_service,
        "root_category": graph.root_cause_category,
        "trajectory": trajectory,
        "actions": [turn["command"] for turn in trajectory],
        "reward_breakdown": obs.reward_breakdown,
    }
