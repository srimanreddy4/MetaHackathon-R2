"""Analyze easy GRPO generations beyond mean reward.

This script is intentionally lightweight: it reads the saved dataset,
checkpoint generations, and trainer report without loading the LLM. It produces
the evaluation views that are useful for the submission story:

- logged reward vs training step
- tool-use/process metrics
- reward by fault type
- reward by root service
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from evaluate_rcaeval_qwen_adapter import resolve_run_dir
from train_unsloth_grpo import MUTATING_TOOLS, READ_ONLY_TOOLS, parse_commands, required_pairs


STATEFUL_SERVICES = {"postgres-primary", "redis-cache"}
UNSAFE_STATEFUL_TOOLS = {"kubectl_rollout_restart", "kubectl_rollout_undo"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def command_tool(command: str) -> str:
    return command.split()[0] if command.split() else ""


def command_service(command: str) -> str:
    parts = command.split()
    return parts[1] if len(parts) > 1 else ""


def is_read_only(command: str) -> bool:
    return command_tool(command) in READ_ONLY_TOOLS


def is_mutating(command: str) -> bool:
    return command_tool(command) in MUTATING_TOOLS


def task_metrics(row: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    completion = str(generation.get("completion", ""))
    commands = generation.get("commands") or parse_commands(completion)
    reward = float(generation.get("reward", 0.0))
    required = required_pairs(row.get("required", []))
    required_commands = {f"{tool} {service}" for tool, service in required}
    command_set = set(commands)
    tools_seen = {command_tool(command) for command in commands}
    services_seen = {command_service(command) for command in commands if command_service(command)}

    exact_required = len(required_commands & command_set)
    required_coverage = exact_required / len(required_commands) if required_commands else 1.0
    required_tool_hit = bool({tool for tool, _ in required} & tools_seen)

    first_read = next((idx for idx, command in enumerate(commands) if is_read_only(command)), None)
    first_mutating = next((idx for idx, command in enumerate(commands) if is_mutating(command)), None)
    diagnostic_before_mutation = first_read is not None and (first_mutating is None or first_read < first_mutating)

    seen_required: set[str] = set()
    early_declare = False
    for command in commands:
        if command in required_commands:
            seen_required.add(command)
        if command == "declare_resolved" and required_commands and seen_required != required_commands:
            early_declare = True
            break

    unsafe_stateful_restart = any(
        command_tool(command) in UNSAFE_STATEFUL_TOOLS and command_service(command) in STATEFUL_SERVICES
        for command in commands
    )

    lower_completion = completion.lower()
    format_valid = "<actions>" in lower_completion and "</actions>" in lower_completion
    root_service = row["root_service"]
    root_service_mentioned = root_service in lower_completion or root_service in services_seen

    return {
        "task_id": row["task_id"],
        "root_service": root_service,
        "root_category": row["root_category"],
        "template": row.get("template", ""),
        "reward": reward,
        "num_commands": len(commands),
        "format_valid": format_valid,
        "has_diagnostic_action": any(is_read_only(command) for command in commands),
        "has_mutating_action": any(is_mutating(command) for command in commands),
        "diagnostic_before_mutation": diagnostic_before_mutation,
        "root_service_mentioned": root_service_mentioned,
        "required_tool_hit": required_tool_hit,
        "required_exact_coverage": required_coverage,
        "all_required_exact": required_coverage >= 0.999,
        "has_declare_resolved": "declare_resolved" in command_set,
        "early_declare_resolved": early_declare,
        "unsafe_stateful_restart": unsafe_stateful_restart,
        "commands": commands,
        "required": row.get("required", []),
    }


def average_bool(rows: list[dict[str, Any]], key: str) -> float:
    return mean([1.0 if row[key] else 0.0 for row in rows]) if rows else 0.0


def average_float(rows: list[dict[str, Any]], key: str) -> float:
    return mean([float(row[key]) for row in rows]) if rows else 0.0


def grouped(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return [
        {
            key: group_key,
            "num_tasks": len(items),
            "mean_reward": average_float(items, "reward"),
            "required_exact_coverage": average_float(items, "required_exact_coverage"),
            "early_declare_rate": average_bool(items, "early_declare_resolved"),
        }
        for group_key, items in sorted(groups.items())
    ]


def reward_log_rows(report: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    latest_checkpoint = report.get("latest_checkpoint")
    if latest_checkpoint:
        candidates = [
            Path(latest_checkpoint) / "trainer_state.json",
            run_dir / Path(latest_checkpoint).name / "trainer_state.json",
        ]
        for state_path in candidates:
            if not state_path.exists():
                continue
            state = load_json(state_path)
            return [
                row
                for row in state.get("log_history", [])
                if row.get("reward") is not None or row.get("rewards/redshift_reward/mean") is not None
            ]
    last_log = report.get("last_log")
    return [last_log] if isinstance(last_log, dict) else []


def write_plots(summary: dict[str, Any], task_rows: list[dict[str, Any]], out_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    log_rows = summary.get("reward_log_history", [])
    if log_rows:
        steps = [int(row.get("step", idx + 1)) for idx, row in enumerate(log_rows)]
        rewards = [float(row.get("reward", row.get("rewards/redshift_reward/mean", 0.0))) for row in log_rows]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot(steps, rewards, marker="o", markersize=3, linewidth=2, color="#1565C0")
        ax.set_title("Easy GRPO Logged Reward vs Training Step")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Logged easy reward")
        ax.grid(alpha=0.25)
        ax.set_ylim(0.0, max(1.0, max(rewards) + 0.05))
        path = out_dir / "easy_grpo_reward_by_step.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    metrics = summary["process_metrics"]
    metric_labels = [
        ("format_valid_rate", "valid XML"),
        ("diagnostic_action_rate", "diagnostic"),
        ("diagnostic_before_mutation_rate", "diagnostic before fix"),
        ("root_service_mention_rate", "root service"),
        ("required_tool_hit_rate", "right tool family"),
        ("all_required_exact_rate", "all required fixes"),
        ("early_declare_rate", "early resolved"),
        ("unsafe_stateful_restart_rate", "unsafe stateful"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [label for _, label in metric_labels]
    values = [float(metrics[key]) for key, _ in metric_labels]
    colors = ["#C62828" if key in {"early_declare_rate", "unsafe_stateful_restart_rate"} else "#2E7D32" for key, _ in metric_labels]
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_title("Easy GRPO Tool-Use Process Metrics")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.015, f"{value:.2f}", ha="center", fontsize=8)
    path = out_dir / "easy_grpo_process_metrics.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    for group_key, filename, title in [
        ("by_fault_type", "easy_grpo_reward_by_fault.png", "Easy GRPO Reward by Fault Type"),
        ("by_root_service", "easy_grpo_reward_by_service.png", "Easy GRPO Reward by Root Service"),
    ]:
        rows = sorted(summary[group_key], key=lambda row: row["mean_reward"])
        fig, ax = plt.subplots(figsize=(max(7, len(rows) * 0.7), 4.8))
        labels = [row["root_category" if group_key == "by_fault_type" else "root_service"] for row in rows]
        values = [float(row["mean_reward"]) for row in rows]
        ax.bar(range(len(labels)), values, color="#00897B")
        ax.set_title(title)
        ax.set_ylabel("Mean reward")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        for idx, value in enumerate(values):
            ax.text(idx, value + 0.012, f"{value:.2f}", ha="center", fontsize=8)
        path = out_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    rewards = [row["reward"] for row in sorted(task_rows, key=lambda row: row["reward"])]
    ax.plot(range(1, len(rewards) + 1), rewards, marker="o", markersize=3, linewidth=1.5, color="#6A1B9A")
    ax.axhline(mean(rewards), color="#C62828", linestyle="--", label=f"mean {mean(rewards):.3f}")
    ax.set_title("Easy GRPO Held-Out Task Reward Distribution")
    ax.set_xlabel("Task sorted by reward")
    ax.set_ylabel("Reward")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    path = out_dir / "easy_grpo_task_reward_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("training_results/unsloth_grpo_qwen3b_easy"))
    parser.add_argument("--artifact-archive", type=Path, default=None)
    parser.add_argument("--extract-dir", type=Path, default=Path("artifacts/models/easy_grpo_qwen3b_extracted"))
    parser.add_argument("--generations", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--checkpoint-report", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results/easy_grpo_analysis"))
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir, args.artifact_archive, args.extract_dir)
    dataset_path = args.dataset or run_dir / "dataset.jsonl"
    generations_path = args.generations or run_dir / "checkpoint_generations.json"
    report_path = args.checkpoint_report or run_dir / "checkpoint_report.json"

    dataset = {row["task_id"]: row for row in load_jsonl(dataset_path)}
    generation_summary = load_json(generations_path)
    generations = generation_summary.get("generations", {})
    task_rows = [
        task_metrics(dataset[task_id], generation)
        for task_id, generation in generations.items()
        if task_id in dataset
    ]
    if not task_rows:
        raise ValueError(f"No overlapping task ids between {dataset_path} and {generations_path}")

    report = load_json(report_path) if report_path.exists() else {}
    reward_history = reward_log_rows(report, run_dir)
    process_metrics = {
        "mean_reward": average_float(task_rows, "reward"),
        "num_tasks": len(task_rows),
        "mean_commands": average_float(task_rows, "num_commands"),
        "format_valid_rate": average_bool(task_rows, "format_valid"),
        "diagnostic_action_rate": average_bool(task_rows, "has_diagnostic_action"),
        "mutating_action_rate": average_bool(task_rows, "has_mutating_action"),
        "diagnostic_before_mutation_rate": average_bool(task_rows, "diagnostic_before_mutation"),
        "root_service_mention_rate": average_bool(task_rows, "root_service_mentioned"),
        "required_tool_hit_rate": average_bool(task_rows, "required_tool_hit"),
        "required_exact_coverage": average_float(task_rows, "required_exact_coverage"),
        "all_required_exact_rate": average_bool(task_rows, "all_required_exact"),
        "declare_resolved_rate": average_bool(task_rows, "has_declare_resolved"),
        "early_declare_rate": average_bool(task_rows, "early_declare_resolved"),
        "unsafe_stateful_restart_rate": average_bool(task_rows, "unsafe_stateful_restart"),
    }

    summary: dict[str, Any] = {
        "status": "complete",
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "generations_path": str(generations_path),
        "checkpoint_report_path": str(report_path),
        "checkpoint_mean_reward": generation_summary.get("mean_reward"),
        "baseline_mean_reward": report.get("baseline_mean_reward"),
        "process_metrics": process_metrics,
        "by_fault_type": grouped(task_rows, "root_category"),
        "by_root_service": grouped(task_rows, "root_service"),
        "by_template": grouped(task_rows, "template"),
        "task_metrics": task_rows,
        "reward_log_history": reward_history,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary["plots"] = write_plots(summary, task_rows, args.out_dir)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"task_metrics", "reward_log_history"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
