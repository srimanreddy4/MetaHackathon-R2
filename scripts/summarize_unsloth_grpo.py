"""Summarize complete or interrupted Unsloth GRPO runs.

For interrupted Kaggle jobs this script is intentionally lightweight: it reads
the saved trainer state and baseline/final JSON files without loading the LLM.
That makes it safe to run after stopping a notebook cell just to recover the
latest checkpoint status, training reward curve, and plots.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reward_value(row: dict[str, Any]) -> float | None:
    value = row.get("reward", row.get("rewards/redshift_reward/mean"))
    return float(value) if value is not None else None


def reward_rows(log_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in log_history if reward_value(row) is not None]


def row_step(row: dict[str, Any], index: int, fallback_every: int = 5) -> int:
    value = row.get("step")
    if value is None:
        return (index + 1) * fallback_every
    return int(value)


def write_plots(report: dict[str, Any], log_history: list[dict[str, Any]], plots_dir: Path, plot_prefix: str) -> list[str]:
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    rows = reward_rows(log_history)
    if rows:
        steps = [row_step(row, index) for index, row in enumerate(rows)]
        rewards = [reward_value(row) for row in rows]

        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot(steps, rewards, marker="o", linewidth=2, markersize=3, label="logged GRPO reward")
        ax.axhline(report.get("best_logged_reward", 0.0), color="#2E7D32", linestyle="--", linewidth=1.5, label="best logged")
        ax.set_title("Easy Qwen2.5 3B GRPO Reward Curve")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Easy reward")
        ax.set_ylim(0.0, max(1.0, max(rewards or [0.0]) + 0.05))
        ax.grid(alpha=0.25)
        ax.legend()
        path = plots_dir / f"{plot_prefix}_reward_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

        clipped = [float(row.get("completions/clipped_ratio", 0.0)) for row in rows]
        lengths = [float(row.get("completion_length", row.get("completions/mean_length", 0.0))) for row in rows]
        fig, ax1 = plt.subplots(figsize=(8, 4.6))
        ax1.plot(steps, clipped, color="#C62828", marker="o", markersize=3, label="clipped ratio")
        ax1.set_xlabel("Training step")
        ax1.set_ylabel("Clipped ratio")
        ax1.set_ylim(0.0, 1.05)
        ax1.grid(alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(steps, lengths, color="#1565C0", linewidth=2, alpha=0.75, label="completion length")
        ax2.set_ylabel("Completion tokens")
        ax1.set_title("Easy GRPO Completion Health")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="lower right")
        path = plots_dir / f"{plot_prefix}_completion_health.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    bar_labels: list[str] = []
    bar_values: list[float] = []
    for key, label in [
        ("baseline_mean_reward", "baseline eval"),
        ("last_logged_reward", "last logged"),
        ("best_logged_reward", "best logged"),
        ("trained_mean_reward", "final eval"),
    ]:
        value = report.get(key)
        if value is not None:
            bar_labels.append(label)
            bar_values.append(float(value))
    if bar_values:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        colors = ["#616161", "#1976D2", "#2E7D32", "#7B1FA2"][: len(bar_values)]
        ax.bar(bar_labels, bar_values, color=colors)
        ax.set_title("Easy Qwen2.5 3B GRPO Checkpoint Results")
        ax.set_ylabel("Reward")
        ax.set_ylim(0.0, max(1.0, max(bar_values) + 0.08))
        for idx, value in enumerate(bar_values):
            ax.text(idx, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        path = plots_dir / f"{plot_prefix}_checkpoint_results.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    return written


def build_report(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not run_dir.exists():
        return {
            "status": "missing",
            "run_dir": str(run_dir),
            "latest_checkpoint": None,
            "checkpoint_steps": [],
            "has_final_summary": False,
        }, []

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        summary["status"] = "complete"
        trainer_state_path = run_dir / "trainer_state.json"
        trainer_state = load_json(trainer_state_path) if trainer_state_path.exists() else {}
        return summary, trainer_state.get("log_history", [])

    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=checkpoint_step)
    latest = checkpoints[-1] if checkpoints else None
    trainer_state_path = latest / "trainer_state.json" if latest else None
    trainer_state = load_json(trainer_state_path) if trainer_state_path and trainer_state_path.exists() else {}
    log_history = trainer_state.get("log_history", [])
    rows = reward_rows(log_history)
    last_reward = rows[-1] if rows else {}
    best_reward = max(
        (reward_value(row) for row in rows),
        default=None,
    )
    baseline_path = run_dir / "baseline_generations.json"
    baseline = load_json(baseline_path) if baseline_path.exists() else {}

    partial = {
        "status": "partial",
        "run_dir": str(run_dir),
        "latest_checkpoint": str(latest) if latest else None,
        "checkpoint_steps": [checkpoint_step(path) for path in checkpoints],
        "global_step": trainer_state.get("global_step"),
        "max_steps": trainer_state.get("max_steps"),
        "epoch": trainer_state.get("epoch"),
        "baseline_mean_reward": baseline.get("mean_reward"),
        "last_logged_reward": reward_value(last_reward) if last_reward else None,
        "best_logged_reward": best_reward,
        "last_logged_clipped_ratio": last_reward.get("completions/clipped_ratio") if last_reward else None,
        "last_logged_completion_length": last_reward.get("completion_length", last_reward.get("completions/mean_length")) if last_reward else None,
        "num_reward_logs": len(rows),
        "last_log": last_reward,
        "has_final_summary": False,
    }
    return partial, log_history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="?", default=Path("training_results/unsloth_grpo_qwen3b_kaggle"))
    parser.add_argument("--write-report", action="store_true", help="Write checkpoint_report.json into the run directory.")
    parser.add_argument("--plots-dir", type=Path, default=None, help="Optional directory for PNG plots.")
    parser.add_argument("--plot-prefix", default=None, help="Filename prefix for generated plots.")
    args = parser.parse_args()

    report, log_history = build_report(args.run_dir)
    if args.write_report and args.run_dir.exists():
        (args.run_dir / "checkpoint_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.plots_dir is not None:
        plot_prefix = args.plot_prefix or args.run_dir.name
        report["plots"] = write_plots(report, log_history, args.plots_dir, plot_prefix)
        if args.write_report and args.run_dir.exists():
            (args.run_dir / "checkpoint_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
