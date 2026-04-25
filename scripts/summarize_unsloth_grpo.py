"""Summarize complete or interrupted Unsloth GRPO runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="?", default=Path("training_results/unsloth_grpo_qwen3b_kaggle"))
    args = parser.parse_args()

    run_dir = args.run_dir
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        summary["status"] = "complete"
        print(json.dumps(summary, indent=2))
        return

    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=checkpoint_step)
    latest = checkpoints[-1] if checkpoints else None
    trainer_state_path = latest / "trainer_state.json" if latest else None
    trainer_state = load_json(trainer_state_path) if trainer_state_path and trainer_state_path.exists() else {}
    log_history = trainer_state.get("log_history", [])
    reward_rows = [
        row for row in log_history
        if "reward" in row or "rewards/redshift_reward/mean" in row
    ]
    last_reward = reward_rows[-1] if reward_rows else {}
    best_reward = max(
        (row.get("reward", row.get("rewards/redshift_reward/mean", float("-inf"))) for row in reward_rows),
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
        "last_logged_reward": last_reward.get("reward", last_reward.get("rewards/redshift_reward/mean")),
        "best_logged_reward": best_reward,
        "last_log": last_reward,
        "has_final_summary": False,
    }
    print(json.dumps(partial, indent=2))


if __name__ == "__main__":
    main()
