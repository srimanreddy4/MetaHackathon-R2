"""Generate turn-level ReAct SFT data from scripted Red Shift experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from react_defender import generate_react_dataset, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum-buffer", type=Path, default=Path("curriculum_results/buffer.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("training_results/react_sft_qwen3b"))
    parser.add_argument("--max-tasks", type=int, default=120)
    parser.add_argument("--train-tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260424)
    args = parser.parse_args()

    rows, metadata = generate_react_dataset(
        curriculum_buffer=args.curriculum_buffer,
        max_tasks=args.max_tasks,
        train_tasks=args.train_tasks,
        seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "react_trajectories.jsonl", rows)
    (args.out_dir / "trajectory_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "dataset_path": str(args.out_dir / "react_trajectories.jsonl"),
                **{key: metadata[key] for key in ["num_tasks", "train_tasks", "eval_tasks", "num_rows", "train_rows", "eval_rows"]},
                "mean_expert_reward": sum(item["expert_reward"] for item in metadata["task_summaries"]) / len(metadata["task_summaries"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
