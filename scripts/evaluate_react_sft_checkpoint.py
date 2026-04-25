"""Evaluate an interrupted ReAct SFT checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from react_defender import generate_react_dataset, write_jsonl
from train_react_sft import evaluate_interactive, evaluate_next_action


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found under {run_dir}")
    return checkpoints[-1]


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_dataset(args) -> list[dict[str, Any]]:
    dataset_path = args.run_dir / "react_trajectories.jsonl"
    metadata_path = args.run_dir / "trajectory_metadata.json"
    if dataset_path.exists():
        return load_rows(dataset_path)

    rows, metadata = generate_react_dataset(
        curriculum_buffer=args.curriculum_buffer,
        max_tasks=args.max_tasks,
        train_tasks=args.train_tasks,
        seed=args.seed,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(dataset_path, rows)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("training_results/react_sft_qwen3b"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    parser.add_argument("--curriculum-buffer", type=Path, default=Path("curriculum_results/buffer.json"))
    parser.add_argument("--max-tasks", type=int, default=120)
    parser.add_argument("--train-tasks", type=int, default=100)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--eval-action-rows", type=int, default=80)
    parser.add_argument("--eval-rollout-tasks", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260424)
    args = parser.parse_args()

    rows = ensure_dataset(args)
    checkpoint = args.checkpoint or latest_checkpoint(args.run_dir)

    from peft import PeftModel
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
    try:
        FastLanguageModel.for_inference(model)
    except Exception:
        pass

    next_action = evaluate_next_action(
        model,
        tokenizer,
        rows,
        args.run_dir / "checkpoint_next_action.json",
        max_rows=args.eval_action_rows,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
    )
    interactive = evaluate_interactive(
        model,
        tokenizer,
        rows,
        args.run_dir / "checkpoint_interactive_rollouts.json",
        max_tasks=args.eval_rollout_tasks,
        max_turns=args.max_turns,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
    )

    summary = {
        "status": "checkpoint_eval",
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step(checkpoint),
        "model_name": args.model_name,
        "num_eval_action_rows": next_action["num_rows"],
        "num_eval_rollout_tasks": interactive["num_tasks"],
        "checkpoint_next_action_accuracy": next_action["exact_command_accuracy"],
        "checkpoint_interactive_mean_reward": interactive["mean_reward"],
        "next_action_path": str(args.run_dir / "checkpoint_next_action.json"),
        "interactive_rollouts_path": str(args.run_dir / "checkpoint_interactive_rollouts.json"),
    }
    (args.run_dir / "checkpoint_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
