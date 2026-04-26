"""Evaluate a saved Unsloth/TRL GRPO checkpoint on the Red Shift eval rows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from train_unsloth_grpo import evaluate_model


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found under {run_dir}")
    return checkpoints[-1]


def load_rows(dataset_path: Path, eval_tasks: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[: min(eval_tasks, len(rows))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("training_results/unsloth_grpo_qwen3b_easy"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    parser.add_argument("--max-seq-length", type=int, default=1280)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--eval-tasks", type=int, default=32)
    parser.add_argument("--reward-mode", choices=["hard", "easy"], default="easy")
    parser.add_argument("--out-name", default="checkpoint_generations.json")
    args = parser.parse_args()

    checkpoint = args.checkpoint or latest_checkpoint(args.run_dir)
    dataset_path = args.run_dir / "dataset.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing {dataset_path}; run training at least until dataset creation.")
    eval_rows = load_rows(dataset_path, args.eval_tasks)

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

    generations_path = args.run_dir / args.out_name
    checkpoint_eval = evaluate_model(
        model,
        tokenizer,
        eval_rows,
        generations_path,
        args.max_completion_length,
        args.reward_mode,
    )

    baseline_path = args.run_dir / "baseline_generations.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    summary = {
        "status": "checkpoint_eval",
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step(checkpoint),
        "model_name": args.model_name,
        "num_eval_tasks": len(eval_rows),
        "reward_mode": args.reward_mode,
        "baseline_mean_reward": baseline.get("mean_reward"),
        "checkpoint_mean_reward": checkpoint_eval["mean_reward"],
        "generations_path": str(generations_path),
    }
    (args.run_dir / "checkpoint_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
