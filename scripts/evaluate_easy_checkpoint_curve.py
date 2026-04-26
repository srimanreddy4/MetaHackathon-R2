"""Evaluate every saved easy-GRPO checkpoint on the same held-out rows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from evaluate_rcaeval_qwen_adapter import resolve_run_dir
from train_unsloth_grpo import evaluate_model


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def load_rows(dataset_path: Path, eval_tasks: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[: min(eval_tasks, len(rows))]


def write_plot(summary: dict[str, Any], out_dir: Path) -> str:
    import matplotlib.pyplot as plt

    rows = summary["checkpoint_results"]
    steps = [row["checkpoint_step"] for row in rows]
    rewards = [row["mean_reward"] for row in rows]
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(steps, rewards, marker="o", linewidth=2.2, color="#1565C0", label="checkpoint eval")
    baseline = summary.get("baseline_mean_reward")
    if baseline is not None:
        ax.axhline(float(baseline), color="#616161", linestyle="--", linewidth=1.5, label=f"baseline {float(baseline):.3f}")
    ax.set_title("Easy GRPO Held-Out Reward by Checkpoint")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Mean held-out reward")
    ax.set_ylim(0.0, max(1.0, max(rewards + ([float(baseline)] if baseline is not None else [])) + 0.05))
    ax.grid(alpha=0.25)
    ax.legend()
    for step, reward in zip(steps, rewards):
        ax.text(step, reward + 0.012, f"{reward:.3f}", ha="center", fontsize=9)
    path = out_dir / "easy_grpo_checkpoint_reward_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("training_results/unsloth_grpo_qwen3b_easy"))
    parser.add_argument("--artifact-archive", type=Path, default=None)
    parser.add_argument("--extract-dir", type=Path, default=Path("artifacts/models/easy_grpo_qwen3b_extracted"))
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--eval-tasks", type=int, default=32)
    parser.add_argument("--reward-mode", choices=["hard", "easy"], default="easy")
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results/easy_grpo_checkpoint_curve"))
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir, args.artifact_archive, args.extract_dir)
    dataset_path = run_dir / "dataset.jsonl"
    eval_rows = load_rows(dataset_path, args.eval_tasks)
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found under {run_dir}")

    from peft import PeftModel
    from unsloth import FastLanguageModel

    results: list[dict[str, Any]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in checkpoints:
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

        out_path = args.out_dir / f"checkpoint_{checkpoint_step(checkpoint)}_generations.json"
        evaluation = evaluate_model(
            model,
            tokenizer,
            eval_rows,
            out_path,
            args.max_completion_length,
            args.reward_mode,
        )
        results.append(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_step": checkpoint_step(checkpoint),
                "mean_reward": evaluation["mean_reward"],
                "generations_path": str(out_path),
            }
        )
        del model

    baseline_path = run_dir / "baseline_generations.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    summary: dict[str, Any] = {
        "status": "complete",
        "run_dir": str(run_dir),
        "model_name": args.model_name,
        "num_eval_tasks": len(eval_rows),
        "reward_mode": args.reward_mode,
        "baseline_mean_reward": baseline.get("mean_reward"),
        "checkpoint_results": results,
    }
    summary["plot"] = write_plot(summary, args.out_dir)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
