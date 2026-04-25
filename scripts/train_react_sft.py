"""Train an interactive ReAct defender with turn-level SFT.

Unlike the all-at-once GRPO defender, this model learns to emit one command at a
time from the current observation history. The resulting checkpoints can be
evaluated as real observation/action loops and used directly in the demo UI.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from react_defender import generate_react_dataset, parse_command, run_interactive_rollout, write_jsonl
from train_unsloth_grpo import ensure_lora_is_trainable


def unique_task_ids(rows: list[dict[str, Any]], split: str, limit: int) -> list[str]:
    task_ids = list(dict.fromkeys(row["task_id"] for row in rows if row["split"] == split))
    return task_ids[: min(limit, len(task_ids))]


def tokenize_dataset(tokenizer, rows: list[dict[str, Any]], max_seq_length: int):
    from datasets import Dataset

    dataset = Dataset.from_list([{"prompt": row["prompt"], "completion": row["completion"]} for row in rows])

    def tokenize(example):
        prompt_ids = tokenizer(example["prompt"] + "\n", add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(example["completion"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + completion_ids)[:max_seq_length]
        labels = ([-100] * len(prompt_ids) + completion_ids)[:max_seq_length]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    return dataset.map(tokenize, remove_columns=dataset.column_names)


class CommandDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
            batch["labels"].append(feature["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def model_command_fn(model, tokenizer, *, max_new_tokens: int):
    import torch

    def command_fn(prompt: str, _: int) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)

    return command_fn


def evaluate_next_action(model, tokenizer, rows: list[dict[str, Any]], out_path: Path, *, max_rows: int, max_new_tokens: int) -> dict[str, Any]:
    eval_rows = [row for row in rows if row["split"] == "eval"][:max_rows]
    command_fn = model_command_fn(model, tokenizer, max_new_tokens=max_new_tokens)
    examples: list[dict[str, Any]] = []
    correct = 0
    for row in eval_rows:
        raw = command_fn(row["prompt"], row["turn_index"])
        predicted = parse_command(raw)
        expected = row["target_command"]
        is_correct = predicted == expected
        correct += int(is_correct)
        examples.append(
            {
                "task_id": row["task_id"],
                "turn_index": row["turn_index"],
                "expected": expected,
                "predicted": predicted,
                "raw_completion": raw,
                "correct": is_correct,
            }
        )
    summary = {
        "num_rows": len(eval_rows),
        "exact_command_accuracy": correct / len(eval_rows) if eval_rows else 0.0,
        "examples": examples,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_interactive(model, tokenizer, rows: list[dict[str, Any]], out_path: Path, *, max_tasks: int, max_turns: int, max_new_tokens: int) -> dict[str, Any]:
    task_ids = unique_task_ids(rows, "eval", max_tasks)
    command_fn = model_command_fn(model, tokenizer, max_new_tokens=max_new_tokens)
    rollouts = [run_interactive_rollout(task_id=task_id, command_fn=command_fn, max_turns=max_turns) for task_id in task_ids]
    summary = {
        "num_tasks": len(rollouts),
        "mean_reward": sum(item["reward"] for item in rollouts) / len(rollouts) if rollouts else 0.0,
        "rollouts": rollouts,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_training(log_history: list[dict[str, Any]], summary: dict[str, Any], plots_dir: Path, plot_prefix: str) -> list[str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    loss_rows = [row for row in log_history if "loss" in row]
    if loss_rows:
        steps = [int(row.get("step", idx + 1)) for idx, row in enumerate(loss_rows)]
        losses = [float(row["loss"]) for row in loss_rows]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot(steps, losses, marker="o", markersize=3, linewidth=2)
        ax.set_title("Interactive ReAct SFT Training Loss")
        ax.set_xlabel("Training step")
        ax.set_ylabel("SFT loss")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = plots_dir / f"{plot_prefix}_loss_curve.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    labels = ["base rollout", "trained rollout", "base action", "trained action"]
    values = [
        summary.get("baseline_interactive_mean_reward"),
        summary.get("trained_interactive_mean_reward"),
        summary.get("baseline_next_action_accuracy"),
        summary.get("trained_next_action_accuracy"),
    ]
    pairs = [(label, value) for label, value in zip(labels, values) if value is not None]
    if pairs:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.bar([item[0] for item in pairs], [float(item[1]) for item in pairs], color=["#616161", "#2E7D32", "#1565C0", "#7B1FA2"][: len(pairs)])
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Interactive ReAct Defender Evaluation")
        for idx, (_, value) in enumerate(pairs):
            ax.text(idx, float(value) + 0.015, f"{float(value):.3f}", ha="center", fontsize=9)
        fig.tight_layout()
        path = plots_dir / f"{plot_prefix}_eval_bar.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    parser.add_argument("--curriculum-buffer", type=Path, default=Path("curriculum_results/buffer.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("training_results/react_sft_qwen3b"))
    parser.add_argument("--max-tasks", type=int, default=120)
    parser.add_argument("--train-tasks", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--eval-action-rows", type=int, default=80)
    parser.add_argument("--eval-rollout-tasks", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260424)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    rows, metadata = generate_react_dataset(
        curriculum_buffer=args.curriculum_buffer,
        max_tasks=args.max_tasks,
        train_tasks=args.train_tasks,
        seed=args.seed,
    )
    dataset_path = args.out_dir / "react_trajectories.jsonl"
    write_jsonl(dataset_path, rows)
    (args.out_dir / "trajectory_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    from transformers import Trainer, TrainingArguments
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    trainable_report = ensure_lora_is_trainable(model)

    baseline_next = evaluate_next_action(
        model,
        tokenizer,
        rows,
        args.out_dir / "baseline_next_action.json",
        max_rows=args.eval_action_rows,
        max_new_tokens=args.max_new_tokens,
    )
    baseline_rollout = evaluate_interactive(
        model,
        tokenizer,
        rows,
        args.out_dir / "baseline_interactive_rollouts.json",
        max_tasks=args.eval_rollout_tasks,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
    )

    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "eval"]
    train_dataset = tokenize_dataset(tokenizer, train_rows, args.max_seq_length)
    eval_dataset = tokenize_dataset(tokenizer, eval_rows[: min(len(eval_rows), args.eval_action_rows)], args.max_seq_length)
    data_collator = CommandDataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=[],
        fp16=True,
        bf16=False,
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    adapter_dir = args.out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    trained_next = evaluate_next_action(
        model,
        tokenizer,
        rows,
        args.out_dir / "trained_next_action.json",
        max_rows=args.eval_action_rows,
        max_new_tokens=args.max_new_tokens,
    )
    trained_rollout = evaluate_interactive(
        model,
        tokenizer,
        rows,
        args.out_dir / "trained_interactive_rollouts.json",
        max_tasks=args.eval_rollout_tasks,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
    )

    summary = {
        "model_name": args.model_name,
        "duration_sec": time.time() - start,
        "max_steps": args.max_steps,
        "dataset_path": str(dataset_path),
        "num_train_rows": len(train_rows),
        "num_eval_rows": len(eval_rows),
        "num_train_tasks": metadata["train_tasks"],
        "num_eval_tasks": metadata["eval_tasks"],
        "baseline_next_action_accuracy": baseline_next["exact_command_accuracy"],
        "trained_next_action_accuracy": trained_next["exact_command_accuracy"],
        "baseline_interactive_mean_reward": baseline_rollout["mean_reward"],
        "trained_interactive_mean_reward": trained_rollout["mean_reward"],
        "adapter_dir": str(adapter_dir),
        "trainable_parameter_report": trainable_report,
    }
    summary["plots"] = plot_training(trainer.state.log_history, summary, Path("docs/plots"), "react_sft_qwen3b")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
