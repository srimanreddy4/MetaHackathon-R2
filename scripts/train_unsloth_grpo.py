"""Train a Red Shift LLM defender with Unsloth + TRL GRPO.

The model learns to emit SRE action sequences. Each sampled completion is parsed
back into simulator commands and rewarded by the OpenEnv environment. This is
the real LLM training track; the lightweight symbolic policy remains useful for
fast baselines and ablations.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from oncallenv import OnCallRedShiftEnv
from oncallenv.core.tools import MUTATING_TOOLS, READ_ONLY_TOOLS
from oncallenv.core.types import Action
from oncallenv.curriculum import RegretBuffer


SERVICES = [
    "api-gateway",
    "checkout-service",
    "payment-service",
    "inventory-service",
    "user-service",
    "postgres-primary",
    "redis-cache",
]

SEED_TASKS = [
    "seed_easy_memory_leak",
    "seed_dns_misconfiguration",
    "seed_cert_expiry",
    "seed_cache_stampede",
    "seed_replica_lag",
    "seed_http_503_loop",
]

COMMAND_RE = re.compile(
    r"\b("
    + "|".join(re.escape(tool) for tool in [*READ_ONLY_TOOLS, *MUTATING_TOOLS, "declare_resolved"])
    + r")\b(?:\s+([a-z0-9_.:/={}\"'-]+))?",
    re.IGNORECASE,
)


def build_rca(service: str, category: str) -> str:
    return json.dumps(
        {
            "root_cause_service": service,
            "root_cause_category": category,
            "timeline": [
                {
                    "timestamp": "2026-04-24T09:00:00Z",
                    "service": service,
                    "description": f"{category} identified from Red Shift telemetry",
                }
            ],
            "five_whys": [
                f"{service} emitted direct {category} symptoms.",
                "The failure propagated through dependent customer-facing services.",
                "The first mitigation needed to target the true faulty component.",
            ],
            "action_items": [f"Add regression alerting and runbook coverage for {service} {category}."],
            "evidence_citations": [{"source": "telemetry", "ref": f"kubectl_logs {service}", "excerpt": category}],
            "blast_radius_description": "Customer-facing requests saw elevated latency or errors before remediation.",
        }
    )


def inspect_task(task_id: str) -> dict[str, Any]:
    env = OnCallRedShiftEnv()
    obs = env.reset(task_id=task_id)
    graph = env._runtime.graph
    service = graph.root_cause_service
    category = graph.root_cause_category
    prompt = f"""You are the on-call SRE for OnCallEnv Red Shift.

Task id: {task_id}
Critical alert: {obs.alerts[0].service} reports {obs.alerts[0].message}
Available services: {", ".join(obs.services)}
Available tools: {", ".join(obs.available_tools)}

Return only a short action plan. Put one simulator command per line inside:
<actions>
...
</actions>

Use real commands such as kubectl_logs SERVICE, promql_query SERVICE,
jaeger_search SERVICE, kubectl_rollout_restart SERVICE,
kubectl_rollout_undo SERVICE, kubectl_scale SERVICE,
feature_flag_toggle SERVICE, traffic_split_update SERVICE,
kubectl_apply_config SERVICE, and declare_resolved.
Do not include prose outside the tags.
"""
    return {
        "prompt": prompt,
        "task_id": task_id,
        "root_service": service,
        "root_category": category,
        "required": sorted(graph.required_remediations),
    }


def load_task_ids(curriculum_buffer: Path | None, max_tasks: int | None, seed: int) -> list[str]:
    task_ids = list(SEED_TASKS)
    if curriculum_buffer and curriculum_buffer.exists():
        buffer = RegretBuffer.load(curriculum_buffer)
        task_ids = list(dict.fromkeys([item.spec.task_id for item in buffer.scenarios]))
    rng = random.Random(seed)
    rng.shuffle(task_ids)
    if max_tasks:
        task_ids = task_ids[:max_tasks]
    return task_ids


def extract_completion_text(completion: Any) -> str:
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


def parse_commands(text: str, max_commands: int = 10) -> list[str]:
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
        if tool in MUTATING_TOOLS or tool in {"kubectl_logs", "kubectl_top", "kubectl_describe_pod", "jaeger_search", "dns_lookup", "check_deploy_history", "curl_service", "promql_query", "logql_query", "istioctl_routes"}:
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


def rollout_reward(task_id: str, completion: Any, root_service: str, root_category: str) -> float:
    text = extract_completion_text(completion)
    commands = parse_commands(text)
    if not commands:
        return -0.25

    env = OnCallRedShiftEnv()
    env.reset(task_id=task_id)
    obs = None
    for command in commands:
        obs = env.step(Action(command=command))
        if obs.done:
            break
    if not any(command == "declare_resolved" for command in commands):
        obs = env.step(Action(command="declare_resolved"))
    obs = env.step(Action(command=f"submit_rca {build_rca(root_service, root_category)}"))

    reward = float(obs.reward or 0.0)
    format_bonus = 0.05 if "<actions>" in text.lower() and "</actions>" in text.lower() else 0.0
    concise_bonus = 0.03 if 2 <= len(commands) <= 8 else 0.0
    return max(-0.25, min(1.1, reward + format_bonus + concise_bonus))


def evaluate_model(model, tokenizer, task_rows: list[dict[str, Any]], out_path: Path, max_new_tokens: int) -> dict[str, Any]:
    import torch

    scores: dict[str, float] = {}
    generations: dict[str, dict[str, Any]] = {}
    model.eval()
    for row in task_rows:
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        reward = rollout_reward(row["task_id"], completion, row["root_service"], row["root_category"])
        scores[row["task_id"]] = reward
        generations[row["task_id"]] = {"completion": completion, "commands": parse_commands(completion), "reward": reward}
    summary = {
        "mean_reward": sum(scores.values()) / len(scores) if scores else 0.0,
        "scores": scores,
        "generations": generations,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def make_grpo_config(args):
    from trl import GRPOConfig

    kwargs = {
        "output_dir": str(args.out_dir),
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "report_to": [],
        "remove_unused_columns": False,
        "fp16": True,
        "bf16": False,
        "seed": args.seed,
    }
    for optional_key, value in {
        "beta": args.beta,
        "scale_rewards": args.scale_rewards,
        "loss_type": args.loss_type,
        "use_vllm": False,
    }.items():
        try:
            test = dict(kwargs)
            test[optional_key] = value
            GRPOConfig(**test)
            kwargs[optional_key] = value
        except TypeError:
            continue
    return GRPOConfig(**kwargs)


def trainable_parameter_report(model) -> dict[str, int]:
    total = 0
    trainable = 0
    lora_names: list[str] = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
        if "lora" in name.lower():
            lora_names.append(name)
    return {"total": total, "trainable": trainable, "lora_tensors": len(lora_names)}


def ensure_lora_is_trainable(model) -> dict[str, int]:
    """Fail early instead of letting AMP crash with an empty optimizer step."""

    report = trainable_parameter_report(model)
    if report["trainable"] > 0:
        return report

    for name, parameter in model.named_parameters():
        if "lora" in name.lower():
            parameter.requires_grad_(True)

    report = trainable_parameter_report(model)
    if report["trainable"] == 0:
        lora_like = [name for name, _ in model.named_parameters() if "lora" in name.lower()]
        raise RuntimeError(
            "No trainable LoRA parameters were found after FastLanguageModel.get_peft_model(). "
            "This would crash GRPO/AMP with 'No inf checks were recorded for this optimizer'. "
            f"LoRA-like tensors found: {lora_like[:12]}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--out-dir", type=Path, default=Path("training_results/unsloth_grpo"))
    parser.add_argument("--curriculum-buffer", type=Path, default=Path("curriculum_results/buffer.json"))
    parser.add_argument("--max-tasks", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--scale-rewards", default="batch")
    parser.add_argument("--loss-type", default="dr_grpo")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-tasks", type=int, default=24)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260424)
    args = parser.parse_args()

    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    task_ids = load_task_ids(args.curriculum_buffer, args.max_tasks, args.seed)
    rows = [inspect_task(task_id) for task_id in task_ids]
    random.Random(args.seed).shuffle(rows)
    eval_rows = rows[: min(args.eval_tasks, len(rows))]

    dataset_path = args.out_dir / "dataset.jsonl"
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    from datasets import Dataset
    from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported

    try:
        PatchFastRL("GRPO", FastLanguageModel)
    except Exception as exc:
        print(f"WARN: PatchFastRL('GRPO') failed or was already applied: {exc}")
    from trl import GRPOTrainer

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
    model.train()
    trainable_report = ensure_lora_is_trainable(model)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    print(f"Trainable parameter report: {trainable_report}")

    if is_bfloat16_supported():
        # V100 normally uses fp16; keep this only for newer GPUs if the script is reused.
        pass

    baseline = evaluate_model(model, tokenizer, eval_rows, args.out_dir / "baseline_generations.json", args.max_completion_length)

    def redshift_reward(completions, task_id, root_service, root_category, **kwargs):
        return [
            rollout_reward(tid, completion, service, category)
            for completion, tid, service, category in zip(completions, task_id, root_service, root_category)
        ]

    train_dataset = Dataset.from_list(rows)
    training_args = make_grpo_config(args)
    trainer_kwargs = {
        "model": model,
        "reward_funcs": redshift_reward,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    trainer_params = inspect.signature(GRPOTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)

    adapter_dir = args.out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    trained = evaluate_model(model, tokenizer, eval_rows, args.out_dir / "trained_generations.json", args.max_completion_length)

    summary = {
        "model_name": args.model_name,
        "duration_sec": time.time() - start,
        "max_steps": args.max_steps,
        "num_train_tasks": len(rows),
        "num_eval_tasks": len(eval_rows),
        "curriculum_buffer": str(args.curriculum_buffer),
        "baseline_mean_reward": baseline["mean_reward"],
        "trained_mean_reward": trained["mean_reward"],
        "adapter_dir": str(adapter_dir),
        "dataset_path": str(dataset_path),
        "trainable_parameter_report": trainable_report,
        "resumed_from_checkpoint": str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
