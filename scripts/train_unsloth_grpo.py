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
from oncallenv.rewards.sre_shaped import (
    build_sre_prompt,
    sre_format_score,
    sre_investigation_score,
    sre_remediation_score,
    sre_shaped_reward,
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


FAULT_RUNBOOK_HINTS = {
    "oom_kill": "memory pressure or OOMKilled usually needs kubectl_rollout_restart on the faulty service",
    "cpu_hog": "CPU saturation usually needs kubectl_scale on the faulty service",
    "network_partition": "network partition symptoms usually need traffic_split_update on the faulty service",
    "dns_misconfig": "DNS or no-route symptoms usually need kubectl_apply_config on the faulty service",
    "replica_lag": "replica lag usually needs feature_flag_toggle on the faulty service",
    "cache_stampede": "cache stampede usually needs feature_flag_toggle on the faulty service",
    "http_503_loop": "HTTP 503 loops usually need kubectl_rollout_undo on the faulty service",
    "deadlock": "deadlocks usually need kubectl_rollout_restart on the faulty service",
    "disk_full": "disk-full configuration incidents usually need kubectl_apply_config on the faulty service",
    "cert_expiry": "certificate expiry usually needs kubectl_apply_config on the faulty service",
    "clock_skew": "clock skew usually needs kubectl_rollout_restart on the faulty service",
    "gc_pause": "GC pause incidents usually need kubectl_rollout_restart on the faulty service",
}

PROMPT_TEMPLATES = [
    "standard",
    "runbook",
    "triage",
]


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


def required_pairs(required: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for item in required:
        if ":" in item:
            tool, service = item.split(":", 1)
            pairs.append((tool, service))
    return pairs


def build_prompt(
    *,
    task_id: str,
    alert_service: str,
    alert_message: str,
    services: list[str],
    tools: list[str],
    root_service: str,
    root_category: str,
    required: list[str],
    prompt_mode: str,
    template: str,
) -> str:
    if prompt_mode == "sre":
        return build_sre_prompt(
            task_id=task_id,
            alert_service=alert_service,
            alert_message=alert_message,
            services=list(services),
        )
    base = f"""You are the on-call SRE for OnCallEnv Red Shift.

Task id: {task_id}
Critical alert: {alert_service} reports {alert_message}
Available services: {", ".join(services)}
Available tools: {", ".join(tools)}

Return only the XML action block below. Put one simulator command per line inside:
<actions>
...
</actions>
Stop immediately after the closing </actions> tag.

Use real commands such as kubectl_logs SERVICE, promql_query SERVICE,
jaeger_search SERVICE, kubectl_rollout_restart SERVICE,
kubectl_rollout_undo SERVICE, kubectl_scale SERVICE,
feature_flag_toggle SERVICE, traffic_split_update SERVICE,
kubectl_apply_config SERVICE, and declare_resolved.
Do not include explanations, markdown, bullets, JSON, RCA text, or prose outside the tags.
"""
    if prompt_mode == "hard":
        return base

    accepted = ", ".join(f"{tool} {service}" for tool, service in required_pairs(required))
    hint = FAULT_RUNBOOK_HINTS.get(root_category, f"{root_category} symptoms should be remediated on {root_service}")
    if template == "runbook":
        return (
            base
            + f"\nTraining runbook hint: suspected faulty service is {root_service}. "
            + f"Fault family is {root_category}. {hint}. "
            + f"Accepted remediation command for this easy curriculum item: {accepted}. "
            + "A good answer is exactly 3-5 command lines and ends with </actions>.\n"
        )
    if template == "triage":
        return (
            base
            + f"\nEasy triage hints: first inspect {root_service}; then apply the remediation matching {root_category}; "
            + f"then declare_resolved. Gold remediation: {accepted}. "
            + "A good answer is exactly 3-5 command lines and ends with </actions>.\n"
        )
    return (
        base
        + f"\nEasy-mode hints: root service = {root_service}; fault = {root_category}; "
        + f"best remediation = {accepted}. Include declare_resolved after the fix. "
        + "A good answer is exactly 3-5 command lines and ends with </actions>.\n"
    )


def inspect_task(task_id: str, *, prompt_mode: str = "hard", template: str = "standard") -> dict[str, Any]:
    env = OnCallRedShiftEnv()
    obs = env.reset(task_id=task_id)
    graph = env._runtime.graph
    service = graph.root_cause_service
    category = graph.root_cause_category
    required = sorted(graph.required_remediations)
    prompt = build_prompt(
        task_id=task_id,
        alert_service=obs.alerts[0].service,
        alert_message=obs.alerts[0].message,
        services=obs.services,
        tools=obs.available_tools,
        root_service=service,
        root_category=category,
        required=required,
        prompt_mode=prompt_mode,
        template=template,
    )
    return {
        "prompt": prompt,
        "task_id": task_id,
        "root_service": service,
        "root_category": category,
        "required": required,
        "prompt_mode": prompt_mode,
        "template": template,
    }


def load_task_ids(curriculum_buffer: Path | None, max_tasks: int | None, seed: int, max_solve_rate: float | None = None) -> list[str]:
    task_ids = list(SEED_TASKS)
    if curriculum_buffer and curriculum_buffer.exists():
        buffer = RegretBuffer.load(curriculum_buffer)
        scenarios = buffer.scenarios
        if max_solve_rate is not None:
            hard = [item for item in scenarios if item.solve_rate < max_solve_rate]
            if hard:
                print(f"[curriculum] keeping {len(hard)}/{len(scenarios)} scenarios with solve_rate < {max_solve_rate}")
                scenarios = hard
            else:
                print(f"[curriculum] WARN: no scenarios with solve_rate < {max_solve_rate}; using full buffer")
        task_ids = list(dict.fromkeys([item.spec.task_id for item in scenarios]))
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


def shaped_easy_reward(completion: Any, commands: list[str], env_reward: float, required: list[str], root_service: str) -> float:
    text = extract_completion_text(completion)
    lower = text.lower()
    pairs = required_pairs(required)
    required_tools = {tool for tool, _ in pairs}
    required_services = {service for _, service in pairs}
    command_set = set(commands)
    tools_seen = {command.split()[0] for command in commands if command.split()}
    services_seen = {service for command in commands for service in SERVICES if service in command}

    score = 0.0
    if "<actions>" in lower and "</actions>" in lower:
        score += 0.10
    if commands:
        score += 0.08
    if any(command.split()[0] in READ_ONLY_TOOLS for command in commands if command.split()):
        score += 0.10
    if root_service in services_seen:
        score += 0.18
    elif required_services & services_seen:
        score += 0.12
    if required_tools & tools_seen:
        score += 0.18
    exact_matches = 0
    for tool, service in pairs:
        if f"{tool} {service}" in command_set:
            exact_matches += 1
    if pairs:
        score += 0.28 * (exact_matches / len(pairs))
    if "declare_resolved" in command_set:
        score += 0.06
    if 2 <= len(commands) <= 8:
        score += 0.04
    if any(command.startswith("submit_rca") for command in commands):
        score -= 0.05

    # Keep a connection to the real environment reward, but make the gradient
    # much denser for early LLM policy learning.
    score = 0.75 * score + 0.25 * max(0.0, env_reward)
    return max(-0.1, min(1.0, score))


def rollout_reward(task_id: str, completion: Any, root_service: str, root_category: str, required: list[str] | None = None, reward_mode: str = "hard") -> float:
    text = extract_completion_text(completion)
    if reward_mode == "sre":
        return float(sre_shaped_reward(text, task_id))
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
    if reward_mode == "easy":
        return shaped_easy_reward(completion, commands, reward, required or [], root_service)
    format_bonus = 0.05 if "<actions>" in text.lower() and "</actions>" in text.lower() else 0.0
    concise_bonus = 0.03 if 2 <= len(commands) <= 8 else 0.0
    return max(-0.25, min(1.1, reward + format_bonus + concise_bonus))


def evaluate_model(model, tokenizer, task_rows: list[dict[str, Any]], out_path: Path, max_new_tokens: int, reward_mode: str = "hard") -> dict[str, Any]:
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
        reward = rollout_reward(row["task_id"], completion, row["root_service"], row["root_category"], row.get("required", []), reward_mode)
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
        "log_completions": False,
        "log_on_each_node": True,
        "disable_tqdm": False,
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
    parser.add_argument("--reward-mode", choices=["hard", "easy", "sre"], default="hard")
    parser.add_argument("--prompt-mode", choices=["hard", "easy", "sre"], default="hard")
    parser.add_argument("--prompt-variants", type=int, default=1)
    parser.add_argument("--max-solve-rate", type=float, default=None,
                        help="If set, only keep curriculum scenarios with solve_rate < this value. "
                             "Recommended 0.25 for SRE-mode training to avoid trivial tasks.")
    parser.add_argument("--logging-steps", type=int, default=1)
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

    task_ids = load_task_ids(args.curriculum_buffer, args.max_tasks, args.seed, args.max_solve_rate)
    rows = []
    for task_id in task_ids:
        for idx in range(max(1, args.prompt_variants)):
            template = PROMPT_TEMPLATES[idx % len(PROMPT_TEMPLATES)]
            rows.append(inspect_task(task_id, prompt_mode=args.prompt_mode, template=template))
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

    baseline = evaluate_model(model, tokenizer, eval_rows, args.out_dir / "baseline_generations.json", args.max_completion_length, args.reward_mode)

    def redshift_reward(completions, task_id, root_service, root_category, **kwargs):
        required = kwargs.get("required", [[] for _ in completions])
        return [
            rollout_reward(tid, completion, service, category, req, args.reward_mode)
            for completion, tid, service, category, req in zip(completions, task_id, root_service, root_category, required)
        ]

    # In SRE mode we expose the 3 disjoint sub-rewards as separate reward funcs
    # so TRL/W&B logs each one as its own column. The total reward used for the
    # GRPO policy gradient is the sum across the list, which matches the
    # original sre_shaped_reward() total exactly (modulo the [-0.6, +1.0] clip
    # which is applied inside _cached_breakdown via sre_shaped_reward).
    def sre_format_reward(completions, task_id, **kwargs):
        return [sre_format_score(extract_completion_text(c), tid) for c, tid in zip(completions, task_id)]

    def sre_investigation_reward(completions, task_id, **kwargs):
        return [sre_investigation_score(extract_completion_text(c), tid) for c, tid in zip(completions, task_id)]

    def sre_remediation_reward(completions, task_id, **kwargs):
        return [sre_remediation_score(extract_completion_text(c), tid) for c, tid in zip(completions, task_id)]

    if args.reward_mode == "sre":
        reward_funcs: Any = [sre_format_reward, sre_investigation_reward, sre_remediation_reward]
    else:
        reward_funcs = redshift_reward

    train_dataset = Dataset.from_list(rows)
    training_args = make_grpo_config(args)
    trainer_kwargs = {
        "model": model,
        "reward_funcs": reward_funcs,
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
    trained = evaluate_model(model, tokenizer, eval_rows, args.out_dir / "trained_generations.json", args.max_completion_length, args.reward_mode)

    summary = {
        "model_name": args.model_name,
        "duration_sec": time.time() - start,
        "max_steps": args.max_steps,
        "num_train_tasks": len(rows),
        "num_eval_tasks": len(eval_rows),
        "curriculum_buffer": str(args.curriculum_buffer),
        "reward_mode": args.reward_mode,
        "prompt_mode": args.prompt_mode,
        "prompt_variants": args.prompt_variants,
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
