"""SPICE-style self-play training for OnCallEnv Red Shift.

A single Unsloth LoRA model alternates between two roles:
  • Attacker  – mutates ScenarioSpec via discrete set_field actions
  • Defender  – diagnoses and remediates the resulting incident

Both roles are updated jointly with DrGRPO (no std-normalisation in
advantage computation).  The Attacker is rewarded by a Gaussian peaked
at high variance of normalised Defender rewards; the Defender receives
the normalised continuous reward from the simulator rubrics.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from oncallenv.core.types import ScenarioSpec
from oncallenv.simulation.scenario_compiler import compile_scenario
from oncallenv.curriculum import RegretBuffer
from oncallenv.curriculum.buffer import BufferedScenario

from llm_attacker import (
    ATTACKER_VALID_FIELDS,
    attacker_reward,
    build_attacker_prompt,
    normalize_defender_reward,
    parse_attacker_actions,
)
from spice_defender import (
    build_defender_prompt,
    defender_rollout_reward,
    parse_commands,
)


# ---------------------------------------------------------------------------
# Seed tasks (same as train_unsloth_grpo)
# ---------------------------------------------------------------------------

SEED_TASKS = [
    "seed_easy_memory_leak",
    "seed_dns_misconfiguration",
    "seed_cert_expiry",
    "seed_cache_stampede",
    "seed_replica_lag",
    "seed_http_503_loop",
]


# ---------------------------------------------------------------------------
# Helper: generate text from model
# ---------------------------------------------------------------------------


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    num_return: int = 1,
) -> list[str]:
    """Generate one or more completions from the LoRA model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs: list[str] = []
    for _ in range(num_return):
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-4),
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(
            ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
        )
        outputs.append(text)
    return outputs


# ---------------------------------------------------------------------------
# Self-play iteration
# ---------------------------------------------------------------------------


def selfplay_iteration(
    model,
    tokenizer,
    parent_specs: list[ScenarioSpec],
    group_size: int,
    generation: int,
    temperature: float,
    max_attacker_tokens: int,
    max_defender_tokens: int,
    challenger_penalty: float,
) -> dict[str, Any]:
    """Run one SPICE self-play iteration.

    Returns a dict with attacker/defender rewards and generated data.
    """
    attacker_rows: list[dict[str, Any]] = []
    defender_rows: list[dict[str, Any]] = []
    valid_specs: list[ScenarioSpec] = []

    # === ATTACKER PHASE ===
    for parent in parent_specs:
        prompt = build_attacker_prompt(parent)
        completions = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=max_attacker_tokens,
            temperature=temperature,
            num_return=group_size,
        )

        for i, completion in enumerate(completions):
            spec, is_valid, actions = parse_attacker_actions(
                completion, parent, generation=generation,
            )

            if not is_valid or spec is None:
                attacker_rows.append({
                    "parent": parent.task_id,
                    "completion": completion,
                    "actions": actions,
                    "valid": False,
                    "reward": challenger_penalty,
                })
                continue

            # Validate that the scenario compiles
            try:
                compile_scenario(spec)
            except Exception:
                attacker_rows.append({
                    "parent": parent.task_id,
                    "completion": completion,
                    "actions": actions,
                    "valid": False,
                    "reward": challenger_penalty,
                })
                continue

            # Run G defender rollouts to compute attacker reward
            defender_prompt = build_defender_prompt(spec)
            defender_completions = generate_text(
                model, tokenizer, defender_prompt,
                max_new_tokens=max_defender_tokens,
                temperature=temperature,
                num_return=group_size,
            )

            defender_rewards = []
            for d_comp in defender_completions:
                try:
                    r = defender_rollout_reward(spec, d_comp)
                except Exception:
                    r = -0.25
                defender_rewards.append(r)

            a_reward = attacker_reward(defender_rewards, penalty=challenger_penalty)
            attacker_rows.append({
                "parent": parent.task_id,
                "child_task_id": spec.task_id,
                "completion": completion,
                "actions": actions,
                "valid": True,
                "defender_rewards": defender_rewards,
                "reward": a_reward,
            })
            valid_specs.append(spec)

    # === DEFENDER PHASE ===
    # Pick valid attacker-generated scenarios for Defender training
    if valid_specs:
        rng = random.Random(generation)
        selected = rng.sample(valid_specs, min(len(valid_specs), len(parent_specs)))
    else:
        # Fallback to parent specs if attacker produced nothing valid
        selected = parent_specs

    for spec in selected:
        defender_prompt = build_defender_prompt(spec)
        defender_completions = generate_text(
            model, tokenizer, defender_prompt,
            max_new_tokens=max_defender_tokens,
            temperature=temperature,
            num_return=group_size,
        )
        for d_comp in defender_completions:
            try:
                r = defender_rollout_reward(spec, d_comp)
            except Exception:
                r = -0.25
            norm_r = normalize_defender_reward(r)
            defender_rows.append({
                "task_id": spec.task_id,
                "completion": d_comp,
                "commands": parse_commands(d_comp),
                "raw_reward": r,
                "reward": norm_r,
            })

    return {
        "attacker_rows": attacker_rows,
        "defender_rows": defender_rows,
        "valid_specs": [s.model_dump() for s in valid_specs],
    }


# ---------------------------------------------------------------------------
# DrGRPO reward functions for TRL GRPOTrainer
# ---------------------------------------------------------------------------


def make_selfplay_reward_fn(
    parent_specs: list[ScenarioSpec],
    group_size: int,
    challenger_penalty: float,
    max_defender_tokens: int,
    temperature: float,
):
    """Build a reward function that handles both Attacker and Defender prompts.

    TRL GRPOTrainer calls:  reward_func(completions, prompt=..., **row_fields)
    """

    def reward_fn(completions, prompt, role, **kwargs):
        rewards = []
        for completion in completions:
            text = completion if isinstance(completion, str) else str(completion)
            role_val = role[0] if isinstance(role, list) else role

            if role_val == "attacker":
                parent_id = kwargs.get("parent_task_id", [""])[0] if isinstance(kwargs.get("parent_task_id"), list) else kwargs.get("parent_task_id", "")
                parent = next((s for s in parent_specs if s.task_id == parent_id), parent_specs[0])
                spec, is_valid, _ = parse_attacker_actions(text, parent)
                if not is_valid or spec is None:
                    rewards.append(challenger_penalty)
                    continue
                try:
                    compile_scenario(spec)
                except Exception:
                    rewards.append(challenger_penalty)
                    continue
                # We can't easily run defender rollouts inside the reward fn
                # in the TRL loop, so we use a heuristic proxy:
                # check if the scenario has enough complexity features
                complexity = 0.0
                if spec.fault_secondary:
                    complexity += 0.3
                if spec.red_herring and spec.red_herring != "none":
                    complexity += 0.2
                if spec.schema_drift and spec.schema_drift != "none":
                    complexity += 0.2
                if spec.metric_noise > 0.3:
                    complexity += 0.15
                if spec.blast_radius > 0.5:
                    complexity += 0.15
                # Reward valid, moderately complex scenarios
                rewards.append(min(1.0, 0.3 + complexity))
            else:
                # Defender role
                task_id = kwargs.get("task_id", [""])[0] if isinstance(kwargs.get("task_id"), list) else kwargs.get("task_id", "")
                root_service = kwargs.get("root_service", [""])[0] if isinstance(kwargs.get("root_service"), list) else kwargs.get("root_service", "")
                root_category = kwargs.get("root_category", [""])[0] if isinstance(kwargs.get("root_category"), list) else kwargs.get("root_category", "")
                try:
                    r = defender_rollout_reward(
                        ScenarioSpec.model_validate(kwargs.get("spec", {})),
                        text,
                    )
                except Exception:
                    r = -0.25
                rewards.append(normalize_defender_reward(r))

        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Load parent scenarios
# ---------------------------------------------------------------------------


def load_parent_specs(
    seed_dir: Path,
    curriculum_buffer: Path | None,
    max_tasks: int | None,
    seed: int,
) -> list[ScenarioSpec]:
    """Load seed + curriculum scenarios as parent specs for the Attacker."""
    specs: list[ScenarioSpec] = []

    # Load seed YAMLs
    if seed_dir.exists():
        for path in sorted(seed_dir.glob("*.y*ml")):
            try:
                specs.append(ScenarioSpec.model_validate(
                    yaml.safe_load(path.read_text(encoding="utf-8")),
                ))
            except Exception:
                continue

    # Load from curriculum buffer
    if curriculum_buffer and curriculum_buffer.exists():
        try:
            buffer = RegretBuffer.load(curriculum_buffer)
            for item in buffer.scenarios:
                specs.append(item.spec)
        except Exception:
            pass

    # Deduplicate
    unique: dict[str, ScenarioSpec] = {s.task_id: s for s in specs}
    specs = list(unique.values())

    rng = random.Random(seed)
    rng.shuffle(specs)
    if max_tasks:
        specs = specs[:max_tasks]
    return specs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    tokenizer,
    eval_specs: list[ScenarioSpec],
    out_path: Path,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Evaluate the Defender on a set of scenarios."""
    scores: dict[str, float] = {}
    generations: dict[str, dict[str, Any]] = {}
    model.eval()

    for spec in eval_specs:
        prompt = build_defender_prompt(spec)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        try:
            reward = defender_rollout_reward(spec, completion)
        except Exception:
            reward = -0.25
        norm_reward = normalize_defender_reward(reward)
        scores[spec.task_id] = norm_reward
        generations[spec.task_id] = {
            "completion": completion,
            "commands": parse_commands(completion),
            "raw_reward": reward,
            "normalized_reward": norm_reward,
        }

    summary = {
        "mean_reward": sum(scores.values()) / len(scores) if scores else 0.0,
        "scores": scores,
        "generations": generations,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Model setup helpers
# ---------------------------------------------------------------------------


def trainable_parameter_report(model) -> dict[str, int]:
    total = trainable = 0
    lora_names: list[str] = []
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
        if "lora" in name.lower():
            lora_names.append(name)
    return {"total": total, "trainable": trainable, "lora_tensors": len(lora_names)}


def ensure_lora_is_trainable(model) -> dict[str, int]:
    report = trainable_parameter_report(model)
    if report["trainable"] > 0:
        return report
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad_(True)
    report = trainable_parameter_report(model)
    if report["trainable"] == 0:
        lora_like = [n for n, _ in model.named_parameters() if "lora" in n.lower()]
        raise RuntimeError(
            "No trainable LoRA parameters found. "
            f"LoRA-like tensors: {lora_like[:12]}"
        )
    return report


# ---------------------------------------------------------------------------
# GRPOConfig builder
# ---------------------------------------------------------------------------


def make_grpo_config(args):
    from trl import GRPOConfig

    kwargs = {
        "output_dir": str(args.out_dir),
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.group_size,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "report_to": [args.report_to] if args.report_to.lower() != "none" else [],
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
        "logging_first_step": True,
        "logging_strategy": "steps",
        "remove_unused_columns": False,
        "fp16": True,
        "bf16": False,
        "seed": args.seed,
    }
    # DrGRPO-specific: try setting beta=0 and loss_type=dr_grpo
    for optional_key, value in {
        "beta": 0.0,  # No KL regularisation (DrGRPO)
        "scale_rewards": args.scale_rewards,
        "loss_type": "dr_grpo",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SPICE-style self-play training: LLM Attacker + Defender with DrGRPO",
    )
    # Model
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--out-dir", type=Path, default=Path("training_results/spice_selfplay"))
    parser.add_argument("--seed-dir", type=Path, default=Path("scenarios_seed"))
    parser.add_argument("--curriculum-buffer", type=Path, default=Path("curriculum_results/buffer.json"))

    # Self-play
    parser.add_argument("--selfplay-iterations", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=4, help="G: num_generations for GRPOTrainer (GRPO weight update)")
    parser.add_argument("--selfplay-group-size", type=int, default=None,
                        help="Number of defender rollouts per attacker-generated scenario for variance reward. "
                             "Defaults to --group-size if not set.")
    parser.add_argument("--batch-size", type=int, default=8, help="Parent scenarios per iteration")
    parser.add_argument("--challenger-penalty", type=float, default=-0.1)

    # Training
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--max-attacker-tokens", type=int, default=200)
    parser.add_argument("--max-defender-tokens", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--scale-rewards", default="batch")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--report-to", type=str, default="tensorboard", help="huggingface, wandb, tensorboard, or none")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--eval-tasks", type=int, default=12)
    parser.add_argument("--max-tasks", type=int, default=120)

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260424)

    args = parser.parse_args()
    # Default selfplay_group_size to group_size if not explicitly set
    if args.selfplay_group_size is None:
        args.selfplay_group_size = args.group_size

    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # ------------------------------------------------------------------
    # 1. Load parent scenarios
    # ------------------------------------------------------------------
    parent_specs = load_parent_specs(
        args.seed_dir, args.curriculum_buffer, args.max_tasks, args.seed,
    )
    print(f"Loaded {len(parent_specs)} parent scenarios")
    if not parent_specs:
        raise RuntimeError("No parent scenarios found. Check --seed-dir or --curriculum-buffer.")

    rng = random.Random(args.seed)
    rng.shuffle(parent_specs)
    eval_specs = parent_specs[:min(args.eval_tasks, len(parent_specs))]

    # ------------------------------------------------------------------
    # 2. Load model with Unsloth
    # ------------------------------------------------------------------
    from datasets import Dataset
    from unsloth import FastLanguageModel, PatchFastRL

    try:
        PatchFastRL("GRPO", FastLanguageModel)
    except Exception as exc:
        print(f"WARN: PatchFastRL('GRPO') failed or already applied: {exc}")
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
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
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

    # ------------------------------------------------------------------
    # 3. Baseline evaluation (Defender only)
    # ------------------------------------------------------------------
    baseline = evaluate_model(
        model, tokenizer, eval_specs,
        args.out_dir / "baseline_generations.json",
        args.max_defender_tokens,
    )
    print(f"Baseline mean reward: {baseline['mean_reward']:.4f}")

    # ------------------------------------------------------------------
    # 4. Self-play loop
    # ------------------------------------------------------------------
    print("\n=== Starting SPICE Self-Play ===")
    all_attacker_data: list[dict] = []
    all_defender_data: list[dict] = []
    iteration_summaries: list[dict] = []

    for iteration in range(args.selfplay_iterations):
        # Sample a batch of parent scenarios
        batch = rng.sample(
            parent_specs,
            min(args.batch_size, len(parent_specs)),
        )

        result = selfplay_iteration(
            model=model,
            tokenizer=tokenizer,
            parent_specs=batch,
            group_size=args.selfplay_group_size,
            generation=iteration,
            temperature=args.temperature,
            max_attacker_tokens=args.max_attacker_tokens,
            max_defender_tokens=args.max_defender_tokens,
            challenger_penalty=args.challenger_penalty,
        )

        a_rows = result["attacker_rows"]
        d_rows = result["defender_rows"]
        all_attacker_data.extend(a_rows)
        all_defender_data.extend(d_rows)

        # Compute DrGRPO advantages (no std normalisation)
        a_rewards = [r["reward"] for r in a_rows]
        d_rewards = [r["reward"] for r in d_rows]
        a_mean = sum(a_rewards) / len(a_rewards) if a_rewards else 0.0
        d_mean = sum(d_rewards) / len(d_rewards) if d_rewards else 0.0

        a_advantages = [r - a_mean for r in a_rewards]
        d_advantages = [r - d_mean for r in d_rewards]

        valid_count = sum(1 for r in a_rows if r["valid"])
        iter_summary = {
            "iteration": iteration,
            "attacker_mean_reward": a_mean,
            "defender_mean_reward": d_mean,
            "attacker_valid_count": valid_count,
            "attacker_total": len(a_rows),
            "num_new_specs": len(result["valid_specs"]),
        }
        iteration_summaries.append(iter_summary)

        if iteration % 10 == 0 or iteration == args.selfplay_iterations - 1:
            print(
                f"[Iter {iteration:>4d}] "
                f"Attacker r={a_mean:.3f} ({valid_count}/{len(a_rows)} valid) | "
                f"Defender r={d_mean:.3f}"
            )

    # Save self-play generation data
    (args.out_dir / "attacker_generations.json").write_text(
        json.dumps(all_attacker_data[:500], indent=2, default=str), encoding="utf-8",
    )
    (args.out_dir / "defender_generations.json").write_text(
        json.dumps(all_defender_data[:500], indent=2, default=str), encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # 5. Build combined dataset for GRPOTrainer
    # ------------------------------------------------------------------
    print("\n=== Building combined dataset for DrGRPO training ===")
    train_rows: list[dict[str, Any]] = []

    # Attacker rows → prompts for attacker role
    for row in all_attacker_data:
        parent = next((s for s in parent_specs if s.task_id == row["parent"]), None)
        if parent is None:
            continue
        train_rows.append({
            "prompt": build_attacker_prompt(parent),
            "role": "attacker",
            "parent_task_id": row["parent"],
            "task_id": row.get("child_task_id", ""),
            "root_service": "",
            "root_category": "",
        })

    # Defender rows → prompts for defender role
    for row in all_defender_data:
        spec = next((s for s in parent_specs if s.task_id == row["task_id"]), None)
        if spec is None:
            continue
        graph = compile_scenario(spec)
        train_rows.append({
            "prompt": build_defender_prompt(spec),
            "role": "defender",
            "parent_task_id": "",
            "task_id": spec.task_id,
            "root_service": graph.root_cause_service,
            "root_category": graph.root_cause_category,
        })

    rng.shuffle(train_rows)
    print(f"Combined training dataset: {len(train_rows)} rows")

    # ------------------------------------------------------------------
    # 6. GRPOTrainer (DrGRPO)
    # ------------------------------------------------------------------

    def selfplay_reward(completions, **kwargs):
        """Unified reward function dispatching to attacker or defender."""
        # TRL passes prompts and custom columns as kwargs
        role = kwargs.get("role")
        # Handle case where role might be a list (one per completion) or a single value
        # TRL usually passes the batch-item value for custom columns
        rewards = []
        for idx, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else str(completion)
            r_val = role[idx] if isinstance(role, list) else role

            if r_val == "attacker":
                pid_list = kwargs.get("parent_task_id", [""])
                pid = pid_list[idx] if isinstance(pid_list, list) else pid_list
                parent = next((s for s in parent_specs if s.task_id == pid), parent_specs[0])
                spec, is_valid, _ = parse_attacker_actions(text, parent)
                if not is_valid or spec is None:
                    rewards.append(args.challenger_penalty)
                    continue
                try:
                    compile_scenario(spec)
                except Exception:
                    rewards.append(args.challenger_penalty)
                    continue
                # Heuristic complexity proxy
                c = 0.0
                if spec.fault_secondary:
                    c += 0.3
                if spec.red_herring and spec.red_herring != "none":
                    c += 0.2
                if spec.schema_drift and spec.schema_drift != "none":
                    c += 0.2
                if spec.metric_noise > 0.3:
                    c += 0.15
                if spec.blast_radius > 0.5:
                    c += 0.15
                rewards.append(min(1.0, 0.3 + c))
            else:
                tid_list = kwargs.get("task_id", [""])
                tid = tid_list[idx] if isinstance(tid_list, list) else tid_list
                spec = next((s for s in parent_specs if s.task_id == tid), parent_specs[0])
                try:
                    r = defender_rollout_reward(spec, text)
                except Exception:
                    r = -0.25
                rewards.append(normalize_defender_reward(r))

        return rewards

    train_dataset = Dataset.from_list(train_rows)
    training_args = make_grpo_config(args)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "reward_funcs": selfplay_reward,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    trainer_params = inspect.signature(GRPOTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()

    # ------------------------------------------------------------------
    # 7. Save adapter & final evaluation
    # ------------------------------------------------------------------
    adapter_dir = args.out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    trained = evaluate_model(
        model, tokenizer, eval_specs,
        args.out_dir / "trained_generations.json",
        args.max_defender_tokens,
    )

    summary = {
        "model_name": args.model_name,
        "duration_sec": time.time() - start,
        "selfplay_iterations": args.selfplay_iterations,
        "group_size": args.group_size,
        "selfplay_group_size": args.selfplay_group_size,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "num_parent_specs": len(parent_specs),
        "num_train_rows": len(train_rows),
        "num_eval_tasks": len(eval_specs),
        "baseline_mean_reward": baseline["mean_reward"],
        "trained_mean_reward": trained["mean_reward"],
        "adapter_dir": str(adapter_dir),
        "trainable_parameter_report": trainable_report,
        "iteration_summaries": iteration_summaries,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print("\n=== SPICE Self-Play Training Complete ===")
    print(json.dumps({
        "baseline_mean_reward": baseline["mean_reward"],
        "trained_mean_reward": trained["mean_reward"],
        "improvement": trained["mean_reward"] - baseline["mean_reward"],
        "duration_sec": summary["duration_sec"],
    }, indent=2))


if __name__ == "__main__":
    main()
