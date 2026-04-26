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
from transformers import TrainerCallback

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
    prompt,
    max_new_tokens: int,
    temperature: float,
    num_return: int = 1,
) -> list[str]:
    """Generate one or more completions from the LoRA model IN PARALLEL."""
    if isinstance(prompt, str):
        prompt = [prompt]
        
    # Inject Chat Template formatting (adds <|im_start|>user ... tags)
    chat_prompts = []
    for p in prompt:
        chat = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        chat_prompts.append(formatted)
        
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True).to(model.device)
    tokenizer.padding_side = old_padding_side
    
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-4),
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_return,
        )
        
    outputs: list[str] = []
    for i, sequence_ids in enumerate(ids):
        prompt_idx = i // num_return
        input_len = inputs["input_ids"][prompt_idx].shape[-1]
        text = tokenizer.decode(
            sequence_ids[input_len:], skip_special_tokens=True,
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
    attacker_prompts = [build_attacker_prompt(p) for p in parent_specs]
    
    all_attacker_completions = generate_text(
        model, tokenizer, attacker_prompts,
        max_new_tokens=max_attacker_tokens,
        temperature=temperature,
        num_return=group_size,
    )
    
    defender_rollout_requests = []
    
    for i, parent in enumerate(parent_specs):
        completions = all_attacker_completions[i * group_size : (i + 1) * group_size]
        
        for completion in completions:
            spec, is_valid, actions = parse_attacker_actions(
                completion, parent, generation=generation,
            )

            if not is_valid or spec is None:
                reward_val = challenger_penalty * 0.5 if "set_field" in completion.lower() or "<actions>" in completion.lower() else challenger_penalty
                attacker_rows.append({
                    "parent": parent.task_id,
                    "completion": completion,
                    "actions": actions,
                    "valid": False,
                    "reward": reward_val,
                })
                continue

            try:
                compile_scenario(spec)
            except Exception:
                reward_val = challenger_penalty * 0.5 if "set_field" in completion.lower() or "<actions>" in completion.lower() else challenger_penalty
                attacker_rows.append({
                    "parent": parent.task_id,
                    "completion": completion,
                    "actions": actions,
                    "valid": False,
                    "reward": reward_val,
                })
                continue
                
            defender_rollout_requests.append((parent, spec, completion, actions))
            valid_specs.append(spec)
            
    if defender_rollout_requests:
        defender_prompts = [build_defender_prompt(req[1]) for req in defender_rollout_requests]
        all_defender_completions = generate_text(
            model, tokenizer, defender_prompts,
            max_new_tokens=max_defender_tokens,
            temperature=temperature,
            num_return=group_size,
        )
        
        for i, req in enumerate(defender_rollout_requests):
            parent, spec, completion, actions = req
            d_comps = all_defender_completions[i * group_size : (i + 1) * group_size]
            
            defender_rewards = []
            for d_comp in d_comps:
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

    # === DEFENDER PHASE ===
    if valid_specs:
        rng = random.Random(generation)
        selected = rng.sample(valid_specs, min(len(valid_specs), len(parent_specs)))
    else:
        selected = parent_specs

    if selected:
        final_defender_prompts = [build_defender_prompt(spec) for spec in selected]
        all_final_defender_completions = generate_text(
            model, tokenizer, final_defender_prompts,
            max_new_tokens=max_defender_tokens,
            temperature=temperature,
            num_return=group_size,
        )
        
        for i, spec in enumerate(selected):
            d_comps = all_final_defender_completions[i * group_size : (i + 1) * group_size]
            for d_comp in d_comps:
                try:
                    r = defender_rollout_reward(spec, d_comp)
                except Exception:
                    r = -0.25
                norm_r = normalize_defender_reward(r)
                defender_rows.append({
                    "task_id": spec.task_id,
                    "spec": spec.model_dump(),
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

    prompts = [build_defender_prompt(spec) for spec in eval_specs]
    
    completions = generate_text(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens,
        temperature=0.0, # Forces do_sample=False inside generate_text
        num_return=1,
    )

    for spec, completion in zip(eval_specs, completions):
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
    parser.add_argument("--group-size", type=int, default=6, help="G: num_generations for GRPOTrainer (GRPO weight update)")
    parser.add_argument("--selfplay-group-size", type=int, default=None,
                        help="Number of defender rollouts per attacker-generated scenario for variance reward. "
                             "Defaults to --group-size if not set.")
    parser.add_argument("--batch-size", type=int, default=8, help="Parent scenarios per iteration")
    parser.add_argument("--challenger-penalty", type=float, default=-0.1)

    # Training
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--per-device-train_batch_size", type=int, default=4)
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
    parser.add_argument("--verbose", action="store_true", help="Print sample completions during training")

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
    from tqdm.auto import tqdm
    all_attacker_data: list[dict] = []
    all_defender_data: list[dict] = []
    iteration_summaries: list[dict] = []

    for iteration in tqdm(range(args.selfplay_iterations), desc="SPICE Generation Phase"):
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
        threshold = 0.45
        pass_rate = sum(1 for r in d_rewards if r >= threshold) / len(d_rewards) if d_rewards else 0.0
        
        iter_summary = {
            "iteration": iteration,
            "attacker_mean_reward": a_mean,
            "defender_mean_reward": d_mean,
            "defender_pass_rate": pass_rate,
            "attacker_valid_count": valid_count,
            "attacker_total": len(a_rows),
            "num_new_specs": len(result["valid_specs"]),
        }
        iteration_summaries.append(iter_summary)

        print(
            f"[Iter {iteration:>4d}] "
            f"Attacker r={a_mean:.3f} ({valid_count}/{len(a_rows)} valid) | "
            f"Defender r={d_mean:.3f} (Pass Rate: {pass_rate:.1%})"
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
        spec = next((s for s in parent_specs if s.task_id == row["parent"]), parent_specs[0])
        train_rows.append({
            "prompt": [{"role": "user", "content": build_attacker_prompt(spec)}],
            "role": "attacker",
            "parent_task_id": row["parent"],
            "task_id": row.get("child_task_id", ""),
            "spec": None,  # Ensure consistent schema
            "root_service": "",
            "root_category": "",
        })

    # Defender rows → prompts for defender role
    for row in all_defender_data:
        spec_dict = row.get("spec")
        if spec_dict:
            spec = ScenarioSpec.model_validate(spec_dict)
        else:
            spec = next((s for s in parent_specs if s.task_id == row["task_id"]), None)
            
        if spec is None:
            continue
        graph = compile_scenario(spec)
        train_rows.append({
            "prompt": [{"role": "user", "content": build_defender_prompt(spec)}],
            "role": "defender",
            "parent_task_id": "",
            "task_id": spec.task_id,
            "spec": spec.model_dump(),
            "root_service": graph.root_cause_service,
            "root_category": graph.root_cause_category,
        })

    attacker_rows = [r for r in train_rows if r['role'] == 'attacker']
    defender_rows = [r for r in train_rows if r['role'] == 'defender']
    train_rows = attacker_rows + defender_rows
    print(f"Combined training dataset: {len(train_rows)} rows (Attacker first, Defender second)")

    # SAVING SELF PLAY GENERATIONS
    dataset_path = args.out_dir / "selfplay_dataset.jsonl"
    dataset_path.write_text("\n".join(json.dumps(row) for row in train_rows) + "\n", encoding="utf-8")
    print(f"Saved generated dataset to {dataset_path}")

    # ------------------------------------------------------------------
    # 6. GRPOTrainer (DrGRPO)
    # ------------------------------------------------------------------

    # --- Shared helper to resolve per-item values from kwargs ---
    def _get(kwargs, key, idx):
        v = kwargs[key]
        return v[idx] if isinstance(v, list) else v

    def attacker_reward(completions, **kwargs):
        """Attacker reward: heuristic complexity of generated scenario.
        Returns challenger_penalty for defender-role rows (masked out).
        """
        from spice_defender import extract_completion_text
        from llm_attacker import parse_attacker_actions
        rewards = []
        comps_list = list(completions)
        r_sample = _get(kwargs, "role", 0)
        if getattr(args, "verbose", False) and comps_list and r_sample == "attacker":
            text = extract_completion_text(comps_list[0])
            _, _, actions = parse_attacker_actions(text, parent_specs[0])
            print(f"\n[VERBOSE] Attacker Sample:\n{comps_list[0]}")
            print(f"[APPLIED ACTIONS]: {actions}\n{'-'*40}\n")
            
        for idx, completion in enumerate(comps_list):
            r_val = _get(kwargs, "role", idx)
            if r_val != "attacker":
                rewards.append(0.0)  # neutral mask for defender rows
                continue
            text = extract_completion_text(completion)
            pid = _get(kwargs, "parent_task_id", idx)
            parent = next(s for s in parent_specs if s.task_id == pid)
            spec, is_valid, _ = parse_attacker_actions(text, parent)
            if not is_valid or spec is None:
                r_val = args.challenger_penalty * 0.5 if "set_field" in text.lower() or "<actions>" in text.lower() else args.challenger_penalty
                rewards.append(r_val)
                continue
            try:
                compile_scenario(spec)
            except Exception:
                r_val = args.challenger_penalty * 0.5 if "set_field" in text.lower() or "<actions>" in text.lower() else args.challenger_penalty
                rewards.append(r_val)
                continue
            # Heuristic complexity proxy - aligned with Gaussian on p intent
            # We reward "medium-high" complexity scenarios as they are more likely to have p=0.5
            c = 0.0
            if spec.fault_secondary:
                c += 0.25
            if spec.red_herring and spec.red_herring != "none":
                c += 0.2
            if spec.schema_drift and spec.schema_drift != "none":
                c += 0.2
            if 0.4 <= spec.metric_noise <= 0.8:
                c += 0.15
            if 0.4 <= spec.blast_radius <= 0.8:
                c += 0.2
            rewards.append(min(1.0, 0.2 + c))
        return rewards

    def defender_reward(completions, **kwargs):
        """Defender reward: normalised simulator rubric score.
        Returns 0.0 for attacker-role rows (masked out).
        """
        from spice_defender import extract_completion_text, parse_commands
        rewards = []
        comps_list = list(completions)
        
        r_sample = _get(kwargs, "role", 0)
        if getattr(args, "verbose", False) and comps_list and r_sample == "defender":
            text = extract_completion_text(comps_list[0])
            cmds = parse_commands(text)
            print(f"\n[VERBOSE] Defender Sample:\n{comps_list[0]}")
            print(f"[PARSED COMMANDS]: {cmds}\n{'-'*40}\n")
            
        for idx, completion in enumerate(comps_list):
            r_val = _get(kwargs, "role", idx)
            if r_val != "defender":
                rewards.append(0.0)  # neutral mask for attacker rows
                continue
            text = extract_completion_text(completion)
            spec_dict = _get(kwargs, "spec", idx)
            spec = ScenarioSpec.model_validate(spec_dict)
            try:
                r = defender_rollout_reward(spec, text)
            except Exception as e:
                if getattr(args, "verbose", False):
                    print(f"[SIMULATION ERROR]: {e}")
                r = -0.25
            rewards.append(normalize_defender_reward(r))
        return rewards

    # --- Callback: prints the same log dict as train_unsloth_grpo.py ---

    class SpiceLogCallback(TrainerCallback):
        def on_log(self, train_args, state, control, logs=None, **cb_kwargs):
            if logs:
                print(logs)

    train_dataset = Dataset.from_list(train_rows)
    training_args = make_grpo_config(args)
    training_args.dataloader_drop_last = False
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "reward_funcs": [attacker_reward, defender_reward],
        "args": training_args,
        "train_dataset": train_dataset,
        "generation_kwargs": {
            "top_p": 0.95,
            "do_sample": True,
            "temperature": args.temperature,
        },
    }
    trainer_params = inspect.signature(GRPOTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)
    # Add callback after construction so Unsloth's patched trainer
    # doesn't swallow it during its own __init__ compilation phase.
    trainer.add_callback(SpiceLogCallback())
    # Ensure TRL's built-in step logs reach stdout even in notebook cells.
    import transformers as _tf
    _tf.logging.set_verbosity_info()
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
