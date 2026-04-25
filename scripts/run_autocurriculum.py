"""Generate evolved Red Shift scenarios and curriculum plots."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from oncallenv.curriculum import AutocurriculumRunner
from oncallenv.curriculum.autocurriculum import LLMDefenderEvaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--seed-dir", type=Path, default=Path("scenarios_seed"))
    parser.add_argument("--out-dir", type=Path, default=Path("curriculum_results"))
    parser.add_argument("--write-yaml", action="store_true")
    parser.add_argument("--difficulty-source", choices=["llm_inference"], default="llm_inference")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--rollouts-per-candidate", type=int, default=3)
    parser.add_argument("--defender-max-steps", type=int, default=18)
    parser.add_argument("--defender-temperature", type=float, default=0.2)
    args = parser.parse_args()

    evaluator = LLMDefenderEvaluator(
        model_name=args.model_name,
        rollout_count=args.rollouts_per_candidate,
        max_steps=args.defender_max_steps,
        temperature=args.defender_temperature,
    )
    runner = AutocurriculumRunner.from_seed_dir(args.seed_dir, seed=args.seed, evaluator=evaluator)
    buffer = runner.evolve(args.iterations)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    buffer.save(args.out_dir / "buffer.json")
    if args.write_yaml:
        runner.write_yaml_scenarios(buffer, args.out_dir / "scenarios")

    counts = Counter(item.spec.fault_primary for item in buffer.scenarios)
    solve_rates = [item.solve_rate for item in buffer.scenarios]
    regrets = [item.regret for item in buffer.scenarios]
    summary = {
        "requested_iterations": args.iterations,
        "buffer_size": len(buffer),
        "evolved_count": sum(1 for item in buffer.scenarios if item.spec.task_id.startswith("evolved_")),
        "difficulty_source": args.difficulty_source,
        "defender_model": args.model_name if args.difficulty_source == "llm_inference" else None,
        "rollouts_per_candidate": args.rollouts_per_candidate if args.difficulty_source == "llm_inference" else 1,
        "fault_counts": dict(sorted(counts.items())),
        "solve_rate_min": min(solve_rates),
        "solve_rate_max": max(solve_rates),
        "solve_rate_mean": sum(solve_rates) / len(solve_rates),
        "regret_mean": sum(regrets) / len(regrets),
    }
    if runner.latest_rollout_stats:
        summary["defender_rollout_stats"] = runner.latest_rollout_stats
        summary["defender_reward_mean"] = mean(item["reward_mean"] for item in runner.latest_rollout_stats)
        summary["defender_reward_std_mean"] = mean(item["reward_std"] for item in runner.latest_rollout_stats)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plots = Path("docs/plots")
    plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    labels = list(summary["fault_counts"])
    values = [summary["fault_counts"][label] for label in labels]
    ax.bar(labels, values)
    ax.set_xlabel("Fault family")
    ax.set_ylabel("Scenarios in regret buffer")
    ax.set_title("ACCEL Autocurriculum Diversity")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plots / "autocurriculum_diversity.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(solve_rates, bins=12, range=(0, 1), alpha=0.8)
    ax.set_xlabel("Estimated solve rate")
    ax.set_ylabel("Scenario count")
    ax.set_title("Autocurriculum Non-Triviality Filter")
    fig.tight_layout()
    fig.savefig(plots / "autocurriculum_solve_rate_hist.png", dpi=160)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

