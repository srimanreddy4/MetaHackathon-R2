"""Autocurriculum loop for generating non-trivial incident scenarios."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Protocol

import yaml
from openai import OpenAI

from oncallenv import OnCallRedShiftEnv
from oncallenv.core.env import DEFAULT_SCENARIO
from oncallenv.core.types import Action, ScenarioSpec
from oncallenv.curriculum.buffer import BufferedScenario, RegretBuffer
from oncallenv.curriculum.mutator import ScenarioMutator


class DifficultyEvaluator(Protocol):
    def __call__(self, spec: ScenarioSpec) -> tuple[float, dict]:
        """Return solve_rate and evaluation metadata for a scenario."""


class RandomPolicyEvaluator:
    """Estimate difficulty by running a random action policy against the env.

    Scenarios where random actions consistently succeed are too easy (solve_rate
    near 1.0); scenarios where nothing works are too hard (solve_rate near 0.0).
    The autocurriculum keeps tasks in the 0.05–0.95 band, which naturally
    selects non-trivial but learnable scenarios without any LLM API calls.
    """

    def __init__(self, rollout_count: int = 3, max_steps: int = 12, seed: int = 0):
        self.rollout_count = rollout_count
        self.max_steps = max_steps
        self._rng = random.Random(seed)

    def __call__(self, spec: ScenarioSpec) -> tuple[float, dict]:
        successes: list[int] = []
        rewards: list[float] = []
        for _ in range(self.rollout_count):
            reward = self._run_rollout(spec)
            rewards.append(reward)
            successes.append(1 if reward >= 0.75 else 0)
        solve_rate = sum(successes) / self.rollout_count
        return solve_rate, {
            "defender": "random_policy",
            "rollout_count": self.rollout_count,
            "successes": successes,
            "rewards": rewards,
        }

    def _run_rollout(self, spec: ScenarioSpec) -> float:
        from oncallenv.core.tools import MUTATING_TOOLS, READ_ONLY_TOOLS

        env = OnCallRedShiftEnv()
        obs = env.reset(task_id=spec.task_id, scenario_spec=spec)
        all_tools = list(READ_ONLY_TOOLS) + list(MUTATING_TOOLS)
        last_obs = obs
        for _ in range(self.max_steps):
            if last_obs.done:
                break
            available = last_obs.available_tools or all_tools
            tool = self._rng.choice(list(available))
            service = self._rng.choice(last_obs.services) if last_obs.services else ""
            cmd = f"{tool} {service}".strip() if service else tool
            last_obs = env.step(Action(command=cmd))
        if not last_obs.done:
            last_obs = env.step(Action(command="declare_resolved"))
            rca_payload = {
                "root_cause_service": obs.services[0] if obs.services else "unknown",
                "root_cause_category": "unknown",
                "timeline": [],
                "five_whys": ["Random policy rollout."],
                "action_items": [],
                "evidence_citations": [],
                "blast_radius_description": "Random policy evaluation.",
            }
            last_obs = env.step(Action(command=f"submit_rca {json.dumps(rca_payload)}"))
        return float(last_obs.reward or 0.0)


class LLMDefenderEvaluator:
    def __init__(
        self,
        model_name: str,
        rollout_count: int = 3,
        max_steps: int = 18,
        temperature: float = 0.2,
        api_base_url: str | None = None,
        api_key: str | None = None,
    ):
        if rollout_count < 1:
            raise ValueError("rollout_count must be >= 1")
        self.model_name = model_name
        self.rollout_count = rollout_count
        self.max_steps = max_steps
        self.temperature = temperature
        self.client = OpenAI(
            base_url=api_base_url or os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
            api_key=api_key or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY"),
        )

    def __call__(self, spec: ScenarioSpec) -> tuple[float, dict]:
        rewards: list[float] = []
        successes: list[int] = []
        for _ in range(self.rollout_count):
            reward = self._run_rollout(spec)
            rewards.append(reward)
            successes.append(1 if reward >= 0.75 else 0)
        solve_rate = sum(successes) / self.rollout_count
        return solve_rate, {
            "defender": "llm_inference",
            "model_name": self.model_name,
            "rollout_count": self.rollout_count,
            "successes": successes,
            "rewards": rewards,
            "reward_mean": mean(rewards),
            "reward_std": pstdev(rewards) if len(rewards) > 1 else 0.0,
        }

    def _run_rollout(self, spec: ScenarioSpec) -> float:
        env = OnCallRedShiftEnv()
        obs = env.reset(task_id=spec.task_id, scenario_spec=spec)
        root_service = obs.alerts[0].service if obs.alerts else (obs.services[0] if obs.services else "payment-service")
        last_obs = obs
        for step in range(1, self.max_steps + 1):
            if last_obs.done:
                break
            command = self._model_command(step, last_obs, root_service)
            last_obs = env.step(Action(command=command))
            if last_obs.done:
                break
        if not last_obs.done:
            env.step(Action(command="declare_resolved"))
            rca_payload = {
                "root_cause_service": root_service,
                "root_cause_category": "unknown",
                "timeline": [],
                "five_whys": ["Telemetry indicates this service path triggered the incident."],
                "action_items": [f"Add safeguards for {root_service}"],
                "evidence_citations": [{"source": "log", "ref": f"kubectl_logs {root_service}", "excerpt": "incident indicators"}],
                "blast_radius_description": "Customer-facing impact observed during the incident window.",
            }
            last_obs = env.step(Action(command=f"submit_rca {json.dumps(rca_payload)}"))
        return float(last_obs.reward or 0.0)

    def _model_command(self, step: int, obs, root_service: str) -> str:
        system_prompt = (
            "You are an SRE incident responder in a simulator. "
            "Respond with exactly one valid command using available tools. "
            "Prefer diagnosis first, then one remediation, then declare_resolved, then submit_rca."
        )
        user_prompt = (
            f"Step: {step}\n"
            f"Task: {obs.task_id}\n"
            f"Goal: {obs.goal}\n"
            f"Services: {', '.join(obs.services)}\n"
            f"Tools: {', '.join(obs.available_tools)}\n"
            f"Last result: {obs.last_action_result}\n"
            f"Primary alert service: {root_service}\n"
            "Return one command only."
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=self.temperature,
                max_tokens=80,
            )
            raw = (completion.choices[0].message.content or "").strip()
            if raw:
                return raw.splitlines()[0].strip()
        except Exception as exc:
            raise RuntimeError(f"LLM completion failed at step {step} for task {obs.task_id}") from exc
        raise RuntimeError(f"LLM returned empty completion at step {step} for task {obs.task_id}")


class AutocurriculumRunner:
    def __init__(
        self,
        seed_specs: list[ScenarioSpec],
        seed: int = 20260424,
        evaluator: DifficultyEvaluator | None = None,
        max_buffer_size: int = 12,
    ):
        self.rng = random.Random(seed)
        self.mutator = ScenarioMutator(self.rng)
        self.max_buffer_size = max_buffer_size
        self.buffer = RegretBuffer(
            [BufferedScenario(spec=spec, regret=0.5, solve_rate=0.5) for spec in seed_specs],
            epsilon=0.08,
        )
        self.archive = {self.mutator.novelty_key(spec) for spec in seed_specs}
        self.evaluator = evaluator if evaluator is not None else RandomPolicyEvaluator(seed=seed)
        self.latest_rollout_stats: list[dict] = []
        self._prune_buffer()

    def _prune_buffer(self):
        if len(self.buffer) <= self.max_buffer_size:
            return
        # Keep all seed tasks (those not prefixed with evolved_)
        seeds = [s for s in self.buffer.scenarios if not s.spec.task_id.startswith("evolved_")]
        evolved = [s for s in self.buffer.scenarios if s.spec.task_id.startswith("evolved_")]
        # Sort evolved by variance descending (highest p*(1-p) first)
        evolved.sort(key=lambda s: s.solve_rate * (1.0 - s.solve_rate), reverse=True)
        # Fill remaining slots with top evolved tasks
        slots_for_evolved = max(0, self.max_buffer_size - len(seeds))
        kept_evolved = evolved[:slots_for_evolved]
        before = len(self.buffer)
        self.buffer.scenarios = seeds + kept_evolved
        print(f"[CURRICULUM_PRUNE] buffer shrunk {before} -> {len(self.buffer)} (kept {len(seeds)} seeds, {len(kept_evolved)} top evolved)")

    @classmethod
    def from_seed_dir(
        cls,
        seed_dir: Path,
        seed: int = 20260424,
        evaluator: DifficultyEvaluator | None = None,
        max_buffer_size: int = 12,
    ) -> "AutocurriculumRunner":
        specs = [DEFAULT_SCENARIO]
        if seed_dir.exists():
            for path in sorted(seed_dir.glob("*.y*ml")):
                specs.append(ScenarioSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))))
        unique = {spec.task_id: spec for spec in specs}
        return cls(list(unique.values()), seed=seed, evaluator=evaluator, max_buffer_size=max_buffer_size)

    def evolve(self, iterations: int) -> RegretBuffer:
        """Run `iterations` mutation rounds, maintaining a fixed-size buffer.

        Selection criterion: keep the K scenarios with the highest Bernoulli
        variance  p*(1-p)  (maximised at p=0.5, i.e. medium difficulty).
        Every novel candidate is evaluated; if the buffer is full it evicts the
        current lowest-variance entry only when the candidate is strictly better.
        """
        generation = 0
        attempts = 0
        added = evicted = skipped_novelty = skipped_variance = 0
        self._prune_buffer()
        buf_str = lambda: f"{len(self.buffer)}/{self.max_buffer_size}" if len(self.buffer) < self.max_buffer_size else f"{len(self.buffer)}"
        print(f"[CURRICULUM_EVOLVE] starting iterations={iterations} buffer={buf_str()}")
        while generation < iterations and attempts < iterations * 20:
            attempts += 1
            parent = self.buffer.sample(self.rng).spec
            candidate, mutated_field = self.mutator.mutate(parent, generation)
            novelty = self.mutator.novelty_key(candidate)
            if novelty in self.archive:
                skipped_novelty += 1
                continue
            solve_rate, eval_meta = self._estimate_solve_rate(candidate)
            variance = solve_rate * (1.0 - solve_rate)
            regret = 1.0 - abs(0.5 - solve_rate) * 2.0
            new_item = BufferedScenario(spec=candidate, regret=regret, solve_rate=solve_rate)
            action = "add"
            if len(self.buffer) < self.max_buffer_size:
                self.buffer.add(new_item)
                added += 1
            else:
                evolved = [
                    (i, item)
                    for i, item in enumerate(self.buffer.scenarios)
                    if item.spec.task_id.startswith("evolved_")
                ]
                if evolved:
                    worst_idx, worst_item = min(evolved, key=lambda t: t[1].solve_rate * (1.0 - t[1].solve_rate))
                    worst_variance = worst_item.solve_rate * (1.0 - worst_item.solve_rate)
                    if variance > worst_variance:
                        action = f"evict:{worst_item.spec.task_id}(var={worst_variance:.4f})"
                        self.buffer.scenarios[worst_idx] = new_item
                        evicted += 1
                    else:
                        skipped_variance += 1
                        continue
                else:
                    self.buffer.add(new_item)
                    added += 1
            print(
                f"[CURRICULUM_MUTATE] gen={generation:04d} "
                f"parent={parent.task_id} field={mutated_field} "
                f"child={candidate.task_id} p={solve_rate:.3f} var={variance:.4f} "
                f"action={action} buffer={buf_str()}"
            )
            meta = dict(eval_meta)
            meta["task_id"] = candidate.task_id
            meta["solve_rate"] = solve_rate
            meta["variance"] = variance
            meta["mutated_field"] = mutated_field
            self.latest_rollout_stats.append(meta)
            self.archive.add(novelty)
            generation += 1
        print(
            f"[CURRICULUM_EVOLVE] done generations={generation} attempts={attempts} "
            f"added={added} evicted={evicted} "
            f"skipped_novelty={skipped_novelty} skipped_variance={skipped_variance} "
            f"buffer={buf_str()}"
        )
        return self.buffer

    def _estimate_solve_rate(self, spec: ScenarioSpec) -> tuple[float, dict]:
        return self.evaluator(spec)

    @staticmethod
    def write_yaml_scenarios(buffer: RegretBuffer, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for item in buffer.scenarios:
            if item.spec.task_id.startswith("evolved_"):
                path = output_dir / f"{item.spec.task_id}.yaml"
                path.write_text(yaml.safe_dump(item.spec.model_dump(), sort_keys=False), encoding="utf-8")

