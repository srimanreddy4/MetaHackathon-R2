"""Autocurriculum loop for generating non-trivial incident scenarios."""

from __future__ import annotations

import random
from pathlib import Path

import yaml

from oncallenv.core.env import DEFAULT_SCENARIO
from models import Action, ScenarioSpec
from oncallenv.curriculum.buffer import BufferedScenario, RegretBuffer
from oncallenv.curriculum.mutator import ScenarioMutator
from oncallenv.simulation.scenario_compiler import compile_scenario


class AutocurriculumRunner:
    def __init__(self, seed_specs: list[ScenarioSpec], seed: int = 20260424):
        self.rng = random.Random(seed)
        self.mutator = ScenarioMutator(self.rng)
        self.buffer = RegretBuffer(
            [BufferedScenario(spec=spec, regret=0.5, solve_rate=0.5) for spec in seed_specs],
            epsilon=0.08,
        )
        self.archive = {self.mutator.novelty_key(spec) for spec in seed_specs}

    @classmethod
    def from_seed_dir(cls, seed_dir: Path, seed: int = 20260424) -> "AutocurriculumRunner":
        specs = [DEFAULT_SCENARIO]
        if seed_dir.exists():
            for path in sorted(seed_dir.glob("*.y*ml")):
                specs.append(ScenarioSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))))
        unique = {spec.task_id: spec for spec in specs}
        return cls(list(unique.values()), seed=seed)

    def evolve(self, iterations: int) -> RegretBuffer:
        generation = 0
        attempts = 0
        while generation < iterations and attempts < iterations * 20:
            attempts += 1
            parent = self.buffer.sample(self.rng).spec
            candidate = self.mutator.mutate(parent, generation)
            novelty = self.mutator.novelty_key(candidate)
            if novelty in self.archive:
                continue
            solve_rate = self._heuristic_solve_rate(candidate)
            if not 0.05 <= solve_rate <= 0.95:
                continue
            regret = 1.0 - abs(0.5 - solve_rate) * 2.0
            self.buffer.add(BufferedScenario(spec=candidate, regret=regret, solve_rate=solve_rate))
            self.archive.add(novelty)
            generation += 1
        return self.buffer

    def _heuristic_solve_rate(self, spec: ScenarioSpec) -> float:
        graph = compile_scenario(spec)
        base = 0.82
        if spec.fault_secondary:
            base -= 0.22
        base -= 0.20 * spec.metric_noise
        base -= 0.18 * spec.blast_radius
        if spec.red_herring:
            base -= 0.08
        if spec.schema_drift and spec.schema_drift != "none":
            base -= 0.10
        if graph.root_cause_service in {"postgres-primary", "redis-cache"}:
            base -= 0.06
        return max(0.02, min(0.98, base + self.rng.uniform(-0.08, 0.08)))

    @staticmethod
    def write_yaml_scenarios(buffer: RegretBuffer, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for item in buffer.scenarios:
            if item.spec.task_id.startswith("evolved_"):
                path = output_dir / f"{item.spec.task_id}.yaml"
                path.write_text(yaml.safe_dump(item.spec.model_dump(), sort_keys=False), encoding="utf-8")

