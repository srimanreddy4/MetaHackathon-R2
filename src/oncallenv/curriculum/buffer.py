"""Regret-prioritized scenario buffer."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from oncallenv.core.types import ScenarioSpec


@dataclass
class BufferedScenario:
    spec: ScenarioSpec
    regret: float
    solve_rate: float
    seen_count: int = 0

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.model_dump(),
            "regret": self.regret,
            "solve_rate": self.solve_rate,
            "seen_count": self.seen_count,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BufferedScenario":
        return cls(
            spec=ScenarioSpec.model_validate(payload["spec"]),
            regret=float(payload["regret"]),
            solve_rate=float(payload["solve_rate"]),
            seen_count=int(payload.get("seen_count", 0)),
        )


class RegretBuffer:
    def __init__(self, scenarios: list[BufferedScenario] | None = None, epsilon: float = 0.08):
        self.scenarios = scenarios or []
        self.epsilon = epsilon

    def __len__(self) -> int:
        return len(self.scenarios)

    def add(self, item: BufferedScenario) -> None:
        self.scenarios.append(item)

    def sample(self, rng: random.Random) -> BufferedScenario:
        if not self.scenarios:
            raise ValueError("cannot sample from empty RegretBuffer")
        if rng.random() < self.epsilon:
            choice = rng.choice(self.scenarios)
        else:
            weights = [max(0.01, item.regret) for item in self.scenarios]
            choice = rng.choices(self.scenarios, weights=weights, k=1)[0]
        choice.seen_count += 1
        return choice

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([item.to_dict() for item in self.scenarios], indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RegretBuffer":
        return cls([BufferedScenario.from_dict(item) for item in json.loads(path.read_text(encoding="utf-8"))])

