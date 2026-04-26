"""Recovery verification rubric."""

from __future__ import annotations

from openenv.core.rubrics import Rubric


class RecoveryRubric(Rubric):
    def forward(self, action, observation) -> float:
        runtime = observation.metadata.get("runtime")
        if runtime is None or not runtime.resolved_declared:
            return 0.0
        if not runtime.graph.is_recovered():
            return 0.0
        success_rate = runtime.graph.synthetic_success_rate()
        if success_rate >= 0.99:
            return 1.0
        if success_rate <= 0.5:
            return 0.0
        return (success_rate - 0.5) / 0.49
