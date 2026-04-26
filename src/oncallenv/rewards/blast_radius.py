"""Blast radius rubric."""

from __future__ import annotations

from openenv.core.rubrics import Rubric


class BlastRadiusRubric(Rubric):
    def forward(self, action, observation) -> float:
        runtime = observation.metadata.get("runtime")
        if runtime is None:
            return 0.0
        return runtime.graph.blast_radius_score()

