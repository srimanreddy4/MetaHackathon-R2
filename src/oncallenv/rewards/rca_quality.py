"""RCA quality rubric with deterministic backbone."""

from __future__ import annotations

from openenv.core.rubrics import Rubric
class RCAQualityRubric(Rubric):
    def forward(self, action, observation) -> float:
        runtime = observation.metadata.get("runtime")
        if runtime is None or runtime.submitted_rca is None:
            return 0.0
        rca = runtime.submitted_rca
        graph = runtime.graph
        
        service_score = 1.0 if rca.root_cause_service == graph.root_cause_service else 0.0
        category_score = 1.0 if rca.root_cause_category == graph.root_cause_category else 0.0
        
        return 0.5 * service_score + 0.5 * category_score
