"""RCA quality rubric with deterministic backbone."""

from __future__ import annotations

from openenv.core.rubrics import Rubric


GENERIC_WHYS = {"unknown", "n/a", "todo", "because", "issue", "problem"}


class RCAQualityRubric(Rubric):
    def forward(self, action, observation) -> float:
        runtime = observation.metadata.get("runtime")
        if runtime is None or runtime.submitted_rca is None:
            return 0.0
        rca = runtime.submitted_rca
        graph = runtime.graph
        timeline_score = 1.0 if any(event.service == graph.root_cause_service for event in rca.timeline) else 0.0
        category_score = 1.0 if rca.root_cause_category == graph.root_cause_category else 0.0
        why_count = sum(1 for why in rca.five_whys if len(why.split()) >= 4 and why.strip().lower() not in GENERIC_WHYS)
        whys_score = min(1.0, why_count / 3.0)
        services = set(graph.service_names())
        action_score = 1.0 if any(any(service in item for service in services) for item in rca.action_items) else 0.0
        return 0.3 * timeline_score + 0.3 * category_score + 0.2 * whys_score + 0.2 * action_score

