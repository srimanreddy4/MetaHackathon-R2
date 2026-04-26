"""Reviewer panel with deterministic fallback.

The full submission plan calls for a three-model LLM judge panel. This module
keeps that interface while making the default path deterministic and offline, so
reward computation and autocurriculum never crash when endpoints are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from pydantic import BaseModel, Field

from models import RCA


class JudgeScore(BaseModel):
    five_whys_score: float = Field(ge=0.0, le=1.0)
    action_items_score: float = Field(ge=0.0, le=1.0)
    narrative_coherence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""

    @property
    def total(self) -> float:
        return 0.4 * self.five_whys_score + 0.35 * self.action_items_score + 0.25 * self.narrative_coherence_score


@dataclass
class ReviewerPanel:
    """Small judge facade with deterministic fallback scoring."""

    enabled_endpoints: tuple[str, ...] = ()

    def score(self, rca: RCA, ground_truth_events: list[dict[str, str]], services: list[str]) -> JudgeScore:
        scores = [self._deterministic_score(rca, ground_truth_events, services)]
        # Future LLM endpoint scores can be appended here. Median keeps the API
        # robust to one noisy judge.
        return JudgeScore(
            five_whys_score=median(score.five_whys_score for score in scores),
            action_items_score=median(score.action_items_score for score in scores),
            narrative_coherence_score=median(score.narrative_coherence_score for score in scores),
            reasoning="median reviewer score with deterministic fallback",
        )

    def _deterministic_score(self, rca: RCA, ground_truth_events: list[dict[str, str]], services: list[str]) -> JudgeScore:
        root_service = ground_truth_events[0]["service"] if ground_truth_events else ""
        five_whys = [why for why in rca.five_whys if len(why.split()) >= 5 and "todo" not in why.lower()]
        cited_services = sum(1 for item in rca.action_items if any(service in item for service in services))
        timeline_match = any(event.service == root_service for event in rca.timeline)
        citation_match = any(citation.ref or citation.excerpt for citation in rca.evidence_citations)
        return JudgeScore(
            five_whys_score=min(1.0, len(five_whys) / 3.0),
            action_items_score=1.0 if cited_services else 0.0,
            narrative_coherence_score=(0.5 if timeline_match else 0.0) + (0.5 if citation_match else 0.0),
            reasoning="deterministic fallback: whys, action-items, timeline, citations",
        )

