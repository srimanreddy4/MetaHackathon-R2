"""Default weighted Red Shift reward."""

from __future__ import annotations

from openenv.core.rubrics import WeightedSum

from oncallenv.rewards.blast_radius import BlastRadiusRubric
from oncallenv.rewards.rca_quality import RCAQualityRubric
from oncallenv.rewards.recovery import RecoveryRubric
from oncallenv.rewards.safety import SafetyRubric


def build_default_rubric() -> WeightedSum:
    return WeightedSum(
        [RecoveryRubric(), RCAQualityRubric(), BlastRadiusRubric(), SafetyRubric()],
        weights=[0.35, 0.30, 0.25, 0.10],
    )

