"""Data models for the OnCallEnv Red Shift environment.

Re-exports the canonical Action and Observation types so that the OpenEnv
CLI tooling (``openenv push``, ``openenv validate``) finds them at the
expected path.
"""

from oncallenv.core.types import Action as OnCallRedShiftAction
from oncallenv.core.types import Observation as OnCallRedShiftObservation

__all__ = ["OnCallRedShiftAction", "OnCallRedShiftObservation"]
