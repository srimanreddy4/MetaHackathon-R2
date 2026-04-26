"""OnCallEnv Red Shift environment client.

Provides a persistent WebSocket-backed client for interacting with a running
OnCallEnv server, compatible with the standard OpenEnv EnvClient protocol.
"""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import OnCallRedShiftAction, OnCallRedShiftObservation


class OnCallRedShiftEnv(
    EnvClient[OnCallRedShiftAction, OnCallRedShiftObservation, State]
):
    """
    Client for the OnCallEnv Red Shift environment.

    Example:
        >>> with OnCallRedShiftEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     result = client.step(OnCallRedShiftAction(command="kubectl_get_pods"))
        ...     print(result.observation.last_action_result)
    """

    def _step_payload(self, action: OnCallRedShiftAction) -> Dict:
        return action.model_dump()

    def _parse_result(
        self, payload: Dict
    ) -> StepResult[OnCallRedShiftObservation]:
        obs_data = payload.get("observation", {})
        observation = OnCallRedShiftObservation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
