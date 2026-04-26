"""OnCallEnv Red Shift — server-side Environment adapter.

Thin wrapper that adapts the core OnCallRedShiftEnv (which inherits from
openenv.core.Environment) to the HTTP-server interface expected by
``openenv.core.env_server.http_server.create_app``.
"""

from __future__ import annotations

from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import OnCallRedShiftAction, OnCallRedShiftObservation
except ImportError:
    from models import OnCallRedShiftAction, OnCallRedShiftObservation

from oncallenv.core.env import OnCallRedShiftEnv


class OnCallRedShiftEnvironment(Environment):
    """HTTP-server-compatible wrapper around the core env."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._env = OnCallRedShiftEnv()
        self._state = State(episode_id=None, step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OnCallRedShiftObservation:
        obs = self._env.reset(seed=seed, episode_id=episode_id, task_id=task_id, **kwargs)
        self._state = State(
            episode_id=episode_id or self._env.state.task_id,
            step_count=0,
        )
        return obs

    def step(self, action: OnCallRedShiftAction, **kwargs: Any) -> OnCallRedShiftObservation:  # type: ignore[override]
        from oncallenv.core.types import Action as CoreAction
        core_action = CoreAction(command=action.command)
        obs = self._env.step(core_action)
        self._state.step_count += 1
        return obs

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        self._env.close()
