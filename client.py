"""Small HTTP client that avoids importing server internals."""

from __future__ import annotations

import requests

from models import Action, Observation


class OnCallRedShiftClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str | None = None) -> Observation:
        response = requests.post(f"{self.base_url}/reset", json={"task_id": task_id}, timeout=30)
        response.raise_for_status()
        return Observation.model_validate(response.json())

    def step(self, action: Action) -> Observation:
        response = requests.post(f"{self.base_url}/step", json={"action": action.model_dump()}, timeout=30)
        response.raise_for_status()
        return Observation.model_validate(response.json())

