"""FastAPI adapter for OnCallEnv Red Shift."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from oncallenv.core.env import OnCallRedShiftEnv
from models import Action


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    action: Action | None = None
    command: Optional[str] = None


app = FastAPI(title="OnCallEnv Red Shift", version="1.0.0")
env = OnCallRedShiftEnv()


@app.get("/")
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "environment": "OnCallEnv Red Shift", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()) -> dict:
    try:
        obs = env.reset(seed=req.seed, episode_id=req.episode_id, task_id=req.task_id)
        return _public_obs(obs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
def step(req: StepRequest) -> dict:
    try:
        action = req.action or Action(command=req.command or "")
        obs = env.step(action)
        return _public_obs(obs)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
def state() -> dict:
    return env.state.model_dump()


@app.post("/close")
def close() -> dict:
    env.close()
    return {"status": "closed"}


@app.get("/tasks")
def tasks() -> list[dict]:
    return env.list_tasks()


def _public_obs(obs) -> dict:
    payload = obs.model_dump()
    payload["metadata"] = {k: v for k, v in payload.get("metadata", {}).items() if k != "runtime"}
    return payload


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)

