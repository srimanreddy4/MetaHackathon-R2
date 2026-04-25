"""OpenEnv-compatible OnCallEnv Red Shift runtime."""

from __future__ import annotations

import os
import random
from typing import Any, Optional

import yaml
from openenv.core import Environment

from oncallenv.core.tools import AVAILABLE_TOOLS, ToolRuntime
from oncallenv.core.types import Action, Alert, Observation, ScenarioSpec, State
from oncallenv.rewards import build_default_rubric
from oncallenv.simulation.scenario_compiler import compile_scenario


DEFAULT_SCENARIO = ScenarioSpec(
    task_id="seed_easy_memory_leak",
    topology="simple_fanout",
    fault_primary="oom_kill",
    inject_service="payment-service",
    latency_ms=4200,
    blast_radius=0.35,
    metric_noise=0.1,
    red_herring="stale_deploy_notice",
    deploy_window="recent_deploy",
    schema_drift="none",
    seed=7,
    max_steps=25,
)


class OnCallRedShiftEnv(Environment[Action, Observation, State]):
    """Three-agent SRE training environment shell with simulator-backed defender loop."""

    def __init__(self, reward_cap: Optional[float] = None):
        super().__init__(rubric=build_default_rubric())
        self.reward_cap = reward_cap if reward_cap is not None else _env_float("REWARD_CAP")
        self._scenario = DEFAULT_SCENARIO
        self._runtime: ToolRuntime | None = None
        self._state = State()
        self._last_reward = 0.0

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, task_id: Optional[str] = None, **kwargs: Any) -> Observation:
        self._reset_rubric()
        scenario_override = kwargs.get("scenario_spec")
        if scenario_override is not None:
            spec = scenario_override if isinstance(scenario_override, ScenarioSpec) else ScenarioSpec.model_validate(scenario_override)
        else:
            spec = self._load_scenario(task_id or episode_id or kwargs.get("task_id"))
        if seed is not None:
            spec = spec.model_copy(update={"seed": seed})
        self._scenario = spec
        self._runtime = ToolRuntime(compile_scenario(spec))
        self._state = State(episode_id=episode_id, task_id=spec.task_id, scenario=spec, context={"episode_id": episode_id})
        self._last_reward = 0.0
        return self._observation("Environment reset. You are the SRE responder. Investigate the alert, remediate safely, declare_resolved, then submit_rca.")

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        if self._runtime is None:
            raise RuntimeError("reset() must be called before step()")
        if self._state.done:
            return self._observation("Episode already finished.")
        self._runtime.graph.tick(30)
        result = self._runtime.execute(action.command)
        self._state.step_count += 1
        self._state.actions_taken.append(action.command)
        self._state.last_action_result = result
        self._state.unsafe_actions = self._runtime.unsafe_actions
        self._state.resolved_declared = self._runtime.resolved_declared
        self._state.rca = self._runtime.submitted_rca
        self._state.remediated = self._runtime.graph.is_recovered()
        done = (
            self._state.step_count >= self._scenario.max_steps
            or (self._runtime.resolved_declared and self._runtime.submitted_rca is not None)
        )
        self._state.done = done
        obs = self._observation(result, done=done)
        reward = float(self._apply_rubric(action, obs))
        if self.reward_cap is not None:
            reward = min(reward, self.reward_cap)
        self._last_reward = reward
        breakdown = self._reward_breakdown()
        self._state.reward_breakdown = breakdown
        obs.reward = reward
        obs.reward_breakdown = breakdown
        obs.metadata = {"task_id": self._scenario.task_id}
        return obs

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        self._runtime = None
        self._state.done = True

    def list_tasks(self) -> list[dict[str, Any]]:
        specs = {spec.task_id: spec for spec in [*self._seed_specs(), *self._curriculum_specs()]}
        return [{"id": spec.task_id, "name": spec.task_id.replace("_", " ").title(), "max_steps": spec.max_steps} for spec in specs.values()]

    def _observation(self, result: str, done: bool = False) -> Observation:
        runtime = self._runtime
        alerts: list[Alert] = []
        services: list[str] = []
        elapsed = 0
        if runtime is not None:
            services = runtime.graph.service_names()
            elapsed = runtime.graph.elapsed_sec
            alerts = [
                Alert(
                    alert_id="ALT-RED-001",
                    severity="critical",
                    service=runtime.graph.root_cause_service,
                    message=f"{runtime.graph.root_cause_category} symptoms detected with elevated p99/error rate",
                    timestamp="2026-04-24T09:00:00Z",
                )
            ]
        return Observation(
            done=done,
            reward=self._last_reward,
            metadata={"runtime": runtime} if runtime else {},
            alerts=alerts,
            last_action_result=result,
            available_tools=AVAILABLE_TOOLS,
            services=services,
            time_elapsed_sec=elapsed,
            goal="Restore customer-facing availability and submit a grounded RCA.",
            rca_required=True,
            reward_breakdown=self._state.reward_breakdown,
            task_id=self._scenario.task_id,
        )

    def _reward_breakdown(self) -> dict[str, float]:
        if self.rubric is None:
            return {}
        return {name: float(rubric.last_score or 0.0) for name, rubric in self.rubric.named_rubrics()}

    def _load_scenario(self, task_id: Optional[str]) -> ScenarioSpec:
        specs = {spec.task_id: spec for spec in [*self._seed_specs(), *self._curriculum_specs()]}
        if task_id in specs:
            return specs[task_id]
        if task_id == "curriculum":
            evolved = self._sample_curriculum_scenario()
            if evolved is not None:
                return evolved
            return DEFAULT_SCENARIO
        if task_id is None:
            return DEFAULT_SCENARIO
        raise ValueError(f"Unknown task_id {task_id}. Available: {', '.join(sorted(specs))}")

    def _sample_curriculum_scenario(self) -> ScenarioSpec | None:
        try:
            from oncallenv.curriculum import RegretBuffer
        except Exception:
            return None
        path = os.getenv("CURRICULUM_BUFFER", os.path.join(os.getcwd(), "curriculum_results", "buffer.json"))
        if not os.path.exists(path):
            return None
        try:
            return RegretBuffer.load(PathLike(path)).sample(random.Random()).spec
        except Exception:
            return None

    def _curriculum_specs(self) -> list[ScenarioSpec]:
        try:
            from oncallenv.curriculum import RegretBuffer
        except Exception:
            return []
        path = os.getenv("CURRICULUM_BUFFER", os.path.join(os.getcwd(), "curriculum_results", "buffer.json"))
        if not os.path.exists(path):
            return []
        try:
            return [item.spec for item in RegretBuffer.load(PathLike(path)).scenarios]
        except Exception:
            return []

    def _seed_specs(self) -> list[ScenarioSpec]:
        seed_dir = os.path.join(os.getcwd(), "scenarios_seed")
        specs = [DEFAULT_SCENARIO]
        if os.path.isdir(seed_dir):
            for name in sorted(os.listdir(seed_dir)):
                if name.endswith((".yaml", ".yml")):
                    with open(os.path.join(seed_dir, name), "r", encoding="utf-8") as handle:
                        specs.append(ScenarioSpec.model_validate(yaml.safe_load(handle)))
        unique: dict[str, ScenarioSpec] = {}
        for spec in specs:
            unique[spec.task_id] = spec
        return list(unique.values())


def _env_float(name: str) -> float | None:
    value = os.getenv(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class PathLike:
    """Tiny adapter to avoid importing pathlib on the hot path above."""

    def __init__(self, value: str):
        self.value = value

    def read_text(self, encoding: str = "utf-8") -> str:
        with open(self.value, "r", encoding=encoding) as handle:
            return handle.read()
