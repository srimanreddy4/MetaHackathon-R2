"""Pydantic models for OnCallEnv Red Shift."""

from __future__ import annotations

from typing import Any, Literal, Optional

from openenv.core import Action as OpenEnvAction
from openenv.core import Observation as OpenEnvObservation
from openenv.core import State as OpenEnvState
from pydantic import BaseModel, Field


FaultName = Literal[
    "oom_kill",
    "cpu_hog",
    "network_partition",
    "dns_misconfig",
    "replica_lag",
    "cache_stampede",
    "http_503_loop",
    "deadlock",
    "disk_full",
    "cert_expiry",
    "clock_skew",
    "gc_pause",
]


class ScenarioSpec(BaseModel):
    task_id: str
    topology: Literal["simple_fanout", "deep_chain", "mesh", "star", "bipartite", "diamond"]
    fault_primary: FaultName
    fault_secondary: Optional[FaultName] = None
    inject_service: str
    latency_ms: int
    blast_radius: float = Field(ge=0.0, le=1.0)
    metric_noise: float = Field(ge=0.0, le=1.0)
    red_herring: Optional[
        Literal[
            "unrelated_alert",
            "stale_deploy_notice",
            "innocent_config_change",
            "flapping_canary",
            "false_correlation",
            "old_anomaly",
            "unused_service_spike",
        ]
    ] = None
    deploy_window: Optional[Literal["recent_deploy", "flag_flip", "config_change", "none"]] = "none"
    schema_drift: Optional[
        Literal[
            "rename_metric",
            "swap_units",
            "rotate_creds",
            "new_required_field",
            "field_type_change",
            "endpoint_version_bump",
            "none",
        ]
    ] = "none"
    seed: int
    max_steps: int = 25


class Alert(BaseModel):
    alert_id: str
    severity: Literal["critical", "warning", "info"]
    service: str
    message: str
    timestamp: str


class Citation(BaseModel):
    source: Literal["log", "metric", "trace", "config"]
    ref: str
    excerpt: str


class TimelineEvent(BaseModel):
    timestamp: str
    service: str
    description: str


class RCA(BaseModel):
    root_cause_service: str
    root_cause_category: str
    timeline: list[TimelineEvent] = Field(default_factory=list)
    five_whys: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    evidence_citations: list[Citation] = Field(default_factory=list)
    blast_radius_description: str = ""


class Reward(BaseModel):
    total: float
    breakdown: dict[str, float] = Field(default_factory=dict)


class Observation(OpenEnvObservation):
    alerts: list[Alert] = Field(default_factory=list)
    last_action_result: str = ""
    available_tools: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    time_elapsed_sec: int = 0
    goal: str = ""
    rca_required: bool = True
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    task_id: str = ""


class Action(OpenEnvAction):
    command: str


class State(OpenEnvState):
    task_id: str = ""
    done: bool = False
    actions_taken: list[str] = Field(default_factory=list)
    resolved_declared: bool = False
    rca: Optional[RCA] = None
    last_action_result: str = ""
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    scenario: Optional[ScenarioSpec] = None
    unsafe_actions: int = 0
    remediated: bool = False
    context: dict[str, Any] = Field(default_factory=dict)

