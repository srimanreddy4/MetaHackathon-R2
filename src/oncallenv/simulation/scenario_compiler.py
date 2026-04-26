"""Compile ScenarioSpec objects into simulator graphs."""

from __future__ import annotations

from oncallenv.core.types import ScenarioSpec
from oncallenv.simulation.faults import FAULTS
from oncallenv.simulation.graph import MicroserviceGraph


def compile_scenario(spec: ScenarioSpec) -> MicroserviceGraph:
    graph = MicroserviceGraph.topology(spec.topology)
    inject_service = spec.inject_service if spec.inject_service in graph.services else "payment-service"
    FAULTS[spec.fault_primary].apply(graph, inject_service)
    graph.get(inject_service).latency_p99 = max(graph.get(inject_service).latency_p99, float(spec.latency_ms))
    if spec.fault_secondary:
        secondary_service = "redis-cache" if inject_service != "redis-cache" else "postgres-primary"
        FAULTS[spec.fault_secondary].apply(graph, secondary_service)
    if spec.red_herring:
        graph.get("user-service").logs.append(f"benign warning: {spec.red_herring}")
    if spec.deploy_window and spec.deploy_window != "none":
        graph.get(inject_service).deploy_history.append(
            {"version": "v3.3.0", "date": "2026-04-24", "status": "current", "notes": spec.deploy_window}
        )
    return graph

