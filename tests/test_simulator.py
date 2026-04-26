import pytest

from oncallenv.core.types import ScenarioSpec
from oncallenv.simulation.faults import FAULTS
from oncallenv.simulation.scenario_compiler import compile_scenario
from oncallenv.telemetry.otlp import json_logs, prometheus_metrics


@pytest.mark.parametrize("fault_name", sorted(FAULTS))
def test_fault_primitives_emit_recognizable_telemetry(fault_name):
    spec = ScenarioSpec(
        task_id=f"test_{fault_name}",
        topology="simple_fanout",
        fault_primary=fault_name,
        inject_service="payment-service",
        latency_ms=1000,
        blast_radius=0.5,
        metric_noise=0.0,
        seed=1,
    )
    graph = compile_scenario(spec)
    logs = json_logs(graph, "payment-service")
    metrics = prometheus_metrics(graph, "payment-service")
    assert fault_name == graph.root_cause_category
    assert "service.name" in logs
    assert "http_requests_total" in metrics

