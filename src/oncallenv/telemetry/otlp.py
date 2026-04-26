"""OpenTelemetry-shaped metrics, logs, and traces."""

from __future__ import annotations

import hashlib
import json

from oncallenv.simulation.graph import MicroserviceGraph


def prometheus_metrics(graph: MicroserviceGraph, service_name: str) -> str:
    service = graph.get(service_name)
    labels = f'service="{service.name}"'
    return "\n".join(
        [
            f'http_requests_total{{{labels},status="200"}} {int(service.rps * (1 - service.error_rate))}',
            f'http_requests_total{{{labels},status="500"}} {int(service.rps * service.error_rate)}',
            f'http_request_duration_seconds{{{labels},quantile="0.50"}} {service.latency_p50 / 1000:.3f}',
            f'http_request_duration_seconds{{{labels},quantile="0.99"}} {service.latency_p99 / 1000:.3f}',
            f'process_cpu_usage{{{labels}}} {service.cpu / 100:.3f}',
            f'process_resident_memory_ratio{{{labels}}} {service.memory / 100:.3f}',
        ]
    )


def json_logs(graph: MicroserviceGraph, service_name: str) -> str:
    service = graph.get(service_name)
    rows = []
    for idx, body in enumerate(service.logs[-20:], start=1):
        digest = hashlib.sha1(f"{service.name}:{idx}:{body}".encode()).hexdigest()
        rows.append(
            {
                "timestamp": f"2026-04-24T09:{idx:02d}:00Z",
                "trace_id": digest[:32],
                "span_id": digest[32:48],
                "service.name": service.name,
                "severity": "ERROR" if any(token in body.lower() for token in ["error", "expired", "killed", "deadline", "503"]) else "WARN",
                "body": body,
                "attributes": {"fault": service.current_fault or "none"},
            }
        )
    return "\n".join(json.dumps(row, sort_keys=True) for row in rows) or "no recent logs"


def jaeger_traces(graph: MicroserviceGraph, service_name: str) -> str:
    service = graph.get(service_name)
    spans = []
    parent = None
    for dep in [service.name, *service.dependencies]:
        span_id = hashlib.md5(dep.encode()).hexdigest()[:16]
        spans.append({"service": dep, "span_id": span_id, "parent_span_id": parent, "duration_ms": graph.get(dep).latency_p99 if dep in graph.services else 20})
        parent = span_id
    return json.dumps({"data": [{"traceID": hashlib.md5(service.name.encode()).hexdigest(), "spans": spans}]}, indent=2)

