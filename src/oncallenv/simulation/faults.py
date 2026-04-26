"""Fault primitives for the Red Shift simulator."""

from __future__ import annotations

from dataclasses import dataclass

from oncallenv.simulation.graph import MicroserviceGraph


@dataclass(frozen=True)
class FaultPrimitive:
    name: str
    log_lines: tuple[str, ...]
    cpu: float
    memory: float
    latency_p99: float
    error_rate: float
    remediation: str

    def apply(self, graph: MicroserviceGraph, service_name: str) -> None:
        service = graph.get(service_name)
        service.current_fault = self.name
        service.cpu = max(service.cpu, self.cpu)
        service.memory = max(service.memory, self.memory)
        service.latency_p50 = max(service.latency_p50, self.latency_p99 / 5)
        service.latency_p99 = max(service.latency_p99, self.latency_p99)
        service.error_rate = max(service.error_rate, self.error_rate)
        service.logs.extend(self.log_lines)
        service.deploy_history.append({"version": "v3.2.1", "date": "2026-04-24", "status": "current", "notes": self.name})
        graph.root_cause_service = service_name
        graph.root_cause_category = self.name
        graph.required_remediations.add(f"{self.remediation}:{service_name}")
        graph.event_log.append(
            {
                "timestamp": "2026-04-24T09:00:00Z",
                "service": service_name,
                "description": f"{self.name} injected into {service_name}",
            }
        )

    def tick(self, graph: MicroserviceGraph, service_name: str, elapsed_sec: int) -> None:
        graph.tick(elapsed_sec)

    def remediation_actions(self) -> set[str]:
        return {self.remediation}

    def ground_truth_rca(self, service_name: str) -> dict[str, str]:
        return {"root_cause_service": service_name, "root_cause_category": self.name}


FAULTS: dict[str, FaultPrimitive] = {
    "oom_kill": FaultPrimitive(
        "oom_kill",
        ("java.lang.OutOfMemoryError: Java heap space", "Container killed by OOMKilled / exit code 137"),
        55.0,
        94.0,
        4200.0,
        0.22,
        "kubectl_rollout_restart",
    ),
    "cpu_hog": FaultPrimitive("cpu_hog", ("run queue saturated", "CPU throttling at 98 percent"), 98.0, 50.0, 2500.0, 0.16, "kubectl_scale"),
    "network_partition": FaultPrimitive("network_partition", ("context deadline exceeded", "Envoy response flag UF upstream failure"), 35.0, 45.0, 5000.0, 0.35, "traffic_split_update"),
    "dns_misconfig": FaultPrimitive("dns_misconfig", ("lookup inventory.internal: no such host", "Envoy response flag NR no route"), 30.0, 40.0, 3600.0, 0.28, "kubectl_apply_config"),
    "replica_lag": FaultPrimitive("replica_lag", ("replication_lag_seconds above threshold", "stale read detected"), 70.0, 72.0, 2100.0, 0.11, "feature_flag_toggle"),
    "cache_stampede": FaultPrimitive("cache_stampede", ("redis miss rate 96 percent", "thundering herd after cache expiry"), 88.0, 76.0, 3100.0, 0.18, "feature_flag_toggle"),
    "http_503_loop": FaultPrimitive("http_503_loop", ("upstream returned HTTP 503", "Envoy response flag UH no healthy upstream"), 66.0, 58.0, 3900.0, 0.41, "kubectl_rollout_undo"),
    "deadlock": FaultPrimitive("deadlock", ("deadlock detected while waiting for lock", "org.postgresql.util.PSQLException: Cannot get connection"), 62.0, 68.0, 4500.0, 0.25, "kubectl_rollout_restart"),
    "disk_full": FaultPrimitive("disk_full", ("no space left on device", "etcdserver: data corruption detected"), 50.0, 60.0, 2400.0, 0.21, "kubectl_apply_config"),
    "cert_expiry": FaultPrimitive("cert_expiry", ("x509: certificate has expired or is not yet valid", "TLS handshake failed"), 25.0, 40.0, 5000.0, 0.50, "kubectl_apply_config"),
    "clock_skew": FaultPrimitive("clock_skew", ("JWT not valid yet due to clock skew", "signature timestamp outside tolerance"), 28.0, 35.0, 1800.0, 0.19, "kubectl_rollout_restart"),
    "gc_pause": FaultPrimitive("gc_pause", ("GC overhead limit exceeded", "called Result::unwrap() on an Err value"), 75.0, 90.0, 4700.0, 0.20, "kubectl_rollout_restart"),
}

