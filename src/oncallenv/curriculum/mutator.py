"""Structured scenario mutation for autocurriculum."""

from __future__ import annotations

import hashlib
import random

from oncallenv.core.types import ScenarioSpec


TOPOLOGIES = ["simple_fanout", "deep_chain", "mesh", "star", "bipartite", "diamond"]
FAULTS = [
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
SERVICES = ["api-gateway", "checkout-service", "payment-service", "inventory-service", "user-service", "postgres-primary", "redis-cache"]
LATENCIES = [0, 50, 200, 1000, 5000]
RED_HERRINGS = [None, "unrelated_alert", "stale_deploy_notice", "innocent_config_change", "flapping_canary", "false_correlation", "old_anomaly", "unused_service_spike"]
DEPLOY_WINDOWS = ["recent_deploy", "flag_flip", "config_change", "none"]
SCHEMA_DRIFTS = ["rename_metric", "swap_units", "rotate_creds", "new_required_field", "field_type_change", "endpoint_version_bump", "none"]


class ScenarioMutator:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def mutate(self, parent: ScenarioSpec, generation: int) -> tuple[ScenarioSpec, str]:
        data = parent.model_dump()
        field = self.rng.choice(
            [
                "topology",
                "fault_primary",
                "fault_secondary",
                "inject_service",
                "latency_ms",
                "blast_radius",
                "metric_noise",
                "red_herring",
                "deploy_window",
                "schema_drift",
            ]
        )
        if field == "topology":
            data[field] = self.rng.choice(TOPOLOGIES)
        elif field == "fault_primary":
            data[field] = self.rng.choice(FAULTS)
        elif field == "fault_secondary":
            data[field] = self.rng.choice([None, *FAULTS])
        elif field == "inject_service":
            data[field] = self.rng.choice(SERVICES)
        elif field == "latency_ms":
            data[field] = self.rng.choice(LATENCIES)
        elif field in {"blast_radius", "metric_noise"}:
            data[field] = round(min(1.0, max(0.0, float(data[field]) + self.rng.uniform(-0.2, 0.2))), 2)
        elif field == "red_herring":
            data[field] = self.rng.choice(RED_HERRINGS)
        elif field == "deploy_window":
            data[field] = self.rng.choice(DEPLOY_WINDOWS)
        elif field == "schema_drift":
            data[field] = self.rng.choice(SCHEMA_DRIFTS)

        fingerprint = hashlib.sha1(repr(sorted(data.items())).encode()).hexdigest()[:10]
        data["task_id"] = f"evolved_{generation:04d}_{fingerprint}"
        data["seed"] = self.rng.randint(1, 2_000_000_000)
        data["max_steps"] = max(10, min(30, int(data.get("max_steps", 25))))
        return ScenarioSpec.model_validate(data), field

    @staticmethod
    def novelty_key(spec: ScenarioSpec) -> tuple:
        return (spec.fault_primary, spec.fault_secondary, spec.inject_service, spec.schema_drift)

