"""Microservice graph state used by the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Service:
    name: str
    cpu: float = 25.0
    memory: float = 40.0
    latency_p50: float = 12.0
    latency_p99: float = 45.0
    error_rate: float = 0.001
    rps: float = 500.0
    dependencies: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    deploy_history: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    current_fault: str | None = None
    remediated: bool = False


class MicroserviceGraph:
    """Small in-memory graph with enough dynamics for RL rollouts."""

    def __init__(self, services: dict[str, Service]):
        self.services = services
        self.elapsed_sec = 0
        self.event_log: list[dict[str, str]] = []
        self.root_cause_service = ""
        self.root_cause_category = ""
        self.required_remediations: set[str] = set()
        self.completed_remediations: set[str] = set()

    @classmethod
    def topology(cls, name: str) -> "MicroserviceGraph":
        base = {
            "api-gateway": Service("api-gateway", dependencies=["checkout-service", "user-service"]),
            "checkout-service": Service("checkout-service", dependencies=["payment-service", "inventory-service"]),
            "payment-service": Service("payment-service", dependencies=["postgres-primary", "redis-cache"]),
            "inventory-service": Service("inventory-service", dependencies=["postgres-primary"]),
            "user-service": Service("user-service", dependencies=["postgres-primary"]),
            "postgres-primary": Service("postgres-primary", cpu=35.0, memory=55.0, rps=1800.0),
            "redis-cache": Service("redis-cache", cpu=18.0, memory=35.0, rps=2600.0),
        }
        if name == "deep_chain":
            base["api-gateway"].dependencies = ["checkout-service"]
            base["checkout-service"].dependencies = ["payment-service"]
            base["payment-service"].dependencies = ["inventory-service"]
            base["inventory-service"].dependencies = ["postgres-primary"]
        elif name == "star":
            for service in base.values():
                if service.name != "api-gateway":
                    service.dependencies = ["api-gateway"]
        elif name == "mesh":
            base["checkout-service"].dependencies.append("user-service")
            base["payment-service"].dependencies.append("inventory-service")
            base["inventory-service"].dependencies.append("redis-cache")
        elif name == "bipartite":
            base["api-gateway"].dependencies = ["checkout-service", "payment-service", "inventory-service"]
            base["checkout-service"].dependencies = ["postgres-primary", "redis-cache"]
            base["payment-service"].dependencies = ["postgres-primary", "redis-cache"]
            base["inventory-service"].dependencies = ["postgres-primary", "redis-cache"]
        elif name == "diamond":
            base["api-gateway"].dependencies = ["checkout-service", "user-service"]
            base["checkout-service"].dependencies = ["payment-service"]
            base["user-service"].dependencies = ["payment-service"]
            base["payment-service"].dependencies = ["postgres-primary"]
        return cls(base)

    def tick(self, seconds: int = 30) -> None:
        self.elapsed_sec += seconds
        for service in self.services.values():
            if service.current_fault and not service.remediated:
                service.latency_p99 *= 1.03
                service.error_rate = min(0.99, service.error_rate + 0.015)

    def service_names(self) -> list[str]:
        return list(self.services)

    def get(self, service_name: str) -> Service:
        if service_name not in self.services:
            raise KeyError(f"unknown service {service_name}")
        return self.services[service_name]

    def mark_remediated(self, action_key: str, service_name: str) -> bool:
        key = f"{action_key}:{service_name}"
        if key not in self.required_remediations:
            return False
        self.completed_remediations.add(key)
        self.services[service_name].remediated = True
        self.services[service_name].current_fault = None
        self.services[service_name].cpu = min(self.services[service_name].cpu, 30.0)
        self.services[service_name].memory = min(self.services[service_name].memory, 45.0)
        self.services[service_name].latency_p50 = min(self.services[service_name].latency_p50, 25.0)
        self.services[service_name].latency_p99 = min(self.services[service_name].latency_p99, 95.0)
        self.services[service_name].error_rate = 0.001
        return self.is_recovered()

    def is_recovered(self) -> bool:
        return bool(self.required_remediations) and self.required_remediations.issubset(self.completed_remediations)

    def synthetic_success_rate(self) -> float:
        affected = self.services.get(self.root_cause_service)
        if affected is None:
            return 0.0
        if self.is_recovered():
            return 1.0
        return max(0.0, 1.0 - affected.error_rate)

    def blast_radius_score(self) -> float:
        affected = self.services.get(self.root_cause_service)
        if affected is None:
            return 0.0
        exposure = min(1.0, self.elapsed_sec / 900.0)
        return max(0.0, 1.0 - (affected.error_rate * 0.65 + exposure * 0.35))

