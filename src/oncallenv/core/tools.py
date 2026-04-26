"""Defender command surface for querying and mutating the simulator."""

from __future__ import annotations

import json
from typing import Callable

from pydantic import ValidationError

from oncallenv.core.types import RCA
from oncallenv.simulation.graph import MicroserviceGraph
from oncallenv.telemetry.otlp import jaeger_traces, json_logs, prometheus_metrics


READ_ONLY_TOOLS = [
    "kubectl_get_pods",
    "kubectl_describe_pod",
    "kubectl_logs",
    "kubectl_top",
    "promql_query",
    "logql_query",
    "jaeger_search",
    "istioctl_proxy_status",
    "istioctl_routes",
    "curl_service",
    "dns_lookup",
    "check_deploy_history",
]
MUTATING_TOOLS = [
    "kubectl_rollout_undo",
    "kubectl_rollout_restart",
    "kubectl_scale",
    "feature_flag_toggle",
    "traffic_split_update",
    "kubectl_apply_config",
]
COMMUNICATION_TOOLS = ["post_status_update"]
TERMINAL_TOOLS = ["declare_resolved", "submit_rca"]
AVAILABLE_TOOLS = READ_ONLY_TOOLS + MUTATING_TOOLS + COMMUNICATION_TOOLS + TERMINAL_TOOLS


class ToolRuntime:
    def __init__(self, graph: MicroserviceGraph):
        self.graph = graph
        self.status_updates: list[str] = []
        self.unsafe_actions = 0
        self.resolved_declared = False
        self.submitted_rca: RCA | None = None

    def execute(self, command: str) -> str:
        parts = command.strip().split(maxsplit=1)
        if not parts:
            return "ERROR: empty command"
        tool = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        handler = self._handlers().get(tool)
        if handler is None:
            return f"ERROR: unknown tool '{tool}'. Available tools: {', '.join(AVAILABLE_TOOLS)}"
        try:
            return handler(args)
        except KeyError as exc:
            return f"ERROR: {exc}"

    def _handlers(self) -> dict[str, Callable[[str], str]]:
        return {
            "kubectl_get_pods": self._get_pods,
            "kubectl_describe_pod": self._describe_pod,
            "kubectl_logs": self._logs,
            "kubectl_top": self._top,
            "promql_query": self._promql,
            "logql_query": self._logql,
            "jaeger_search": self._jaeger,
            "istioctl_proxy_status": self._proxy_status,
            "istioctl_routes": self._routes,
            "curl_service": self._curl,
            "dns_lookup": self._dns,
            "check_deploy_history": self._deploy_history,
            "kubectl_rollout_undo": lambda a: self._fix("kubectl_rollout_undo", a),
            "kubectl_rollout_restart": lambda a: self._fix("kubectl_rollout_restart", a),
            "kubectl_scale": lambda a: self._fix("kubectl_scale", a.split()[0] if a else a),
            "feature_flag_toggle": lambda a: self._fix("feature_flag_toggle", a.split()[0] if a else a),
            "traffic_split_update": lambda a: self._fix("traffic_split_update", a.split()[0] if a else a),
            "kubectl_apply_config": lambda a: self._fix("kubectl_apply_config", a.split()[0] if a else a),
            "post_status_update": self._status,
            "declare_resolved": self._declare,
            "submit_rca": self._submit_rca,
        }

    def _service_arg(self, args: str) -> str:
        return args.split()[0] if args.strip() else self.graph.root_cause_service

    def _get_pods(self, _: str) -> str:
        lines = ["NAME READY STATUS RESTARTS AGE"]
        for service in self.graph.services.values():
            status = "Running" if not service.current_fault else ("CrashLoopBackOff" if service.current_fault == "oom_kill" else "Degraded")
            lines.append(f"{service.name}-7d9c 1/1 {status} {3 if service.current_fault else 0} 42m")
        return "\n".join(lines)

    def _describe_pod(self, args: str) -> str:
        service = self.graph.get(self._service_arg(args))
        return f"Name: {service.name}\nDependencies: {', '.join(service.dependencies) or 'none'}\nFault: {service.current_fault or 'none'}\nConfig: {json.dumps(service.config, sort_keys=True)}"

    def _logs(self, args: str) -> str:
        return json_logs(self.graph, self._service_arg(args))

    def _top(self, args: str) -> str:
        service = self.graph.get(self._service_arg(args))
        return f"NAME CPU% MEMORY% LATENCY_P99_MS ERROR_RATE\n{service.name} {service.cpu:.1f} {service.memory:.1f} {service.latency_p99:.0f} {service.error_rate:.3f}"

    def _promql(self, args: str) -> str:
        service = self._service_arg(args.replace("{", " ").replace("}", " ").replace("service=", " "))
        return prometheus_metrics(self.graph, service if service in self.graph.services else self.graph.root_cause_service)

    def _logql(self, args: str) -> str:
        return self._logs(args)

    def _jaeger(self, args: str) -> str:
        return jaeger_traces(self.graph, self._service_arg(args))

    def _proxy_status(self, _: str) -> str:
        return "\n".join(f"{svc.name} SYNCED {'STALE' if svc.current_fault else 'HEALTHY'}" for svc in self.graph.services.values())

    def _routes(self, args: str) -> str:
        service = self.graph.get(self._service_arg(args))
        return json.dumps({"service": service.name, "routes": service.dependencies, "envoy_flags": "UH UF URX NR" if service.current_fault else ""})

    def _curl(self, args: str) -> str:
        service = self.graph.get(self._service_arg(args))
        if service.error_rate > 0.3 and not service.remediated:
            return "HTTP/1.1 503 Service Unavailable\ncontext deadline exceeded"
        return "HTTP/1.1 200 OK\nok"

    def _dns(self, args: str) -> str:
        service = self.graph.get(self._service_arg(args))
        if service.current_fault == "dns_misconfig":
            return f"{service.name}.svc.cluster.local NXDOMAIN"
        return f"{service.name}.svc.cluster.local 10.43.12.7"

    def _deploy_history(self, args: str) -> str:
        return json.dumps(self.graph.get(self._service_arg(args)).deploy_history, indent=2)

    def _fix(self, tool: str, args: str) -> str:
        service = self._service_arg(args)
        correct = self.graph.mark_remediated(tool, service)
        if correct:
            return f"OK: {tool} applied to {service}; synthetic health checks are green"
        if service in {"postgres-primary", "redis-cache"} and tool in {"kubectl_rollout_restart", "kubectl_rollout_undo"}:
            self.unsafe_actions += 1
        return f"WARN: {tool} applied to {service}, but incident symptoms persist"

    def _status(self, args: str) -> str:
        self.status_updates.append(args)
        return "status update posted"

    def _declare(self, _: str) -> str:
        self.resolved_declared = True
        return "resolution declared; submit_rca is still required"

    def _submit_rca(self, args: str) -> str:
        try:
            self.submitted_rca = RCA.model_validate_json(args)
        except (ValidationError, ValueError) as exc:
            return f"ERROR: invalid RCA JSON: {exc}"
        return "RCA accepted"

