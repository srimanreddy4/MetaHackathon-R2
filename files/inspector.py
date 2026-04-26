"""Environment Inspector — exposes simulator state for the third tab.

Shows judges what the agent is reasoning over: the service graph, current
metrics, recent telemetry, and active fault state.
"""

from __future__ import annotations

import pandas as pd

from demo_runner import _try_real_env, get_scenario_info


# Canonical service catalog — used as fallback if real env isn't available
DEMO_SERVICES = [
    {"name": "api-gateway",        "type": "edge",       "replicas": 3, "deps": "checkout-service, user-service"},
    {"name": "checkout-service",   "type": "domain",     "replicas": 2, "deps": "payment-service, inventory-service"},
    {"name": "payment-service",    "type": "domain",     "replicas": 2, "deps": "postgres-primary"},
    {"name": "inventory-service",  "type": "domain",     "replicas": 2, "deps": "postgres-primary, redis-cache"},
    {"name": "user-service",       "type": "domain",     "replicas": 2, "deps": "postgres-primary"},
    {"name": "postgres-primary",   "type": "stateful",   "replicas": 1, "deps": "—"},
    {"name": "redis-cache",        "type": "stateful",   "replicas": 1, "deps": "—"},
]


def get_services_df(scenario_id: str = "") -> pd.DataFrame:
    """Service inventory with current health hint based on scenario."""
    info = get_scenario_info(scenario_id) if scenario_id else {}
    fault = info.get("fault_label", "")

    rows = []
    for s in DEMO_SERVICES:
        # Naive: mark service as degraded if name appears in fault label
        is_affected = s["name"] in fault.lower() if fault else False
        rows.append({
            "Service": s["name"],
            "Type": s["type"],
            "Replicas": s["replicas"],
            "Dependencies": s["deps"],
            "Status": "🔴 degraded" if is_affected else "🟢 healthy",
        })
    return pd.DataFrame(rows)


def get_alerts_df(scenario_id: str) -> pd.DataFrame:
    """Active alerts for the scenario."""
    info = get_scenario_info(scenario_id)
    alerts = info.get("alerts", [])
    if not alerts:
        return pd.DataFrame(columns=["Severity", "Service", "Message"])

    rows = []
    for a in alerts:
        sev = a.get("severity", "info")
        sev_icon = {"critical": "🔴 critical", "warning": "🟡 warning", "info": "🔵 info"}.get(sev, sev)
        rows.append({
            "Severity": sev_icon,
            "Service": a.get("service", "—"),
            "Message": a.get("message", ""),
        })
    return pd.DataFrame(rows)


def get_fault_catalog_df() -> pd.DataFrame:
    """The 12 fault families our env supports — useful overview for judges."""
    rows = [
        ("oom_kill",          "Memory",   "Container OOMKilled, exit code 137",       "rollout_undo / scale_memory"),
        ("cpu_hog",           "Compute",  "Sustained CPU saturation",                  "scale / restart"),
        ("network_partition", "Network",  "Service-to-service connectivity loss",      "restart / route_update"),
        ("dns_misconfig",     "Network",  "Service name resolution failures",          "config_apply"),
        ("replica_lag",       "Data",     "Postgres replica behind primary",           "investigate primary load"),
        ("cache_stampede",    "Cache",    "Mass cache miss → upstream overload",        "feature_flag_toggle"),
        ("http_503_loop",     "Service",  "Circuit breaker trip cascading 503s",       "rollout_undo"),
        ("deadlock",          "Data",     "DB transaction deadlock loop",              "restart / config_apply"),
        ("disk_full",         "Storage",  "Persistent volume at capacity",             "scale storage"),
        ("cert_expiry",       "Security", "x509 expired in TLS handshake",             "config_apply (cert rotation)"),
        ("clock_skew",        "Infra",    "Node clocks drifted, JWT validation fails", "investigate node sync"),
        ("gc_pause",          "JVM",      "Long stop-the-world GC pauses",             "rollout_undo / config tune"),
    ]
    return pd.DataFrame(rows, columns=[
        "Fault", "Domain", "Symptom", "Typical Remediation",
    ])


def get_action_surface_df() -> pd.DataFrame:
    """The agent's tool surface — useful for showing scope to judges."""
    rows = [
        # Read-only
        ("kubectl_get_pods",        "READ",     "List pods across the cluster"),
        ("kubectl_describe_pod",    "READ",     "Pod details: events, mounts, status"),
        ("kubectl_logs",            "READ",     "Recent log lines from a service"),
        ("kubectl_top",             "READ",     "Live CPU/memory usage"),
        ("promql_query",            "READ",     "Metric query against Prometheus"),
        ("logql_query",             "READ",     "Structured log query against Loki"),
        ("jaeger_search",           "READ",     "Distributed trace search"),
        ("istioctl_proxy_status",   "READ",     "Sidecar proxy sync state"),
        ("istioctl_routes",         "READ",     "Effective traffic routes"),
        ("curl_service",            "READ",     "Probe an endpoint directly"),
        ("dns_lookup",              "READ",     "Resolve service hostname"),
        ("check_deploy_history",    "READ",     "Recent deploys and rollouts"),
        # Mutating
        ("kubectl_rollout_undo",    "MUTATE",   "Roll back to previous revision"),
        ("kubectl_rollout_restart", "MUTATE",   "Restart all pods of a deployment"),
        ("kubectl_scale",           "MUTATE",   "Change replica count"),
        ("feature_flag_toggle",     "MUTATE",   "Flip a feature flag"),
        ("traffic_split_update",    "MUTATE",   "Adjust traffic weights"),
        ("kubectl_apply_config",    "MUTATE",   "Apply a config change"),
        # Lifecycle
        ("post_status_update",      "TERMINAL", "Status page communication"),
        ("declare_resolved",        "TERMINAL", "Mark incident resolved (verified)"),
        ("submit_rca",              "TERMINAL", "Structured root-cause analysis"),
    ]
    return pd.DataFrame(rows, columns=["Tool", "Category", "Purpose"])


def get_env_config_html(scenario_id: str = "") -> str:
    """Configuration summary card for the inspector sidebar."""
    info = get_scenario_info(scenario_id) if scenario_id else {}
    fault = info.get("fault_label", "(no scenario selected)")
    n_alerts = len(info.get("alerts", []))

    real_status = "live" if _try_real_env() is not None else "demo (canned trajectories)"

    return f"""
    <div style="
      font-family: 'Inter', sans-serif;
      padding: 14px;
      background: #131a2e;
      border: 1px solid #1f2940;
      border-radius: 6px;
    ">
      <div style="
        font-size: 11px;
        font-weight: 600;
        color: #7aa7ff;
        letter-spacing: 0.06em;
        margin-bottom: 10px;
      ">ENVIRONMENT CONFIG</div>

      <div style="display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; font-size: 12px;">
        <span style="color:#7d8ba1;">Scenario</span>
        <span style="font-family:'JetBrains Mono',monospace; color:#d6deeb;">{scenario_id or '—'}</span>

        <span style="color:#7d8ba1;">Fault</span>
        <span style="color:#d6deeb;">{fault}</span>

        <span style="color:#7d8ba1;">Active alerts</span>
        <span style="color:#d6deeb;">{n_alerts}</span>

        <span style="color:#7d8ba1;">Services</span>
        <span style="color:#d6deeb;">{len(DEMO_SERVICES)} (7-service e-commerce graph)</span>

        <span style="color:#7d8ba1;">Faults supported</span>
        <span style="color:#d6deeb;">12 families</span>

        <span style="color:#7d8ba1;">Backend mode</span>
        <span style="color:{'#10b981' if 'live' in real_status else '#f59e0b'};">{real_status}</span>
      </div>
    </div>
    """
