"""LLM-based Attacker (Challenger) for SPICE-style self-play.

The Attacker reads a parent ScenarioSpec and outputs discrete `set_field`
actions to mutate it into a harder scenario for the Defender.  Its reward
is a Gaussian centered on high variance of the Defender's (normalised)
continuous rewards — scenarios at the exact learning frontier score highest.
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics
from typing import Any

import yaml

from oncallenv.core.types import ScenarioSpec


# ---------------------------------------------------------------------------
# Valid field values (mirrors ScenarioSpec + mutator.py constants)
# ---------------------------------------------------------------------------

ATTACKER_VALID_FIELDS: dict[str, list[Any]] = {
    "topology": ["simple_fanout", "deep_chain", "mesh", "star", "bipartite", "diamond"],
    "fault_primary": [
        "oom_kill", "cpu_hog", "network_partition", "dns_misconfig",
        "replica_lag", "cache_stampede", "http_503_loop", "deadlock",
        "disk_full", "cert_expiry", "clock_skew", "gc_pause",
    ],
    "fault_secondary": [
        "none", "oom_kill", "cpu_hog", "network_partition", "dns_misconfig",
        "replica_lag", "cache_stampede", "http_503_loop", "deadlock",
        "disk_full", "cert_expiry", "clock_skew", "gc_pause",
    ],
    "inject_service": [
        "api-gateway", "checkout-service", "payment-service",
        "inventory-service", "user-service", "postgres-primary", "redis-cache",
    ],
    "latency_ms": [0, 50, 200, 500, 1000, 2000, 3000, 5000],
    "blast_radius": [round(x * 0.1, 1) for x in range(11)],  # 0.0 .. 1.0
    "metric_noise": [round(x * 0.1, 1) for x in range(11)],
    "red_herring": [
        "none", "unrelated_alert", "stale_deploy_notice",
        "innocent_config_change", "flapping_canary", "false_correlation",
        "old_anomaly", "unused_service_spike",
    ],
    "deploy_window": ["none", "recent_deploy", "flag_flip", "config_change"],
    "schema_drift": [
        "none", "rename_metric", "swap_units", "rotate_creds",
        "new_required_field", "field_type_change", "endpoint_version_bump",
    ],
    "max_steps": list(range(10, 31)),  # 10 .. 30
}

SET_FIELD_RE = re.compile(
    r"set_field\s+(\w+)\s+(.+)", re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_FIELD_DOC = "\n".join(
    f"  {field}: {vals}" for field, vals in ATTACKER_VALID_FIELDS.items()
)

ATTACKER_SYSTEM = (
    "You are the Chaos Attacker for an SRE training environment. "
    "Your goal is to mutate incident scenarios so they are maximally "
    "challenging for the Defender — not trivially easy, not impossibly hard. "
    "Scenarios at the edge of the Defender's capability earn the highest reward."
)


def build_attacker_prompt(parent_spec: ScenarioSpec) -> str:
    """Return the user prompt for the Attacker role."""
    parent_yaml = yaml.safe_dump(parent_spec.model_dump(), sort_keys=False).strip()
    return f"""{ATTACKER_SYSTEM}

Below is the current parent scenario YAML:

```yaml
{parent_yaml}
```

You are a strict code execution agent. You may mutate the scenario by issuing one or more `set_field` commands.
Each command changes a single field to a new valid value.

Valid fields and their allowed values:
{_FIELD_DOC}

STRICT RULES:
1. You MUST output ONLY the commands inside <actions> and </actions> tags.
2. DO NOT write ANY conversational text or markdown blocks outside the tags.
3. One command per line: `set_field FIELD_NAME VALUE`.
4. You may issue 1 to 5 set_field commands.

Example Output:
<actions>
set_field fault_primary deadlock
set_field topology mesh
set_field blast_radius 0.8
</actions>
"""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _coerce_value(field: str, raw: str) -> Any:
    """Coerce a raw string value to the type expected by ScenarioSpec."""
    raw = raw.strip().strip("'\"")
    valid = ATTACKER_VALID_FIELDS.get(field)
    if valid is None:
        return None  # unknown field

    # Handle "none" → None for optional fields
    if raw.lower() == "none" and field in (
        "fault_secondary", "red_herring", "deploy_window", "schema_drift",
    ):
        return None

    # Numeric fields
    if field in ("latency_ms", "max_steps"):
        try:
            val = int(raw)
            return val if val in valid else None
        except ValueError:
            return None
    if field in ("blast_radius", "metric_noise"):
        try:
            val = round(float(raw), 1)
            return val if 0.0 <= val <= 1.0 else None
        except ValueError:
            return None

    # String enum fields
    if raw in valid:
        return raw
    return None


def parse_attacker_actions(
    text: str,
    parent_spec: ScenarioSpec,
    generation: int = 0,
) -> tuple[ScenarioSpec | None, bool, list[str]]:
    """Parse LLM output into a mutated ScenarioSpec.

    Returns (spec_or_None, is_valid, list_of_applied_actions).
    """
    # Extract ALL <actions> blocks - resilient to clipping
    matches = re.findall(r"<actions>(.*?)(?:</actions>|$)", text, flags=re.IGNORECASE | re.DOTALL)
    bodies = matches if matches else [text]

    data = parent_spec.model_dump()
    applied: list[str] = []

    for body in bodies:
        for line in body.splitlines():
            line = line.strip().strip("-* ")
            if not line:
                continue
            m = SET_FIELD_RE.match(line)
            if not m:
                continue
            field, raw_value = m.group(1), m.group(2)
            if field not in ATTACKER_VALID_FIELDS:
                continue
            value = _coerce_value(field, raw_value)
            if value is None and field not in (
                "fault_secondary", "red_herring", "deploy_window", "schema_drift",
            ):
                continue  # skip invalid non-optional values
            data[field] = value
            applied.append(f"{field}={value}")

    if not applied:
        return None, False, applied

    # Generate unique task_id
    fingerprint = hashlib.sha1(repr(sorted(data.items())).encode()).hexdigest()[:10]
    data["task_id"] = f"spice_{generation:04d}_{fingerprint}"
    data["seed"] = hash(data["task_id"]) % 2_000_000_000

    try:
        spec = ScenarioSpec.model_validate(data)
        return spec, True, applied
    except Exception:
        return None, False, applied


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

DEFENDER_REWARD_MIN = 0.0
DEFENDER_REWARD_MAX = 1.0
DEFENDER_REWARD_RANGE = 1.0


def normalize_defender_reward(r: float) -> float:
    """Clip defender reward to [0, 1]."""
    return max(0.0, min(1.0, r))


def attacker_reward(
    defender_rewards: list[float],
    penalty: float = -0.1,
    threshold: float = 0.3,
    tau: float = 0.1,
) -> float:
    """Gaussian binary-pass-rate reward for the Attacker.

    Peaks at 1.0 when the pass rate p is 0.5 (maximum uncertainty/learning frontier).
    """
    if not defender_rewards:
        return penalty

    # Binarize based on threshold
    passes = [1.0 if r >= threshold else 0.0 for r in defender_rewards]
    p = sum(passes) / len(passes)

    # Gaussian centered at p=0.5
    reward = math.exp(-((p - 0.5) ** 2) / (2 * tau))
    return reward
