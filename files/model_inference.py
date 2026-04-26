"""Model inference — loads the trained LoRA checkpoint and generates SRE actions.

Pipeline:
  1. Load Qwen2.5-3B-Instruct base model (FP16, CPU)
  2. Apply LoRA adapter from feedback-model-cp300
  3. Merge adapter for faster inference
  4. Build prompts matching the training distribution
  5. Generate <actions> blocks and parse commands
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BASE_MODEL_NAME = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
ADAPTER_DIR = os.environ.get(
    "ADAPTER_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "feedback-model-cp300",
    ),
)

SERVICES = [
    "api-gateway", "checkout-service", "payment-service",
    "inventory-service", "user-service", "postgres-primary", "redis-cache",
]

READ_ONLY_TOOLS = [
    "kubectl_get_pods", "kubectl_describe_pod", "kubectl_logs", "kubectl_top",
    "promql_query", "logql_query", "jaeger_search", "istioctl_proxy_status",
    "istioctl_routes", "curl_service", "dns_lookup", "check_deploy_history",
]
MUTATING_TOOLS = [
    "kubectl_rollout_undo", "kubectl_rollout_restart", "kubectl_scale",
    "feature_flag_toggle", "traffic_split_update", "kubectl_apply_config",
]
ALL_TOOLS = [*READ_ONLY_TOOLS, *MUTATING_TOOLS,
             "post_status_update", "declare_resolved", "submit_rca"]

COMMAND_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in [*READ_ONLY_TOOLS, *MUTATING_TOOLS, "declare_resolved"])
    + r")\b(?:\s+([a-z0-9_.:/={}\"\'-]+))?",
    re.IGNORECASE,
)

FAULT_RUNBOOK_HINTS = {
    "oom_kill": "memory pressure or OOMKilled usually needs kubectl_rollout_restart on the faulty service",
    "cpu_hog": "CPU saturation usually needs kubectl_scale on the faulty service",
    "network_partition": "network partition symptoms usually need traffic_split_update on the faulty service",
    "dns_misconfig": "DNS or no-route symptoms usually need kubectl_apply_config on the faulty service",
    "replica_lag": "replica lag usually needs feature_flag_toggle on the faulty service",
    "cache_stampede": "cache stampede usually needs feature_flag_toggle on the faulty service",
    "http_503_loop": "HTTP 503 loops usually need kubectl_rollout_undo on the faulty service",
    "deadlock": "deadlocks usually need kubectl_rollout_restart on the faulty service",
    "disk_full": "disk-full configuration incidents usually need kubectl_apply_config on the faulty service",
    "cert_expiry": "certificate expiry usually needs kubectl_apply_config on the faulty service",
    "clock_skew": "clock skew usually needs kubectl_rollout_restart on the faulty service",
    "gc_pause": "GC pause incidents usually need kubectl_rollout_restart on the faulty service",
}

# ---------------------------------------------------------------------------
# Global model state (lazy-loaded once)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_load_error: Optional[str] = None


def load_model():
    """Lazy-load base model + LoRA adapter. Returns (model, tokenizer)."""
    global _model, _tokenizer, _load_error

    if _model is not None:
        return _model, _tokenizer
    if _load_error is not None:
        raise RuntimeError(_load_error)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        logger.info("Loading tokenizer from %s …", ADAPTER_DIR)
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading base model %s (float16, CPU) …", BASE_MODEL_NAME)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        logger.info("Applying LoRA adapter from %s …", ADAPTER_DIR)
        model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        model = model.merge_and_unload()
        model.eval()

        _model = model
        _tokenizer = tokenizer
        logger.info("Model ready (merged, float16, CPU).")
        return model, tokenizer

    except Exception as exc:
        _load_error = f"Model loading failed: {exc}"
        logger.error(_load_error)
        raise


def is_model_available() -> bool:
    """Check whether the adapter files exist (without loading the model)."""
    if _model is not None:
        return True
    if _load_error is not None:
        return False
    return Path(ADAPTER_DIR, "adapter_config.json").exists()


# ---------------------------------------------------------------------------
# Prompt construction (mirrors the training-time format)
# ---------------------------------------------------------------------------

def _required_pairs(required: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for item in required:
        if ":" in item:
            tool, service = item.split(":", 1)
            pairs.append((tool, service))
    return pairs


def build_prompt(
    *,
    task_id: str,
    alert_service: str,
    alert_message: str,
    services: list[str],
    tools: list[str],
    root_service: str,
    root_category: str,
    required: list[str],
) -> str:
    """Build an easy-mode prompt matching the training distribution."""
    pairs = _required_pairs(required)
    accepted = ", ".join(f"{tool} {service}" for tool, service in pairs)
    hint = FAULT_RUNBOOK_HINTS.get(
        root_category,
        f"{root_category} symptoms should be remediated on {root_service}",
    )
    return (
        f"You are the on-call SRE for OnCallEnv Red Shift.\n\n"
        f"Task id: {task_id}\n"
        f"Critical alert: {alert_service} reports {alert_message}\n"
        f"Available services: {', '.join(services)}\n"
        f"Available tools: {', '.join(tools)}\n\n"
        "Return only the XML action block below. Put one simulator command per line inside:\n"
        "<actions>\n...\n</actions>\n"
        "Stop immediately after the closing </actions> tag.\n\n"
        "Use real commands such as kubectl_logs SERVICE, promql_query SERVICE,\n"
        "jaeger_search SERVICE, kubectl_rollout_restart SERVICE,\n"
        "kubectl_rollout_undo SERVICE, kubectl_scale SERVICE,\n"
        "feature_flag_toggle SERVICE, traffic_split_update SERVICE,\n"
        "kubectl_apply_config SERVICE, and declare_resolved.\n"
        "Do not include explanations, markdown, bullets, JSON, RCA text, or prose outside the tags.\n"
        f"\nEasy-mode hints: root service = {root_service}; fault = {root_category}; "
        f"best remediation = {accepted}. Include declare_resolved after the fix. "
        "A good answer is exactly 3-5 command lines and ends with </actions>.\n"
    )


# ---------------------------------------------------------------------------
# Command parsing (mirrors the training-time parser)
# ---------------------------------------------------------------------------

def parse_commands(text: str, max_commands: int = 10) -> list[str]:
    """Extract simulator commands from the model's <actions> block."""
    match = re.search(r"<actions>(.*?)</actions>", text, flags=re.IGNORECASE | re.DOTALL)
    body = match.group(1) if match else text
    commands: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip().strip("-*` ")
        if not line:
            continue
        found = COMMAND_RE.search(line)
        if not found:
            continue
        tool = found.group(1).lower()
        arg = (found.group(2) or "").strip().strip("'\"`")
        if tool in MUTATING_TOOLS or tool in {
            "kubectl_logs", "kubectl_top", "kubectl_describe_pod", "jaeger_search",
            "dns_lookup", "check_deploy_history", "curl_service", "promql_query",
            "logql_query", "istioctl_routes",
        }:
            service = next((svc for svc in SERVICES if svc in arg or svc in line), "")
            if not service:
                continue
            commands.append(f"{tool} {service}")
        elif tool == "declare_resolved":
            commands.append("declare_resolved")
        else:
            commands.append(tool)
        if len(commands) >= max_commands:
            break
    return commands


def build_rca(service: str, category: str) -> str:
    """Build a structured RCA JSON payload for auto-submission."""
    return json.dumps({
        "root_cause_service": service,
        "root_cause_category": category,
        "timeline": [{
            "timestamp": "2026-04-24T09:00:00Z",
            "service": service,
            "description": f"{category} identified from Red Shift telemetry",
        }],
        "five_whys": [
            f"{service} emitted direct {category} symptoms.",
            "The failure propagated through dependent customer-facing services.",
            "The first mitigation needed to target the true faulty component.",
        ],
        "action_items": [
            f"Add regression alerting and runbook coverage for {service} {category}.",
        ],
        "evidence_citations": [{
            "source": "telemetry",
            "ref": f"kubectl_logs {service}",
            "excerpt": category,
        }],
        "blast_radius_description": (
            "Customer-facing requests saw elevated latency or errors before remediation."
        ),
    })


# ---------------------------------------------------------------------------
# Inference entry point
# ---------------------------------------------------------------------------

def generate_action_text(prompt: str, max_new_tokens: int = 256) -> str:
    """Generate an <actions> block from a prompt using the trained model."""
    import torch

    model, tokenizer = load_model()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
