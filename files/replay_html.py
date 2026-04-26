"""Replay HTML renderer for OnCallEnv Red Shift episodes.

Renders an agent episode as a vertically-scrolling incident timeline with:
- Phase headers (Triage / Investigation / Diagnosis / Remediation / Resolution)
- Per-step action cards color-coded by tool category
- Reward breakdown bars at the end

This is the storytelling centerpiece of the Space — judges spend most of their
time reading these replays. Make every visual decision serve clarity.
"""

from __future__ import annotations

import html as _html
from typing import Any


def _esc(text: Any) -> str:
    return _html.escape(str(text))


# --- Categorize actions for color coding -----------------------------------

READ_ACTIONS = {
    "kubectl_get_pods", "kubectl_describe_pod", "kubectl_logs", "kubectl_top",
    "promql_query", "logql_query", "jaeger_search",
    "istioctl_proxy_status", "istioctl_routes",
    "curl_service", "dns_lookup", "check_deploy_history",
}

MUTATE_ACTIONS = {
    "kubectl_rollout_undo", "kubectl_rollout_restart", "kubectl_scale",
    "feature_flag_toggle", "traffic_split_update", "kubectl_apply_config",
}

TERMINAL_ACTIONS = {"declare_resolved", "submit_rca", "post_status_update"}


def _categorize(action_name: str) -> str:
    if action_name in READ_ACTIONS:
        return "read"
    if action_name in MUTATE_ACTIONS:
        return "mutate"
    if action_name in TERMINAL_ACTIONS:
        return "terminal"
    return "unknown"


CATEGORY_STYLE = {
    "read": {
        "color": "#7aa7ff", "bg": "rgba(122, 167, 255, 0.06)",
        "icon": "▷", "label": "INVESTIGATE",
    },
    "mutate": {
        "color": "#f59e0b", "bg": "rgba(245, 158, 11, 0.08)",
        "icon": "✎", "label": "REMEDIATE",
    },
    "terminal": {
        "color": "#10b981", "bg": "rgba(16, 185, 129, 0.08)",
        "icon": "✓", "label": "RESOLVE",
    },
    "unknown": {
        "color": "#7d8ba1", "bg": "rgba(125, 139, 161, 0.05)",
        "icon": "·", "label": "ACTION",
    },
}


# --- Episode phase inference ----------------------------------------------

def _infer_phase(step_idx: int, total_steps: int, action: str) -> str:
    """Infer high-level phase from position and action category."""
    cat = _categorize(action)
    if cat == "terminal":
        return "RESOLUTION"
    if cat == "mutate":
        return "REMEDIATION"
    if step_idx <= 1:
        return "TRIAGE"
    if step_idx <= total_steps // 2:
        return "INVESTIGATION"
    return "DIAGNOSIS"


PHASE_STYLE = {
    "TRIAGE": {"color": "#ef4444", "icon": "⚠", "desc": "Alert received — assessing scope"},
    "INVESTIGATION": {"color": "#7aa7ff", "icon": "◇", "desc": "Querying telemetry, narrowing hypothesis"},
    "DIAGNOSIS": {"color": "#8b5cf6", "icon": "◆", "desc": "Root cause hypothesis converging"},
    "REMEDIATION": {"color": "#f59e0b", "icon": "✎", "desc": "Applying corrective action"},
    "RESOLUTION": {"color": "#10b981", "icon": "✓", "desc": "Validating recovery, writing RCA"},
}


# ---------------------------------------------------------------------------

def format_replay_html(
    log: list[dict],
    rewards: dict[str, float],
    scenario_id: str = "",
    fault_label: str = "",
) -> str:
    """Render an agent episode as an SRE-styled incident timeline.

    Args:
        log: list of step dicts with keys {step, action, result, reward, error}
        rewards: dict of reward components {recovery, rca_quality, blast_radius, safety, total}
        scenario_id: optional scenario identifier for the header
        fault_label: optional human-readable fault description
    """
    if not log:
        return _empty_state()

    total_steps = len(log)

    parts: list[str] = []
    parts.append(_outer_open(scenario_id, fault_label))

    current_phase = None
    for idx, entry in enumerate(log):
        action = entry.get("action", "")
        action_name = action.split()[0] if action else "(empty)"
        phase = _infer_phase(idx, total_steps, action_name)

        if phase != current_phase:
            current_phase = phase
            parts.append(_phase_header(phase, idx + 1))

        parts.append(_action_card(idx + 1, entry, action_name))

    parts.append(_reward_panel(rewards))
    parts.append(_outer_close())

    return "".join(parts)


def _empty_state() -> str:
    return """
    <div style="
      padding: 48px 24px;
      text-align: center;
      color: #7d8ba1;
      font-family: 'Inter', sans-serif;
      background: #131a2e;
      border: 1px dashed #1f2940;
      border-radius: 8px;
    ">
      <div style="font-size: 14px; margin-bottom: 8px;">No episode loaded</div>
      <div style="font-size: 12px; color: #5b6577;">
        Configure a scenario and click <b>Run Episode</b> to begin.
      </div>
    </div>
    """


def _outer_open(scenario_id: str, fault_label: str) -> str:
    meta_line = ""
    if scenario_id or fault_label:
        meta_parts = []
        if scenario_id:
            meta_parts.append(f"<span style='color:#7aa7ff;'>scenario</span> <code>{_esc(scenario_id)}</code>")
        if fault_label:
            meta_parts.append(f"<span style='color:#7aa7ff;'>fault</span> <code>{_esc(fault_label)}</code>")
        meta_line = f"""
        <div style="
          margin-top: 6px;
          font-size: 11px;
          color: #7d8ba1;
          font-family: 'JetBrains Mono', monospace;
          letter-spacing: 0.02em;
        ">{' &middot; '.join(meta_parts)}</div>
        """

    return f"""
    <div style="
      font-family: 'Inter', sans-serif;
      background: #0f1626;
      color: #d6deeb;
      padding: 20px;
      border-radius: 8px;
      border: 1px solid #1f2940;
    ">
      <div style="
        padding-bottom: 12px;
        margin-bottom: 16px;
        border-bottom: 1px solid #1f2940;
      ">
        <div style="
          font-size: 13px;
          font-weight: 600;
          color: #e6edf7;
          letter-spacing: 0.02em;
        ">INCIDENT REPLAY</div>
        {meta_line}
      </div>
    """


def _outer_close() -> str:
    return "</div>"


def _phase_header(phase: str, step_num: int) -> str:
    style = PHASE_STYLE[phase]
    return f"""
    <div style="
      margin: 18px 0 10px 0;
      padding: 8px 12px;
      background: linear-gradient(90deg,
        {style['color']}1a 0%,
        {style['color']}05 100%);
      border-left: 3px solid {style['color']};
      border-radius: 0 6px 6px 0;
      display: flex;
      align-items: center;
      gap: 12px;
    ">
      <span style="
        font-size: 14px;
        color: {style['color']};
      ">{style['icon']}</span>
      <div style="flex: 1;">
        <div style="
          font-size: 11px;
          font-weight: 600;
          color: {style['color']};
          letter-spacing: 0.06em;
        ">{phase}</div>
        <div style="
          font-size: 11px;
          color: #7d8ba1;
          margin-top: 2px;
        ">{style['desc']}</div>
      </div>
      <div style="
        font-size: 10px;
        color: #5b6577;
        font-family: 'JetBrains Mono', monospace;
      ">step {step_num:02d}+</div>
    </div>
    """


def _action_card(step_num: int, entry: dict, action_name: str) -> str:
    cat = _categorize(action_name)
    style = CATEGORY_STYLE[cat]

    raw_action = entry.get("action", "")
    result = entry.get("result", "")
    is_error = entry.get("error", False)
    step_reward = entry.get("reward", None)

    # Truncate long results, but keep enough to show the agent's work
    if isinstance(result, str) and len(result) > 280:
        result_display = result[:280] + "…"
    else:
        result_display = str(result)

    error_badge = ""
    if is_error:
        error_badge = """
        <span style="
          padding: 1px 6px;
          background: rgba(239, 68, 68, 0.12);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 3px;
          font-size: 9px;
          color: #ef4444;
          font-weight: 600;
          letter-spacing: 0.05em;
          margin-left: 6px;
        ">ERROR</span>
        """

    reward_badge = ""
    if step_reward is not None:
        rcolor = "#10b981" if step_reward > 0 else ("#ef4444" if step_reward < 0 else "#7d8ba1")
        reward_badge = f"""
        <span style="
          padding: 1px 6px;
          background: rgba(125, 139, 161, 0.08);
          border-radius: 3px;
          font-size: 9px;
          color: {rcolor};
          font-family: 'JetBrains Mono', monospace;
          font-weight: 600;
          margin-left: 6px;
        ">{step_reward:+.2f}</span>
        """

    return f"""
    <div style="
      margin: 6px 0;
      padding: 10px 12px;
      background: {style['bg']};
      border-left: 2px solid {style['color']};
      border-radius: 0 4px 4px 0;
      display: flex;
      align-items: flex-start;
      gap: 12px;
    ">
      <div style="
        min-width: 32px;
        padding-top: 1px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #5b6577;
      ">{step_num:02d}</div>

      <div style="flex: 1; min-width: 0;">
        <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 4px;">
          <span style="
            color: {style['color']};
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.06em;
          ">{style['label']}</span>
          <span style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: #e6edf7;
            font-weight: 500;
          ">{_esc(raw_action)}</span>
          {error_badge}{reward_badge}
        </div>

        <div style="
          margin-top: 6px;
          padding: 6px 10px;
          background: #0b1020;
          border: 1px solid #1f2940;
          border-radius: 4px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
          color: #a8b3c7;
          white-space: pre-wrap;
          word-break: break-word;
          line-height: 1.5;
          max-height: 180px;
          overflow-y: auto;
        ">{_esc(result_display) if result_display else '<span style="color:#5b6577;">(no output)</span>'}</div>
      </div>
    </div>
    """


def _reward_panel(rewards: dict[str, float]) -> str:
    components = [
        ("Recovery",     rewards.get("recovery", 0.0),     0.35, "#10b981"),
        ("RCA Quality",  rewards.get("rca_quality", 0.0),  0.30, "#7aa7ff"),
        ("Blast Radius", rewards.get("blast_radius", 0.0), 0.25, "#f59e0b"),
        ("Safety",       rewards.get("safety", 0.0),       0.10, "#8b5cf6"),
    ]
    total = rewards.get("total", sum(c[1] * c[2] for c in components))

    if total >= 0.75:
        total_color = "#10b981"
        verdict = "RESOLVED"
    elif total >= 0.40:
        total_color = "#f59e0b"
        verdict = "PARTIAL"
    else:
        total_color = "#ef4444"
        verdict = "FAILED"

    bars: list[str] = []
    for name, value, weight, color in components:
        pct = max(0, min(value * 100, 100))
        bars.append(f"""
        <div style="margin-bottom: 10px;">
          <div style="
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 4px;
            font-size: 11px;
          ">
            <span style="color: {color}; font-weight: 600;">{name}</span>
            <span style="
              font-family: 'JetBrains Mono', monospace;
              color: #a8b3c7;
            ">{value:.3f} <span style="color:#5b6577;">× {weight:.2f}</span></span>
          </div>
          <div style="
            height: 4px;
            background: #1f2940;
            border-radius: 2px;
            overflow: hidden;
          ">
            <div style="
              height: 100%;
              width: {pct:.1f}%;
              background: {color};
              border-radius: 2px;
            "></div>
          </div>
        </div>
        """)

    return f"""
    <div style="
      margin-top: 24px;
      padding: 16px;
      background: #131a2e;
      border: 1px solid #1f2940;
      border-radius: 6px;
    ">
      <div style="
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid #1f2940;
      ">
        <div style="
          font-size: 12px;
          font-weight: 600;
          color: #e6edf7;
          letter-spacing: 0.04em;
        ">REWARD BREAKDOWN</div>
        <div style="display: flex; align-items: baseline; gap: 8px;">
          <span style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 22px;
            font-weight: 700;
            color: {total_color};
          ">{total:.3f}</span>
          <span style="
            padding: 2px 8px;
            background: {total_color}1a;
            border: 1px solid {total_color}66;
            border-radius: 3px;
            font-size: 10px;
            color: {total_color};
            font-weight: 600;
            letter-spacing: 0.06em;
          ">{verdict}</span>
        </div>
      </div>
      {''.join(bars)}
    </div>
    """


# --- Compact verdict / scores HTML for sidebars ----------------------------

def format_scores_html(rewards: dict[str, float]) -> str:
    """Compact reward summary for the sidebar."""
    total = rewards.get("total", 0.0)
    if total >= 0.75:
        color, label = "#10b981", "RESOLVED"
    elif total >= 0.40:
        color, label = "#f59e0b", "PARTIAL"
    else:
        color, label = "#ef4444", "FAILED"

    components = [
        ("Recovery", rewards.get("recovery", 0.0)),
        ("RCA", rewards.get("rca_quality", 0.0)),
        ("Blast", rewards.get("blast_radius", 0.0)),
        ("Safety", rewards.get("safety", 0.0)),
    ]
    rows = "".join(
        f"""
        <div style="
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          padding: 3px 0;
          color: #a8b3c7;
        ">
          <span>{name}</span>
          <span style="font-family: 'JetBrains Mono', monospace; color: #d6deeb;">{val:.3f}</span>
        </div>
        """
        for name, val in components
    )

    return f"""
    <div style="font-family: 'Inter', sans-serif;">
      <div style="
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        margin-bottom: 10px;
      ">
        <span style="
          font-family: 'JetBrains Mono', monospace;
          font-size: 28px;
          font-weight: 700;
          color: {color};
        ">{total:.3f}</span>
        <span style="
          padding: 2px 8px;
          background: {color}1a;
          border: 1px solid {color}66;
          border-radius: 3px;
          font-size: 10px;
          color: {color};
          font-weight: 600;
          letter-spacing: 0.06em;
        ">{label}</span>
      </div>
      <div style="border-top: 1px solid #1f2940; padding-top: 8px;">{rows}</div>
    </div>
    """


def format_comparison_verdict_html(
    baseline_rewards: dict[str, float],
    trained_rewards: dict[str, float],
) -> str:
    """Side-by-side scoreboard for the Baseline vs Trained tab."""
    b = baseline_rewards.get("total", 0.0)
    t = trained_rewards.get("total", 0.0)
    delta = t - b
    delta_pct = (delta / max(b, 0.01)) * 100 if b > 0 else 0

    delta_color = "#10b981" if delta > 0 else ("#ef4444" if delta < 0 else "#7d8ba1")
    delta_arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")

    return f"""
    <div style="
      font-family: 'Inter', sans-serif;
      padding: 12px;
      background: #131a2e;
      border: 1px solid #1f2940;
      border-radius: 6px;
    ">
      <div style="
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        gap: 16px;
        align-items: center;
      ">
        <div style="text-align: center;">
          <div style="font-size: 10px; color: #7d8ba1; letter-spacing: 0.05em;">BASELINE</div>
          <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 24px;
            font-weight: 700;
            color: #d6deeb;
            margin-top: 2px;
          ">{b:.3f}</div>
        </div>

        <div style="
          text-align: center;
          padding: 0 8px;
          border-left: 1px solid #1f2940;
          border-right: 1px solid #1f2940;
        ">
          <div style="
            font-size: 10px;
            color: {delta_color};
            letter-spacing: 0.05em;
            font-weight: 600;
          ">{delta_arrow} Δ</div>
          <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            font-weight: 700;
            color: {delta_color};
            margin-top: 2px;
          ">{delta:+.3f}</div>
          <div style="
            font-size: 10px;
            color: {delta_color};
            margin-top: 1px;
          ">{delta_pct:+.1f}%</div>
        </div>

        <div style="text-align: center;">
          <div style="font-size: 10px; color: #7aa7ff; letter-spacing: 0.05em;">TRAINED</div>
          <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 24px;
            font-weight: 700;
            color: #7aa7ff;
            margin-top: 2px;
          ">{t:.3f}</div>
        </div>
      </div>
    </div>
    """
