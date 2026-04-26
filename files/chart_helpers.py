"""Chart helpers for OnCallEnv Red Shift Space.

Builds pandas DataFrames consumed by gr.LinePlot / gr.BarPlot. Native Gradio
plots are used over matplotlib because they're interactive, theme-respecting,
and don't blow out the memory budget on free-tier Spaces.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Single-episode plots
# ---------------------------------------------------------------------------

def build_reward_progression_df(log: list[dict]) -> pd.DataFrame:
    """Cumulative + per-step reward over episode steps."""
    if not log:
        return pd.DataFrame(columns=["step", "reward", "metric"])

    rows = []
    cumulative = 0.0
    for entry in log:
        step = entry.get("step", 0)
        r = entry.get("reward", 0.0) or 0.0
        cumulative += r
        rows.append({"step": step, "reward": r, "metric": "step"})
        rows.append({"step": step, "reward": cumulative, "metric": "cumulative"})

    return pd.DataFrame(rows)


def build_action_distribution_df(log: list[dict]) -> pd.DataFrame:
    """Counts of action types — to reveal investigate vs remediate balance."""
    if not log:
        return pd.DataFrame(columns=["action", "count", "category"])

    from replay_html import _categorize  # reuse classification

    counts: dict[str, dict] = {}
    for entry in log:
        action = (entry.get("action") or "").split()[0] if entry.get("action") else "unknown"
        if not action:
            continue
        cat = _categorize(action)
        if action not in counts:
            counts[action] = {"action": action, "count": 0, "category": cat}
        counts[action]["count"] += 1

    return pd.DataFrame(list(counts.values())).sort_values("count", ascending=False)


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def build_comparison_df(
    baseline_rewards: dict[str, float],
    trained_rewards: dict[str, float],
) -> pd.DataFrame:
    """Reward components side-by-side for the comparison bar chart."""
    rows = []
    for component in ["recovery", "rca_quality", "blast_radius", "safety", "total"]:
        rows.append({
            "component": component.replace("_", " ").title(),
            "score": baseline_rewards.get(component, 0.0),
            "model": "Baseline",
        })
        rows.append({
            "component": component.replace("_", " ").title(),
            "score": trained_rewards.get(component, 0.0),
            "model": "Trained",
        })
    return pd.DataFrame(rows)


def build_dual_progression_df(
    baseline_log: list[dict],
    trained_log: list[dict],
) -> pd.DataFrame:
    """Two cumulative-reward curves on shared axes for visual A/B."""
    rows = []
    for label, log in [("Baseline", baseline_log), ("Trained", trained_log)]:
        cumulative = 0.0
        for entry in log:
            cumulative += (entry.get("reward", 0.0) or 0.0)
            rows.append({
                "step": entry.get("step", 0),
                "reward": cumulative,
                "model": label,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Method ablation (the headline plot for the About / Methodology tab)
# ---------------------------------------------------------------------------

def build_method_ablation_df() -> pd.DataFrame:
    """Static comparison of the three curriculum methods from your runs.

    These numbers are placeholders matching your reported results; replace
    them with the live numbers from your held-out eval before submission.
    """
    rows = [
        # Method 1 — No feedback (static curriculum)
        {"method": "M1: No Feedback", "step": 0,   "reward": 0.638, "kind": "mean"},
        {"method": "M1: No Feedback", "step": 200, "reward": 0.872, "kind": "mean"},

        # Method 2 — ACCEL-style feedback
        {"method": "M2: ACCEL Feedback", "step": 0,   "reward": 0.638, "kind": "mean"},
        {"method": "M2: ACCEL Feedback", "step": 100, "reward": 0.781, "kind": "mean"},
        {"method": "M2: ACCEL Feedback", "step": 200, "reward": 0.857, "kind": "mean"},
        {"method": "M2: ACCEL Feedback", "step": 300, "reward": 0.905, "kind": "mean"},

        # Method 3 — SPICE-style LLM self-play (defender side)
        {"method": "M3: SPICE Self-Play", "step": 0,   "reward": 0.638, "kind": "mean"},
        {"method": "M3: SPICE Self-Play", "step": 100, "reward": 0.420, "kind": "mean"},
        {"method": "M3: SPICE Self-Play", "step": 200, "reward": 0.350, "kind": "mean"},
        {"method": "M3: SPICE Self-Play", "step": 300, "reward": 0.310, "kind": "mean"},

        # Reference lines
        {"method": "Scripted Ceiling", "step": 0,   "reward": 0.921, "kind": "ref"},
        {"method": "Scripted Ceiling", "step": 300, "reward": 0.921, "kind": "ref"},
        {"method": "Random Floor",     "step": 0,   "reward": 0.270, "kind": "ref"},
        {"method": "Random Floor",     "step": 300, "reward": 0.270, "kind": "ref"},
    ]
    return pd.DataFrame(rows)


def build_held_out_comparison_df() -> pd.DataFrame:
    """Bar chart for held-out evaluation across methods."""
    rows = [
        {"method": "Random",          "score": 0.270},
        {"method": "Untrained 3B",    "score": 0.638},
        {"method": "M1: No Feedback", "score": 0.872},
        {"method": "M2: ACCEL",       "score": 0.905},
        {"method": "M3: SPICE",       "score": 0.310},
        {"method": "Scripted",        "score": 0.921},
    ]
    return pd.DataFrame(rows)
