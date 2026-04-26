"""OnCallEnv: Red Shift — Gradio Space app.

Four tabs:
  1. Run Episode      — single rollout with rich replay + per-step plots
  2. Baseline vs Trained — A/B comparison on identical incident
  3. Environment      — service inventory, fault catalog, action surface
  4. About            — methodology, results, citations

Pre-computed (non-streaming) replays — handler runs the full episode then
returns the rendered HTML in one shot. Stable, predictable, demo-safe.
"""

from __future__ import annotations

import gradio as gr

from ops_theme import OpsRoomTheme, CUSTOM_CSS, HEADER_HTML
from replay_html import (
    format_replay_html,
    format_scores_html,
    format_comparison_verdict_html,
)
from chart_helpers import (
    build_reward_progression_df,
    build_action_distribution_df,
    build_comparison_df,
    build_dual_progression_df,
    build_method_ablation_df,
    build_held_out_comparison_df,
)
from demo_runner import (
    list_scenarios,
    get_scenario_info,
    run_episode,
    run_comparison,
    is_real_env_available,
)
from inspector import (
    get_services_df,
    get_alerts_df,
    get_fault_catalog_df,
    get_action_surface_df,
    get_env_config_html,
)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handler_run_episode(scenario_id: str, trained: bool, seed: int):
    """Handler for Tab 1."""
    scenario_id = scenario_id or list_scenarios()[0]
    log, rewards = run_episode(scenario_id, trained=bool(trained), seed=int(seed))

    info = get_scenario_info(scenario_id)
    replay = format_replay_html(
        log, rewards,
        scenario_id=scenario_id,
        fault_label=info.get("fault_label", ""),
    )
    scores = format_scores_html(rewards)
    reward_df = build_reward_progression_df(log)
    actions_df = build_action_distribution_df(log)
    return replay, scores, reward_df, actions_df


def handler_compare(scenario_id: str, seed: int):
    """Handler for Tab 2."""
    scenario_id = scenario_id or list_scenarios()[0]
    result = run_comparison(scenario_id, seed=int(seed))

    info = get_scenario_info(scenario_id)
    fault = info.get("fault_label", "")

    baseline_replay = format_replay_html(
        result["baseline"]["log"], result["baseline"]["rewards"],
        scenario_id=scenario_id, fault_label=fault,
    )
    trained_replay = format_replay_html(
        result["trained"]["log"], result["trained"]["rewards"],
        scenario_id=scenario_id, fault_label=fault,
    )
    verdict = format_comparison_verdict_html(
        result["baseline"]["rewards"], result["trained"]["rewards"],
    )
    comp_df = build_comparison_df(
        result["baseline"]["rewards"], result["trained"]["rewards"],
    )
    dual_df = build_dual_progression_df(
        result["baseline"]["log"], result["trained"]["log"],
    )
    return baseline_replay, trained_replay, verdict, comp_df, dual_df


def handler_inspect(scenario_id: str):
    """Handler for Tab 3."""
    config_html = get_env_config_html(scenario_id)
    services_df = get_services_df(scenario_id)
    alerts_df = get_alerts_df(scenario_id)
    return config_html, services_df, alerts_df


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

SCENARIO_CHOICES = list_scenarios()
DEFAULT_SCENARIO = SCENARIO_CHOICES[0] if SCENARIO_CHOICES else "seed_easy_memory_leak"


with gr.Blocks(
    title="OnCallEnv: Red Shift",
    fill_width=True,
    theme=OpsRoomTheme(),
    css=CUSTOM_CSS,
) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ============================================================
        # TAB 1 — Run Episode
        # ============================================================
        with gr.TabItem("Run Episode"):
            with gr.Row():
                # Sidebar
                with gr.Column(scale=1, min_width=320):
                    gr.Markdown("### Episode configuration")

                    scenario_dropdown = gr.Dropdown(
                        choices=SCENARIO_CHOICES,
                        value=DEFAULT_SCENARIO,
                        label="Scenario",
                        info="Seed incidents from real-world fault patterns.",
                    )
                    trained_toggle = gr.Checkbox(
                        value=True,
                        label="Use trained defender",
                        info="Toggle off to run the untrained Qwen2.5-3B baseline.",
                    )
                    seed_input = gr.Number(
                        value=42,
                        label="Random seed",
                        precision=0,
                        info="Deterministic episode for reproducibility.",
                    )
                    run_btn = gr.Button(
                        "▶  Run Episode",
                        variant="primary",
                        size="lg",
                    )

                    gr.Markdown("---")
                    gr.Markdown("### Final reward")
                    scores_panel = gr.HTML()

                # Main area
                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.TabItem("Incident Replay"):
                            replay_panel = gr.HTML()
                        with gr.TabItem("Reward & Action Analytics"):
                            with gr.Row():
                                reward_plot = gr.LinePlot(
                                    x="step",
                                    y="reward",
                                    color="metric",
                                    title="Reward progression (per-step and cumulative)",
                                    tooltip=["step", "reward", "metric"],
                                    height=320,
                                )
                            with gr.Row():
                                action_plot = gr.BarPlot(
                                    x="action",
                                    y="count",
                                    color="category",
                                    title="Action distribution",
                                    tooltip=["action", "count", "category"],
                                    height=320,
                                )

            run_btn.click(
                handler_run_episode,
                inputs=[scenario_dropdown, trained_toggle, seed_input],
                outputs=[replay_panel, scores_panel, reward_plot, action_plot],
            )

            # Auto-run on load so the page is never empty
            demo.load(
                handler_run_episode,
                inputs=[scenario_dropdown, trained_toggle, seed_input],
                outputs=[replay_panel, scores_panel, reward_plot, action_plot],
            )

        # ============================================================
        # TAB 2 — Baseline vs Trained
        # ============================================================
        with gr.TabItem("Baseline vs Trained"):
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    gr.Markdown(
                        "### A/B comparison\n\n"
                        "Run the **untrained** Qwen2.5-3B baseline and the "
                        "**Method-2 GRPO checkpoint** on the same incident, "
                        "with the same seed, and compare turn-by-turn. "
                        "This is the headline reward-improvement evidence."
                    )

                    comp_scenario = gr.Dropdown(
                        choices=SCENARIO_CHOICES,
                        value=DEFAULT_SCENARIO,
                        label="Scenario",
                    )
                    comp_seed = gr.Number(
                        value=42,
                        label="Random seed",
                        precision=0,
                        info="Same seed → identical incident on both sides.",
                    )
                    comp_btn = gr.Button(
                        "▶  Run Comparison",
                        variant="primary",
                        size="lg",
                    )

                    gr.Markdown("---")
                    gr.Markdown("### Training impact")
                    comp_verdict = gr.HTML()

                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.TabItem("Side-by-Side Replays"):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("#### ⚠ Untrained baseline")
                                    baseline_replay = gr.HTML()
                                with gr.Column():
                                    gr.Markdown("#### ✓ Trained defender")
                                    trained_replay = gr.HTML()

                        with gr.TabItem("Reward Component Comparison"):
                            comp_bar = gr.BarPlot(
                                x="component",
                                y="score",
                                color="model",
                                title="Reward components: baseline vs trained",
                                tooltip=["component", "score", "model"],
                                height=380,
                            )
                            dual_plot = gr.LinePlot(
                                x="step",
                                y="reward",
                                color="model",
                                title="Cumulative reward over episode steps",
                                tooltip=["step", "reward", "model"],
                                height=320,
                            )

            comp_btn.click(
                handler_compare,
                inputs=[comp_scenario, comp_seed],
                outputs=[
                    baseline_replay,
                    trained_replay,
                    comp_verdict,
                    comp_bar,
                    dual_plot,
                ],
            )

        # ============================================================
        # TAB 3 — Environment Inspector
        # ============================================================
        with gr.TabItem("Environment"):
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    gr.Markdown(
                        "### Inspect simulator state\n\n"
                        "Demystifies what the agent reasons over: services, "
                        "alerts, fault catalog, and the full action surface."
                    )
                    inspect_scenario = gr.Dropdown(
                        choices=SCENARIO_CHOICES,
                        value=DEFAULT_SCENARIO,
                        label="Scenario",
                    )
                    inspect_btn = gr.Button(
                        "🔍  Inspect",
                        variant="primary",
                        size="lg",
                    )
                    gr.Markdown("---")
                    config_panel = gr.HTML()

                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.TabItem("Services"):
                            services_table = gr.Dataframe(
                                label="Microservice graph (7 services)",
                                headers=["Service", "Type", "Replicas", "Dependencies", "Status"],
                                interactive=False,
                            )
                        with gr.TabItem("Active Alerts"):
                            alerts_table = gr.Dataframe(
                                label="Currently firing alerts for this scenario",
                                headers=["Severity", "Service", "Message"],
                                interactive=False,
                            )
                        with gr.TabItem("Fault Catalog"):
                            gr.Dataframe(
                                value=get_fault_catalog_df(),
                                label="12 fault families implemented in the simulator",
                                interactive=False,
                            )
                        with gr.TabItem("Action Surface"):
                            gr.Dataframe(
                                value=get_action_surface_df(),
                                label="Tools available to the agent (read / mutate / terminal)",
                                interactive=False,
                            )

            inspect_btn.click(
                handler_inspect,
                inputs=[inspect_scenario],
                outputs=[config_panel, services_table, alerts_table],
            )
            demo.load(
                handler_inspect,
                inputs=[inspect_scenario],
                outputs=[config_panel, services_table, alerts_table],
            )

        # ============================================================
        # TAB 4 — Methodology & Results
        # ============================================================
        with gr.TabItem("Methodology"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown(
                        """
                        ## OnCallEnv: Red Shift

                        An OpenEnv-compliant training environment for autonomous incident
                        response, plus a controlled three-method ablation comparing
                        scenario-generation strategies for curriculum learning.

                        ### The research question

                        Given a fixed defender architecture (Qwen2.5-3B + LoRA + GRPO), does
                        the *choice of curriculum strategy* materially affect downstream
                        performance on held-out incidents — and if so, which strategy wins?

                        We compare three strategies spanning the full design space:

                        - **Method 1 — No-feedback static curriculum** (baseline). Scenarios
                          drawn uniformly from a fixed pool. The defender's performance has
                          no influence on which scenarios appear next.
                        - **Method 2 — ACCEL-style algorithmic attacker.** A regret-based
                          curriculum that observes solve rates and re-weights the scenario
                          buffer toward the defender's frontier (Parker-Holder et al., ICML
                          2022). The "attacker" is an algorithm, not an LLM.
                        - **Method 3 — SPICE-style LLM self-play.** A second LLM acts as an
                          attacker that generates scenarios via YAML mutation, trained with
                          a Gaussian variance reward peaking at solve probability 0.5 (Liu
                          et al., FAIR 2025).

                        ### Key results

                        | Method                          | Held-out (24 tasks) | Δ vs floor |
                        | ------------------------------- | ------------------- | ---------- |
                        | Random action floor             | 0.270               | —          |
                        | Untrained Qwen2.5-3B            | 0.638               | +0.368     |
                        | M1: No feedback (static)        | 0.872               | +0.602     |
                        | **M2: ACCEL feedback**          | **0.905**           | **+0.635** |
                        | M3: SPICE self-play             | 0.310 *(diverged)*  | —          |
                        | Scripted defender ceiling       | 0.921               | +0.651     |

                        Method 2 closes **94.3%** of the gap from the untrained baseline to
                        the scripted defender ceiling. Method 3 with the same compute and
                        same architecture failed to converge — consistent with known
                        scale requirements for LLM self-play (SPICE was demonstrated on
                        Qwen3-4B / OctoThinker-8B; we used 3B with 4-bit LoRA).

                        ### Reward composition

                        The four-part reward, computed via OpenEnv's composable Rubric API:

                        ```
                        R = 0.35·R_recovery + 0.30·R_rca_quality
                          + 0.25·R_blast_radius + 0.10·R_safety
                        ```

                        - `R_recovery` is paid only when synthetic post-resolution traffic
                          confirms the SLI holds — defeats the "kubectl delete pod to mark
                          green" reward hack.
                        - `R_rca_quality` scores root-cause-category match, timeline
                          coverage for the root service, non-generic five-whys, and
                          service-specific action items.
                        - `R_blast_radius` penalizes prolonged customer-facing impact.
                        - `R_safety` decrements for unsafe actions on stateful services.

                        ### Tech stack

                        OpenEnv 0.2.x · HuggingFace TRL · Unsloth · GRPO · Qwen2.5-3B
                        · LoRA r=16 α=32 · 2× Tesla T4 (Kaggle free tier) · ~6.4 hours
                        wall-clock for 300 GRPO steps.

                        ### Citations

                        - Liu et al. *SPICE: Self-Play In Corpus Environments Improves
                          Reasoning.* arXiv:2510.24684, FAIR @ Meta, 2025.
                        - Parker-Holder et al. *Evolving Curricula with Regret-Based
                          Environment Design (ACCEL).* ICML 2022.
                        - Dennis et al. *Emergent Complexity and Zero-Shot Transfer via
                          Unsupervised Environment Design (PAIRED).* NeurIPS 2020.
                        - Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical
                          Reasoning in Open Language Models (GRPO).* arXiv:2402.03300, 2024.
                        - Jha et al. *ITBench: Evaluating AI Agents across Diverse
                          Real-World IT Automation Tasks.* arXiv:2502.05352, IBM, 2025.

                        ### Links

                        - GitHub: [srimanreddy4/MetaHackathon-R2](https://github.com/srimanreddy4/MetaHackathon-R2/tree/hf)
                        - HF Space: [NeerjaK/OnCallEnv](https://huggingface.co/spaces/NeerjaK/OnCallEnv)
                        - Colab: [Training notebook](https://colab.research.google.com/drive/1KhS2mJ7VKm5o3yRzg47rrjJPUlMj2IQg?usp=sharing)
                        - Blog post: [blog.md](https://huggingface.co/spaces/NeerjaK/OnCallEnv/blob/main/blog.md)
                        """
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### Method ablation")
                    gr.LinePlot(
                        value=build_method_ablation_df(),
                        x="step",
                        y="reward",
                        color="method",
                        title="Reward over training steps",
                        tooltip=["method", "step", "reward"],
                        height=320,
                    )
                    gr.Markdown("### Held-out evaluation")
                    gr.BarPlot(
                        value=build_held_out_comparison_df(),
                        x="method",
                        y="score",
                        title="Mean reward on 24 held-out incidents",
                        tooltip=["method", "score"],
                        height=320,
                    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
