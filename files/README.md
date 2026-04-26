---
title: OnCallEnv Red Shift
emoji: 🚨
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - openenv
  - reinforcement-learning
  - sre
  - aiops
  - grpo
  - self-play
  - curriculum-learning
short_description: An OpenEnv RL environment for training LLMs as on-call SREs.
---

# OnCallEnv: Red Shift

An OpenEnv-compliant training environment for autonomous incident response,
plus a controlled three-method ablation comparing scenario-generation
strategies for curriculum learning.

This Space lets you:

1. **Run an episode** — see the agent investigate, remediate, and resolve a
   simulated incident with a turn-by-turn replay.
2. **Compare baseline vs trained** — A/B view of the untrained Qwen2.5-3B
   versus our Method-2 GRPO checkpoint on identical incidents.
3. **Inspect the environment** — service inventory, alerts, fault catalog,
   and the agent's full action surface.
4. **Read the methodology** — research framing, reward composition, key
   results, and citations.

See the full blog post `<hf-blog-link>` and code `<github-link>` for details.

## Quick local run

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:7860`.

## Architecture

Built with Gradio 4 + a custom Ops-Room theme. The replay renderer in
`replay_html.py` is the storytelling centerpiece — every turn shown as a
phase-aware action card with reward attribution.

The Space gracefully degrades when the real `OnCallRedShiftEnv` package
isn't importable (e.g., during early deployment): canned trajectories are
served while showing a "demo mode" badge so judges always see a live UI.

## Citations

- Liu et al. *SPICE: Self-Play In Corpus Environments Improves Reasoning.*
  arXiv:2510.24684, FAIR @ Meta, 2025.
- Parker-Holder et al. *Evolving Curricula with Regret-Based Environment
  Design (ACCEL).* ICML 2022.
- Shao et al. *DeepSeekMath: GRPO.* arXiv:2402.03300, 2024.
