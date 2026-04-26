"""OnCallEnv Red Shift — Ops Room Gradio theme.

Visual language: Datadog / Grafana / Linear. Cool grays, single electric-blue
accent for primary actions, amber for warnings, red for critical. No neon, no
scanlines, no cyberpunk flourishes. Optimized for projector readability during
a 3-minute pitch.
"""

from __future__ import annotations

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes


class OpsRoomTheme(Base):
    """Production-grade observability aesthetic for OnCallEnv Red Shift."""

    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.blue,
        secondary_hue: colors.Color | str = colors.amber,
        neutral_hue: colors.Color | str = colors.slate,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str | list = (
            fonts.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | list = (
            fonts.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )

        # Body / surface
        self.body_background_fill = "#0b1020"
        self.body_background_fill_dark = "#0b1020"
        self.body_text_color = "#d6deeb"
        self.body_text_color_dark = "#d6deeb"
        self.body_text_color_subdued = "#7d8ba1"
        self.body_text_color_subdued_dark = "#7d8ba1"

        # Blocks / panels
        self.block_background_fill = "#131a2e"
        self.block_background_fill_dark = "#131a2e"
        self.block_border_color = "#1f2940"
        self.block_border_color_dark = "#1f2940"
        self.block_border_width = "1px"
        self.block_label_background_fill = "#0f1626"
        self.block_label_background_fill_dark = "#0f1626"
        self.block_label_text_color = "#7aa7ff"
        self.block_label_text_color_dark = "#7aa7ff"
        self.block_shadow = "0 1px 3px rgba(0,0,0,0.25)"
        self.block_shadow_dark = "0 1px 3px rgba(0,0,0,0.25)"
        self.block_title_text_color = "#e6edf7"
        self.block_title_text_color_dark = "#e6edf7"

        # Borders
        self.border_color_accent = "#3b82f6"
        self.border_color_accent_dark = "#3b82f6"
        self.border_color_primary = "#1f2940"
        self.border_color_primary_dark = "#1f2940"
        self.panel_background_fill = "#0f1626"
        self.panel_background_fill_dark = "#0f1626"
        self.panel_border_color = "#1f2940"
        self.panel_border_color_dark = "#1f2940"

        # Primary button (electric blue)
        self.button_primary_background_fill = "#3b82f6"
        self.button_primary_background_fill_dark = "#3b82f6"
        self.button_primary_background_fill_hover = "#60a5fa"
        self.button_primary_background_fill_hover_dark = "#60a5fa"
        self.button_primary_text_color = "#ffffff"
        self.button_primary_text_color_dark = "#ffffff"
        self.button_primary_border_color = "#2563eb"
        self.button_primary_border_color_dark = "#2563eb"

        # Secondary button
        self.button_secondary_background_fill = "#1f2940"
        self.button_secondary_background_fill_dark = "#1f2940"
        self.button_secondary_background_fill_hover = "#2a3656"
        self.button_secondary_background_fill_hover_dark = "#2a3656"
        self.button_secondary_text_color = "#d6deeb"
        self.button_secondary_text_color_dark = "#d6deeb"
        self.button_secondary_border_color = "#2a3656"
        self.button_secondary_border_color_dark = "#2a3656"

        # Inputs
        self.input_background_fill = "#0f1626"
        self.input_background_fill_dark = "#0f1626"
        self.input_border_color = "#1f2940"
        self.input_border_color_dark = "#1f2940"
        self.input_border_color_focus = "#3b82f6"
        self.input_border_color_focus_dark = "#3b82f6"
        self.input_placeholder_color = "#5b6577"
        self.input_placeholder_color_dark = "#5b6577"
        self.input_text_color = "#d6deeb"
        self.input_text_color_dark = "#d6deeb"

        # Checkbox
        self.checkbox_background_color = "#0f1626"
        self.checkbox_background_color_dark = "#0f1626"
        self.checkbox_background_color_selected = "#3b82f6"
        self.checkbox_background_color_selected_dark = "#3b82f6"
        self.checkbox_border_color = "#2a3656"
        self.checkbox_border_color_dark = "#2a3656"
        self.checkbox_border_color_selected = "#3b82f6"
        self.checkbox_border_color_selected_dark = "#3b82f6"

        # Tables / code
        self.table_border_color = "#1f2940"
        self.table_border_color_dark = "#1f2940"
        self.table_even_background_fill = "#131a2e"
        self.table_even_background_fill_dark = "#131a2e"
        self.table_odd_background_fill = "#0f1626"
        self.table_odd_background_fill_dark = "#0f1626"
        self.code_background_fill = "#0f1626"
        self.code_background_fill_dark = "#0f1626"

        # Shadows
        self.shadow_drop = "0 1px 3px rgba(0,0,0,0.3)"
        self.shadow_drop_lg = "0 4px 12px rgba(0,0,0,0.4)"


# --------------------------------------------------------------------------
# Custom CSS — overrides + bespoke components
# --------------------------------------------------------------------------

CUSTOM_CSS = """
:root {
  --ops-bg: #0b1020;
  --ops-surface: #131a2e;
  --ops-surface-alt: #0f1626;
  --ops-border: #1f2940;
  --ops-border-strong: #2a3656;
  --ops-text: #d6deeb;
  --ops-text-dim: #7d8ba1;
  --ops-text-bright: #e6edf7;

  --ops-blue: #3b82f6;
  --ops-blue-bright: #60a5fa;
  --ops-amber: #f59e0b;
  --ops-red: #ef4444;
  --ops-green: #10b981;
  --ops-purple: #8b5cf6;
}

footer { display: none !important; }

.gradio-container {
  background: var(--ops-bg) !important;
  max-width: 100% !important;
  padding: 0 !important;
}

.gradio-container > .main {
  max-width: 1600px;
  margin: 0 auto;
  padding: 0 24px;
}

/* ============== TAB HEADERS ============== */
.tab-nav button {
  background: transparent !important;
  color: var(--ops-text-dim) !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.875rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.01em;
  padding: 12px 20px !important;
  transition: all 0.15s ease;
}

.tab-nav button:hover {
  color: var(--ops-text-bright) !important;
}

.tab-nav button.selected {
  color: var(--ops-blue-bright) !important;
  border-bottom: 2px solid var(--ops-blue) !important;
  background: transparent !important;
}

/* ============== HEADER BANNER ============== */
.ops-header {
  padding: 28px 24px 24px;
  background: linear-gradient(180deg, #0d1428 0%, var(--ops-bg) 100%);
  border-bottom: 1px solid var(--ops-border);
  margin-bottom: 16px;
}

.ops-header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  flex-wrap: wrap;
}

.ops-header h1 {
  font-family: 'Inter', sans-serif;
  font-size: 1.875rem;
  font-weight: 700;
  color: var(--ops-text-bright);
  margin: 0 0 4px 0;
  letter-spacing: -0.02em;
}

.ops-header .subtitle {
  font-family: 'Inter', sans-serif;
  font-size: 0.875rem;
  color: var(--ops-text-dim);
  letter-spacing: 0.01em;
}

.ops-header-meta {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.ops-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 500;
  letter-spacing: 0.04em;
  border: 1px solid var(--ops-border-strong);
  background: var(--ops-surface-alt);
  color: var(--ops-text-dim);
}

.ops-pill-blue {
  color: var(--ops-blue-bright);
  border-color: rgba(59, 130, 246, 0.3);
  background: rgba(59, 130, 246, 0.08);
}

.ops-pill-green {
  color: var(--ops-green);
  border-color: rgba(16, 185, 129, 0.3);
  background: rgba(16, 185, 129, 0.08);
}

.ops-pill .dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
}

/* ============== CARDS ============== */
.ops-card {
  background: var(--ops-surface) !important;
  border: 1px solid var(--ops-border) !important;
  border-radius: 8px !important;
  padding: 16px !important;
}

/* ============== PRIMARY BUTTON ============== */
button.primary {
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
  transition: all 0.15s ease !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
}

button.primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(59, 130, 246, 0.25) !important;
}

/* ============== TYPOGRAPHY ============== */
.prose h1, .prose h2, .prose h3, .prose h4 {
  color: var(--ops-text-bright) !important;
  font-family: 'Inter', sans-serif !important;
  letter-spacing: -0.01em;
}

.prose h2 { font-size: 1.25rem !important; margin-top: 1.5rem !important; }
.prose h3 { font-size: 1.05rem !important; margin-top: 1.25rem !important; }

.prose p, .prose li {
  color: var(--ops-text) !important;
  line-height: 1.65;
}

.prose strong {
  color: var(--ops-text-bright) !important;
  font-weight: 600;
}

.prose a {
  color: var(--ops-blue-bright) !important;
}

.prose code {
  background: var(--ops-surface-alt) !important;
  color: var(--ops-blue-bright) !important;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.875em;
  border: 1px solid var(--ops-border);
}

.prose pre {
  background: var(--ops-surface-alt) !important;
  border: 1px solid var(--ops-border) !important;
  border-radius: 6px;
}

.prose hr {
  border-color: var(--ops-border) !important;
}

.prose table {
  border-collapse: collapse;
  font-size: 0.875rem;
}

.prose th {
  background: var(--ops-surface-alt) !important;
  color: var(--ops-text-bright) !important;
  border: 1px solid var(--ops-border) !important;
  padding: 8px 12px !important;
  text-align: left;
}

.prose td {
  border: 1px solid var(--ops-border) !important;
  padding: 6px 12px !important;
}

/* ============== SCROLLBAR ============== */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--ops-bg); }
::-webkit-scrollbar-thumb {
  background: var(--ops-border-strong);
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: #3a4a6a; }
"""


HEADER_HTML = """
<div class="ops-header">
  <div class="ops-header-row">
    <div>
      <h1>OnCallEnv: Red Shift</h1>
      <div class="subtitle">RL training environment for autonomous incident response</div>
    </div>
    <div class="ops-header-meta">
      <span class="ops-pill ops-pill-green"><span class="dot"></span>ENV READY</span>
      <span class="ops-pill ops-pill-blue">QWEN2.5-3B · GRPO</span>
      <span class="ops-pill">OPENENV 0.2</span>
    </div>
  </div>
</div>
"""
