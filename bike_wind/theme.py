"""Shared GustyPower appearance.

One light palette drives ``.streamlit/config.toml`` and the CSS injected on
every page. The Dark toggle still switches a high-contrast dark palette for
charts and this stylesheet. Widget text does not stay on Streamlit's default
grey: captions in Streamlit use 60% opacity, which dropped helper text under
WCAG AA, and the old dark canvas sat under light-theme body text (~1.5:1).

``st.html`` is required. ``st.markdown(..., unsafe_allow_html=True)`` strips
``<style>`` on current Streamlit and would print the stylesheet as text.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThemeColors:
    """Colors for one appearance. Hex values are lowercase."""

    bg0: str
    bg1: str
    surface: str
    sidebar: str
    text: str
    muted: str
    accent: str
    on_accent: str
    link: str
    line: str
    border: str
    metric: str
    chart_bg: str
    spine: str
    actual: str
    power: str
    shade: str
    placeholder: str
    code_bg: str
    input_text: str
    glow: str
    glow2: str


# Keep these in sync with .streamlit/config.toml (the locked light theme).
LIGHT = ThemeColors(
    bg0="#f4f7f4",
    bg1="#eef3f0",
    surface="#ffffff",
    sidebar="#e6efe9",
    text="#14241c",
    muted="#3e5348",
    accent="#0f6e4e",
    on_accent="#ffffff",
    link="#0c5c42",
    line="rgba(20, 36, 28, 0.12)",
    border="#6e8b7e",
    metric="#0e241c",
    chart_bg="#f7faf8",
    spine="#4f675c",
    actual="#1e4d6b",
    power="#7a4e12",
    shade="#c45c26",
    placeholder="#4a6156",
    code_bg="#e7efe9",
    input_text="#14241c",
    glow="rgba(15, 110, 78, 0.10)",
    glow2="rgba(30, 77, 107, 0.07)",
)

DARK = ThemeColors(
    bg0="#1c2823",
    bg1="#24312b",
    surface="#2a3832",
    sidebar="#18211d",
    text="#f4faf6",
    muted="#d5e4dc",
    accent="#3ecf8e",
    on_accent="#062018",
    link="#9aebc8",
    line="rgba(244, 250, 246, 0.16)",
    border="#8eaea0",
    metric="#f7fbf8",
    chart_bg="#1c2823",
    spine="#8eaea0",
    actual="#d5e4dc",
    power="#e0b15a",
    shade="#e07a3d",
    placeholder="#4a6156",
    code_bg="#18211d",
    input_text="#14241c",
    glow="rgba(62, 207, 142, 0.14)",
    glow2="rgba(30, 77, 107, 0.20)",
)


def chart_palette(dark: bool) -> ThemeColors:
    """Matplotlib colors for the active appearance."""
    return DARK if dark else LIGHT


def theme_is_dark() -> bool:
    """Session value of the main-page Dark toggle. Light when unset."""
    import streamlit as st

    return bool(st.session_state.get("bw_dark", False))


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG relative-luminance contrast. Both arguments are ``#rrggbb``."""

    def _channel(value: int) -> float:
        c = value / 255
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    def _lum(hex_color: str) -> float:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)

    lighter = max(_lum(foreground), _lum(background))
    darker = min(_lum(foreground), _lum(background))
    return (lighter + 0.05) / (darker + 0.05)


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Syne:wght@600;700;800&display=swap');

.stApp,
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 520px at 0% -10%, @@glow@@, transparent 58%),
    radial-gradient(820px 420px at 100% 0%, @@glow2@@, transparent 52%),
    linear-gradient(180deg, @@bg0@@ 0%, @@bg1@@ 100%) !important;
  color: @@text@@;
  font-family: "Figtree", sans-serif;
}
[data-testid="stHeader"] {
  background: color-mix(in srgb, @@bg0@@ 88%, transparent) !important;
}
[data-testid="stHeader"] svg,
[data-testid="stToolbar"] svg,
[data-testid="stToolbar"] button,
[data-testid="stStatusWidget"] {
  color: @@text@@ !important;
  fill: @@text@@;
}
[data-testid="stDecoration"] {
  background: @@accent@@ !important;
}
section[data-testid="stSidebar"] {
  background: @@sidebar@@ !important;
  color: @@text@@ !important;
  border-right: 1px solid @@line@@;
}
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] span,
[data-testid="stSidebarNav"] p,
[data-testid="stPageLink"] a,
[data-testid="stPageLink-NavLink"],
[data-testid="stPageLink"] span {
  color: @@text@@ !important;
  opacity: 1 !important;
}
h1, h2, h3, h4, h5, h6,
[data-testid="stHeading"] {
  font-family: "Syne", sans-serif !important;
  letter-spacing: -0.02em;
  color: @@text@@ !important;
}
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  color: @@text@@;
}
[data-testid="stMarkdownContainer"] a {
  color: @@link@@ !important;
}
[data-testid="stCaptionContainer"] {
  opacity: 1 !important;
  color: @@muted@@ !important;
}
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span {
  color: @@muted@@ !important;
  opacity: 1 !important;
}
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] label {
  color: @@text@@ !important;
  opacity: 1 !important;
}
[data-testid="stTooltipIcon"] {
  color: @@muted@@ !important;
  opacity: 1 !important;
}
[data-testid="stMetricValue"] {
  color: @@metric@@ !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] label {
  color: @@muted@@ !important;
  opacity: 1 !important;
}
[data-testid="stAlert"] {
  border-radius: 12px;
}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
[data-testid="stAlert"] p {
  opacity: 1 !important;
}
[data-testid="stExpander"] {
  background: @@surface@@;
  border: 1px solid @@line@@ !important;
  border-radius: 12px;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
  color: @@text@@ !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  background: @@surface@@;
  border: 1px solid @@line@@ !important;
  border-radius: 14px;
  padding: 0.9rem 1rem 1rem;
  box-shadow: 0 1px 2px rgba(20, 36, 28, 0.05);
}
[data-testid="stTable"],
[data-testid="stDataFrame"] {
  color: @@text@@;
}
div.stButton > button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-tertiary"],
[data-testid="stBaseLinkButton-secondary"] {
  min-height: 44px;
  border-radius: 10px;
  font-family: "Figtree", sans-serif;
  font-weight: 600;
  background: @@surface@@ !important;
  color: @@text@@ !important;
  border: 1px solid @@border@@ !important;
}
div.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"],
[data-testid="stBaseLinkButton-primary"] {
  background: @@accent@@ !important;
  border: 1px solid @@accent@@ !important;
  color: @@on_accent@@ !important;
}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stDateInput"] input,
[data-baseweb="input"],
[data-baseweb="textarea"],
[data-baseweb="select"] > div {
  background-color: #ffffff !important;
  color: @@input_text@@ !important;
  -webkit-text-fill-color: @@input_text@@;
  border-color: @@border@@ !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
  color: @@placeholder@@ !important;
  opacity: 1 !important;
  -webkit-text-fill-color: @@placeholder@@;
}
.stCode, pre, code {
  color: @@text@@ !important;
  background: @@code_bg@@ !important;
}
.bw-hero {
  padding: 0.35rem 0 1rem 0;
  border-bottom: 1px solid @@line@@;
  margin-bottom: 1.15rem;
}
.bw-hero-compact {
  margin-bottom: 0.85rem;
}
.bw-brand {
  font-family: "Syne", sans-serif;
  font-size: clamp(2.2rem, 6vw, 3.15rem);
  line-height: 1.05;
  margin: 0;
  color: @@text@@;
  font-weight: 700;
  letter-spacing: -0.03em;
}
.bw-hero-compact .bw-brand {
  font-size: clamp(1.55rem, 4vw, 1.9rem);
}
.bw-accent-word {
  color: @@accent@@;
}
.bw-tagline {
  margin: 0.5rem 0 0 0;
  color: @@muted@@;
  font-size: 1.05rem;
  max-width: 36rem;
  line-height: 1.45;
}
.bw-section-label {
  font-family: "Syne", sans-serif;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: @@accent@@;
  margin: 0 0 0.35rem 0;
}
.bw-time-row {
  display: flex;
  flex-wrap: wrap;
  gap: 1.25rem 2rem;
  margin: 0.15rem 0 0.85rem 0;
  align-items: flex-end;
}
.bw-time-block {
  min-width: 7.5rem;
}
.bw-time-label {
  margin: 0;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: @@muted@@;
}
.bw-time-value {
  margin: 0.15rem 0 0 0;
  font-family: "Syne", sans-serif;
  font-size: clamp(1.85rem, 5vw, 2.55rem);
  line-height: 1.05;
  font-weight: 700;
  color: @@metric@@;
  letter-spacing: -0.02em;
}
.bw-pace-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.85rem 1.75rem;
  margin: 0.2rem 0 0.15rem 0;
}
.bw-pace-value {
  margin: 0.1rem 0 0 0;
  font-family: "Syne", sans-serif;
  font-size: 1.35rem;
  line-height: 1.1;
  font-weight: 700;
  color: @@metric@@;
  letter-spacing: -0.02em;
}
.bw-run-wrap {
  position: sticky;
  bottom: 0.5rem;
  z-index: 20;
  padding: 0.65rem 0 0.25rem;
  background: linear-gradient(180deg, transparent, @@bg1@@ 35%);
}
@media (max-width: 640px) {
  .bw-brand { font-size: 2.2rem; }
  .bw-tagline { font-size: 1rem; }
}
</style>
"""


def _stylesheet(colors: ThemeColors) -> str:
    css = _CSS
    for name, value in colors.__dict__.items():
        css = css.replace(f"@@{name}@@", value)
    return css


def apply_theme(dark: bool | None = None) -> None:
    """Inject the shared stylesheet. Call once at the top of every page."""
    import streamlit as st

    if dark is None:
        dark = bool(st.session_state.get("bw_dark", False))
    # st.html keeps <style>; markdown would strip tags and leak CSS as text.
    st.html(_stylesheet(DARK if dark else LIGHT))
