"""WCAG AA contrast for the shared light and dark palettes."""

from __future__ import annotations

import tomllib
from pathlib import Path

from bike_wind.theme import DARK, LIGHT, contrast_ratio

ROOT = Path(__file__).resolve().parents[1]


def _assert_body(colors, name: str) -> None:
    assert contrast_ratio(colors.text, colors.bg0) >= 4.5, name
    assert contrast_ratio(colors.text, colors.surface) >= 4.5, name
    assert contrast_ratio(colors.text, colors.sidebar) >= 4.5, name
    assert contrast_ratio(colors.muted, colors.bg0) >= 4.5, name
    assert contrast_ratio(colors.muted, colors.surface) >= 4.5, name
    assert contrast_ratio(colors.muted, colors.sidebar) >= 4.5, name
    assert contrast_ratio(colors.accent, colors.bg0) >= 4.5, name
    assert contrast_ratio(colors.on_accent, colors.accent) >= 4.5, name
    assert contrast_ratio(colors.link, colors.surface) >= 4.5, name
    assert contrast_ratio(colors.text, colors.chart_bg) >= 4.5, name
    assert contrast_ratio(colors.actual, colors.chart_bg) >= 4.5, name
    assert contrast_ratio(colors.power, colors.chart_bg) >= 4.5, name
    assert contrast_ratio(colors.muted, colors.chart_bg) >= 4.5, name
    assert contrast_ratio(colors.spine, colors.chart_bg) >= 3.0, name
    assert contrast_ratio(colors.placeholder, "#ffffff") >= 4.5, name


def test_light_palette_meets_aa() -> None:
    _assert_body(LIGHT, "light")


def test_dark_palette_meets_aa() -> None:
    _assert_body(DARK, "dark")


def test_config_toml_matches_light_palette() -> None:
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    theme = cfg["theme"]
    sidebar = cfg["theme"]["sidebar"]
    assert theme["base"] == "light"
    assert theme["textColor"].lower() == LIGHT.text
    assert theme["backgroundColor"].lower() == LIGHT.bg0
    assert theme["secondaryBackgroundColor"].lower() == LIGHT.surface
    assert theme["primaryColor"].lower() == LIGHT.accent
    assert theme["linkColor"].lower() == LIGHT.link
    assert theme["grayTextColor"].lower() == LIGHT.muted
    assert sidebar["textColor"].lower() == LIGHT.text
    assert sidebar["backgroundColor"].lower() == LIGHT.sidebar
