"""Tests for Windy embed URL / pin helpers."""

from __future__ import annotations

from bike_wind.local_wind_map import (
    DEFAULT_LAT,
    DEFAULT_LON,
    compass_label,
    windy_embed_url,
    windy_iframe_html,
)


def test_windy_embed_url_has_wind_overlay() -> None:
    url = windy_embed_url(DEFAULT_LAT, DEFAULT_LON)
    assert "embed.windy.com" in url
    assert "overlay=wind" in url
    assert f"lat={DEFAULT_LAT:.4f}" in url
    assert "metricWind=mph" in url


def test_windy_iframe_html_contains_iframe() -> None:
    html = windy_iframe_html(26.7, -80.0, height_px=280)
    assert "<iframe" in html
    assert "280px" in html


def test_compass_label_cardinals() -> None:
    assert compass_label(0) == "N"
    assert compass_label(90) == "E"
    assert compass_label(180) == "S"
    assert compass_label(270) == "W"
