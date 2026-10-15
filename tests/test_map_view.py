"""Tests for wind→color map helpers."""

from __future__ import annotations

from bike_wind.map_view import (
    DARK_TILES,
    LIGHT_TILES,
    RED,
    TEAL,
    build_wind_route_map,
    wind_to_color,
)


def test_tailwind_is_tealish() -> None:
    color = wind_to_color(5.0)
    assert color.lower() == f"#{TEAL[0]:02x}{TEAL[1]:02x}{TEAL[2]:02x}"


def test_headwind_is_redish() -> None:
    color = wind_to_color(-5.0)
    assert color.lower() == f"#{RED[0]:02x}{RED[1]:02x}{RED[2]:02x}"


def test_calm_is_between() -> None:
    calm = wind_to_color(0.0)
    assert calm.startswith("#")
    assert calm != wind_to_color(5.0)
    assert calm != wind_to_color(-5.0)


def test_build_map_dark_and_light() -> None:
    lats = [26.12, 26.13, 26.14]
    lons = [-80.14, -80.13, -80.12]
    winds = [2.0, -3.0, 0.5]
    dark = build_wind_route_map(lats, lons, winds, dark=True)
    light = build_wind_route_map(lats, lons, winds, dark=False)
    dark_html = dark.get_root().render()
    light_html = light.get_root().render()
    assert "World_Dark_Gray_Base" in dark_html
    assert "World_Light_Gray_Base" in light_html
    children = list(dark._children.values())
    assert sum(1 for c in children if type(c).__name__ == "PolyLine") == 2


def test_build_map_without_weather_uses_neutral() -> None:
    fmap = build_wind_route_map(
        [26.0, 26.1],
        [-80.0, -80.1],
        [-4.0, -4.0],
        weather_ok=False,
    )
    assert fmap is not None
