"""Folium map: dark/light tiles with teal→red along-track wind coloring."""

from __future__ import annotations

from typing import Sequence

import folium
from branca.element import MacroElement, Template

# Tailwind (positive along-track) → teal; headwind (negative) → red
TEAL = (20, 184, 166)  # #14b8a6
RED = (239, 68, 68)  # #ef4444
NEUTRAL = (148, 163, 184)  # slate for calm / no weather

# Esri gray canvases — no API key required for standard usage
DARK_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)
LIGHT_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)
DARK_ATTR = "Tiles © Esri — Dark Gray Canvas"
LIGHT_ATTR = "Tiles © Esri — Light Gray Canvas"

# Fixed scale for color mapping (± m/s); values beyond clamp
WIND_SCALE_MPS = 5.0
MAP_MIN_HEIGHT_PX = 300


def wind_to_color(along_mps: float, *, scale_mps: float = WIND_SCALE_MPS) -> str:
    """Map along-track wind to hex: teal (tail) → slate (calm) → red (head)."""
    if scale_mps <= 0:
        return _rgb_hex(NEUTRAL)
    t = max(-1.0, min(1.0, along_mps / scale_mps))
    if t >= 0:
        # calm → teal
        r, g, b = _lerp_rgb(NEUTRAL, TEAL, t)
    else:
        # calm → red
        r, g, b = _lerp_rgb(NEUTRAL, RED, -t)
    return _rgb_hex((r, g, b))


def build_wind_route_map(
    lats: Sequence[float],
    lons: Sequence[float],
    wind_along_mps: Sequence[float],
    *,
    dark: bool = True,
    height_px: int = MAP_MIN_HEIGHT_PX,
    weather_ok: bool = True,
) -> folium.Map:
    """Build a Folium map with the route colored by along-track wind."""
    if len(lats) < 2 or len(lats) != len(lons):
        raise ValueError("Need at least two lat/lon points")
    if len(wind_along_mps) != len(lats):
        # Allow one-shorter wind (segment values); pad last
        winds = list(wind_along_mps)
        while len(winds) < len(lats):
            winds.append(winds[-1] if winds else 0.0)
        wind_along_mps = winds[: len(lats)]

    mid_lat = sum(lats) / len(lats)
    mid_lon = sum(lons) / len(lons)
    fmap = folium.Map(
        location=[mid_lat, mid_lon],
        zoom_start=12,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles=DARK_TILES if dark else LIGHT_TILES,
        attr=DARK_ATTR if dark else LIGHT_ATTR,
        name="basemap",
        control=False,
    ).add_to(fmap)

    # Color each consecutive segment by wind at the segment start
    for i in range(len(lats) - 1):
        along = wind_along_mps[i] if weather_ok else 0.0
        color = wind_to_color(along) if weather_ok else _rgb_hex(NEUTRAL)
        folium.PolyLine(
            locations=[[lats[i], lons[i]], [lats[i + 1], lons[i + 1]]],
            color=color,
            weight=5,
            opacity=0.92,
            tooltip=(
                f"Along-track {along:+.1f} m/s"
                if weather_ok
                else "Wind unavailable"
            ),
        ).add_to(fmap)

    # Start / end markers
    folium.CircleMarker(
        location=[lats[0], lons[0]],
        radius=6,
        color="#22c55e",
        fill=True,
        fill_color="#22c55e",
        tooltip="Start",
    ).add_to(fmap)
    folium.CircleMarker(
        location=[lats[-1], lons[-1]],
        radius=6,
        color="#a78bfa",
        fill=True,
        fill_color="#a78bfa",
        tooltip="End",
    ).add_to(fmap)

    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(24, 24))
    _add_legend(fmap, weather_ok=weather_ok, dark=dark)

    # Ensure min height via folium Figure size (streamlit-folium also sets height)
    fmap.get_root().html.add_child(
        folium.Element(
            f"<style>.folium-map {{ min-height: {height_px}px !important; }}</style>"
        )
    )
    return fmap


def _add_legend(fmap: folium.Map, *, weather_ok: bool, dark: bool) -> None:
    bg = "#161b22" if dark else "#ffffff"
    fg = "#e5e7eb" if dark else "#111827"
    border = "#30363d" if dark else "#e5e7eb"
    body = (
        "Teal = tailwind · Red = headwind"
        if weather_ok
        else "Neutral route (wind unavailable)"
    )
    template = Template(
        f"""
        {{% macro html(this, kwargs) %}}
        <div style="
            position: fixed; bottom: 28px; left: 28px; z-index: 9999;
            background: {bg}; color: {fg}; border: 1px solid {border};
            border-radius: 8px; padding: 8px 12px; font-size: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25); max-width: 220px;
        ">
          <div style="font-weight: 600; margin-bottom: 4px;">Along-track wind</div>
          <div style="display:flex; align-items:center; gap:8px; margin: 4px 0;">
            <span style="display:inline-block;width:28px;height:6px;border-radius:3px;
              background: linear-gradient(90deg, #14b8a6, #94a3b8, #ef4444);"></span>
            <span>tail → head</span>
          </div>
          <div style="opacity: 0.85;">{body}</div>
        </div>
        {{% endmacro %}}
        """
    )
    macro = MacroElement()
    macro._template = template
    fmap.get_root().add_child(macro)


def _lerp_rgb(
    a: tuple[int, int, int], b: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
