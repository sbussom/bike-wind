"""Local area wind overview: Windy.com embed + Open-Meteo pin sample."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import requests

# South Florida default (Boca / Palm Beach–ish)
DEFAULT_LAT = 26.7
DEFAULT_LON = -80.0
DEFAULT_ZOOM = 9
WINDY_EMBED_HEIGHT_PX = 320
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass(frozen=True)
class PinWind:
    """Current / nearest-hour wind at the map pin."""

    lat: float
    lon: float
    speed_mps: float
    direction_deg: float  # meteorological: from which direction wind blows
    temperature_c: float | None = None

    @property
    def speed_mph(self) -> float:
        return self.speed_mps * 2.236936

    @property
    def speed_kmh(self) -> float:
        return self.speed_mps * 3.6


def windy_embed_url(
    lat: float,
    lon: float,
    *,
    zoom: int = DEFAULT_ZOOM,
    metric_wind: str = "mph",
) -> str:
    """Build a Windy embed2 URL with wind overlay centered on lat/lon."""
    params = {
        "lat": f"{lat:.4f}",
        "lon": f"{lon:.4f}",
        "detailLat": f"{lat:.4f}",
        "detailLon": f"{lon:.4f}",
        "zoom": str(zoom),
        "level": "surface",
        "overlay": "wind",
        "product": "ecmwf",
        "menu": "",
        "message": "true",
        "marker": "true",
        "calendar": "now",
        "pressure": "",
        "type": "map",
        "location": "coordinates",
        "detail": "true",
        "metricWind": metric_wind,
        "metricTemp": "°F",
        "radarRange": "-1",
    }
    return f"https://embed.windy.com/embed2.html?{urlencode(params)}"


def windy_iframe_html(
    lat: float,
    lon: float,
    *,
    height_px: int = WINDY_EMBED_HEIGHT_PX,
    zoom: int = DEFAULT_ZOOM,
    metric_wind: str = "mph",
) -> str:
    """Responsive iframe HTML for Streamlit ``st.components.v1.html``."""
    src = windy_embed_url(lat, lon, zoom=zoom, metric_wind=metric_wind)
    return f"""
<div style="width:100%;min-height:{height_px}px;border-radius:10px;overflow:hidden;
            border:1px solid #30363d;">
  <iframe
    src="{src}"
    title="Local wind — Windy"
    style="width:100%;height:{height_px}px;border:0;display:block;"
    loading="lazy"
    allowfullscreen
  ></iframe>
</div>
"""


def fetch_pin_wind(
    lat: float,
    lon: float,
    *,
    session: requests.Session | None = None,
    timeout_s: float = 12,
) -> PinWind | None:
    """Nearest-hour 10 m wind at the pin from Open-Meteo (no secrets required)."""
    http = session or requests.Session()
    try:
        resp = http.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "wind_speed_10m,wind_direction_10m,temperature_2m",
                "wind_speed_unit": "ms",
                "timezone": "UTC",
            },
            timeout=timeout_s,
        )
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        cur = payload.get("current") or {}
        speed = float(cur.get("wind_speed_10m") or 0.0)
        direction = float(cur.get("wind_direction_10m") or 0.0)
        temp = cur.get("temperature_2m")
        return PinWind(
            lat=lat,
            lon=lon,
            speed_mps=speed,
            direction_deg=direction,
            temperature_c=float(temp) if temp is not None else None,
        )
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return None


def compass_label(direction_deg: float) -> str:
    """16-point compass from meteorological wind-from direction."""
    dirs = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    idx = int((direction_deg % 360) / 22.5 + 0.5) % 16
    return dirs[idx]
