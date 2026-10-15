"""Unified weather fetch: Open-Meteo primary, optional HRRR with fallback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import requests

from bike_wind.physics.constants import RIDER_HEIGHT_M, Z0_OPEN_GRASS
from bike_wind.weather.cache import ForecastCache
from bike_wind.weather.hrrr import fetch_hrrr_along_route, route_in_conus
from bike_wind.weather.open_meteo import (
    DEFAULT_TIMEOUT_S,
    ForecastPoint,
    WeatherFetchError,
    fetch_forecast_along_route,
)


@dataclass(frozen=True)
class WeatherFetchResult:
    points: list[ForecastPoint]
    source: str  # "open-meteo" | "hrrr"
    message: str | None = None


def fetch_weather_along_route(
    lats: Sequence[float],
    lons: Sequence[float],
    bearings_deg: Sequence[float],
    when: datetime,
    *,
    prefer_hrrr: bool = False,
    max_samples: int = 12,
    z0_m: float = Z0_OPEN_GRASS,
    z_m: float = RIDER_HEIGHT_M,
    session: requests.Session | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    cache: ForecastCache | None = None,
) -> WeatherFetchResult:
    """Fetch wind along a route.

    Open-Meteo is always available as primary/fallback. When ``prefer_hrrr`` is
    True and the route is in CONUS, try HRRR first; on failure or out-of-domain,
    fall back to Open-Meteo with a clear message.
    """
    cache = cache or ForecastCache()
    http = session or requests.Session()

    if prefer_hrrr:
        if not route_in_conus(lats, lons):
            points = fetch_forecast_along_route(
                lats,
                lons,
                bearings_deg,
                when,
                max_samples=max_samples,
                z0_m=z0_m,
                z_m=z_m,
                session=http,
                timeout_s=timeout_s,
                cache=cache,
            )
            return WeatherFetchResult(
                points=points,
                source="open-meteo",
                message="HRRR is CONUS-only — using Open-Meteo forecast.",
            )
        try:
            points = fetch_hrrr_along_route(
                lats,
                lons,
                bearings_deg,
                when,
                max_samples=max_samples,
                z0_m=z0_m,
                z_m=z_m,
                session=http,
                timeout_s=timeout_s,
                cache=cache,
            )
            return WeatherFetchResult(points=points, source="hrrr")
        except WeatherFetchError as exc:
            points = fetch_forecast_along_route(
                lats,
                lons,
                bearings_deg,
                when,
                max_samples=max_samples,
                z0_m=z0_m,
                z_m=z_m,
                session=http,
                timeout_s=timeout_s,
                cache=cache,
            )
            return WeatherFetchResult(
                points=points,
                source="open-meteo",
                message=f"HRRR unavailable ({exc}); fell back to Open-Meteo.",
            )

    points = fetch_forecast_along_route(
        lats,
        lons,
        bearings_deg,
        when,
        max_samples=max_samples,
        z0_m=z0_m,
        z_m=z_m,
        session=http,
        timeout_s=timeout_s,
        cache=cache,
    )
    return WeatherFetchResult(points=points, source="open-meteo")
