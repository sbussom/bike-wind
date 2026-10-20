"""Unified weather fetch: finest available wind, with automatic cache upgrade."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

import requests

from bike_wind.physics.constants import RIDER_HEIGHT_M, Z0_OPEN_GRASS
from bike_wind.weather.archive import fetch_archive_along_route
from bike_wind.weather.cache import ForecastCache
from bike_wind.weather.hrrr import fetch_hrrr_along_route, route_in_conus
from bike_wind.weather.open_meteo import (
    DEFAULT_TIMEOUT_S,
    ForecastPoint,
    WeatherFetchError,
    as_utc,
    fetch_forecast_15min_along_route,
    fetch_forecast_along_route,
    past_days_for,
)


@dataclass(frozen=True)
class WeatherFetchResult:
    points: list[ForecastPoint]
    source: str  # "open-meteo-15min" | "open-meteo" | "hrrr" | "open-meteo-archive"
    message: str | None = None


def fetch_weather_along_route(
    lats: Sequence[float],
    lons: Sequence[float],
    bearings_deg: Sequence[float],
    when: datetime,
    *,
    prefer_hrrr: bool = False,
    use_archive: bool = False,
    max_samples: int = 12,
    z0_m: float = Z0_OPEN_GRASS,
    z_m: float = RIDER_HEIGHT_M,
    session: requests.Session | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    cache: ForecastCache | None = None,
) -> WeatherFetchResult:
    """Fetch the most granular wind available for ``when`` along the route.

    Preference order (automatic — no user picker):
    1. Open-Meteo 15-minutely when the API returns it (also covers recent past
       via ``past_days``); writes/upgrades the weather SQLite cache.
    2. HRRR via Open-Meteo when ``prefer_hrrr`` and the route is CONUS
       (finer spatial grid; hourly).
    3. Open-Meteo hourly forecast.
    4. Open-Meteo historical archive (hourly) for older past times, or when
       ``use_archive`` is forced.
    """
    cache = cache or ForecastCache()
    http = session or requests.Session()
    when_utc = as_utc(when)
    now = datetime.now(timezone.utc)
    age = now - when_utc
    # Forecast product with past_days covers ~92 days; older → archive.
    too_old_for_forecast = age > timedelta(days=90)
    force_archive = use_archive or too_old_for_forecast

    if force_archive and age > timedelta(hours=1):
        points = fetch_archive_along_route(
            lats,
            lons,
            bearings_deg,
            when_utc,
            max_samples=max_samples,
            z0_m=z0_m,
            z_m=z_m,
            session=http,
            timeout_s=timeout_s,
            cache=cache,
        )
        return WeatherFetchResult(points=points, source="open-meteo-archive")

    # 1) Finest temporal: 15-min Open-Meteo (also works for recent past).
    try:
        points = fetch_forecast_15min_along_route(
            lats,
            lons,
            bearings_deg,
            when_utc,
            max_samples=max_samples,
            z0_m=z0_m,
            z_m=z_m,
            session=http,
            timeout_s=timeout_s,
            cache=cache,
        )
        return WeatherFetchResult(points=points, source="open-meteo-15min")
    except WeatherFetchError:
        pass

    # 2) Optional HRRR (CONUS spatial fidelity) when requested.
    hrrr_msg: str | None = None
    if prefer_hrrr and route_in_conus(lats, lons):
        try:
            points = fetch_hrrr_along_route(
                lats,
                lons,
                bearings_deg,
                when_utc,
                max_samples=max_samples,
                z0_m=z0_m,
                z_m=z_m,
                session=http,
                timeout_s=timeout_s,
                cache=cache,
            )
            return WeatherFetchResult(points=points, source="hrrr")
        except WeatherFetchError as exc:
            hrrr_msg = f"HRRR unavailable ({exc}); falling back."
    elif prefer_hrrr:
        hrrr_msg = "HRRR is CONUS-only — using Open-Meteo forecast."

    # 3) Hourly forecast (with past_days when needed).
    if past_days_for(when_utc, now=now) <= 92 and not too_old_for_forecast:
        try:
            points = fetch_forecast_along_route(
                lats,
                lons,
                bearings_deg,
                when_utc,
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
                message=hrrr_msg,
            )
        except WeatherFetchError:
            if age <= timedelta(hours=1):
                raise

    # 4) Archive for past rides when forecast products fail / are too old.
    points = fetch_archive_along_route(
        lats,
        lons,
        bearings_deg,
        when_utc,
        max_samples=max_samples,
        z0_m=z0_m,
        z_m=z_m,
        session=http,
        timeout_s=timeout_s,
        cache=cache,
    )
    msg = hrrr_msg
    if msg:
        msg = f"{msg} Using archive reanalysis."
    return WeatherFetchResult(
        points=points,
        source="open-meteo-archive",
        message=msg,
    )
