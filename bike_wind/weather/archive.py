"""Open-Meteo Historical Weather (archive) client for past-ride validation."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Sequence

import requests

from bike_wind.physics.constants import RIDER_HEIGHT_M, Z0_OPEN_GRASS
from bike_wind.weather.cache import CachedForecastRaw, ForecastCache
from bike_wind.weather.open_meteo import (
    DEFAULT_TIMEOUT_S,
    ForecastPoint,
    WeatherFetchError,
    as_utc,
    nearest_hour_index,
    point_from_raw,
    reproject_point,
    sample_indices,
)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
SOURCE_KEY = "open-meteo-archive"


def speed_dir_to_uv(speed_mps: float, direction_deg: float) -> tuple[float, float]:
    """Meteorological wind FROM ``direction_deg`` → U (east) / V (north)."""
    rad = math.radians(direction_deg)
    u = -speed_mps * math.sin(rad)
    v = -speed_mps * math.cos(rad)
    return u, v


def fetch_archive_along_route(
    lats: Sequence[float],
    lons: Sequence[float],
    bearings_deg: Sequence[float],
    when: datetime,
    *,
    max_samples: int = 12,
    z0_m: float = Z0_OPEN_GRASS,
    z_m: float = RIDER_HEIGHT_M,
    session: requests.Session | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    cache: ForecastCache | None = None,
) -> list[ForecastPoint]:
    """Historical wind along a route at ``when`` (UTC)."""
    if not lats or len(lats) != len(lons) or len(lats) != len(bearings_deg):
        raise ValueError("lats, lons, and bearings_deg must be equal non-empty lengths")

    when_utc = as_utc(when)
    sample_idxs = sample_indices(len(lats), max_samples)
    http = session or requests.Session()
    cache = cache or ForecastCache()
    forecast_ts = ForecastCache.hour_key(when_utc)
    day = when_utc.date().isoformat()

    samples: dict[int, ForecastPoint] = {}
    try:
        for idx in sample_idxs:
            samples[idx] = _fetch_one_archive(
                http,
                lat=lats[idx],
                lon=lons[idx],
                bearing_deg=bearings_deg[idx],
                when_utc=when_utc,
                forecast_ts=forecast_ts,
                day=day,
                z0_m=z0_m,
                z_m=z_m,
                timeout_s=timeout_s,
                cache=cache,
            )
    except (requests.RequestException, KeyError, ValueError, IndexError) as exc:
        raise WeatherFetchError(f"Open-Meteo archive failed: {exc}") from exc

    out: list[ForecastPoint] = []
    for i in range(len(lats)):
        nearest = min(sample_idxs, key=lambda s: abs(s - i))
        out.append(reproject_point(samples[nearest], lats[i], lons[i], bearings_deg[i]))
    return out


def _fetch_one_archive(
    http: requests.Session,
    *,
    lat: float,
    lon: float,
    bearing_deg: float,
    when_utc: datetime,
    forecast_ts: str,
    day: str,
    z0_m: float,
    z_m: float,
    timeout_s: float,
    cache: ForecastCache,
) -> ForecastPoint:
    cached = cache.get(lat, lon, forecast_ts, SOURCE_KEY)
    if cached is not None:
        return point_from_raw(cached, lat, lon, bearing_deg, z_m=z_m, z0_m=z0_m)

    # Pad one day so nearest-hour lookup near midnight still works.
    start = (when_utc.date() - timedelta(days=0)).isoformat()
    end = day
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(
            [
                "temperature_2m",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
            ]
        ),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    resp = http.get(ARCHIVE_URL, params=params, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    hourly = payload["hourly"]
    times: list[str] = hourly["time"]
    ti = nearest_hour_index(times, when_utc)
    rows: list[CachedForecastRaw] = []
    requested: CachedForecastRaw | None = None
    pressure = hourly.get("surface_pressure")
    gusts = hourly.get("wind_gusts_10m")
    for i, stamp in enumerate(times):
        try:
            speed = float(hourly["wind_speed_10m"][i])
            direction = float(hourly["wind_direction_10m"][i])
            temp = float(hourly["temperature_2m"][i])
        except (TypeError, ValueError, IndexError):
            continue
        if not (math.isfinite(speed) and math.isfinite(direction) and math.isfinite(temp)):
            continue
        u10, v10 = speed_dir_to_uv(speed, direction)
        pressure_hpa = 1013.25
        if pressure is not None:
            try:
                pressure_hpa = float(pressure[i])
            except (TypeError, ValueError, IndexError):
                pressure_hpa = 1013.25
        gust_10 = None
        if gusts is not None:
            try:
                gust_10 = float(gusts[i])
            except (TypeError, ValueError, IndexError):
                gust_10 = None
            if gust_10 is not None and not math.isfinite(gust_10):
                gust_10 = None
        slot = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if slot.tzinfo is None:
            slot = slot.replace(tzinfo=timezone.utc)
        raw = CachedForecastRaw(
            lat=lat,
            lon=lon,
            forecast_ts=ForecastCache.hour_key(slot),
            source=SOURCE_KEY,
            temperature_c=temp,
            pressure_hpa=pressure_hpa,
            u10_mps=u10,
            v10_mps=v10,
            gust_10_mps=gust_10,
        )
        rows.append(raw)
        if i == ti:
            requested = raw
    if requested is None:
        raise ValueError("archive hour missing wind")
    if requested.forecast_ts != forecast_ts:
        rows.append(
            CachedForecastRaw(
                lat=lat,
                lon=lon,
                forecast_ts=forecast_ts,
                source=SOURCE_KEY,
                temperature_c=requested.temperature_c,
                pressure_hpa=requested.pressure_hpa,
                u10_mps=requested.u10_mps,
                v10_mps=requested.v10_mps,
                gust_10_mps=requested.gust_10_mps,
            )
        )
    cache.put_many(rows)
    return point_from_raw(requested, lat, lon, bearing_deg, z_m=z_m, z0_m=z0_m)
