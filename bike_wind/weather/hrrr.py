"""Optional NOAA HRRR (~3 km, CONUS) via Open-Meteo's gfs_hrrr model.

Open-Meteo remains the transport; this avoids shipping Herbie/cfgrib while still
exposing a higher-fidelity CONUS wind source. Outside CONUS or on API failure,
callers must fall back to the default Open-Meteo forecast.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import requests

from bike_wind.physics.constants import RIDER_HEIGHT_M, Z0_OPEN_GRASS
from bike_wind.settings import get_bool_setting
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

# Approximate CONUS coverage for HRRR
CONUS_LAT = (24.0, 50.0)
CONUS_LON = (-125.0, -66.0)

HRRR_URL = "https://api.open-meteo.com/v1/gfs"
HRRR_MODEL = "gfs_hrrr"
SOURCE_KEY = "hrrr"


def hrrr_enabled_by_env() -> bool:
    """True when BIKE_WIND_HRRR is set in env or Streamlit secrets."""
    return get_bool_setting("BIKE_WIND_HRRR", default=False)


def in_conus(lat: float, lon: float) -> bool:
    return CONUS_LAT[0] <= lat <= CONUS_LAT[1] and CONUS_LON[0] <= lon <= CONUS_LON[1]


def route_in_conus(lats: Sequence[float], lons: Sequence[float]) -> bool:
    if not lats:
        return False
    return all(in_conus(la, lo) for la, lo in zip(lats, lons))


def fetch_hrrr_along_route(
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
    """Fetch HRRR (via Open-Meteo gfs_hrrr) for a CONUS route."""
    if not lats or len(lats) != len(lons) or len(lats) != len(bearings_deg):
        raise ValueError("lats, lons, and bearings_deg must be equal non-empty lengths")
    if not route_in_conus(lats, lons):
        raise WeatherFetchError("HRRR is CONUS-only; route is outside coverage")

    when_utc = as_utc(when)
    sample_idxs = sample_indices(len(lats), max_samples)
    http = session or requests.Session()
    cache = cache or ForecastCache()
    forecast_ts = ForecastCache.hour_key(when_utc)

    samples: dict[int, ForecastPoint] = {}
    try:
        for idx in sample_idxs:
            samples[idx] = _fetch_hrrr_one(
                http,
                lat=lats[idx],
                lon=lons[idx],
                bearing_deg=bearings_deg[idx],
                when_utc=when_utc,
                forecast_ts=forecast_ts,
                z0_m=z0_m,
                z_m=z_m,
                timeout_s=timeout_s,
                cache=cache,
            )
    except (requests.RequestException, KeyError, ValueError, IndexError) as exc:
        raise WeatherFetchError(f"HRRR forecast failed: {exc}") from exc

    out: list[ForecastPoint] = []
    for i in range(len(lats)):
        nearest = min(sample_idxs, key=lambda s: abs(s - i))
        out.append(reproject_point(samples[nearest], lats[i], lons[i], bearings_deg[i]))
    return out


def _fetch_hrrr_one(
    http: requests.Session,
    *,
    lat: float,
    lon: float,
    bearing_deg: float,
    when_utc: datetime,
    forecast_ts: str,
    z0_m: float,
    z_m: float,
    timeout_s: float,
    cache: ForecastCache,
) -> ForecastPoint:
    cached = cache.get(lat, lon, forecast_ts, SOURCE_KEY)
    if cached is not None:
        return point_from_raw(cached, lat, lon, bearing_deg, z_m=z_m, z0_m=z0_m)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(
            [
                "temperature_2m",
                "surface_pressure",
                "wind_u_component_10m",
                "wind_v_component_10m",
            ]
        ),
        "models": HRRR_MODEL,
        "timezone": "UTC",
        "forecast_days": 2,
    }
    resp = http.get(HRRR_URL, params=params, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    hourly = payload.get("hourly")
    if not hourly or not hourly.get("time"):
        raise WeatherFetchError("HRRR returned empty hourly data")

    times: list[str] = hourly["time"]
    ti = nearest_hour_index(times, when_utc)
    raw = CachedForecastRaw(
        lat=lat,
        lon=lon,
        forecast_ts=forecast_ts,
        source=SOURCE_KEY,
        temperature_c=float(hourly["temperature_2m"][ti]),
        pressure_hpa=float(hourly["surface_pressure"][ti]),
        u10_mps=float(hourly["wind_u_component_10m"][ti]),
        v10_mps=float(hourly["wind_v_component_10m"][ti]),
    )
    cache.put(raw)
    return point_from_raw(raw, lat, lon, bearing_deg, z_m=z_m, z0_m=z0_m)
