"""Open-Meteo forecast client (hourly wind U/V, temperature, pressure)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import requests

from bike_wind.physics.forces import WindComponents, air_density
from bike_wind.physics.constants import RIDER_HEIGHT_M, RHO_DEFAULT, Z0_OPEN_GRASS
from bike_wind.weather.wind_profile import wind_uv_at_rider_height

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT_S = 20


@dataclass(frozen=True)
class ForecastPoint:
    """Weather + derived fields at one route sample."""

    lat: float
    lon: float
    temperature_c: float
    pressure_hpa: float
    u10_mps: float
    v10_mps: float
    u_rider_mps: float
    v_rider_mps: float
    rho: float
    along_track_mps: float
    cross_track_mps: float

    @property
    def wind(self) -> WindComponents:
        return WindComponents(
            along_track=self.along_track_mps,
            cross_track=abs(self.cross_track_mps),
        )


class WeatherFetchError(RuntimeError):
    """Raised when the forecast API cannot be reached or returns bad data."""


def fetch_forecast_along_route(
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
) -> list[ForecastPoint]:
    """Query Open-Meteo forecast at subsampled route points for ``when``.

    Returns one ForecastPoint per input coordinate (nearest subsample broadcast
    to neighbors). Raises WeatherFetchError on network / API failure.
    """
    if not lats or len(lats) != len(lons) or len(lats) != len(bearings_deg):
        raise ValueError("lats, lons, and bearings_deg must be equal non-empty lengths")

    when_utc = _as_utc(when)
    sample_idxs = _sample_indices(len(lats), max_samples)
    http = session or requests.Session()

    samples: dict[int, ForecastPoint] = {}
    try:
        for idx in sample_idxs:
            samples[idx] = _fetch_one(
                http,
                lat=lats[idx],
                lon=lons[idx],
                bearing_deg=bearings_deg[idx],
                when_utc=when_utc,
                z0_m=z0_m,
                z_m=z_m,
                timeout_s=timeout_s,
            )
    except (requests.RequestException, KeyError, ValueError, IndexError) as exc:
        raise WeatherFetchError(f"Open-Meteo forecast failed: {exc}") from exc

    # Broadcast nearest sample to every route point
    out: list[ForecastPoint] = []
    for i in range(len(lats)):
        nearest = min(sample_idxs, key=lambda s: abs(s - i))
        base = samples[nearest]
        # Recompute along/cross for this point's bearing using same U/V
        along, cross = _along_cross(base.u_rider_mps, base.v_rider_mps, bearings_deg[i])
        out.append(
            ForecastPoint(
                lat=lats[i],
                lon=lons[i],
                temperature_c=base.temperature_c,
                pressure_hpa=base.pressure_hpa,
                u10_mps=base.u10_mps,
                v10_mps=base.v10_mps,
                u_rider_mps=base.u_rider_mps,
                v_rider_mps=base.v_rider_mps,
                rho=base.rho,
                along_track_mps=along,
                cross_track_mps=cross,
            )
        )
    return out


def _fetch_one(
    http: requests.Session,
    *,
    lat: float,
    lon: float,
    bearing_deg: float,
    when_utc: datetime,
    z0_m: float,
    z_m: float,
    timeout_s: float,
) -> ForecastPoint:
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
        "timezone": "UTC",
        "forecast_days": 16,
    }
    resp = http.get(OPEN_METEO_URL, params=params, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    hourly = payload["hourly"]
    times: list[str] = hourly["time"]
    ti = _nearest_hour_index(times, when_utc)

    temp_c = float(hourly["temperature_2m"][ti])
    # surface_pressure is hPa in Open-Meteo
    pressure_hpa = float(hourly["surface_pressure"][ti])
    u10 = float(hourly["wind_u_component_10m"][ti])
    v10 = float(hourly["wind_v_component_10m"][ti])
    u_r, v_r = wind_uv_at_rider_height(u10, v10, z_m=z_m, z0_m=z0_m)
    along, cross = _along_cross(u_r, v_r, bearing_deg)
    rho = air_density(temp_c, pressure_hpa) if pressure_hpa > 0 else RHO_DEFAULT

    return ForecastPoint(
        lat=lat,
        lon=lon,
        temperature_c=temp_c,
        pressure_hpa=pressure_hpa,
        u10_mps=u10,
        v10_mps=v10,
        u_rider_mps=u_r,
        v_rider_mps=v_r,
        rho=rho,
        along_track_mps=along,
        cross_track_mps=cross,
    )


def _along_cross(u: float, v: float, bearing_deg: float) -> tuple[float, float]:
    """Project meteorological U (east)/V (north) onto route bearing.

    along: positive = tailwind (wind blowing in travel direction).
    cross: signed left/right; callers typically take abs for drag.
    """
    br = math.radians(bearing_deg)
    # Unit travel vector: east = sin(bearing), north = cos(bearing)
    along = u * math.sin(br) + v * math.cos(br)
    cross = u * math.cos(br) - v * math.sin(br)
    return along, cross


def _sample_indices(n: int, max_samples: int) -> list[int]:
    if n <= max_samples:
        return list(range(n))
    if max_samples == 1:
        return [n // 2]
    step = (n - 1) / (max_samples - 1)
    return sorted({int(round(i * step)) for i in range(max_samples)})


def _as_utc(when: datetime) -> datetime:
    if when.tzinfo is None:
        return when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc)


def _nearest_hour_index(times: Sequence[str], when_utc: datetime) -> int:
    target = when_utc.replace(minute=0, second=0, microsecond=0)
    best_i = 0
    best_dt = None
    for i, t in enumerate(times):
        # Open-Meteo: "2024-01-15T14:00"
        parsed = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        delta = abs((parsed - target).total_seconds())
        if best_dt is None or delta < best_dt:
            best_dt = delta
            best_i = i
    return best_i
