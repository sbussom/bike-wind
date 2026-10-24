"""Open-Meteo forecast client (15-min when available, else hourly wind)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import requests

from bike_wind.physics.constants import RIDER_HEIGHT_M, RHO_DEFAULT, Z0_OPEN_GRASS
from bike_wind.physics.forces import WindComponents, air_density
from bike_wind.weather.cache import CachedForecastRaw, ForecastCache
from bike_wind.weather.wind_profile import wind_uv_at_rider_height

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT_S = 20
SOURCE_KEY = "open-meteo"
SOURCE_KEY_15MIN = "open-meteo-15min"


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
    # Open-Meteo wind_gusts_10m projected the same way as sustained wind.
    # None → no gust product; band falls back to sustained.
    gust_along_track_mps: float | None = None
    gust_cross_track_mps: float | None = None
    wind_speed_10m: float | None = None
    gust_speed_10m: float | None = None

    @property
    def wind(self) -> WindComponents:
        return WindComponents(
            along_track=self.along_track_mps,
            cross_track=abs(self.cross_track_mps),
        )

    @property
    def gust_wind(self) -> WindComponents:
        """Stronger same-direction wind from gust speed (or sustained if missing)."""
        if self.gust_along_track_mps is None:
            return self.wind
        return WindComponents(
            along_track=self.gust_along_track_mps,
            cross_track=abs(self.gust_cross_track_mps or 0.0),
        )

    @property
    def light_gust_wind(self) -> WindComponents:
        """Weaker bound: scale sustained down by sustained/gust (toward calmer)."""
        spd = self.wind_speed_10m
        gst = self.gust_speed_10m
        if (
            spd is None
            or gst is None
            or spd <= 1e-6
            or gst <= spd
            or self.gust_along_track_mps is None
        ):
            return self.wind
        factor = spd / gst  # < 1
        return WindComponents(
            along_track=self.along_track_mps * factor,
            cross_track=abs(self.cross_track_mps) * factor,
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
    cache: ForecastCache | None = None,
) -> list[ForecastPoint]:
    """Query Open-Meteo forecast at subsampled route points for ``when``.

    Uses SQLite cache keyed by (lat, lon, forecast_ts). Raises WeatherFetchError
    on network / API failure.
    """
    if not lats or len(lats) != len(lons) or len(lats) != len(bearings_deg):
        raise ValueError("lats, lons, and bearings_deg must be equal non-empty lengths")

    when_utc = as_utc(when)
    sample_idxs = sample_indices(len(lats), max_samples)
    http = session or requests.Session()
    cache = cache or ForecastCache()
    forecast_ts = ForecastCache.hour_key(when_utc)

    samples: dict[int, ForecastPoint] = {}
    try:
        for idx in sample_idxs:
            samples[idx] = fetch_one(
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
        raise WeatherFetchError(f"Open-Meteo forecast failed: {exc}") from exc

    out: list[ForecastPoint] = []
    for i in range(len(lats)):
        nearest = min(sample_idxs, key=lambda s: abs(s - i))
        out.append(reproject_point(samples[nearest], lats[i], lons[i], bearings_deg[i]))
    return out


def fetch_one(
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

    params: dict[str, object] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(
            [
                "temperature_2m",
                "surface_pressure",
                "wind_u_component_10m",
                "wind_v_component_10m",
                "wind_gusts_10m",
            ]
        ),
        "timezone": "UTC",
        "forecast_days": 16,
    }
    past_days = past_days_for(when_utc)
    if past_days > 0:
        params["past_days"] = past_days
    resp = http.get(OPEN_METEO_URL, params=params, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    hourly = payload["hourly"]
    times: list[str] = hourly["time"]
    ti = nearest_hour_index(times, when_utc)
    gusts = hourly.get("wind_gusts_10m")
    gust_10 = float(gusts[ti]) if gusts else None

    raw = CachedForecastRaw(
        lat=lat,
        lon=lon,
        forecast_ts=forecast_ts,
        source=SOURCE_KEY,
        temperature_c=float(hourly["temperature_2m"][ti]),
        pressure_hpa=float(hourly["surface_pressure"][ti]),
        u10_mps=float(hourly["wind_u_component_10m"][ti]),
        v10_mps=float(hourly["wind_v_component_10m"][ti]),
        gust_10_mps=gust_10,
    )
    cache.put(raw)
    return point_from_raw(raw, lat, lon, bearing_deg, z_m=z_m, z0_m=z0_m)


def point_from_raw(
    raw: CachedForecastRaw,
    lat: float,
    lon: float,
    bearing_deg: float,
    *,
    z_m: float,
    z0_m: float,
) -> ForecastPoint:
    u_r, v_r = wind_uv_at_rider_height(raw.u10_mps, raw.v10_mps, z_m=z_m, z0_m=z0_m)
    along, cross = along_cross(u_r, v_r, bearing_deg)
    rho = (
        air_density(raw.temperature_c, raw.pressure_hpa)
        if raw.pressure_hpa > 0
        else RHO_DEFAULT
    )
    speed_10 = math.hypot(raw.u10_mps, raw.v10_mps)
    gust_along = None
    gust_cross = None
    gust_10 = raw.gust_10_mps
    if gust_10 is not None and math.isfinite(gust_10) and speed_10 > 1e-6:
        # Same direction as sustained; magnitude from Open-Meteo gust product.
        factor = max(1.0, float(gust_10) / speed_10)
        u_g, v_g = wind_uv_at_rider_height(
            raw.u10_mps * factor, raw.v10_mps * factor, z_m=z_m, z0_m=z0_m
        )
        gust_along, gust_cross = along_cross(u_g, v_g, bearing_deg)
    return ForecastPoint(
        lat=lat,
        lon=lon,
        temperature_c=raw.temperature_c,
        pressure_hpa=raw.pressure_hpa,
        u10_mps=raw.u10_mps,
        v10_mps=raw.v10_mps,
        u_rider_mps=u_r,
        v_rider_mps=v_r,
        rho=rho,
        along_track_mps=along,
        cross_track_mps=cross,
        gust_along_track_mps=gust_along,
        gust_cross_track_mps=gust_cross,
        wind_speed_10m=speed_10,
        gust_speed_10m=float(gust_10) if gust_10 is not None else None,
    )


def along_cross(u: float, v: float, bearing_deg: float) -> tuple[float, float]:
    """Project meteorological U (east)/V (north) onto route bearing."""
    br = math.radians(bearing_deg)
    along = u * math.sin(br) + v * math.cos(br)
    cross = u * math.cos(br) - v * math.sin(br)
    return along, cross


def reproject_point(
    base: ForecastPoint, lat: float, lon: float, bearing_deg: float
) -> ForecastPoint:
    """Copy thermo/wind from ``base`` onto a new sample bearing."""
    along, cross = along_cross(base.u_rider_mps, base.v_rider_mps, bearing_deg)
    gust_along = None
    gust_cross = None
    if (
        base.gust_speed_10m is not None
        and base.wind_speed_10m is not None
        and base.wind_speed_10m > 1e-6
        and base.gust_speed_10m >= base.wind_speed_10m
    ):
        factor = base.gust_speed_10m / base.wind_speed_10m
        gust_along, gust_cross = along_cross(
            base.u_rider_mps * factor, base.v_rider_mps * factor, bearing_deg
        )
    return ForecastPoint(
        lat=lat,
        lon=lon,
        temperature_c=base.temperature_c,
        pressure_hpa=base.pressure_hpa,
        u10_mps=base.u10_mps,
        v10_mps=base.v10_mps,
        u_rider_mps=base.u_rider_mps,
        v_rider_mps=base.v_rider_mps,
        rho=base.rho,
        along_track_mps=along,
        cross_track_mps=cross,
        gust_along_track_mps=gust_along,
        gust_cross_track_mps=gust_cross,
        wind_speed_10m=base.wind_speed_10m,
        gust_speed_10m=base.gust_speed_10m,
    )


def sample_indices(n: int, max_samples: int) -> list[int]:
    if n <= max_samples:
        return list(range(n))
    if max_samples == 1:
        return [n // 2]
    step = (n - 1) / (max_samples - 1)
    return sorted({int(round(i * step)) for i in range(max_samples)})


def as_utc(when: datetime) -> datetime:
    if when.tzinfo is None:
        return when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc)


def nearest_hour_index(times: Sequence[str], when_utc: datetime) -> int:
    target = when_utc.replace(minute=0, second=0, microsecond=0)
    return nearest_time_index(times, target)


def nearest_time_index(times: Sequence[str], when_utc: datetime) -> int:
    best_i = 0
    best_dt: float | None = None
    for i, t in enumerate(times):
        parsed = datetime.fromisoformat(t.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        delta = abs((parsed - when_utc).total_seconds())
        if best_dt is None or delta < best_dt:
            best_dt = delta
            best_i = i
    return best_i


def past_days_for(when_utc: datetime, *, now: datetime | None = None) -> int:
    """How many past_days the forecast API needs to cover ``when_utc``."""
    now = now or datetime.now(timezone.utc)
    if when_utc >= now:
        return 0
    delta_h = (now - when_utc).total_seconds() / 3600.0
    return max(1, min(92, int(math.ceil(delta_h / 24.0)) + 1))


def fetch_forecast_15min_along_route(
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
    """Open-Meteo 15-minutely wind along a route (updates cache under finer source)."""
    if not lats or len(lats) != len(lons) or len(lats) != len(bearings_deg):
        raise ValueError("lats, lons, and bearings_deg must be equal non-empty lengths")

    when_utc = as_utc(when)
    sample_idxs = sample_indices(len(lats), max_samples)
    http = session or requests.Session()
    cache = cache or ForecastCache()
    forecast_ts = ForecastCache.quarter_key(when_utc)
    past_days = past_days_for(when_utc)

    samples: dict[int, ForecastPoint] = {}
    try:
        for idx in sample_idxs:
            samples[idx] = fetch_one_15min(
                http,
                lat=lats[idx],
                lon=lons[idx],
                bearing_deg=bearings_deg[idx],
                when_utc=when_utc,
                forecast_ts=forecast_ts,
                past_days=past_days,
                z0_m=z0_m,
                z_m=z_m,
                timeout_s=timeout_s,
                cache=cache,
            )
    except (requests.RequestException, KeyError, ValueError, IndexError) as exc:
        raise WeatherFetchError(f"Open-Meteo 15-min forecast failed: {exc}") from exc

    out: list[ForecastPoint] = []
    for i in range(len(lats)):
        nearest = min(sample_idxs, key=lambda s: abs(s - i))
        out.append(reproject_point(samples[nearest], lats[i], lons[i], bearings_deg[i]))
    return out


def fetch_one_15min(
    http: requests.Session,
    *,
    lat: float,
    lon: float,
    bearing_deg: float,
    when_utc: datetime,
    forecast_ts: str,
    past_days: int,
    z0_m: float,
    z_m: float,
    timeout_s: float,
    cache: ForecastCache,
) -> ForecastPoint:
    cached = cache.get(lat, lon, forecast_ts, SOURCE_KEY_15MIN)
    if cached is not None:
        return point_from_raw(cached, lat, lon, bearing_deg, z_m=z_m, z0_m=z0_m)

    params: dict[str, object] = {
        "latitude": lat,
        "longitude": lon,
        "minutely_15": ",".join(
            [
                "temperature_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
            ]
        ),
        "hourly": "surface_pressure",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "forecast_days": 16,
    }
    if past_days > 0:
        params["past_days"] = past_days
    resp = http.get(OPEN_METEO_URL, params=params, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    m15 = payload.get("minutely_15") or {}
    times: list[str] = list(m15.get("time") or [])
    if len(times) < 2:
        raise ValueError("minutely_15 series empty")
    ti = nearest_time_index(times, when_utc)
    speed = float(m15["wind_speed_10m"][ti])
    direction = float(m15["wind_direction_10m"][ti])
    # Meteorological FROM → U east / V north
    rad = math.radians(direction)
    u10 = -speed * math.sin(rad)
    v10 = -speed * math.cos(rad)
    temp = float(m15["temperature_2m"][ti])
    gusts = m15.get("wind_gusts_10m")
    gust_10 = float(gusts[ti]) if gusts else None

    pressure_hpa = 1013.25
    hourly = payload.get("hourly") or {}
    h_times = list(hourly.get("time") or [])
    h_press = hourly.get("surface_pressure")
    if h_times and h_press:
        hi = nearest_hour_index(h_times, when_utc)
        pressure_hpa = float(h_press[hi])

    raw = CachedForecastRaw(
        lat=lat,
        lon=lon,
        forecast_ts=forecast_ts,
        source=SOURCE_KEY_15MIN,
        temperature_c=temp,
        pressure_hpa=pressure_hpa,
        u10_mps=u10,
        v10_mps=v10,
        gust_10_mps=gust_10,
    )
    # Always write the finer slot — upgrades a prior hourly-only cache for this place/time.
    cache.put(raw)
    return point_from_raw(raw, lat, lon, bearing_deg, z_m=z_m, z0_m=z0_m)


# Back-compat aliases used by hrrr module
_along_cross = along_cross
_sample_indices = sample_indices
_as_utc = as_utc
_nearest_hour_index = nearest_hour_index
