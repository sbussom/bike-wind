"""Actual-vs-predicted helpers: GPX times, speeds, 15-min weather cache."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

from bike_wind.gpx.parse import RouteStreams, parse_gpx, speeds_from_times
from bike_wind.predict import downsample_route, predict_speeds
from bike_wind.physics.forces import RiderBikeParams
from bike_wind.weather.cache import ForecastCache
from bike_wind.weather.fetch import fetch_weather_along_route
from bike_wind.weather.open_meteo import past_days_for


TIMED_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
  <trk><trkseg>
    <trkpt lat="26.1200" lon="-80.1370"><ele>2.0</ele>
      <time>2024-06-01T14:00:00Z</time></trkpt>
    <trkpt lat="26.1210" lon="-80.1370"><ele>4.0</ele>
      <time>2024-06-01T14:00:30Z</time></trkpt>
    <trkpt lat="26.1220" lon="-80.1370"><ele>6.0</ele>
      <time>2024-06-01T14:01:00Z</time></trkpt>
  </trkseg></trk>
</gpx>
"""


def test_parse_gpx_with_times_and_speeds() -> None:
    route = parse_gpx(TIMED_GPX.encode())
    assert route.has_actual_speeds
    assert route.start_time is not None
    assert route.start_time.year == 2024
    assert route.actual_speeds_mps is not None
    assert route.actual_speeds_mps[0] > 0.0
    # ~111 m in 30 s ≈ 3.7 m/s
    assert 2.0 < route.actual_speeds_mps[0] < 6.0
    assert route.mean_grade_pct() > 0.0


def test_speeds_from_times_zero_dt() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [t0, t0, t0 + timedelta(seconds=10)]
    speeds = speeds_from_times([0.0, 10.0, 20.0], times)
    assert len(speeds) == 3
    assert speeds[2] == 1.0


def test_downsample_keeps_actual_speeds() -> None:
    n = 50
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    route = RouteStreams(
        lats=[26.0 + i * 0.001 for i in range(n)],
        lons=[-80.0] * n,
        elevs_m=[float(i) for i in range(n)],
        distances_m=[float(i * 10) for i in range(n)],
        times_utc=[t0 + timedelta(seconds=i) for i in range(n)],
        actual_speeds_mps=[10.0] * n,
    )
    thin = downsample_route(route, max_points=10)
    assert len(thin) <= 10
    assert thin.has_actual_speeds
    assert thin.start_time == t0


def test_quarter_key() -> None:
    when = datetime(2026, 10, 15, 15, 42, tzinfo=timezone.utc)
    assert ForecastCache.quarter_key(when) == "2026-10-15T15:30"


def test_past_days_for() -> None:
    now = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
    assert past_days_for(now + timedelta(hours=3), now=now) == 0
    assert past_days_for(now - timedelta(days=2), now=now) >= 2


def test_fetch_prefers_15min_and_caches(tmp_path: Path) -> None:
    cache = ForecastCache(tmp_path / "wx.sqlite")
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "minutely_15": {
            "time": ["2026-10-15T15:00", "2026-10-15T15:15", "2026-10-15T15:30"],
            "temperature_2m": [27.0, 27.1, 27.2],
            "wind_speed_10m": [4.0, 4.5, 5.0],
            "wind_direction_10m": [90.0, 90.0, 90.0],
        },
        "hourly": {
            "time": ["2026-10-15T15:00"],
            "surface_pressure": [1013.0],
        },
    }
    session.get.return_value = resp
    when = datetime(2026, 10, 15, 15, 20, tzinfo=timezone.utc)
    result = fetch_weather_along_route(
        [26.1, 26.11],
        [-80.1, -80.11],
        [90.0, 90.0],
        when,
        max_samples=1,
        session=session,
        cache=cache,
    )
    assert result.source == "open-meteo-15min"
    assert session.get.call_count == 1
    # Finer cache slot written (max_samples=1 hits the mid index).
    assert cache.get(26.11, -80.11, "2026-10-15T15:15", "open-meteo-15min") is not None

    session.get.reset_mock()
    result2 = fetch_weather_along_route(
        [26.1, 26.11],
        [-80.1, -80.11],
        [90.0, 90.0],
        when,
        max_samples=1,
        session=session,
        cache=cache,
    )
    assert result2.source == "open-meteo-15min"
    session.get.assert_not_called()


def test_predict_uses_ride_start_and_grade(monkeypatch) -> None:
    route = parse_gpx(TIMED_GPX.encode())
    calls: list[datetime] = []

    def fake_wx(lats, lons, bearings, when, **kwargs):
        from bike_wind.weather.fetch import WeatherFetchResult
        from bike_wind.weather.open_meteo import ForecastPoint

        calls.append(when)
        n = len(lats)
        pts = [
            ForecastPoint(
                lat=lats[i],
                lon=lons[i],
                temperature_c=25.0,
                pressure_hpa=1013.0,
                u10_mps=1.0,
                v10_mps=0.0,
                u_rider_mps=0.8,
                v_rider_mps=0.0,
                rho=1.2,
                along_track_mps=0.5,
                cross_track_mps=0.1,
            )
            for i in range(n)
        ]
        return WeatherFetchResult(points=pts, source="open-meteo-15min")

    monkeypatch.setattr("bike_wind.predict.fetch_weather_along_route", fake_wx)
    params = RiderBikeParams(rider_mass_kg=75.0, power_w=200.0)
    ui_when = datetime(2030, 1, 1, tzinfo=timezone.utc)
    result = predict_speeds(route, params, ui_when)
    assert result.used_ride_start_time
    assert calls and calls[0].year == 2024
    assert result.has_actual_speeds
    assert result.mean_grade_pct > 0.0
    assert len(result.grades) == len(route)
