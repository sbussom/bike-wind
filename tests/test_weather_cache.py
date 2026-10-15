"""Tests for forecast SQLite cache and HRRR CONUS / fallback helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bike_wind.weather.cache import CachedForecastRaw, ForecastCache
from bike_wind.weather.fetch import fetch_weather_along_route
from bike_wind.weather.hrrr import in_conus, route_in_conus
from bike_wind.weather.open_meteo import WeatherFetchError, fetch_forecast_along_route


def test_forecast_cache_roundtrip(tmp_path: Path) -> None:
    cache = ForecastCache(tmp_path / "wx.sqlite")
    ts = "2026-10-15T15:00"
    raw = CachedForecastRaw(
        lat=26.1224,
        lon=-80.1373,
        forecast_ts=ts,
        source="open-meteo",
        temperature_c=28.0,
        pressure_hpa=1012.0,
        u10_mps=3.0,
        v10_mps=-1.0,
    )
    cache.put(raw)
    got = cache.get(26.1224, -80.1373, ts, "open-meteo")
    assert got is not None
    assert got.u10_mps == 3.0
    # Nearby coords round to same key
    assert cache.get(26.12241, -80.13729, ts, "open-meteo") is not None


def test_hour_key() -> None:
    when = datetime(2026, 10, 15, 15, 42, tzinfo=timezone.utc)
    assert ForecastCache.hour_key(when) == "2026-10-15T15:00"


def test_conus_bounds() -> None:
    assert in_conus(26.1, -80.1) is True
    assert in_conus(51.5, -0.1) is False  # London
    assert route_in_conus([26.0, 26.1], [-80.1, -80.0]) is True
    assert route_in_conus([26.0, 51.5], [-80.1, -0.1]) is False


def test_open_meteo_uses_cache(tmp_path: Path) -> None:
    cache = ForecastCache(tmp_path / "wx.sqlite")
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "hourly": {
            "time": ["2026-10-15T15:00"],
            "temperature_2m": [27.0],
            "surface_pressure": [1013.0],
            "wind_u_component_10m": [2.0],
            "wind_v_component_10m": [1.0],
        }
    }
    resp.raise_for_status = MagicMock()
    session.get.return_value = resp

    when = datetime(2026, 10, 15, 15, 0, tzinfo=timezone.utc)
    pts = fetch_forecast_along_route(
        [26.1, 26.11],
        [-80.1, -80.11],
        [90.0, 90.0],
        when,
        max_samples=1,
        session=session,
        cache=cache,
    )
    assert len(pts) == 2
    assert session.get.call_count == 1

    session.get.reset_mock()
    pts2 = fetch_forecast_along_route(
        [26.1, 26.11],
        [-80.1, -80.11],
        [90.0, 90.0],
        when,
        max_samples=1,
        session=session,
        cache=cache,
    )
    assert len(pts2) == 2
    session.get.assert_not_called()


def test_hrrr_outside_conus_falls_back(tmp_path: Path) -> None:
    cache = ForecastCache(tmp_path / "wx.sqlite")
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "hourly": {
            "time": ["2026-10-15T15:00"],
            "temperature_2m": [10.0],
            "surface_pressure": [1015.0],
            "wind_u_component_10m": [1.0],
            "wind_v_component_10m": [0.0],
        }
    }
    session.get.return_value = resp
    when = datetime(2026, 10, 15, 15, 0, tzinfo=timezone.utc)
    # London — outside CONUS
    result = fetch_weather_along_route(
        [51.5, 51.51],
        [-0.12, -0.11],
        [90.0, 90.0],
        when,
        prefer_hrrr=True,
        max_samples=1,
        session=session,
        cache=cache,
    )
    assert result.source == "open-meteo"
    assert result.message is not None
    assert "CONUS" in result.message


def test_hrrr_failure_falls_back(tmp_path: Path) -> None:
    cache = ForecastCache(tmp_path / "wx.sqlite")
    session = MagicMock()

    def side_effect(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "gfs" in url:
            resp.status_code = 200
            resp.json.return_value = {"hourly": {}}  # empty → HRRR error
            return resp
        resp.status_code = 200
        resp.json.return_value = {
            "hourly": {
                "time": ["2026-10-15T15:00"],
                "temperature_2m": [28.0],
                "surface_pressure": [1012.0],
                "wind_u_component_10m": [4.0],
                "wind_v_component_10m": [-2.0],
            }
        }
        return resp

    session.get.side_effect = side_effect
    when = datetime(2026, 10, 15, 15, 0, tzinfo=timezone.utc)
    result = fetch_weather_along_route(
        [26.1, 26.11],
        [-80.1, -80.11],
        [0.0, 0.0],
        when,
        prefer_hrrr=True,
        max_samples=1,
        session=session,
        cache=cache,
    )
    assert result.source == "open-meteo"
    assert result.message is not None
    assert "HRRR" in result.message
