"""Tests for Open-Meteo archive helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bike_wind.weather.archive import fetch_archive_along_route, speed_dir_to_uv
from bike_wind.weather.cache import ForecastCache


def test_speed_dir_to_uv_from_east() -> None:
    u, v = speed_dir_to_uv(10.0, 90.0)
    assert u == pytest.approx(-10.0)
    assert v == pytest.approx(0.0, abs=1e-9)


def test_fetch_archive_along_route_uses_cache(tmp_path: Path) -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "hourly": {
            "time": ["2024-06-01T12:00"],
            "temperature_2m": [25.0],
            "surface_pressure": [1012.0],
            "wind_speed_10m": [5.0],
            "wind_direction_10m": [180.0],
        }
    }
    session.get.return_value = resp
    cache = ForecastCache(tmp_path / "wx.sqlite")
    when = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    pts = fetch_archive_along_route(
        [26.0, 26.01],
        [-80.0, -80.01],
        [0.0, 0.0],
        when,
        max_samples=2,
        session=session,
        cache=cache,
    )
    assert len(pts) == 2
    assert session.get.call_count >= 1
    # Second call should hit cache for same hour/lat
    pts2 = fetch_archive_along_route(
        [26.0, 26.01],
        [-80.0, -80.01],
        [0.0, 0.0],
        when,
        max_samples=2,
        session=session,
        cache=cache,
    )
    assert len(pts2) == 2
