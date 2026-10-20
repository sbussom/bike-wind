"""Tests for past-ride comparison helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bike_wind.gpx.parse import RouteStreams
from bike_wind.past_ride import (
    compare_activity,
    parse_activity_start,
    ride_activity_label,
)
from bike_wind.predict import PredictionResult
from bike_wind.physics.forces import RiderBikeParams, WindComponents
from bike_wind.strava.client import StravaClient, StravaConfig
from bike_wind.strava.models import ActivitySummary


def test_parse_activity_start_z() -> None:
    dt = parse_activity_start("2024-03-15T14:30:00Z")
    assert dt == datetime(2024, 3, 15, 14, 30, tzinfo=timezone.utc)


def test_ride_activity_label_name_date_distance() -> None:
    act = ActivitySummary(
        activity_id=1,
        name="Morning",
        sport_type="Ride",
        start_date="2024-03-15T14:30:00Z",
        distance_m=25000,
        moving_time_s=3600,
        elapsed_time_s=3800,
        average_watts=210.0,
    )
    label = ride_activity_label(act)
    assert "Morning" in label
    assert "2024-03-15" in label
    assert "25.0 km" in label
    assert "210 W" not in label


def test_compare_activity_uses_fallback_power(tmp_path: Path) -> None:
    act = ActivitySummary(
        activity_id=9,
        name="No power",
        sport_type="Ride",
        start_date="2024-03-15T14:30:00Z",
        distance_m=10000,
        moving_time_s=1800,
        elapsed_time_s=1900,
        average_watts=None,
    )
    route = RouteStreams(
        lats=[26.0, 26.01, 26.02],
        lons=[-80.0, -80.01, -80.02],
        elevs_m=[0.0, 1.0, 2.0],
        distances_m=[0.0, 500.0, 1000.0],
    )
    cfg = StravaConfig(
        access_token="t",
        client_id="1",
        client_secret="s",
        redirect_uri=None,
        refresh_token=None,
        cache_dir=tmp_path,
    )
    client = StravaClient(cfg)
    fake_result = PredictionResult(
        distances_m=route.distances_m,
        speeds_with_wind_mps=[8.0, 8.0, 8.0],
        speeds_no_wind_mps=[8.0, 8.0, 8.0],
        grades=[0.0, 0.0, 0.0],
        wind_along_mps=[0.0, 0.0, 0.0],
        wind_cross_mps=[0.0, 0.0, 0.0],
        winds=[WindComponents()] * 3,
        params=RiderBikeParams(rider_mass_kg=75, power_w=200, bike_mass_kg=8),
        rho_mean=1.2,
        weather_ok=True,
        weather_source="open-meteo-archive",
    )
    with patch.object(client, "load_activity_route", return_value=route):
        with patch("bike_wind.past_ride.predict_speeds", return_value=fake_result):
            cmp = compare_activity(
                client,
                act,
                rider_mass_kg=75.0,
                bike_mass_kg=8.0,
                position_key="hoods",
                fallback_power_w=220.0,
                height_cm=175.0,
            )
    assert cmp.power_from_activity is False
    assert cmp.power_w == 220.0
    assert "No activity power" in cmp.note
