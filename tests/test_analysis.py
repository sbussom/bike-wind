"""Unit tests for Phase 2 analysis helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bike_wind.analysis import (
    MPH_TO_MPS,
    build_segment_bins,
    choose_bin_count,
    forecast_horizon_days,
    format_duration,
    parse_target_time,
    power_for_target_time,
    time_uncertainty_band,
    wind_along_uncertainty_mps,
    wind_label,
)
from bike_wind.physics import RiderBikeParams, WindComponents


def _flat_route(n: int = 20, spacing_m: float = 500.0):
    distances = [i * spacing_m for i in range(n)]
    grades = [0.0] * n
    winds = [WindComponents(along_track=-3.0, cross_track=1.0)] * n
    along = [-3.0] * n
    return distances, grades, winds, along


def test_parse_target_time_seconds() -> None:
    assert parse_target_time("750") == 750.0
    assert parse_target_time("") is None
    assert parse_target_time("  ") is None


def test_parse_target_time_mmss() -> None:
    assert parse_target_time("12:30") == 750.0
    assert parse_target_time("1:02:03") == 3723.0


def test_parse_target_time_rejects_bad() -> None:
    with pytest.raises(ValueError):
        parse_target_time("12:30:00:00")
    with pytest.raises(ValueError):
        parse_target_time("0")


def test_format_duration() -> None:
    assert format_duration(750) == "12:30"
    assert format_duration(3723) == "1:02:03"


def test_wind_uncertainty_at_2_3_days_near_1_mph() -> None:
    u = wind_along_uncertainty_mps(2.5)
    assert u == pytest.approx(MPH_TO_MPS, rel=0.05)


def test_wind_uncertainty_grows_past_day_5() -> None:
    assert wind_along_uncertainty_mps(6.0) > wind_along_uncertainty_mps(5.0)
    assert wind_along_uncertainty_mps(0.5) < wind_along_uncertainty_mps(2.5)


def test_forecast_horizon_days() -> None:
    now = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
    when = now + timedelta(days=2, hours=12)
    assert forecast_horizon_days(when, now=now) == pytest.approx(2.5)


def test_power_for_target_faster_needs_more_power() -> None:
    distances, grades, winds, _ = _flat_route()
    params = RiderBikeParams(rider_mass_kg=75.0, power_w=200.0)
    easy = power_for_target_time(3600.0, distances, grades, winds, params)
    hard = power_for_target_time(900.0, distances, grades, winds, params)
    assert hard.required_power_w > easy.required_power_w


def test_implausible_flag_for_crazy_kom() -> None:
    distances, grades, winds, _ = _flat_route(n=10, spacing_m=1000.0)
    params = RiderBikeParams(rider_mass_kg=75.0, power_w=180.0)
    # 10 km in 8 minutes ≈ 75 km/h — should be implausible at 180 W
    result = power_for_target_time(480.0, distances, grades, winds, params)
    assert result.implausible is True
    assert result.required_power_w > params.power_w


def test_achievable_target_near_sustainable() -> None:
    distances, grades, winds, _ = _flat_route()
    params = RiderBikeParams(rider_mass_kg=75.0, power_w=250.0)
    # Use a very loose target so low power suffices
    result = power_for_target_time(7200.0, distances, grades, winds, params)
    assert result.implausible is False
    assert result.required_power_w <= params.power_w


def test_time_band_brackets_nominal() -> None:
    distances, grades, winds, _ = _flat_route()
    params = RiderBikeParams(rider_mass_kg=75.0, power_w=220.0)
    when = datetime.now(timezone.utc) + timedelta(days=2.5)
    band = time_uncertainty_band(distances, grades, winds, params, when)
    assert band.eta_low_s <= band.eta_s <= band.eta_high_s
    assert band.along_uncertainty_mps == pytest.approx(MPH_TO_MPS, rel=0.1)


def test_choose_bin_count() -> None:
    assert choose_bin_count(500.0) == 1
    assert choose_bin_count(8000.0) >= 2
    assert choose_bin_count(20000.0) <= 8


def test_build_segment_bins_multiple() -> None:
    distances, grades, winds, along = _flat_route(n=21, spacing_m=500.0)
    params = RiderBikeParams(rider_mass_kg=75.0, power_w=220.0)
    speeds = [8.0] * 21
    bins = build_segment_bins(
        distances, grades, along, speeds, speeds, params, winds
    )
    assert len(bins) > 1
    assert bins[0].start_km == pytest.approx(0.0)
    assert all(b.wind_label.startswith("Head") for b in bins)


def test_wind_label() -> None:
    assert "Head" in wind_label(-4.0)
    assert "Tail" in wind_label(3.0)
