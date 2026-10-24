"""Grade-weighted KOM pacing."""

from __future__ import annotations

from bike_wind.analysis import (
    grade_power_weights,
    pace_plan_for_target_time,
    power_for_target_time,
)
from bike_wind.physics.forces import RiderBikeParams, WindComponents
from bike_wind.strava.client import parse_kom_time_s


def test_parse_kom_time_s() -> None:
    assert parse_kom_time_s("1:23") == 83.0
    assert parse_kom_time_s("1:02:03") == 3723.0
    assert parse_kom_time_s(120) == 120.0
    assert parse_kom_time_s(None) is None


def test_grade_weights_harder_on_climbs() -> None:
    w = grade_power_weights([0.0, 0.05, -0.05])
    assert w[1] > w[0] > w[2]


def test_pace_plan_peaks_on_climb() -> None:
    # Flat then climb then flat; calm wind.
    distances = [0.0, 500.0, 1000.0, 1500.0]
    grades = [0.0, 0.06, 0.0]
    # grades length must match samples for solver — pad
    grades = [0.0, 0.06, 0.06, 0.0]
    winds = [WindComponents()] * 4
    params = RiderBikeParams(rider_mass_kg=70.0, power_w=200.0)
    flat = power_for_target_time(400.0, distances, grades, winds, params)
    plan = pace_plan_for_target_time(400.0, distances, grades, winds, params)
    assert plan.peak_power_w >= plan.avg_power_w
    assert plan.peak_power_w > flat.required_power_w * 0.9
    # Climb samples should get more watts than flat samples
    assert plan.powers_w[1] > plan.powers_w[0]
