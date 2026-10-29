"""Grade-weighted KOM pacing."""

from __future__ import annotations

import pytest

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


def test_wind_split_pushes_the_headwind() -> None:
    from bike_wind.analysis import wind_pace_split

    n = 11
    spacing = 400.0
    distances = [i * spacing for i in range(n)]
    grades = [0.0] * n
    winds = [WindComponents() for _ in range(n)]
    for j in range(1, 6):
        winds[j] = WindComponents(along_track=-5.0)
    for j in range(6, n):
        winds[j] = WindComponents(along_track=5.0)
    params = RiderBikeParams(
        rider_mass_kg=69.0,
        power_w=240.5,
        bike_mass_kg=9.0,
        cd=1.0,
        area_m2=0.505,
        crr=0.0052,
    )
    plan = wind_pace_split(240.5, distances, grades, winds, params)
    assert plan.split_useful
    assert plan.headwind_power_w > plan.avg_power_w > plan.tailwind_power_w
    assert plan.seconds_saved > 0.5
    total = plan.headwind_distance_m + plan.tailwind_distance_m + plan.neutral_distance_m
    weighted = (
        plan.headwind_power_w * plan.headwind_distance_m
        + plan.tailwind_power_w * plan.tailwind_distance_m
        + plan.avg_power_w * plan.neutral_distance_m
    )
    assert weighted / total == pytest.approx(240.5, abs=0.05)


def test_crosswind_has_no_split() -> None:
    from bike_wind.analysis import wind_pace_split

    distances = [0.0, 1000.0, 2000.0, 3000.0]
    grades = [0.0, 0.0, 0.0, 0.0]
    winds = [WindComponents(cross_track=4.0) for _ in range(4)]
    params = RiderBikeParams(rider_mass_kg=70.0, power_w=200.0)
    plan = wind_pace_split(200.0, distances, grades, winds, params)
    assert not plan.split_useful
    assert plan.seconds_saved == 0.0
    assert plan.headwind_power_w == pytest.approx(200.0)
    assert plan.tailwind_power_w == pytest.approx(200.0)


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
