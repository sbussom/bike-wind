"""Unit tests for cycling physics forces and velocity solver."""

from __future__ import annotations

import math

import pytest

from bike_wind.physics.constants import BIKE_MASS_ROAD_KG, G, RHO_DEFAULT, TUCK_AREA_FACTOR
from bike_wind.physics.forces import (
    RiderBikeParams,
    WindComponents,
    aero_force,
    air_density,
    effective_area,
    gravity_force,
    power_required,
    rolling_force,
    total_force,
)
from bike_wind.physics.solver import solve_speed, solve_speeds
from bike_wind.weather.wind_profile import correct_wind_speed, wind_uv_at_rider_height


def _flat_params(**kwargs: float) -> RiderBikeParams:
    base = dict(rider_mass_kg=75.0, power_w=200.0)
    base.update(kwargs)
    return RiderBikeParams(**base)  # type: ignore[arg-type]


def test_air_density_sea_level_approx() -> None:
    rho = air_density(15.0, 1013.25)
    assert 1.20 < rho < 1.23


def test_rolling_force_flat() -> None:
    p = _flat_params()
    f = rolling_force(p, grade=0.0)
    expected = 0.004 * (75.0 + BIKE_MASS_ROAD_KG) * G
    assert f == pytest.approx(expected, rel=1e-6)


def test_gravity_force_climb() -> None:
    p = _flat_params()
    grade = 0.05  # ~5%
    f = gravity_force(p, grade=grade)
    theta = math.atan(grade)
    expected = p.total_mass_kg * G * math.sin(theta)
    assert f == pytest.approx(expected, rel=1e-6)
    assert f > 0


def test_gravity_force_descent_negative() -> None:
    p = _flat_params()
    assert gravity_force(p, grade=-0.06) < 0


def test_aero_zero_at_still_air_zero_speed() -> None:
    p = _flat_params()
    assert aero_force(0.0, p) == 0.0


def test_aero_scales_with_speed_squared() -> None:
    p = _flat_params()
    f10 = aero_force(10.0, p)
    f20 = aero_force(20.0, p)
    assert f20 == pytest.approx(4.0 * f10, rel=1e-6)


def test_headwind_increases_aero_drag() -> None:
    p = _flat_params()
    calm = aero_force(10.0, p, wind=WindComponents(along_track=0.0))
    head = aero_force(10.0, p, wind=WindComponents(along_track=-5.0))
    assert head > calm


def test_tailwind_reduces_aero_drag() -> None:
    p = _flat_params()
    calm = aero_force(10.0, p, wind=WindComponents(along_track=0.0))
    tail = aero_force(10.0, p, wind=WindComponents(along_track=5.0))
    assert tail < calm


def test_crosswind_adds_drag_not_free_speed() -> None:
    p = _flat_params()
    calm = aero_force(10.0, p, wind=WindComponents(along_track=0.0, cross_track=0.0))
    cross = aero_force(10.0, p, wind=WindComponents(along_track=0.0, cross_track=8.0))
    assert cross > calm


def test_tuck_reduces_area_on_descent() -> None:
    p = _flat_params()
    assert effective_area(p, grade=-0.03) == pytest.approx(p.area_m2 * TUCK_AREA_FACTOR)
    assert effective_area(p, grade=0.03) == pytest.approx(p.area_m2)


def test_power_equals_force_times_velocity() -> None:
    p = _flat_params()
    v = 8.0
    assert power_required(v, p, grade=0.02) == pytest.approx(
        total_force(v, p, grade=0.02) * v, rel=1e-9
    )


def test_solve_speed_matches_power_on_flat() -> None:
    p = _flat_params(power_w=250.0)
    v = solve_speed(p, grade=0.0)
    assert v > 0
    assert power_required(v, p, grade=0.0) == pytest.approx(250.0, abs=0.1)


def test_higher_power_yields_higher_speed() -> None:
    slow = solve_speed(_flat_params(power_w=150.0))
    fast = solve_speed(_flat_params(power_w=300.0))
    assert fast > slow


def test_climb_slower_than_flat() -> None:
    p = _flat_params(power_w=250.0)
    flat = solve_speed(p, grade=0.0)
    climb = solve_speed(p, grade=0.06)
    assert 0 < climb < flat


def test_headwind_slower_than_calm() -> None:
    p = _flat_params(power_w=250.0)
    calm = solve_speed(p, wind=WindComponents(along_track=0.0))
    head = solve_speed(p, wind=WindComponents(along_track=-6.0))
    assert head < calm


def test_impossible_climb_returns_zero() -> None:
    p = _flat_params(power_w=5.0)
    v = solve_speed(p, grade=0.25)  # 25% wall — crawling still needs ≫5 W
    assert v == 0.0


def test_solve_speeds_length() -> None:
    p = _flat_params()
    grades = [0.0, 0.02, -0.01]
    speeds = solve_speeds(p, grades)
    assert len(speeds) == 3
    assert all(s >= 0 for s in speeds)


def test_solve_speeds_variable_power() -> None:
    p = _flat_params(power_w=150.0)
    grades = [0.0, 0.0, 0.0]
    speeds = solve_speeds(p, grades, powers_w=[100.0, 300.0, 100.0])
    assert speeds[1] > speeds[0]
    assert speeds[1] > speeds[2]


def test_log_profile_reduces_10m_wind() -> None:
    u10 = 10.0
    u12 = correct_wind_speed(u10, z_ref_m=10.0, z_m=1.2, z0_m=0.03)
    assert 0 < u12 < u10
    # Manual: ln(1.2/0.03)/ln(10/0.03) * 10
    expected = 10.0 * math.log(1.2 / 0.03) / math.log(10.0 / 0.03)
    assert u12 == pytest.approx(expected, rel=1e-9)


def test_wind_uv_preserves_direction() -> None:
    u, v = wind_uv_at_rider_height(3.0, 4.0)
    assert abs(u / v - 3.0 / 4.0) < 1e-9
    assert math.hypot(u, v) < 5.0


def test_rho_default_constant() -> None:
    assert RHO_DEFAULT == 1.2
