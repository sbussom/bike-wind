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


def test_yaw_cda_factor_extra_drag_when_enabled() -> None:
    from bike_wind.physics.forces import yaw_cda_factor

    assert yaw_cda_factor(10.0, 0.0) == pytest.approx(1.0)
    assert yaw_cda_factor(5.0, 5.0) > 1.0
    assert yaw_cda_factor(0.0, 20.0) == pytest.approx(1.25)
    off = _flat_params(yaw_cda=False)
    on = _flat_params(yaw_cda=True)
    wind = WindComponents(along_track=0.0, cross_track=8.0)
    assert aero_force(10.0, on, wind=wind) > aero_force(10.0, off, wind=wind)
    # No crosswind → yaw factor is 1, same as off.
    assert aero_force(10.0, on) == pytest.approx(aero_force(10.0, off))


def test_clothing_cda_scale_default_unchanged() -> None:
    p = _flat_params()
    assert p.clothing_cda_scale == 1.0
    assert p.cda_m2 == pytest.approx(p.cd * p.area_m2)


def test_tuck_reduces_area_on_descent() -> None:
    p = _flat_params()
    assert effective_area(p, grade=-0.03) == pytest.approx(p.area_m2 * TUCK_AREA_FACTOR)
    assert effective_area(p, grade=0.03) == pytest.approx(p.area_m2)


def test_power_equals_force_times_velocity() -> None:
    p = _flat_params()
    v = 8.0
    mech = total_force(v, p, grade=0.02) * v
    assert power_required(v, p, grade=0.02) == pytest.approx(
        mech / p.drivetrain_efficiency, rel=1e-9
    )


def test_drivetrain_loss_slows_speed() -> None:
    perfect = _flat_params(power_w=250.0)
    perfect = RiderBikeParams(
        rider_mass_kg=perfect.rider_mass_kg,
        power_w=250.0,
        bike_mass_kg=perfect.bike_mass_kg,
        cd=perfect.cd,
        area_m2=perfect.area_m2,
        crr=perfect.crr,
        drivetrain_efficiency=1.0,
    )
    lossy = RiderBikeParams(
        rider_mass_kg=perfect.rider_mass_kg,
        power_w=250.0,
        bike_mass_kg=perfect.bike_mass_kg,
        cd=perfect.cd,
        area_m2=perfect.area_m2,
        crr=perfect.crr,
        drivetrain_efficiency=0.97,
    )
    assert solve_speed(lossy) < solve_speed(perfect)


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


def test_inertia_power_step_approaches_equilibrium() -> None:
    """A power step must not jump to the new eq on the next sample."""
    p = _flat_params(power_w=150.0)
    # Flat course, 5 m samples — long enough to see a lag, short of full settle.
    n = 41
    distances = [float(i * 5) for i in range(n)]
    grades = [0.0] * n
    step_at = 10
    powers = [150.0] * step_at + [300.0] * (n - step_at)
    # Seed near the low-power cruise so the step lag is measurable.
    v_lo = solve_speed(_flat_params(power_w=150.0), grade=0.0)
    v_hi = solve_speed(_flat_params(power_w=300.0), grade=0.0)
    speeds = solve_speeds(
        p,
        grades,
        powers_w=powers,
        distances_m=distances,
        initial_speed_mps=v_lo,
    )
    assert v_hi > v_lo

    # First sample after the step: still closer to the old equilibrium.
    v_just_after = speeds[step_at]
    assert v_just_after < 0.5 * (v_lo + v_hi)
    assert v_just_after < v_hi * 0.92

    # Far downstream: nearer the high-power equilibrium than right after the step.
    assert speeds[-1] > v_just_after
    assert abs(speeds[-1] - v_hi) < abs(v_just_after - v_hi)
    assert speeds[-1] > 0.5 * (v_lo + v_hi)


def test_short_coast_keeps_momentum() -> None:
    """A brief 0 W must not collapse to a stop or a 0.1 m/s crawl."""
    from bike_wind.analysis import eta_seconds

    p = _flat_params(power_w=250.0)
    v0 = solve_speed(_flat_params(power_w=250.0), grade=0.0)
    # 1 Hz-ish: 8 m gaps, five seconds of coast after a cruise.
    n = 12
    distances = [float(i * 8) for i in range(n)]
    grades = [0.0] * n
    powers = [250.0] * 6 + [0.0] * (n - 6)
    gap: list[float] = []
    speeds = solve_speeds(
        p,
        grades,
        powers_w=powers,
        distances_m=distances,
        initial_speed_mps=v0,
        gap_dt=gap,
    )
    # Still rolling at the end of a 48 m coast. Steady-state 0 W is a stop.
    assert speeds[-1] > 3.0
    assert min(speeds[6:]) > 3.0
    # Time is a few seconds, not distance / 0.1 m/s.
    elapsed = eta_seconds(distances, speeds, gap_dt=gap)
    assert elapsed < 40.0
    crawl = sum(
        (distances[i + 1] - distances[i]) / 0.1 for i in range(len(distances) - 1)
    )
    assert elapsed < crawl * 0.25


def test_planned_zero_watts_is_coasting() -> None:
    """A planned 0 W segment uses the same integrator: coast, don't false-stop."""
    p = _flat_params(power_w=200.0)
    v0 = solve_speed(_flat_params(power_w=200.0), grade=0.0)
    n = 16
    distances = [float(i * 10) for i in range(n)]
    grades = [0.0] * n
    powers = [200.0] * 4 + [0.0] * 6 + [200.0] * (n - 10)
    speeds = solve_speeds(
        p,
        grades,
        powers_w=powers,
        distances_m=distances,
        initial_speed_mps=v0,
    )
    # Mid-coast sample is still near the cruise, not under 0.5 m/s.
    assert speeds[7] > 0.7 * v0
    # Power coming back does not require a standing start.
    assert speeds[11] > 2.0


def test_recorded_stop_is_not_moving_time() -> None:
    from bike_wind.analysis import eta_seconds

    distances = [0.0, 10.0, 10.2, 20.0]
    speeds = [8.0, 0.0, 8.0, 8.0]
    # Middle gap is a recorded stop (actual < 0.5). It must not add crawl time.
    actual = [8.0, 0.1, 8.0, 8.0]
    moving = eta_seconds(distances, speeds, actual_speeds_mps=actual)
    # 10 m at 8 m/s + ~10 m at 8 m/s. The 0.2 m stop is excluded.
    assert moving == pytest.approx(20.0 / 8.0, rel=0.05)


def test_zero_watts_coasts_not_default_power() -> None:
    """Recorded 0 W must coast; must not substitute sustained params.power_w."""
    p = _flat_params(power_w=250.0)
    n = 20
    distances = [float(i * 10) for i in range(n)]
    grades = [0.0] * n
    # Cruise then explicit zeros (coasting).
    powers = [250.0] * 5 + [0.0] * (n - 5)
    v0 = solve_speed(_flat_params(power_w=250.0), grade=0.0)
    speeds = solve_speeds(
        p,
        grades,
        powers_w=powers,
        distances_m=distances,
        initial_speed_mps=v0,
    )
    # After several zero-watt samples, speed must drop well below cruise.
    assert speeds[-1] < v0 * 0.7
    assert speeds[-1] < speeds[4]


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
