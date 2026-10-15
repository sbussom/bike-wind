"""Tests for rider presets, aero toggles, and unit helpers."""

from __future__ import annotations

import pytest

from bike_wind.physics import RiderBikeParams, solve_speed
from bike_wind.physics.constants import BIKE_MASS_ROAD_KG, BIKE_MASS_TT_KG
from bike_wind.rider import (
    POSITION_PRESETS,
    kg_to_lb,
    lb_to_kg,
    resolve_aero,
    smart_default_height_cm,
    standing_height_to_wind_z_m,
)


def test_position_presets_match_spec() -> None:
    assert POSITION_PRESETS["tt_aero"].cd == 0.6
    assert POSITION_PRESETS["tt_aero"].area_m2 == 0.32
    assert POSITION_PRESETS["tt_aero"].bike_mass_kg == BIKE_MASS_TT_KG
    assert POSITION_PRESETS["drops"].cd == 0.85
    assert POSITION_PRESETS["drops"].area_m2 == 0.38
    assert POSITION_PRESETS["drops"].bike_mass_kg == BIKE_MASS_ROAD_KG
    assert POSITION_PRESETS["hoods"].cd == 0.95
    assert POSITION_PRESETS["hoods"].area_m2 == 0.42
    assert POSITION_PRESETS["upright"].cd == 1.1
    assert POSITION_PRESETS["upright"].area_m2 == 0.50


def test_aero_kit_deltas() -> None:
    base = resolve_aero("drops")
    aero = resolve_aero("drops", full_aero_kit=True)
    assert aero.cd == pytest.approx(base.cd - 0.05)
    assert aero.area_m2 == pytest.approx(base.area_m2 - 0.02)


def test_non_aero_delta() -> None:
    base = resolve_aero("hoods")
    non = resolve_aero("hoods", non_aero=True)
    assert non.cd == pytest.approx(base.cd + 0.03)
    assert non.area_m2 == pytest.approx(base.area_m2)


def test_both_toggles_prefers_aero_kit() -> None:
    resolved = resolve_aero("drops", full_aero_kit=True, non_aero=True)
    assert resolved.full_aero_kit is True
    assert resolved.non_aero is False
    assert resolved.cd == pytest.approx(0.85 - 0.05)


def test_tt_default_bike_mass() -> None:
    assert resolve_aero("tt_aero").bike_mass_kg == BIKE_MASS_TT_KG


def test_manual_bike_mass_override() -> None:
    assert resolve_aero("drops", bike_mass_kg=9.2).bike_mass_kg == 9.2


def test_kg_lb_roundtrip() -> None:
    assert lb_to_kg(kg_to_lb(75.0)) == pytest.approx(75.0, rel=1e-9)


def test_smart_height_at_75kg() -> None:
    assert smart_default_height_cm(75.0) == 170.0


def test_wind_z_near_default_at_170cm() -> None:
    assert standing_height_to_wind_z_m(170.0) == pytest.approx(1.19, abs=0.01)


def test_lower_cda_is_faster() -> None:
    drops = resolve_aero("drops")
    tt = resolve_aero("tt_aero")
    p_drops = RiderBikeParams(
        rider_mass_kg=75.0,
        power_w=250.0,
        bike_mass_kg=drops.bike_mass_kg,
        cd=drops.cd,
        area_m2=drops.area_m2,
    )
    p_tt = RiderBikeParams(
        rider_mass_kg=75.0,
        power_w=250.0,
        bike_mass_kg=tt.bike_mass_kg,
        cd=tt.cd,
        area_m2=tt.area_m2,
    )
    assert solve_speed(p_tt) > solve_speed(p_drops)
