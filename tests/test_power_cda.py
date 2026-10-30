"""Power-dependent Cd·A on the fitted gravel-hoods path."""

import math

from bike_wind.calibration.cda_defaults import FITTED_CDA, power_cda, power_cda_factor
from bike_wind.physics.forces import RiderBikeParams, aero_force
from bike_wind.rider import resolve_aero


def test_power_cda_curve_points():
    assert math.isclose(power_cda(180.0), 0.5125)
    assert math.isclose(power_cda(250.0), 0.4775)
    # Clamped below 100 W (coasting) and above 350 W.
    assert math.isclose(power_cda(0.0), power_cda(100.0))
    assert math.isclose(power_cda(500.0), power_cda(350.0))
    assert power_cda(300.0) < power_cda(200.0) < power_cda(120.0)
    assert math.isclose(power_cda_factor(195.0) * FITTED_CDA, power_cda(195.0))


def test_fitted_path_turns_power_cda_on_and_presets_off():
    assert resolve_aero("gravel_hoods").power_cda is True
    assert resolve_aero("gravel_hoods", full_aero_kit=True).power_cda is False
    assert resolve_aero("gravel_hoods", apply_fitted_cda=False).power_cda is False


def test_aero_force_scales_with_power():
    base = RiderBikeParams(rider_mass_kg=69, power_w=250, cd=1.0, area_m2=FITTED_CDA)
    on = base.with_updates(power_cda=True)
    f_off = aero_force(10.0, base)
    f_on = aero_force(10.0, on)
    assert math.isclose(f_on / f_off, power_cda(250.0) / FITTED_CDA)
    assert f_on < f_off
