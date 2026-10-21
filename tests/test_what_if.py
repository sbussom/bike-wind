"""What-if sliders: flat power, wind speed scale, grade scale."""

from __future__ import annotations

from datetime import datetime, timezone

from bike_wind.physics.forces import RiderBikeParams, WindComponents
from bike_wind.predict import PredictionResult, apply_what_if


def _base_result(*, mean_wind_mps: float = 4.0) -> PredictionResult:
    n = 4
    winds = [
        WindComponents(along_track=mean_wind_mps * 0.8, cross_track=mean_wind_mps * 0.2)
        for _ in range(n)
    ]
    params = RiderBikeParams(
        rider_mass_kg=69.0,
        power_w=200.0,
        bike_mass_kg=9.0,
        cd=1.0,
        area_m2=0.45,
        crr=0.008,
    )
    dists = [0.0, 100.0, 200.0, 300.0]
    grades = [0.0, 0.02, 0.02, 0.0]
    speeds = [8.0] * n
    return PredictionResult(
        distances_m=dists,
        speeds_with_wind_mps=list(speeds),
        speeds_no_wind_mps=list(speeds),
        grades=grades,
        wind_along_mps=[w.along_track for w in winds],
        wind_cross_mps=[w.cross_track for w in winds],
        winds=winds,
        params=params,
        rho_mean=1.2,
        weather_ok=True,
        weather_source="open-meteo-15min",
        weather_when=datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc),
        mean_wind_speed_mps=mean_wind_mps,
        mean_wind_from_deg=90.0,
        watts=[180.0, 220.0, 200.0, 190.0],
        power_source="meter",
    )


def test_default_whatif_power_uses_mean_watts() -> None:
    result = _base_result()
    assert result.default_whatif_power_w == 197.5


def test_higher_power_faster_eta() -> None:
    result = _base_result()
    slow = apply_what_if(result, power_w=150.0, wind_speed_mps=4.0, grade_scale=1.0)
    fast = apply_what_if(result, power_w=300.0, wind_speed_mps=4.0, grade_scale=1.0)
    assert fast.eta_with_wind_s < slow.eta_with_wind_s
    assert fast.speeds_with_wind_mph[0] > slow.speeds_with_wind_mph[0]


def test_more_wind_head_component_changes_speed() -> None:
    # Strong headwind along-track negative
    result = _base_result(mean_wind_mps=5.0)
    result.winds[:] = [WindComponents(along_track=-5.0, cross_track=0.0) for _ in result.winds]
    result.mean_wind_speed_mps = 5.0
    calmish = apply_what_if(result, power_w=220.0, wind_speed_mps=1.0, grade_scale=1.0)
    hard = apply_what_if(result, power_w=220.0, wind_speed_mps=10.0, grade_scale=1.0)
    assert hard.eta_with_wind_s > calmish.eta_with_wind_s


def test_steeper_climbs_slower() -> None:
    result = _base_result()
    flat = apply_what_if(result, power_w=220.0, wind_speed_mps=4.0, grade_scale=0.0)
    steep = apply_what_if(result, power_w=220.0, wind_speed_mps=4.0, grade_scale=2.0)
    assert steep.eta_with_wind_s > flat.eta_with_wind_s


def test_what_if_uses_flat_power_not_watt_stream() -> None:
    result = _base_result()
    # Extreme watt stream would have made middle point much faster if used.
    result.watts = [50.0, 500.0, 50.0, 50.0]
    wi = apply_what_if(result, power_w=200.0, wind_speed_mps=0.0, grade_scale=0.0)
    # Flat power + flat grade → nearly uniform speeds
    assert max(wi.speeds_no_wind_mps) - min(wi.speeds_no_wind_mps) < 0.5
