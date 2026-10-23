"""3-second power alignment and recorded-power what-if mode."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bike_wind.physics.forces import RiderBikeParams, WindComponents
from bike_wind.predict import PredictionResult, apply_what_if, power_series_3s


def test_power_series_3s_skips_roll_when_already_slow() -> None:
    t0 = datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc)
    # 3 s samples — already near the head-unit average window.
    times = [t0 + timedelta(seconds=3 * i) for i in range(5)]
    watts = [100.0, 200.0, 300.0, 200.0, 100.0]
    out, window = power_series_3s(watts, times, window_s=3.0)
    assert window is None
    assert out == watts


def test_power_series_3s_rolls_fast_samples() -> None:
    t0 = datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc)
    # 1 Hz samples — roll a trailing 3 s mean.
    times = [t0 + timedelta(seconds=i) for i in range(6)]
    watts = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    out, window = power_series_3s(watts, times, window_s=3.0)
    assert window == 3.0
    # Index 0: only itself
    assert out[0] == 100.0
    # Index 2: samples at t=0,1,2 → (100+200+300)/3
    assert abs(out[2] - 200.0) < 1e-9
    # Index 5: t=3,4,5 within last 3s from t=5 (t=2 is exactly 3s back, included)
    # window is [t5-3, t5] = [t2, t5] → 300+400+500+600
    assert abs(out[5] - 450.0) < 1e-9


def test_power_series_3s_no_times_trusts_raw() -> None:
    watts = [150.0, 250.0]
    out, window = power_series_3s(watts, None)
    assert window is None
    assert out == watts


def _result_with_watts() -> PredictionResult:
    n = 4
    winds = [WindComponents(along_track=0.0, cross_track=0.0) for _ in range(n)]
    params = RiderBikeParams(
        rider_mass_kg=69.0,
        power_w=200.0,
        bike_mass_kg=9.0,
        cd=1.0,
        area_m2=0.45,
        crr=0.008,
    )
    return PredictionResult(
        distances_m=[0.0, 100.0, 200.0, 300.0],
        speeds_with_wind_mps=[8.0] * n,
        speeds_no_wind_mps=[8.0] * n,
        grades=[0.0, 0.0, 0.0, 0.0],
        wind_along_mps=[0.0] * n,
        wind_cross_mps=[0.0] * n,
        winds=winds,
        params=params,
        rho_mean=1.2,
        weather_ok=True,
        weather_source="open-meteo-15min",
        weather_when=datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc),
        mean_wind_speed_mps=0.0,
        mean_wind_from_deg=90.0,
        watts=[50.0, 400.0, 50.0, 50.0],
        power_source="meter",
        times_utc=[
            datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 1, 14, 0, 30, tzinfo=timezone.utc),
            datetime(2024, 6, 1, 14, 1, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 1, 14, 1, 30, tzinfo=timezone.utc),
        ],
    )


def test_apply_what_if_recorded_uses_watt_stream() -> None:
    result = _result_with_watts()
    wi = apply_what_if(
        result,
        power_w=200.0,
        wind_speed_mps=0.0,
        grade_scale=0.0,
        use_recorded_power=True,
    )
    assert wi.used_recorded_power
    assert wi.powers_w == [50.0, 400.0, 50.0, 50.0]
    # Middle point at 400 W much faster than neighbors at 50 W.
    assert wi.speeds_no_wind_mps[1] > wi.speeds_no_wind_mps[0] * 1.5
    assert abs(wi.power_w - 137.5) < 1e-9  # mean caption


def test_apply_what_if_flat_ignores_watt_stream() -> None:
    result = _result_with_watts()
    wi = apply_what_if(
        result,
        power_w=200.0,
        wind_speed_mps=0.0,
        grade_scale=0.0,
        use_recorded_power=False,
    )
    assert not wi.used_recorded_power
    assert wi.powers_w is None
    assert max(wi.speeds_no_wind_mps) - min(wi.speeds_no_wind_mps) < 0.5


def test_actual_elapsed_from_times() -> None:
    result = _result_with_watts()
    assert result.actual_elapsed_s == 90.0
