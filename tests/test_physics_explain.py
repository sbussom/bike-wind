"""Physics explain panel uses the same force helpers as the solver."""

from __future__ import annotations

from datetime import datetime, timezone

from bike_wind.gpx.parse import RouteStreams
from bike_wind.physics.forces import RiderBikeParams
from bike_wind.physics_explain import explain_prediction
from bike_wind.predict import apply_what_if, predict_speeds


def test_explain_prediction_fills_ride_numbers(monkeypatch) -> None:
    from bike_wind.weather.fetch import WeatherFetchResult
    from bike_wind.weather.open_meteo import ForecastPoint

    def fake_wx(lats, lons, bearings, when, **kwargs):
        n = len(lats)
        pts = [
            ForecastPoint(
                lat=lats[i],
                lon=lons[i],
                temperature_c=25.0,
                pressure_hpa=1013.0,
                u10_mps=2.0,
                v10_mps=0.0,
                u_rider_mps=1.5,
                v_rider_mps=0.0,
                rho=1.2,
                along_track_mps=0.8,
                cross_track_mps=0.2,
                gust_along_track_mps=1.2,
                gust_cross_track_mps=0.3,
                wind_speed_10m=2.0,
                gust_speed_10m=3.0,
            )
            for i in range(n)
        ]
        return WeatherFetchResult(points=pts, source="open-meteo")

    monkeypatch.setattr("bike_wind.predict.fetch_weather_along_route", fake_wx)
    t0 = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    route = RouteStreams(
        lats=[26.1, 26.11, 26.12],
        lons=[-80.1, -80.1, -80.1],
        elevs_m=[2.0, 3.0, 5.0],
        distances_m=[0.0, 100.0, 200.0],
        times_utc=[t0, t0, t0],
        actual_speeds_mps=[8.0, 8.5, 9.0],
        watts=[180.0, 200.0, 190.0],
    )
    params = RiderBikeParams(rider_mass_kg=70.0, power_w=200.0, area_m2=0.408, cd=1.0)
    result = predict_speeds(route, params, t0)
    whatif = apply_what_if(
        result, power_w=200.0, wind_speed_mps=result.mean_wind_speed_mps, use_recorded_power=True
    )
    explain = explain_prediction(result, whatif, grade_scale=1.0)
    assert explain.lines
    blob = " ".join(line.latex_or_text for line in explain.lines)
    assert "Cd·A" in blob or "Cd" in blob
    assert "W" in blob
    assert explain.distance_mi >= 0.0
