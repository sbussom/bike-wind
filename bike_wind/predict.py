"""End-to-end prediction: GPX + weather → speed profile."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from bike_wind.gpx.parse import RouteStreams
from bike_wind.physics.forces import RiderBikeParams, WindComponents
from bike_wind.physics.solver import solve_speeds
from bike_wind.weather.open_meteo import WeatherFetchError, fetch_forecast_along_route


@dataclass
class PredictionResult:
    distances_m: list[float]
    speeds_with_wind_mps: list[float]
    speeds_no_wind_mps: list[float]
    grades: list[float]
    wind_along_mps: list[float]
    wind_cross_mps: list[float]
    rho_mean: float
    weather_ok: bool
    weather_message: str | None = None

    @property
    def distances_km(self) -> list[float]:
        return [d / 1000.0 for d in self.distances_m]

    @property
    def speeds_with_wind_kmh(self) -> list[float]:
        return [v * 3.6 for v in self.speeds_with_wind_mps]

    @property
    def speeds_no_wind_kmh(self) -> list[float]:
        return [v * 3.6 for v in self.speeds_no_wind_mps]

    @property
    def eta_with_wind_s(self) -> float:
        return _eta_seconds(self.distances_m, self.speeds_with_wind_mps)

    @property
    def eta_no_wind_s(self) -> float:
        return _eta_seconds(self.distances_m, self.speeds_no_wind_mps)


def predict_speeds(
    route: RouteStreams,
    params: RiderBikeParams,
    when: datetime,
    *,
    max_weather_samples: int = 12,
) -> PredictionResult:
    """Solve speeds with and without wind for a route at ``when``."""
    grades = route.grades()
    bearings = route.bearings_deg()

    # No-wind baseline (uses default or caller-supplied rho on params)
    speeds_calm = solve_speeds(params, grades)

    wind_along = [0.0] * len(route)
    wind_cross = [0.0] * len(route)
    rho_mean = params.rho
    weather_ok = False
    weather_message: str | None = None
    speeds_wind = list(speeds_calm)

    try:
        forecast = fetch_forecast_along_route(
            route.lats,
            route.lons,
            bearings,
            when,
            max_samples=max_weather_samples,
        )
        winds = [fp.wind for fp in forecast]
        wind_along = [fp.along_track_mps for fp in forecast]
        wind_cross = [abs(fp.cross_track_mps) for fp in forecast]
        rhos = [fp.rho for fp in forecast]
        rho_mean = sum(rhos) / len(rhos)
        # Use mean density for the winded solve (Phase 0 simplicity)
        params_wind = RiderBikeParams(
            rider_mass_kg=params.rider_mass_kg,
            power_w=params.power_w,
            bike_mass_kg=params.bike_mass_kg,
            cd=params.cd,
            area_m2=params.area_m2,
            crr=params.crr,
            rho=rho_mean,
            tuck_on_descent=params.tuck_on_descent,
        )
        speeds_wind = solve_speeds(params_wind, grades, winds)
        weather_ok = True
    except WeatherFetchError as exc:
        weather_message = (
            f"Weather unavailable ({exc}). Showing calm-air prediction only — "
            "check network access to api.open-meteo.com."
        )
        speeds_wind = list(speeds_calm)

    return PredictionResult(
        distances_m=list(route.distances_m),
        speeds_with_wind_mps=speeds_wind,
        speeds_no_wind_mps=speeds_calm,
        grades=grades,
        wind_along_mps=wind_along,
        wind_cross_mps=wind_cross,
        rho_mean=rho_mean,
        weather_ok=weather_ok,
        weather_message=weather_message,
    )


def downsample_route(route: RouteStreams, max_points: int = 400) -> RouteStreams:
    """Thin a dense GPX for faster charting / weather fan-out."""
    n = len(route)
    if n <= max_points:
        return route
    step = (n - 1) / (max_points - 1)
    idxs = sorted({int(round(i * step)) for i in range(max_points)})
    return RouteStreams(
        lats=[route.lats[i] for i in idxs],
        lons=[route.lons[i] for i in idxs],
        elevs_m=[route.elevs_m[i] for i in idxs],
        distances_m=[route.distances_m[i] for i in idxs],
    )


def _eta_seconds(distances_m: Sequence[float], speeds_mps: Sequence[float]) -> float:
    if len(distances_m) < 2:
        return 0.0
    total = 0.0
    for i in range(len(distances_m) - 1):
        dd = distances_m[i + 1] - distances_m[i]
        v = max(speeds_mps[i], 0.1)
        total += dd / v
    return total
