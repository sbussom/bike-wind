"""End-to-end prediction: GPX + weather → speed profile + Phase 2 analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from bike_wind.analysis import (
    SegmentBin,
    TargetPowerResult,
    TimeBand,
    build_segment_bins,
    eta_seconds,
    head_tail_summary,
    power_for_target_time,
    time_uncertainty_band,
)
from bike_wind.gpx.parse import RouteStreams
from bike_wind.physics.constants import RIDER_HEIGHT_M
from bike_wind.physics.forces import RiderBikeParams, WindComponents
from bike_wind.physics.solver import solve_speeds
from bike_wind.weather.fetch import fetch_weather_along_route
from bike_wind.weather.open_meteo import WeatherFetchError


@dataclass
class PredictionResult:
    distances_m: list[float]
    speeds_with_wind_mps: list[float]
    speeds_no_wind_mps: list[float]
    grades: list[float]
    wind_along_mps: list[float]
    wind_cross_mps: list[float]
    winds: list[WindComponents]
    params: RiderBikeParams
    rho_mean: float
    weather_ok: bool
    lats: list[float] = field(default_factory=list)
    lons: list[float] = field(default_factory=list)
    weather_message: str | None = None
    weather_source: str = "none"
    time_band: TimeBand | None = None
    target_power: TargetPowerResult | None = None
    bins: list[SegmentBin] = field(default_factory=list)
    mean_wind_along_mps: float = 0.0
    mean_headwind_mps: float = 0.0
    mean_tailwind_mps: float = 0.0

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
        return eta_seconds(self.distances_m, self.speeds_with_wind_mps)

    @property
    def eta_no_wind_s(self) -> float:
        return eta_seconds(self.distances_m, self.speeds_no_wind_mps)


def predict_speeds(
    route: RouteStreams,
    params: RiderBikeParams,
    when: datetime,
    *,
    max_weather_samples: int = 12,
    wind_height_m: float = RIDER_HEIGHT_M,
    target_time_s: float | None = None,
    prefer_hrrr: bool = False,
    use_archive: bool = False,
) -> PredictionResult:
    """Solve speeds with and without wind; attach time band, KOM power, bins."""
    grades = route.grades()
    bearings = route.bearings_deg()

    speeds_calm = solve_speeds(params, grades)

    wind_along = [0.0] * len(route)
    wind_cross = [0.0] * len(route)
    winds: list[WindComponents] = [WindComponents()] * len(route)
    rho_mean = params.rho
    weather_ok = False
    weather_message: str | None = None
    weather_source = "none"
    speeds_wind = list(speeds_calm)
    params_used = params

    try:
        wx = fetch_weather_along_route(
            route.lats,
            route.lons,
            bearings,
            when,
            prefer_hrrr=prefer_hrrr and not use_archive,
            use_archive=use_archive,
            max_samples=max_weather_samples,
            z_m=wind_height_m,
        )
        forecast = wx.points
        weather_source = wx.source
        weather_message = wx.message
        winds = [fp.wind for fp in forecast]
        wind_along = [fp.along_track_mps for fp in forecast]
        wind_cross = [abs(fp.cross_track_mps) for fp in forecast]
        rhos = [fp.rho for fp in forecast]
        rho_mean = sum(rhos) / len(rhos)
        params_used = RiderBikeParams(
            rider_mass_kg=params.rider_mass_kg,
            power_w=params.power_w,
            bike_mass_kg=params.bike_mass_kg,
            cd=params.cd,
            area_m2=params.area_m2,
            crr=params.crr,
            rho=rho_mean,
            tuck_on_descent=params.tuck_on_descent,
        )
        speeds_wind = solve_speeds(params_used, grades, winds)
        weather_ok = True
    except WeatherFetchError as exc:
        weather_message = (
            f"Weather unavailable ({exc}). Showing calm-air prediction only — "
            "check network access to api.open-meteo.com."
        )
        speeds_wind = list(speeds_calm)

    band = time_uncertainty_band(
        route.distances_m,
        grades,
        winds,
        params_used,
        when,
    )

    target: TargetPowerResult | None = None
    if target_time_s is not None:
        target = power_for_target_time(
            target_time_s,
            route.distances_m,
            grades,
            winds,
            params_used,
        )

    bins = build_segment_bins(
        route.distances_m,
        grades,
        wind_along,
        speeds_wind,
        speeds_calm,
        params_used,
        winds,
        target=target,
    )

    mean_along, mean_head, mean_tail = head_tail_summary(wind_along)

    return PredictionResult(
        distances_m=list(route.distances_m),
        speeds_with_wind_mps=speeds_wind,
        speeds_no_wind_mps=speeds_calm,
        grades=grades,
        wind_along_mps=wind_along,
        wind_cross_mps=wind_cross,
        winds=winds,
        params=params_used,
        rho_mean=rho_mean,
        weather_ok=weather_ok,
        lats=list(route.lats),
        lons=list(route.lons),
        weather_message=weather_message,
        weather_source=weather_source,
        time_band=band,
        target_power=target,
        bins=bins,
        mean_wind_along_mps=mean_along,
        mean_headwind_mps=mean_head,
        mean_tailwind_mps=mean_tail,
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
