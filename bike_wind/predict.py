"""End-to-end prediction: GPX + weather → speed profile + Phase 2 analysis."""

from __future__ import annotations

import math
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
from bike_wind.local_wind_map import format_wind_speed_dir, wind_from_uv
from bike_wind.physics.constants import RIDER_HEIGHT_M
from bike_wind.physics.forces import RiderBikeParams, WindComponents
from bike_wind.physics.solver import solve_speeds
from bike_wind.weather.fetch import fetch_weather_along_route
from bike_wind.weather.open_meteo import WeatherFetchError

MPS_TO_MPH = 2.236936


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
    actual_speeds_mps: list[float] | None = None
    watts: list[float] | None = None
    weather_when: datetime | None = None
    mean_grade_pct: float = 0.0
    used_ride_start_time: bool = False
    power_source: str = "entered"  # "meter" | "entered"
    mean_wind_speed_mps: float = 0.0
    mean_wind_from_deg: float = 0.0

    @property
    def distances_km(self) -> list[float]:
        return [d / 1000.0 for d in self.distances_m]

    @property
    def distances_mi(self) -> list[float]:
        return [d / 1609.344 for d in self.distances_m]

    @property
    def speeds_with_wind_kmh(self) -> list[float]:
        return [v * 3.6 for v in self.speeds_with_wind_mps]

    @property
    def speeds_no_wind_kmh(self) -> list[float]:
        return [v * 3.6 for v in self.speeds_no_wind_mps]

    @property
    def speeds_with_wind_mph(self) -> list[float]:
        return [v * MPS_TO_MPH for v in self.speeds_with_wind_mps]

    @property
    def speeds_no_wind_mph(self) -> list[float]:
        return [v * MPS_TO_MPH for v in self.speeds_no_wind_mps]

    @property
    def actual_speeds_kmh(self) -> list[float] | None:
        if not self.actual_speeds_mps:
            return None
        return [v * 3.6 for v in self.actual_speeds_mps]

    @property
    def actual_speeds_mph(self) -> list[float] | None:
        if not self.actual_speeds_mps:
            return None
        return [v * MPS_TO_MPH for v in self.actual_speeds_mps]

    @property
    def has_actual_speeds(self) -> bool:
        if not self.actual_speeds_mps or len(self.actual_speeds_mps) != len(
            self.distances_m
        ):
            return False
        return any(math.isfinite(v) for v in self.actual_speeds_mps)

    @property
    def has_watts(self) -> bool:
        if not self.watts or len(self.watts) != len(self.distances_m):
            return False
        return any(math.isfinite(w) and w > 0.0 for w in self.watts)

    @property
    def eta_with_wind_s(self) -> float:
        return eta_seconds(self.distances_m, self.speeds_with_wind_mps)

    @property
    def eta_no_wind_s(self) -> float:
        return eta_seconds(self.distances_m, self.speeds_no_wind_mps)

    @property
    def power_source_note(self) -> str:
        if self.power_source == "meter":
            return "Model uses recorded power-meter watts along the route."
        return "Model uses entered rider power (no power meter on this ride)."

    def wind_label(self, *, unit: str = "mph") -> str:
        """Source, time, and speed+direction used for the with-wind case."""
        when = self.weather_when
        when_s = when.strftime("%Y-%m-%d %H:%M UTC") if when else "—"
        if not self.weather_ok:
            return f"Wind unavailable · calm prediction only ({when_s})"
        spd_dir = format_wind_speed_dir(
            self.mean_wind_speed_mps,
            self.mean_wind_from_deg,
            unit=unit,
        )
        return f"{spd_dir} · {self.weather_source} at {when_s}"

    @property
    def default_whatif_power_w(self) -> float:
        """Flat power default: mean recorded watts, else sustained params power."""
        if self.has_watts and self.watts is not None:
            finite = [w for w in self.watts if math.isfinite(w) and w > 0.0]
            if finite:
                return sum(finite) / len(finite)
        return float(self.params.power_w)

    @property
    def mean_wind_speed_mph(self) -> float:
        return self.mean_wind_speed_mps * MPS_TO_MPH


@dataclass(frozen=True)
class WhatIfResult:
    """Speeds/ETAs after power, wind-speed, and grade-scale what-if sliders."""

    speeds_with_wind_mps: list[float]
    speeds_no_wind_mps: list[float]
    eta_with_wind_s: float
    eta_no_wind_s: float
    power_w: float
    wind_speed_mps: float
    wind_from_deg: float
    grade_scale: float  # 1.0 = 100% of route grades

    @property
    def speeds_with_wind_mph(self) -> list[float]:
        return [v * MPS_TO_MPH for v in self.speeds_with_wind_mps]

    @property
    def speeds_no_wind_mph(self) -> list[float]:
        return [v * MPS_TO_MPH for v in self.speeds_no_wind_mps]

    @property
    def wind_speed_mph(self) -> float:
        return self.wind_speed_mps * MPS_TO_MPH


def apply_what_if(
    result: PredictionResult,
    *,
    power_w: float,
    wind_speed_mps: float,
    grade_scale: float = 1.0,
) -> WhatIfResult:
    """Re-solve with flat power, scaled wind speed (same direction), scaled grades.

    Does not use the recorded watts stream for the model — that stays on the
    chart as the Power trace. Grade scale of 1.0 is the route as recorded.
    """
    n = len(result.grades)
    scale_g = max(0.0, float(grade_scale))
    grades = [g * scale_g for g in result.grades]

    base_wind = max(0.0, float(result.mean_wind_speed_mps))
    target_wind = max(0.0, float(wind_speed_mps))
    if result.weather_ok and base_wind > 1e-6:
        wind_scale = target_wind / base_wind
        winds = [
            WindComponents(
                along_track=w.along_track * wind_scale,
                cross_track=w.cross_track * wind_scale,
            )
            for w in result.winds
        ]
    elif result.weather_ok and target_wind <= 1e-6:
        winds = [WindComponents()] * n
    elif result.weather_ok:
        # Forecast was calm; keep calm along-track structure (still zero).
        winds = list(result.winds) if len(result.winds) == n else [WindComponents()] * n
    else:
        winds = [WindComponents()] * n

    params = RiderBikeParams(
        rider_mass_kg=result.params.rider_mass_kg,
        power_w=max(1.0, float(power_w)),
        bike_mass_kg=result.params.bike_mass_kg,
        cd=result.params.cd,
        area_m2=result.params.area_m2,
        crr=result.params.crr,
        rho=result.params.rho,
        tuck_on_descent=result.params.tuck_on_descent,
        drivetrain_efficiency=result.params.drivetrain_efficiency,
    )
    # Flat power only (no powers_w) for the what-if model.
    speeds_calm = solve_speeds(params, grades)
    speeds_wind = (
        solve_speeds(params, grades, winds) if result.weather_ok else list(speeds_calm)
    )
    return WhatIfResult(
        speeds_with_wind_mps=speeds_wind,
        speeds_no_wind_mps=speeds_calm,
        eta_with_wind_s=eta_seconds(result.distances_m, speeds_wind),
        eta_no_wind_s=eta_seconds(result.distances_m, speeds_calm),
        power_w=params.power_w,
        wind_speed_mps=target_wind,
        wind_from_deg=result.mean_wind_from_deg,
        grade_scale=scale_g,
    )


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
    prefer_ride_start_time: bool = True,
) -> PredictionResult:
    """Solve speeds with and without wind; attach time band, KOM power, bins.

    Grade (rise/run from elevation) is always passed into the ground-speed
    solver. When the route has a recorded start time and
    ``prefer_ride_start_time`` is True, weather is fetched for that instant
    instead of ``when``. When the route has power-meter ``watts``, those drive
    the model; otherwise ``params.power_w`` is used.
    """
    grades = route.grades()
    bearings = route.bearings_deg()
    used_ride_start = False
    weather_when = when
    if prefer_ride_start_time and route.start_time is not None:
        weather_when = route.start_time
        used_ride_start = True

    powers_w: list[float] | None = None
    power_source = "entered"
    if route.has_watts and route.watts is not None:
        powers_w = list(route.watts)
        power_source = "meter"
        finite = [w for w in powers_w if math.isfinite(w) and w > 0.0]
        if finite:
            params = RiderBikeParams(
                rider_mass_kg=params.rider_mass_kg,
                power_w=sum(finite) / len(finite),
                bike_mass_kg=params.bike_mass_kg,
                cd=params.cd,
                area_m2=params.area_m2,
                crr=params.crr,
                rho=params.rho,
                tuck_on_descent=params.tuck_on_descent,
                drivetrain_efficiency=params.drivetrain_efficiency,
            )

    speeds_calm = solve_speeds(params, grades, powers_w=powers_w)

    wind_along = [0.0] * len(route)
    wind_cross = [0.0] * len(route)
    winds: list[WindComponents] = [WindComponents()] * len(route)
    rho_mean = params.rho
    weather_ok = False
    weather_message: str | None = None
    weather_source = "none"
    speeds_wind = list(speeds_calm)
    params_used = params
    mean_wind_speed = 0.0
    mean_wind_from = 0.0

    try:
        wx = fetch_weather_along_route(
            route.lats,
            route.lons,
            bearings,
            weather_when,
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
            drivetrain_efficiency=params.drivetrain_efficiency,
        )
        # Grade from elevation is required input to the speed solver.
        speeds_wind = solve_speeds(params_used, grades, winds, powers_w=powers_w)
        weather_ok = True
        if forecast:
            u_mean = sum(fp.u_rider_mps for fp in forecast) / len(forecast)
            v_mean = sum(fp.v_rider_mps for fp in forecast) / len(forecast)
            mean_wind_speed, mean_wind_from = wind_from_uv(u_mean, v_mean)
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
        weather_when,
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
    actual = list(route.actual_speeds_mps) if route.has_actual_speeds else None
    watts_out = list(route.watts) if route.has_watts and route.watts is not None else None

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
        actual_speeds_mps=actual,
        watts=watts_out,
        weather_when=weather_when,
        mean_grade_pct=route.mean_grade_pct(),
        used_ride_start_time=used_ride_start,
        power_source=power_source,
        mean_wind_speed_mps=mean_wind_speed,
        mean_wind_from_deg=mean_wind_from,
    )


def downsample_route(route: RouteStreams, max_points: int = 400) -> RouteStreams:
    """Thin a dense GPX for faster charting / weather fan-out."""
    n = len(route)
    if n <= max_points:
        return route
    step = (n - 1) / (max_points - 1)
    idxs = sorted({int(round(i * step)) for i in range(max_points)})
    times = (
        [route.times_utc[i] for i in idxs] if route.times_utc is not None else None
    )
    speeds = (
        [route.actual_speeds_mps[i] for i in idxs]
        if route.actual_speeds_mps is not None
        else None
    )
    watts = (
        [route.watts[i] for i in idxs] if route.watts is not None else None
    )
    return RouteStreams(
        lats=[route.lats[i] for i in idxs],
        lons=[route.lons[i] for i in idxs],
        elevs_m=[route.elevs_m[i] for i in idxs],
        distances_m=[route.distances_m[i] for i in idxs],
        times_utc=times,
        actual_speeds_mps=speeds,
        watts=watts,
    )
