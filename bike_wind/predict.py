"""End-to-end prediction: GPX + weather → speed profile + Phase 2 analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
    watts: list[float] | None = None  # series used for the model / power chart
    times_utc: list[datetime] | None = None
    weather_when: datetime | None = None
    mean_grade_pct: float = 0.0
    used_ride_start_time: bool = False
    power_source: str = "entered"  # "meter" | "entered"
    mean_wind_speed_mps: float = 0.0
    mean_wind_from_deg: float = 0.0
    power_window_s: float | None = None  # 3.0 if rolled; None if raw / entered
    # Gust envelope around the sustained-wind line (not a spiked alternate path).
    speeds_gust_high_mps: list[float] | None = None
    speeds_gust_low_mps: list[float] | None = None
    mean_gust_speed_mps: float | None = None

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
    def speeds_gust_high_mph(self) -> list[float] | None:
        if self.speeds_gust_high_mps is None:
            return None
        return [v * MPS_TO_MPH for v in self.speeds_gust_high_mps]

    @property
    def speeds_gust_low_mph(self) -> list[float] | None:
        if self.speeds_gust_low_mps is None:
            return None
        return [v * MPS_TO_MPH for v in self.speeds_gust_low_mps]

    @property
    def has_gust_band(self) -> bool:
        return (
            self.speeds_gust_high_mps is not None
            and self.speeds_gust_low_mps is not None
            and len(self.speeds_gust_high_mps) == len(self.distances_m)
            and len(self.speeds_gust_low_mps) == len(self.distances_m)
        )

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

    @property
    def actual_elapsed_s(self) -> float | None:
        """Elapsed time from recorded timestamps when present."""
        if not self.times_utc or len(self.times_utc) < 2:
            return None
        dt = (self.times_utc[-1] - self.times_utc[0]).total_seconds()
        return dt if dt > 0 else None


def _initial_speed_mps(actual_speeds_mps: list[float] | None) -> float | None:
    """First finite recorded speed as the inertia integrator's initial state."""
    if not actual_speeds_mps:
        return None
    for v in actual_speeds_mps:
        if math.isfinite(v) and v >= 0.0:
            return float(v)
    return None


def power_series_3s(
    watts: Sequence[float],
    times_utc: Sequence[datetime] | None = None,
    *,
    window_s: float = 3.0,
) -> tuple[list[float], float | None]:
    """Align power to a ~3 s average for the speed solver.

    Head units often already emit a ~3 s average. If the median sample interval
    is already near that window, return the recorded series unchanged (do not
    double-smooth). Faster streams get a trailing ``window_s`` rolling mean.

    Returns ``(powers, window_applied_or_None)``.
    """
    n = len(watts)
    if n == 0:
        return [], None

    def _finite(w: float) -> float:
        return float(w) if math.isfinite(w) else float("nan")

    raw = [_finite(w) for w in watts]
    if times_utc is None or len(times_utc) != n or n < 2:
        # No timing → trust the recorded series (often already smoothed).
        return raw, None

    dts: list[float] = []
    for i in range(1, n):
        dt = (times_utc[i] - times_utc[i - 1]).total_seconds()
        if dt > 0.0:
            dts.append(dt)
    if not dts:
        return raw, None
    dts_sorted = sorted(dts)
    median_dt = dts_sorted[len(dts_sorted) // 2]
    # Already ~3 s (or slower): use as-is.
    if median_dt >= window_s * 0.75:
        return raw, None

    out: list[float] = []
    for i in range(n):
        t_end = times_utc[i]
        t_start = t_end - timedelta(seconds=window_s)
        vals: list[float] = []
        for j in range(i, -1, -1):
            if times_utc[j] < t_start:
                break
            w = raw[j]
            if math.isfinite(w) and w >= 0.0:
                vals.append(w)
        if vals:
            out.append(sum(vals) / len(vals))
        else:
            out.append(raw[i])
    return out, window_s


@dataclass(frozen=True)
class WhatIfResult:
    """Speeds/ETAs after power mode, wind-speed, and grade-scale what-if."""

    speeds_with_wind_mps: list[float]
    speeds_no_wind_mps: list[float]
    eta_with_wind_s: float
    eta_no_wind_s: float
    power_w: float  # mean / flat watts used for captions
    wind_speed_mps: float
    wind_from_deg: float
    grade_scale: float  # 1.0 = 100% of route grades
    powers_w: list[float] | None = None  # series plotted / used when recorded
    used_recorded_power: bool = False
    speeds_gust_high_mps: list[float] | None = None
    speeds_gust_low_mps: list[float] | None = None

    @property
    def speeds_with_wind_mph(self) -> list[float]:
        return [v * MPS_TO_MPH for v in self.speeds_with_wind_mps]

    @property
    def speeds_no_wind_mph(self) -> list[float]:
        return [v * MPS_TO_MPH for v in self.speeds_no_wind_mps]

    @property
    def speeds_gust_high_mph(self) -> list[float] | None:
        if self.speeds_gust_high_mps is None:
            return None
        return [v * MPS_TO_MPH for v in self.speeds_gust_high_mps]

    @property
    def speeds_gust_low_mph(self) -> list[float] | None:
        if self.speeds_gust_low_mps is None:
            return None
        return [v * MPS_TO_MPH for v in self.speeds_gust_low_mps]

    @property
    def has_gust_band(self) -> bool:
        return (
            self.speeds_gust_high_mps is not None
            and self.speeds_gust_low_mps is not None
            and len(self.speeds_gust_high_mps) == len(self.speeds_with_wind_mps)
        )

    @property
    def wind_speed_mph(self) -> float:
        return self.wind_speed_mps * MPS_TO_MPH


def apply_what_if(
    result: PredictionResult,
    *,
    power_w: float,
    wind_speed_mps: float,
    grade_scale: float = 1.0,
    use_recorded_power: bool = False,
) -> WhatIfResult:
    """Re-solve with power (recorded series or flat), scaled wind, scaled grades.

    When ``use_recorded_power`` and the result has a watts series, each point
    uses that power with grade and wind. Otherwise uses flat ``power_w``.
    Grade scale of 1.0 is the route as recorded.
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
        winds = list(result.winds) if len(result.winds) == n else [WindComponents()] * n
    else:
        winds = [WindComponents()] * n

    use_series = (
        use_recorded_power
        and result.has_watts
        and result.watts is not None
        and len(result.watts) == n
    )
    powers_w: list[float] | None = list(result.watts) if use_series else None
    flat = max(1.0, float(power_w))
    if use_series and powers_w is not None:
        finite = [w for w in powers_w if math.isfinite(w) and w > 0.0]
        caption_power = sum(finite) / len(finite) if finite else flat
    else:
        caption_power = flat

    params = result.params.with_updates(
        power_w=caption_power if use_series else flat,
    )
    v0 = _initial_speed_mps(result.actual_speeds_mps)
    speeds_calm = solve_speeds(
        params,
        grades,
        powers_w=powers_w,
        distances_m=result.distances_m,
        initial_speed_mps=v0,
    )
    speeds_wind = (
        solve_speeds(
            params,
            grades,
            winds,
            powers_w=powers_w,
            distances_m=result.distances_m,
            initial_speed_mps=v0,
        )
        if result.weather_ok
        else list(speeds_calm)
    )
    gust_high = None
    gust_low = None
    if result.weather_ok and result.has_gust_band and result.mean_gust_speed_mps:
        # Scale the stored gust envelope by the same wind-speed slider factor.
        base_gust = max(0.0, float(result.mean_gust_speed_mps))
        base_wind = max(0.0, float(result.mean_wind_speed_mps))
        if base_wind > 1e-6 and base_gust > base_wind:
            # Rebuild light/strong winds from sustained winds × gust/sustained.
            gfactor = base_gust / base_wind
            winds_strong = [
                WindComponents(
                    along_track=w.along_track * gfactor,
                    cross_track=w.cross_track * gfactor,
                )
                for w in winds
            ]
            winds_light = [
                WindComponents(
                    along_track=w.along_track / gfactor,
                    cross_track=w.cross_track / gfactor,
                )
                for w in winds
            ]
            s_strong = solve_speeds(
                params,
                grades,
                winds_strong,
                powers_w=powers_w,
                distances_m=result.distances_m,
                initial_speed_mps=v0,
            )
            s_light = solve_speeds(
                params,
                grades,
                winds_light,
                powers_w=powers_w,
                distances_m=result.distances_m,
                initial_speed_mps=v0,
            )
            gust_high = [max(a, b, c) for a, b, c in zip(speeds_wind, s_strong, s_light)]
            gust_low = [min(a, b, c) for a, b, c in zip(speeds_wind, s_strong, s_light)]
    return WhatIfResult(
        speeds_with_wind_mps=speeds_wind,
        speeds_no_wind_mps=speeds_calm,
        eta_with_wind_s=eta_seconds(result.distances_m, speeds_wind),
        eta_no_wind_s=eta_seconds(result.distances_m, speeds_calm),
        power_w=caption_power,
        wind_speed_mps=target_wind,
        wind_from_deg=result.mean_wind_from_deg,
        grade_scale=scale_g,
        powers_w=powers_w,
        used_recorded_power=use_series,
        speeds_gust_high_mps=gust_high,
        speeds_gust_low_mps=gust_low,
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
    power_window_s: float | None = None
    if route.has_watts and route.watts is not None:
        powers_w, power_window_s = power_series_3s(route.watts, route.times_utc)
        power_source = "meter"
        finite = [w for w in powers_w if math.isfinite(w) and w > 0.0]
        if finite:
            params = params.with_updates(power_w=sum(finite) / len(finite))

    v0 = _initial_speed_mps(
        list(route.actual_speeds_mps) if route.has_actual_speeds else None
    )
    speeds_calm = solve_speeds(
        params,
        grades,
        powers_w=powers_w,
        distances_m=route.distances_m,
        initial_speed_mps=v0,
    )

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
    speeds_gust_high: list[float] | None = None
    speeds_gust_low: list[float] | None = None
    mean_gust_speed: float | None = None

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
        params_used = params.with_updates(rho=rho_mean)
        # Sustained wind drives the drawn speed line and ETA.
        speeds_wind = solve_speeds(
            params_used,
            grades,
            winds,
            powers_w=powers_w,
            distances_m=route.distances_m,
            initial_speed_mps=v0,
        )
        weather_ok = True
        if forecast:
            u_mean = sum(fp.u_rider_mps for fp in forecast) / len(forecast)
            v_mean = sum(fp.v_rider_mps for fp in forecast) / len(forecast)
            mean_wind_speed, mean_wind_from = wind_from_uv(u_mean, v_mean)
            gust_vals = [
                fp.gust_speed_10m
                for fp in forecast
                if fp.gust_speed_10m is not None and math.isfinite(fp.gust_speed_10m)
            ]
            if gust_vals and any(
                fp.gust_along_track_mps is not None for fp in forecast
            ):
                winds_strong = [fp.gust_wind for fp in forecast]
                winds_light = [fp.light_gust_wind for fp in forecast]
                s_strong = solve_speeds(
                    params_used,
                    grades,
                    winds_strong,
                    powers_w=powers_w,
                    distances_m=route.distances_m,
                    initial_speed_mps=v0,
                )
                s_light = solve_speeds(
                    params_used,
                    grades,
                    winds_light,
                    powers_w=powers_w,
                    distances_m=route.distances_m,
                    initial_speed_mps=v0,
                )
                speeds_gust_high = [
                    max(a, b, c) for a, b, c in zip(speeds_wind, s_strong, s_light)
                ]
                speeds_gust_low = [
                    min(a, b, c) for a, b, c in zip(speeds_wind, s_strong, s_light)
                ]
                mean_gust_speed = sum(gust_vals) / len(gust_vals)
            else:
                speeds_gust_high = None
                speeds_gust_low = None
                mean_gust_speed = None
        else:
            speeds_gust_high = None
            speeds_gust_low = None
            mean_gust_speed = None
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
    # Store the series used by the solver (3s-aligned when rolled).
    watts_out = list(powers_w) if powers_w is not None else None
    times_out = list(route.times_utc) if route.times_utc is not None else None

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
        times_utc=times_out,
        weather_when=weather_when,
        mean_grade_pct=route.mean_grade_pct(),
        used_ride_start_time=used_ride_start,
        power_source=power_source,
        mean_wind_speed_mps=mean_wind_speed,
        mean_wind_from_deg=mean_wind_from,
        power_window_s=power_window_s,
        speeds_gust_high_mps=speeds_gust_high,
        speeds_gust_low_mps=speeds_gust_low,
        mean_gust_speed_mps=mean_gust_speed,
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
