"""Phase 2 analysis: target-time power, uncertainty bands, segment bins."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from bike_wind.physics.forces import RiderBikeParams, WindComponents, power_required
from bike_wind.physics.solver import solve_speeds

# ~1 mph in m/s — reference headwind error at 2–3 day horizon for ~20° shift
MPH_TO_MPS = 0.44704
IMPLAUSIBLE_POWER_RATIO = 1.35  # vs sustainable
IMPLAUSIBLE_W_PER_KG = 7.5  # absolute ceiling for sustained route effort
MIN_BIN_M = 800.0
MAX_BINS = 8


@dataclass(frozen=True)
class TimeBand:
    """Predicted ETA with low/high bounds from wind-forecast uncertainty."""

    eta_s: float
    eta_low_s: float
    eta_high_s: float
    horizon_days: float
    along_uncertainty_mps: float
    note: str


@dataclass(frozen=True)
class TargetPowerResult:
    """Power needed to hit a target/KOM time under given winds."""

    target_time_s: float
    required_power_w: float
    sustainable_power_w: float
    achievable: bool
    implausible: bool
    ratio: float
    message: str


@dataclass(frozen=True)
class SegmentBin:
    """Distance-bin summary for the results table."""

    index: int
    start_km: float
    end_km: float
    distance_m: float
    avg_grade_pct: float
    avg_wind_along_mps: float  # + tailwind, − headwind
    wind_label: str
    speed_with_wind_kmh: float
    speed_no_wind_kmh: float
    time_with_wind_s: float
    required_power_w: float | None
    implausible: bool


def parse_target_time(text: str) -> float | None:
    """Parse seconds or mm:ss / h:mm:ss into total seconds. Empty → None."""
    raw = (text or "").strip()
    if not raw:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", raw):
        seconds = float(raw)
        if seconds <= 0:
            raise ValueError("Target time must be positive")
        return seconds
    parts = raw.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        total = int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        total = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        raise ValueError("Use seconds or mm:ss (or h:mm:ss)")
    if total <= 0:
        raise ValueError("Target time must be positive")
    return float(total)


def format_duration(seconds: float) -> str:
    if seconds <= 0 or not math.isfinite(seconds):
        return "—"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def forecast_horizon_days(when: datetime, *, now: datetime | None = None) -> float:
    """Days from now until the ride datetime (UTC)."""
    now = now or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return max(0.0, (when - now).total_seconds() / 86400.0)


def wind_along_uncertainty_mps(horizon_days: float) -> float:
    """Along-track wind speed uncertainty from forecast horizon.

    Anchored to ~1 mph (~0.45 m/s) at 2–3 days for a ~20° direction error
    on headwind-exposed segments; grows past day 5.
    """
    d = max(0.0, horizon_days)
    if d <= 1.0:
        return 0.20 + 0.10 * d  # ~0.2–0.3 m/s
    if d <= 2.5:
        # 1 → 2.5 days: climb to ~1 mph (20° direction error on headwind segs)
        return 0.30 + (MPH_TO_MPS - 0.30) * ((d - 1.0) / 1.5)
    if d <= 5.0:
        return MPH_TO_MPS + (0.70 - MPH_TO_MPS) * ((d - 2.5) / 2.5)
    # Past day 5: steeper degradation
    return min(1.20, 0.70 + 0.12 * (d - 5.0))


def eta_seconds(
    distances_m: Sequence[float],
    speeds_mps: Sequence[float],
) -> float:
    if len(distances_m) < 2:
        return 0.0
    total = 0.0
    for i in range(len(distances_m) - 1):
        dd = distances_m[i + 1] - distances_m[i]
        v = max(speeds_mps[i], 0.1)
        total += dd / v
    return total


def solve_speeds_for_power(
    power_w: float,
    base: RiderBikeParams,
    grades: Sequence[float],
    winds: Sequence[WindComponents],
    distances_m: Sequence[float] | None = None,
) -> list[float]:
    params = RiderBikeParams(
        rider_mass_kg=base.rider_mass_kg,
        power_w=power_w,
        bike_mass_kg=base.bike_mass_kg,
        cd=base.cd,
        area_m2=base.area_m2,
        crr=base.crr,
        rho=base.rho,
        tuck_on_descent=base.tuck_on_descent,
        drivetrain_efficiency=base.drivetrain_efficiency,
    )
    return solve_speeds(params, grades, winds, distances_m=distances_m)


def power_for_target_time(
    target_time_s: float,
    distances_m: Sequence[float],
    grades: Sequence[float],
    winds: Sequence[WindComponents],
    params: RiderBikeParams,
    *,
    p_min: float = 50.0,
    p_max: float = 800.0,
    tol_s: float = 0.5,
    max_iter: int = 40,
) -> TargetPowerResult:
    """Binary-search sustained power that matches ``target_time_s`` on the route."""
    sustainable = params.power_w
    if target_time_s <= 0:
        raise ValueError("target_time_s must be positive")

    def eta_at(power: float) -> float:
        speeds = solve_speeds_for_power(
            power, params, grades, winds, distances_m=distances_m
        )
        return eta_seconds(distances_m, speeds)

    eta_hi = eta_at(p_max)  # more power → lower ETA
    eta_lo = eta_at(p_min)

    if eta_hi > target_time_s:
        # Even p_max is too slow
        ratio = p_max / max(sustainable, 1.0)
        return TargetPowerResult(
            target_time_s=target_time_s,
            required_power_w=p_max,
            sustainable_power_w=sustainable,
            achievable=False,
            implausible=True,
            ratio=ratio,
            message=(
                f"Target {format_duration(target_time_s)} needs >{p_max:.0f} W "
                f"(>{ratio:.1f}× your {sustainable:.0f} W) — physically implausible here."
            ),
        )

    if eta_lo <= target_time_s:
        # Even minimal power is fast enough (e.g. big descent / tailwind)
        return TargetPowerResult(
            target_time_s=target_time_s,
            required_power_w=p_min,
            sustainable_power_w=sustainable,
            achievable=True,
            implausible=False,
            ratio=p_min / max(sustainable, 1.0),
            message=(
                f"Target {format_duration(target_time_s)} is easy at ≤{p_min:.0f} W "
                f"under these conditions (terrain/wind dominated)."
            ),
        )

    lo, hi = p_min, p_max
    best = p_max
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        eta = eta_at(mid)
        best = mid
        if abs(eta - target_time_s) <= tol_s:
            break
        if eta > target_time_s:
            lo = mid  # too slow → more power
        else:
            hi = mid

    ratio = best / max(sustainable, 1.0)
    w_per_kg = best / max(params.total_mass_kg, 1.0)
    implausible = ratio >= IMPLAUSIBLE_POWER_RATIO or w_per_kg >= IMPLAUSIBLE_W_PER_KG
    achievable = best <= sustainable * 1.05  # small slack

    if implausible:
        msg = (
            f"Hitting {format_duration(target_time_s)} needs ~{best:.0f} W "
            f"({ratio:.2f}× your {sustainable:.0f} W, {w_per_kg:.1f} W/kg) — "
            "flagged as implausible for sustained effort."
        )
    elif achievable:
        msg = (
            f"Target {format_duration(target_time_s)} needs ~{best:.0f} W "
            f"vs your {sustainable:.0f} W sustainable — within reach."
        )
    else:
        msg = (
            f"Target {format_duration(target_time_s)} needs ~{best:.0f} W "
            f"({ratio:.2f}× your {sustainable:.0f} W) — stretch / unlikely to hold."
        )

    return TargetPowerResult(
        target_time_s=target_time_s,
        required_power_w=best,
        sustainable_power_w=sustainable,
        achievable=achievable,
        implausible=implausible,
        ratio=ratio,
        message=msg,
    )


def time_uncertainty_band(
    distances_m: Sequence[float],
    grades: Sequence[float],
    winds: Sequence[WindComponents],
    params: RiderBikeParams,
    when: datetime,
    *,
    now: datetime | None = None,
) -> TimeBand:
    """ETA ± band from along-track wind uncertainty vs forecast horizon."""
    horizon = forecast_horizon_days(when, now=now)
    delta = wind_along_uncertainty_mps(horizon)

    speeds_nom = solve_speeds(params, grades, winds, distances_m=distances_m)
    eta = eta_seconds(distances_m, speeds_nom)

    # Favorable: add delta to along-track (more tail / less head)
    winds_fast = [
        WindComponents(along_track=w.along_track + delta, cross_track=w.cross_track)
        for w in winds
    ]
    # Adverse: subtract delta (more head / less tail)
    winds_slow = [
        WindComponents(along_track=w.along_track - delta, cross_track=w.cross_track)
        for w in winds
    ]
    eta_fast = eta_seconds(
        distances_m, solve_speeds(params, grades, winds_fast, distances_m=distances_m)
    )
    eta_slow = eta_seconds(
        distances_m, solve_speeds(params, grades, winds_slow, distances_m=distances_m)
    )

    note = (
        f"Horizon {horizon:.1f} d → ±{delta:.2f} m/s along-track wind "
        f"(~{delta / MPH_TO_MPS:.1f} mph). At 2–3 days a ~20° direction shift "
        "is ~1 mph on headwind segments. Skill degrades past day 5."
    )
    return TimeBand(
        eta_s=eta,
        eta_low_s=min(eta_fast, eta_slow, eta),
        eta_high_s=max(eta_fast, eta_slow, eta),
        horizon_days=horizon,
        along_uncertainty_mps=delta,
        note=note,
    )


def wind_label(along_mps: float) -> str:
    if along_mps >= 1.0:
        return f"Tail {along_mps:.1f} m/s"
    if along_mps <= -1.0:
        return f"Head {abs(along_mps):.1f} m/s"
    return f"Cross/light {along_mps:+.1f} m/s"


def choose_bin_count(total_distance_m: float) -> int:
    """Whole-route summary if short; otherwise up to MAX_BINS ~1 km bins."""
    if total_distance_m < MIN_BIN_M * 1.5:
        return 1
    n = int(math.ceil(total_distance_m / 1000.0))
    return max(2, min(MAX_BINS, n))


def build_segment_bins(
    distances_m: Sequence[float],
    grades: Sequence[float],
    wind_along_mps: Sequence[float],
    speeds_with_wind_mps: Sequence[float],
    speeds_no_wind_mps: Sequence[float],
    params: RiderBikeParams,
    winds: Sequence[WindComponents],
    *,
    target: TargetPowerResult | None = None,
) -> list[SegmentBin]:
    """Aggregate the route into distance bins for a compact table."""
    n = len(distances_m)
    if n < 2:
        return []
    total = distances_m[-1] - distances_m[0]
    n_bins = choose_bin_count(total)
    edges = [distances_m[0] + total * i / n_bins for i in range(n_bins + 1)]

    bins: list[SegmentBin] = []
    for b in range(n_bins):
        start_d, end_d = edges[b], edges[b + 1]
        idxs = [
            i
            for i in range(n - 1)
            if distances_m[i] >= start_d - 1e-6 and distances_m[i] < end_d - 1e-6
        ]
        if not idxs:
            idxs = [min(n - 2, max(0, int(round((n - 1) * (b + 0.5) / n_bins))))]

        dist = end_d - start_d
        avg_grade = sum(grades[i] for i in idxs) / len(idxs)
        avg_along = sum(wind_along_mps[i] for i in idxs) / len(idxs)
        t_wind = 0.0
        t_calm = 0.0
        covered = 0.0
        for i in idxs:
            dd = distances_m[i + 1] - distances_m[i]
            if dd <= 0:
                continue
            covered += dd
            t_wind += dd / max(speeds_with_wind_mps[i], 0.1)
            t_calm += dd / max(speeds_no_wind_mps[i], 0.1)
        use_dist = covered if covered > 0 else dist
        speed_w = (use_dist / t_wind * 3.6) if t_wind > 0 else 0.0
        speed_c = (use_dist / t_calm * 3.6) if t_calm > 0 else 0.0

        mid = idxs[len(idxs) // 2]
        v_req = use_dist / t_wind if t_wind > 0 else 0.0
        p_req = (
            power_required(v_req, params, grade=grades[mid], wind=winds[mid])
            if v_req > 0
            else None
        )

        implausible = False
        if target is not None and target.implausible and avg_grade > 0.02:
            implausible = True
        if p_req is not None and params.total_mass_kg > 0:
            if p_req / params.total_mass_kg >= IMPLAUSIBLE_W_PER_KG and avg_grade > 0.03:
                implausible = True

        bins.append(
            SegmentBin(
                index=b + 1,
                start_km=start_d / 1000.0,
                end_km=end_d / 1000.0,
                distance_m=dist,
                avg_grade_pct=avg_grade * 100.0,
                avg_wind_along_mps=avg_along,
                wind_label=wind_label(avg_along),
                speed_with_wind_kmh=speed_w,
                speed_no_wind_kmh=speed_c,
                time_with_wind_s=t_wind,
                required_power_w=p_req,
                implausible=implausible,
            )
        )
    return bins


def head_tail_summary(wind_along_mps: Sequence[float]) -> tuple[float, float, float]:
    """Return (mean_along, mean_headwind_magnitude, mean_tailwind)."""
    if not wind_along_mps:
        return 0.0, 0.0, 0.0
    mean_along = sum(wind_along_mps) / len(wind_along_mps)
    heads = [-w for w in wind_along_mps if w < 0]
    tails = [w for w in wind_along_mps if w > 0]
    mean_head = sum(heads) / len(heads) if heads else 0.0
    mean_tail = sum(tails) / len(tails) if tails else 0.0
    return mean_along, mean_head, mean_tail
