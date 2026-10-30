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
class PacePlan:
    """Grade-aware watts to match a target time (harder on climbs)."""

    target_time_s: float
    flat_power_w: float
    avg_power_w: float
    peak_power_w: float
    mean_speed_mps: float
    powers_w: list[float]
    message: str

    @property
    def mean_speed_mph(self) -> float:
        return self.mean_speed_mps * 2.236936


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


# Recorded speed below this is a stop. Moving time does not include it.
# About 1.1 mph — under the 3 mph speed-error gate, and above GPS creep.
RECORDED_STOP_MPS = 0.5


def eta_seconds(
    distances_m: Sequence[float],
    speeds_mps: Sequence[float],
    *,
    gap_dt: Sequence[float] | None = None,
    actual_speeds_mps: Sequence[float] | None = None,
) -> float:
    """Seconds to cover the route at ``speeds_mps``.

    Uses integrator gap times when ``gap_dt`` is provided (the seconds the
    bike was actually rolling). Otherwise distance / speed, and a modeled
    stop (speed under 0.05 m/s) adds nothing — it is not crawled at 0.1 m/s.

    A recorded stop (actual speed under 0.5 m/s) is left out of the sum.
    That is the moving-time clock. A low predicted speed while the recording
    still shows the rider rolling is coasting, and it keeps its own time.
    """
    n = len(distances_m)
    if n < 2:
        return 0.0
    total = 0.0
    use_gap = gap_dt is not None and len(gap_dt) == n - 1
    for i in range(n - 1):
        dd = distances_m[i + 1] - distances_m[i]
        if dd <= 0.0:
            continue
        if actual_speeds_mps is not None and i < len(actual_speeds_mps):
            av = actual_speeds_mps[i]
            if math.isfinite(av) and av < RECORDED_STOP_MPS:
                continue
        if use_gap:
            dt = float(gap_dt[i])
            if math.isfinite(dt) and dt > 0.0:
                total += dt
            continue
        v = float(speeds_mps[i]) if i < len(speeds_mps) else 0.0
        if not math.isfinite(v) or v < 0.05:
            continue
        total += dd / v
    return total


def solve_speeds_for_power(
    power_w: float,
    base: RiderBikeParams,
    grades: Sequence[float],
    winds: Sequence[WindComponents],
    distances_m: Sequence[float] | None = None,
) -> list[float]:
    params = base.with_updates(power_w=power_w)
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
        gap: list[float] = []
        speeds = solve_speeds(
            params.with_updates(power_w=power),
            grades,
            winds,
            distances_m=distances_m,
            gap_dt=gap,
        )
        return eta_seconds(distances_m, speeds, gap_dt=gap)

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


def grade_power_weights(
    grades: Sequence[float],
    *,
    climb_gain: float = 2.5,
    descent_cut: float = 0.35,
) -> list[float]:
    """Relative watts by grade: more on climbs, less on descents."""
    out: list[float] = []
    for g in grades:
        if g >= 0.0:
            out.append(1.0 + climb_gain * float(g))
        else:
            out.append(max(0.35, 1.0 + descent_cut * float(g)))
    return out


def pace_plan_for_target_time(
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
) -> PacePlan:
    """Same ETA as flat watts, but harder on steep grades and easier on flats.

    Searches a scale on grade weights so inertia-limited ETA matches
    ``target_time_s``. Reports average and peak watts for the plan.
    """
    flat = power_for_target_time(
        target_time_s, distances_m, grades, winds, params,
        p_min=p_min, p_max=p_max, tol_s=tol_s, max_iter=max_iter,
    )
    weights = grade_power_weights(grades)
    n = len(grades)

    def powers_at(scale: float) -> list[float]:
        return [max(0.0, scale * w) for w in weights]

    def eta_at(scale: float) -> float:
        powers = powers_at(scale)
        gap: list[float] = []
        speeds = solve_speeds(
            params,
            grades,
            winds,
            powers_w=powers,
            distances_m=distances_m,
            gap_dt=gap,
        )
        return eta_seconds(distances_m, speeds, gap_dt=gap)

    # Bracket scale so mean power ≈ flat required, then refine on ETA.
    mean_w = sum(weights) / max(len(weights), 1)
    lo = p_min / max(mean_w, 1e-6)
    hi = p_max / max(mean_w, 1e-6)
    # Seed near flat solution
    mid0 = flat.required_power_w / max(mean_w, 1e-6)
    lo, hi = min(lo, mid0 * 0.5), max(hi, mid0 * 1.5)
    best_scale = mid0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        eta = eta_at(mid)
        best_scale = mid
        if abs(eta - target_time_s) <= tol_s:
            break
        if eta > target_time_s:
            lo = mid
        else:
            hi = mid

    powers = powers_at(best_scale)
    gap_plan: list[float] = []
    speeds = solve_speeds(
        params,
        grades,
        winds,
        powers_w=powers,
        distances_m=distances_m,
        gap_dt=gap_plan,
    )
    eta = eta_seconds(distances_m, speeds, gap_dt=gap_plan)
    avg = sum(powers) / max(len(powers), 1)
    peak = max(powers) if powers else 0.0
    dist = (
        float(distances_m[-1] - distances_m[0]) if len(distances_m) >= 2 else 0.0
    )
    mean_v = dist / eta if eta > 0 else 0.0
    msg = (
        f"KOM pace ~{format_duration(target_time_s)}: "
        f"{mean_v * 2.236936:.1f} mph · "
        f"{avg:.0f} W avg (peak {peak:.0f} W; flat would be {flat.required_power_w:.0f} W). "
        f"Harder on climbs, easier on flats."
    )
    return PacePlan(
        target_time_s=target_time_s,
        flat_power_w=flat.required_power_w,
        avg_power_w=avg,
        peak_power_w=peak,
        mean_speed_mps=mean_v,
        powers_w=powers,
        message=msg,
    )


# Along-track wind inside this band is treated as crosswind, not a pacing section.
_SPLIT_ALONG_MPS = 0.5


@dataclass(frozen=True)
class WindPaceSplit:
    """Headwind / tailwind watts at one average power, versus steady power.

    The average is distance-weighted along the route: equal-length out and
    back legs at ``P+d`` and ``P-d`` still average ``P``. A power meter reads
    the time-weighted number, which is higher when the hard leg takes longer.
    """

    avg_power_w: float
    headwind_power_w: float
    tailwind_power_w: float
    headwind_distance_m: float
    tailwind_distance_m: float
    neutral_distance_m: float
    steady_time_s: float
    paced_time_s: float
    seconds_saved: float
    steady_speed_mps: float
    paced_speed_mps: float
    meter_avg_power_w: float
    split_useful: bool
    message: str

    @property
    def steady_speed_mph(self) -> float:
        return self.steady_speed_mps * 2.236936

    @property
    def paced_speed_mph(self) -> float:
        return self.paced_speed_mps * 2.236936


def _gap_distance(distances_m: Sequence[float], i: int) -> float:
    return max(0.0, float(distances_m[i + 1]) - float(distances_m[i]))


def wind_pace_split(
    avg_power_w: float,
    distances_m: Sequence[float],
    grades: Sequence[float],
    winds: Sequence[WindComponents],
    params: RiderBikeParams,
    *,
    along_threshold_mps: float = _SPLIT_ALONG_MPS,
    p_min: float = 50.0,
    p_max: float = 800.0,
) -> WindPaceSplit:
    """Best headwind/tailwind split that keeps the same distance-averaged power.

    Positive along-track wind is a tailwind. Samples inside
    ``along_threshold_mps`` stay at the average. The search only moves watts
    between the headwind and tailwind distance, then keeps the plan that
    finishes sooner.
    """
    n = len(grades)
    if n < 2 or len(distances_m) != n or len(winds) != n:
        raise ValueError("distances, grades, and winds must share one length")
    avg = float(avg_power_w)
    if not math.isfinite(avg) or avg <= 0.0:
        raise ValueError("avg_power_w must be positive")

    # Gap i is ridden at the power and wind of sample i+1 (same as the solver).
    kind = [0] * n  # 1 head, -1 tail, 0 neutral / start
    dist_h = 0.0
    dist_t = 0.0
    dist_n = 0.0
    for i in range(n - 1):
        dd = _gap_distance(distances_m, i)
        if dd <= 0.0:
            continue
        along = float(winds[i + 1].along_track)
        j = i + 1
        if along <= -along_threshold_mps:
            kind[j] = 1
            dist_h += dd
        elif along >= along_threshold_mps:
            kind[j] = -1
            dist_t += dd
        else:
            dist_n += dd
    total = dist_h + dist_t + dist_n

    def eta_of(powers: list[float]) -> tuple[float, list[float]]:
        gap: list[float] = []
        speeds = solve_speeds(
            params,
            grades,
            winds,
            powers_w=powers,
            distances_m=distances_m,
            gap_dt=gap,
        )
        return eta_seconds(distances_m, speeds, gap_dt=gap), gap

    steady_powers = [avg] * n
    steady_eta, _steady_gap = eta_of(steady_powers)
    steady_v = total / steady_eta if steady_eta > 0.0 else 0.0

    def finish(
        *,
        head_w: float,
        tail_w: float,
        paced_eta: float,
        meter_w: float,
        useful: bool,
        message: str,
    ) -> WindPaceSplit:
        saved = max(0.0, steady_eta - paced_eta)
        paced_v = total / paced_eta if paced_eta > 0.0 else 0.0
        return WindPaceSplit(
            avg_power_w=avg,
            headwind_power_w=head_w,
            tailwind_power_w=tail_w,
            headwind_distance_m=dist_h,
            tailwind_distance_m=dist_t,
            neutral_distance_m=dist_n,
            steady_time_s=steady_eta,
            paced_time_s=paced_eta if useful or paced_eta > 0 else steady_eta,
            seconds_saved=saved if useful else 0.0,
            steady_speed_mps=steady_v,
            paced_speed_mps=paced_v if useful else steady_v,
            meter_avg_power_w=meter_w if useful else avg,
            split_useful=useful,
            message=message,
        )

    share_h = dist_h / total if total > 0.0 else 0.0
    share_t = dist_t / total if total > 0.0 else 0.0
    if dist_h < 100.0 or dist_t < 100.0 or share_h < 0.08 or share_t < 0.08:
        return finish(
            head_w=avg,
            tail_w=avg,
            paced_eta=steady_eta,
            meter_w=avg,
            useful=False,
            message=(
                "Wind stays cross or one-sided, so a headwind/tailwind split "
                "does not change the time. Steady power is the plan."
            ),
        )

    ratio = dist_h / dist_t
    # Keep both section powers inside [p_min, p_max].
    delta_lo = max(p_min - avg, (avg - p_max) / ratio)
    delta_hi = min(p_max - avg, (avg - p_min) / ratio)
    if delta_hi < delta_lo + 1.0:
        return finish(
            head_w=avg,
            tail_w=avg,
            paced_eta=steady_eta,
            meter_w=avg,
            useful=False,
            message=(
                "There is no room to move watts between the headwind and the "
                "tailwind without leaving a rideable power. Steady power is the plan."
            ),
        )

    def powers_at(delta: float) -> tuple[list[float], float, float]:
        p_h = avg + delta
        p_t = avg - delta * ratio
        out = [avg] * n
        for j in range(1, n):
            if kind[j] == 1:
                out[j] = p_h
            elif kind[j] == -1:
                out[j] = p_t
        out[0] = out[1]
        return out, p_h, p_t

    def meter_avg(powers: list[float], gap: list[float]) -> float:
        num = 0.0
        den = 0.0
        for i, dt in enumerate(gap):
            if dt > 0.0:
                num += powers[i + 1] * dt
                den += dt
        return num / den if den > 0.0 else avg

    best_delta = 0.0
    best_eta = steady_eta
    best_powers = steady_powers
    best_head = avg
    best_tail = avg
    best_gap: list[float] = []

    def consider(delta: float) -> None:
        nonlocal best_delta, best_eta, best_powers, best_head, best_tail, best_gap
        delta = min(delta_hi, max(delta_lo, delta))
        powers, p_h, p_t = powers_at(delta)
        eta, gap = eta_of(powers)
        if eta < best_eta - 1e-6:
            best_delta = delta
            best_eta = eta
            best_powers = powers
            best_head = p_h
            best_tail = p_t
            best_gap = gap

    steps = 8
    span = delta_hi - delta_lo
    for i in range(steps + 1):
        consider(delta_lo + span * i / steps)
    consider(0.0)
    refine = span / steps
    for i in range(7):
        consider(best_delta + refine * (i - 3) / 3.0)

    if best_gap:
        meter = meter_avg(best_powers, best_gap)
    else:
        meter = avg
        best_eta = steady_eta
    saved = max(0.0, steady_eta - best_eta)
    useful = saved >= 0.5 and abs(best_head - best_tail) >= 1.0
    if not useful:
        return finish(
            head_w=avg,
            tail_w=avg,
            paced_eta=steady_eta,
            meter_w=avg,
            useful=False,
            message=(
                "Moving watts between the headwind and the tailwind does not "
                "save a meaningful amount of time. Steady power is the plan."
            ),
        )
    return finish(
        head_w=best_head,
        tail_w=best_tail,
        paced_eta=best_eta,
        meter_w=meter,
        useful=True,
        message=(
            f"Headwind {best_head:.0f} W, tailwind {best_tail:.0f} W, "
            f"average {avg:.0f} W along the route. "
            f"Saves {saved:.0f} s versus steady "
            f"({format_duration(steady_eta)} → {format_duration(best_eta)})."
        ),
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

    gap_nom: list[float] = []
    speeds_nom = solve_speeds(
        params, grades, winds, distances_m=distances_m, gap_dt=gap_nom
    )
    eta = eta_seconds(distances_m, speeds_nom, gap_dt=gap_nom)

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
    gap_fast: list[float] = []
    gap_slow: list[float] = []
    speeds_fast = solve_speeds(
        params, grades, winds_fast, distances_m=distances_m, gap_dt=gap_fast
    )
    speeds_slow = solve_speeds(
        params, grades, winds_slow, distances_m=distances_m, gap_dt=gap_slow
    )
    eta_fast = eta_seconds(distances_m, speeds_fast, gap_dt=gap_fast)
    eta_slow = eta_seconds(distances_m, speeds_slow, gap_dt=gap_slow)

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
            t_wind += dd / speeds_with_wind_mps[i] if speeds_with_wind_mps[i] >= 0.05 else 0.0
            t_calm += dd / speeds_no_wind_mps[i] if speeds_no_wind_mps[i] >= 0.05 else 0.0
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
