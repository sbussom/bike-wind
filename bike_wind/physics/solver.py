"""Ground-speed solver: steady-state and inertia-limited along a route."""

from __future__ import annotations

import math
from typing import Sequence

from bike_wind.physics.forces import (
    RiderBikeParams,
    WindComponents,
    power_required,
    total_force,
)

# Floor for propulsion F = ηP/v (avoids singularity at standstill).
_V_DRIVE_FLOOR_MPS = 0.5
_V_MIN_MPS = 0.05
_V_MAX_MPS = 40.0
# Sub-step length for Euler integration along a long sample gap.
_INTEGRATE_STEP_M = 5.0


def solve_speed(
    params: RiderBikeParams,
    grade: float = 0.0,
    wind: WindComponents | None = None,
    *,
    v_min: float = _V_MIN_MPS,
    v_max: float = _V_MAX_MPS,
    tol_w: float = 0.05,
    max_iter: int = 80,
) -> float:
    """Solve for steady-state ground speed (m/s) given sustained power.

    Uses bisection on P(v) − P_target. Returns 0.0 when even crawling speed
    requires more power than available (e.g. very steep climb).
    """
    wind = wind or WindComponents()
    target = params.power_w

    # If power at v_min still exceeds target and force is positive, can't move.
    p_lo = power_required(v_min, params, grade=grade, wind=wind)
    if p_lo > target and _force_positive(v_min, params, grade, wind):
        return 0.0

    p_hi = power_required(v_max, params, grade=grade, wind=wind)
    # With strong tailwind / steep descent, power at high speed may be negative
    # (rider brakes). Clamp to a sensible freewheel-ish max via searching where
    # required power crosses target from below.
    lo, hi = v_min, v_max
    if p_hi < target:
        # Can go faster than v_max at this power — return v_max.
        return v_max

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = power_required(mid, params, grade=grade, wind=wind)
        if abs(p_mid - target) <= tol_w:
            return mid
        # On descents with tailwind, P(v) can be non-monotonic near zero; still
        # treat rising limb: if p_mid < target need more speed.
        if p_mid < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_speeds(
    params: RiderBikeParams,
    grades: Sequence[float],
    winds: Sequence[WindComponents] | None = None,
    powers_w: Sequence[float] | None = None,
    distances_m: Sequence[float] | None = None,
    *,
    initial_speed_mps: float | None = None,
    gap_dt: list[float] | None = None,
    surface_classes: Sequence[int] | None = None,
    point_crr: Sequence[float] | None = None,
) -> list[float]:
    """Ground speed at each sample (m/s).

    When ``distances_m`` is provided, integrate along the route with inertia:
    ``a = F_net / m`` from power, aero, rolling, and grade. Positive power
    may be capped at the steady-state speed so a coarse step cannot jump
    past equilibrium. **0 W is coasting**: momentum is carried and the bike
    slows under drag, rolling resistance, and gravity. There is no steady-state
    solve at 0 W, and speed is not snapped to a stop.

    Without distances, each sample is solved at equilibrium (legacy / unit
    helpers). A 0 W sample then has no propulsion; the helper still reports
    the force balance, which is 0 m/s on flat ground.

    ``powers_w``: finite values are used as-is, including **0 W coasting**.
    Non-finite samples fall back to ``params.power_w``. (A previous bug treated
    0 W as missing and substituted the default sustained watts.)

    ``initial_speed_mps``: optional start state for inertia (e.g. recorded speed
    at the first sample on a known effort). When omitted, a powered first
    sample starts at its steady speed and a 0 W first sample starts from rest.

    ``gap_dt``: if a list is passed, it is replaced with the seconds spent
    crossing each gap (length ``n - 1``). Time is the integral of ``dt`` while
    the bike is actually moving. A gap that runs out of speed contributes only
    the seconds until it stops — not a crawl of the leftover distance.

    ``surface_classes``: optional per-sample OSM class (1 paved, 0 unpaved,
    -1 off-map). Used only when ``params.surface_aware``; each class then
    gets its own Crr and Cd·A scale from ``cda_defaults.surface_setup``.

    ``point_crr``: optional per-sample rolling resistance. When set, it
    replaces Crr and leaves Cd·A alone. Omitted on the production path.
    """
    n = len(grades)
    if winds is None:
        winds = [WindComponents()] * n
    if len(winds) != n:
        raise ValueError("grades and winds must have the same length")
    if powers_w is not None and len(powers_w) != n:
        raise ValueError("grades and powers_w must have the same length")
    if distances_m is not None and len(distances_m) != n:
        raise ValueError("grades and distances_m must have the same length")

    if surface_classes is not None and len(surface_classes) != n:
        raise ValueError("grades and surface_classes must have the same length")
    if point_crr is not None and len(point_crr) != n:
        raise ValueError("grades and point_crr must have the same length")

    point_params = _point_params(params, powers_w, n, surface_classes, point_crr)

    if distances_m is None:
        return [
            solve_speed(pp, grade=g, wind=w)
            for pp, g, w in zip(point_params, grades, winds)
        ]

    speeds, dts = _integrate_speeds(
        point_params,
        grades,
        winds,
        distances_m,
        initial_speed_mps=initial_speed_mps,
    )
    if gap_dt is not None:
        gap_dt.clear()
        gap_dt.extend(dts)
    return speeds


def _point_params(
    params: RiderBikeParams,
    powers_w: Sequence[float] | None,
    n: int,
    surface_classes: Sequence[int] | None = None,
    point_crr: Sequence[float] | None = None,
) -> list[RiderBikeParams]:
    per_class: dict[int, RiderBikeParams] | None = None
    if surface_classes is not None and params.surface_aware:
        from bike_wind.calibration.cda_defaults import surface_setup

        per_class = {}
        for c in set(int(x) for x in surface_classes):
            crr, cda_scale = surface_setup(c)
            per_class[c] = params.with_updates(crr=crr, area_m2=params.area_m2 * cda_scale)

    def _at(i: int) -> RiderBikeParams:
        base = params if per_class is None else per_class[int(surface_classes[i])]  # type: ignore[index]
        if point_crr is not None:
            base = base.with_updates(crr=float(point_crr[i]))
        return base

    if powers_w is None:
        if per_class is None and point_crr is None:
            return [params] * n
        return [_at(i) for i in range(n)]
    out: list[RiderBikeParams] = []
    for i, pw in enumerate(powers_w):
        base = _at(i)
        if math.isfinite(pw) and pw >= 0.0:
            # Explicit 0 W = coast / stop. Do not substitute sustained watts.
            power = float(pw)
        else:
            power = float(params.power_w)
        out.append(base.with_updates(power_w=power))
    return out


def _integrate_speeds(
    point_params: Sequence[RiderBikeParams],
    grades: Sequence[float],
    winds: Sequence[WindComponents],
    distances_m: Sequence[float],
    *,
    initial_speed_mps: float | None = None,
) -> tuple[list[float], list[float]]:
    """Integrate dv/dt = F_net/m. Returns speeds and per-gap moving seconds.

    0 W never calls the steady-state solver. The bike keeps its speed and
    sheds it only as drag, rolling resistance, and gravity require. A powered
    step may be capped at that step's equilibrium so one coarse distance
    chunk cannot leap past the force balance.
    """
    n = len(grades)
    speeds = [0.0] * n
    gap_dt = [0.0] * max(0, n - 1)
    if (
        initial_speed_mps is not None
        and math.isfinite(initial_speed_mps)
        and initial_speed_mps >= 0.0
    ):
        speeds[0] = min(_V_MAX_MPS, float(initial_speed_mps))
    elif point_params[0].power_w > 0.0:
        speeds[0] = solve_speed(point_params[0], grade=grades[0], wind=winds[0])
        if speeds[0] < _V_MIN_MPS:
            speeds[0] = _V_MIN_MPS
    else:
        # Planned or recorded 0 W with no known start speed: start from rest.
        speeds[0] = 0.0

    for i in range(n - 1):
        dd = float(distances_m[i + 1]) - float(distances_m[i])
        v = speeds[i]
        if dd <= 0.0:
            speeds[i + 1] = v
            gap_dt[i] = 0.0
            continue
        pp = point_params[i + 1]
        grade = float(grades[i + 1])
        wind = winds[i + 1]
        power = max(0.0, float(pp.power_w))
        # Equilibrium cap only while pedaling. Coasting has no target speed.
        v_eq: float | None
        if power > 0.0:
            v_eq = solve_speed(pp, grade=grade, wind=wind)
        else:
            v_eq = None
        remaining = dd
        spent = 0.0
        guard = 0
        while remaining > 1e-6 and guard < 10000:
            guard += 1
            step = min(remaining, _INTEGRATE_STEP_M)
            v, dt, covered = _coast_step(v, step, pp, grade, wind, power, v_eq)
            spent += dt
            if covered <= 1e-8:
                # Out of speed with distance left. That is a real stop for
                # this gap, not a 0.1 m/s crawl of the rest of it.
                break
            remaining -= covered
        speeds[i + 1] = v
        gap_dt[i] = spent

    return speeds, gap_dt


def _coast_step(
    v: float,
    step: float,
    params: RiderBikeParams,
    grade: float,
    wind: WindComponents,
    power: float,
    v_eq: float | None,
) -> tuple[float, float, float]:
    """Advance up to ``step`` meters. Returns ``(v_new, dt, covered)``."""
    v = 0.0 if not math.isfinite(v) or v < 0.0 else min(_V_MAX_MPS, v)
    a = _accel(v, params, grade, wind, power)
    if v <= 1e-4 and a <= 1e-4:
        return 0.0, 0.0, 0.0
    # Constant-acceleration kinematics over this chunk: s = v t + ½ a t².
    if abs(a) < 1e-8:
        dt = step / v
        return v, dt, step
    disc = v * v + 2.0 * a * step
    if disc <= 0.0:
        # Stops inside the chunk. Time to v=0 is finite; leftover distance
        # is not ridden.
        dt = v / (-a) if a < 0.0 else 0.0
        covered = max(0.0, v * dt + 0.5 * a * dt * dt)
        return 0.0, dt, covered
    dt = (-v + math.sqrt(disc)) / a
    if dt < 0.0:
        dt = step / max(v, 1e-3)
    v_new = v + a * dt
    if v_eq is not None and v_eq >= 0.0:
        # Do not cross the powered equilibrium inside one chunk.
        if v_eq > v:
            v_new = min(v_new, v_eq)
        elif v_eq < v:
            v_new = max(v_new, v_eq)
    if v_new < 0.0:
        v_new = 0.0
    elif v_new > _V_MAX_MPS:
        v_new = _V_MAX_MPS
    return v_new, max(0.0, dt), step


def _accel(
    v: float,
    params: RiderBikeParams,
    grade: float,
    wind: WindComponents,
    power: float,
) -> float:
    """Along-track acceleration (m/s²). Positive speeds the bike up."""
    mass = max(params.total_mass_kg, 1.0)
    if power > 0.0:
        eta = params.drivetrain_efficiency if params.drivetrain_efficiency > 0.0 else 1.0
        f_drive = (eta * power) / max(v, _V_DRIVE_FLOOR_MPS)
        v_force = max(v, _V_MIN_MPS)
    else:
        f_drive = 0.0
        v_force = max(v, 0.0)
    f_resist = total_force(v_force, params, grade=grade, wind=wind)
    return (f_drive - f_resist) / mass


def _force_positive(
    v: float,
    params: RiderBikeParams,
    grade: float,
    wind: WindComponents,
) -> bool:
    return total_force(v, params, grade=grade, wind=wind) > 0.0
