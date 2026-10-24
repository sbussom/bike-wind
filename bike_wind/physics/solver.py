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
) -> list[float]:
    """Ground speed at each sample (m/s).

    When ``distances_m`` is provided, integrate along the route with inertia:
    ``a = F_net / m`` from power, aero, rolling, and grade — speed approaches
    the steady-state value instead of jumping to it. Without distances, each
    sample is solved at equilibrium (legacy / unit helpers).

    ``powers_w``: finite values are used as-is, including **0 W coasting**.
    Non-finite samples fall back to ``params.power_w``. (A previous bug treated
    0 W as missing and substituted the default sustained watts.)

    ``initial_speed_mps``: optional start state for inertia (e.g. recorded speed
    at the first sample on a known effort). When omitted, starts at the first
    sample's steady-state speed.
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

    point_params = _point_params(params, powers_w, n)

    if distances_m is None:
        return [
            solve_speed(pp, grade=g, wind=w)
            for pp, g, w in zip(point_params, grades, winds)
        ]

    return _integrate_speeds(
        point_params,
        grades,
        winds,
        distances_m,
        initial_speed_mps=initial_speed_mps,
    )


def _point_params(
    params: RiderBikeParams,
    powers_w: Sequence[float] | None,
    n: int,
) -> list[RiderBikeParams]:
    if powers_w is None:
        return [params] * n
    out: list[RiderBikeParams] = []
    for pw in powers_w:
        if math.isfinite(pw) and pw >= 0.0:
            # Explicit 0 W = coast / stop. Do not substitute sustained watts.
            power = float(pw)
        else:
            power = float(params.power_w)
        out.append(params.with_updates(power_w=power))
    return out


def _integrate_speeds(
    point_params: Sequence[RiderBikeParams],
    grades: Sequence[float],
    winds: Sequence[WindComponents],
    distances_m: Sequence[float],
    *,
    initial_speed_mps: float | None = None,
) -> list[float]:
    """Euler integration: dv/dt = F_net/m along successive distance samples."""
    n = len(grades)
    speeds = [0.0] * n
    if (
        initial_speed_mps is not None
        and math.isfinite(initial_speed_mps)
        and initial_speed_mps >= 0.0
    ):
        # Known effort: use the recorded start speed as the integrator state.
        speeds[0] = min(_V_MAX_MPS, float(initial_speed_mps))
    else:
        speeds[0] = solve_speed(point_params[0], grade=grades[0], wind=winds[0])
        if speeds[0] < _V_MIN_MPS:
            speeds[0] = _V_MIN_MPS if point_params[0].power_w > 0.0 else 0.0

    for i in range(n - 1):
        dd = float(distances_m[i + 1]) - float(distances_m[i])
        v = speeds[i]
        if dd <= 0.0:
            speeds[i + 1] = v
            continue
        # Conditions at the destination sample (power/grade/wind at i+1).
        pp = point_params[i + 1]
        grade = grades[i + 1]
        wind = winds[i + 1]
        v_eq = solve_speed(pp, grade=grade, wind=wind)
        mass = max(pp.total_mass_kg, 1.0)
        eta = pp.drivetrain_efficiency if pp.drivetrain_efficiency > 0.0 else 1.0
        power = max(0.0, float(pp.power_w))

        remaining = dd
        while remaining > 1e-9:
            step = min(remaining, _INTEGRATE_STEP_M)
            if v <= 0.0 and v_eq <= 0.0:
                v = 0.0
                break
            v_state = max(v, _V_MIN_MPS) if v > 0.0 or power > 0.0 else 0.0
            if v_state <= 0.0:
                v = 0.0
                break
            v_drive = max(v_state, _V_DRIVE_FLOOR_MPS)
            f_drive = (eta * power) / v_drive
            f_resist = total_force(v_state, pp, grade=grade, wind=wind)
            # Acceleration capped exactly by this force balance (no extra boost).
            a = (f_drive - f_resist) / mass
            dt = step / v_state
            v_new = v_state + a * dt
            # Approach steady state; do not jump past equilibrium in one step.
            if v_eq > v_state:
                v_new = min(v_new, v_eq)
            elif v_eq < v_state:
                v_new = max(v_new, v_eq)
            if v_eq <= 0.0 and v_new < _V_MIN_MPS:
                v_new = 0.0
            else:
                v_new = min(_V_MAX_MPS, max(_V_MIN_MPS if v_eq > 0.0 else 0.0, v_new))
            v = v_new
            remaining -= step
        speeds[i + 1] = v

    return speeds


def _force_positive(
    v: float,
    params: RiderBikeParams,
    grade: float,
    wind: WindComponents,
) -> bool:
    return total_force(v, params, grade=grade, wind=wind) > 0.0
