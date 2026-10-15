"""Iterative ground-speed solver for sustained power."""

from __future__ import annotations

from typing import Sequence

from bike_wind.physics.forces import (
    RiderBikeParams,
    WindComponents,
    power_required,
)


def solve_speed(
    params: RiderBikeParams,
    grade: float = 0.0,
    wind: WindComponents | None = None,
    *,
    v_min: float = 0.05,
    v_max: float = 40.0,
    tol_w: float = 0.05,
    max_iter: int = 80,
) -> float:
    """Solve for ground speed (m/s) given sustained power.

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
) -> list[float]:
    """Solve ground speed for each segment grade (and optional wind)."""
    n = len(grades)
    if winds is None:
        winds = [WindComponents()] * n
    if len(winds) != n:
        raise ValueError("grades and winds must have the same length")
    return [solve_speed(params, grade=g, wind=w) for g, w in zip(grades, winds)]


def _force_positive(
    v: float,
    params: RiderBikeParams,
    grade: float,
    wind: WindComponents,
) -> bool:
    from bike_wind.physics.forces import total_force

    return total_force(v, params, grade=grade, wind=wind) > 0.0
