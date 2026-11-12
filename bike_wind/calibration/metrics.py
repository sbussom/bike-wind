"""Accuracy checks.

``speed_errors_mph`` is the corpus checker: point speed error in mph, with no
wind in the comparison. Forecast logs and post-ride checks use
``score_segment_time`` instead, and those two results stay separate so a bad
forecast is not mixed into the post-ride model error.

A segment-time check passes within ±5% (``SEGMENT_TIME_PASS_PCT``). That
target used to live only in the README reliability note.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

MPS_TO_MPH = 2.236936

# Flat-road reliability target: predicted segment time vs actual.
SEGMENT_TIME_PASS_PCT = 5.0


@dataclass(frozen=True)
class RideSpeedError:
    n: int
    mean_signed_mph: float
    mae_mph: float

    @property
    def ok(self) -> bool:
        return self.n > 0


def speed_errors_mph(
    predicted_mps: Sequence[float],
    actual_mps: Sequence[float],
    *,
    min_actual_mps: float = 0.5,
) -> RideSpeedError | None:
    """Pointwise predicted−actual in mph; skip slow/invalid actual samples."""
    if len(predicted_mps) != len(actual_mps):
        raise ValueError("predicted and actual must align")
    errs: list[float] = []
    for p, a in zip(predicted_mps, actual_mps):
        if not (math.isfinite(p) and math.isfinite(a)):
            continue
        if a < min_actual_mps:
            continue
        errs.append((p - a) * MPS_TO_MPH)
    if not errs:
        return None
    return RideSpeedError(
        n=len(errs),
        mean_signed_mph=sum(errs) / len(errs),
        mae_mph=sum(abs(e) for e in errs) / len(errs),
    )


@dataclass(frozen=True)
class SegmentTimeScore:
    """Predicted segment time minus actual, as a signed percent of actual."""

    predicted_s: float
    actual_s: float
    error_s: float
    error_pct: float
    passed: bool


def score_segment_time(
    predicted_s: float,
    actual_s: float,
    *,
    pass_pct: float = SEGMENT_TIME_PASS_PCT,
) -> SegmentTimeScore | None:
    """Grade predicted vs actual segment time. None if the actual clock is unusable."""
    if not (math.isfinite(predicted_s) and math.isfinite(actual_s)):
        return None
    if actual_s <= 0.0 or predicted_s < 0.0:
        return None
    error_s = float(predicted_s) - float(actual_s)
    error_pct = 100.0 * error_s / float(actual_s)
    return SegmentTimeScore(
        predicted_s=float(predicted_s),
        actual_s=float(actual_s),
        error_s=error_s,
        error_pct=error_pct,
        passed=abs(error_pct) <= float(pass_pct),
    )


def direction_error_deg(forecast_from_deg: float, actual_from_deg: float) -> float:
    """Signed smallest angle, forecast minus actual, in (-180, 180]."""
    return (float(forecast_from_deg) - float(actual_from_deg) + 180.0) % 360.0 - 180.0
