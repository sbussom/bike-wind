"""Speed-error metrics (predicted − actual) in mph."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

MPS_TO_MPH = 2.236936


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
