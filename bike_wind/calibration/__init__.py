"""Offline calibration against recorded rides with power.

Corpus files are athlete-private and live outside git (see
``CorpusPaths``). This package only evaluates / tunes parameters.
"""

from bike_wind.calibration.metrics import (
    RideSpeedError,
    SegmentTimeScore,
    score_segment_time,
    speed_errors_mph,
)
from bike_wind.calibration.eval_corpus import evaluate_corpus

__all__ = [
    "RideSpeedError",
    "SegmentTimeScore",
    "evaluate_corpus",
    "score_segment_time",
    "speed_errors_mph",
]
