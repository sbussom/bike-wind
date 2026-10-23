"""Calibration metrics + corpus eval smoke."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bike_wind.calibration.eval_corpus import evaluate_corpus
from bike_wind.calibration.metrics import speed_errors_mph


def test_speed_errors_mph_basic() -> None:
    # pred 1 m/s high on every sample
    pred = [10.0, 11.0, 12.0]
    actual = [9.0, 10.0, 11.0]
    err = speed_errors_mph(pred, actual)
    assert err is not None
    assert err.n == 3
    assert abs(err.mean_signed_mph - 2.236936) < 1e-6
    assert abs(err.mae_mph - 2.236936) < 1e-6


def test_evaluate_corpus_skips_no_watts(tmp_path: Path) -> None:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    n = 60
    streams = {
        "time": list(range(n)),
        "altitude": [0.0] * n,
        "distance": [float(i * 10) for i in range(n)],
        "latlng": [[27.3, -80.3]] * n,
        "velocity_smooth": [8.0] * n,
        "watts": [0] * n,  # no positive watts → skip
    }
    (tmp_path / "ride_1_streams.json").write_text(json.dumps(streams))
    # one with power
    streams2 = dict(streams)
    streams2["watts"] = [200] * n
    (tmp_path / "ride_2_streams.json").write_text(json.dumps(streams2))
    summary = evaluate_corpus(tmp_path, min_points=20)
    assert summary.n_rides == 1
    assert summary.rides[0].activity_id == 2
