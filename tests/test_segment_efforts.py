"""Segment efforts on a ride → slice activity streams for actual speed + watts."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from bike_wind.gpx.parse import RouteStreams
from bike_wind.strava.client import (
    StravaClient,
    StravaConfig,
    route_with_effort_actuals,
    slice_route_for_effort,
)
from bike_wind.strava.models import SegmentEffortSummary


def _config(tmp_path: Path) -> StravaConfig:
    return StravaConfig(
        access_token="tok",
        client_id="1",
        client_secret="s",
        redirect_uri="https://gustypower.streamlit.app",
        refresh_token="r",
        cache_dir=tmp_path,
    )


def test_list_activity_segment_efforts(tmp_path: Path) -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "id": 10,
        "segment_efforts": [
            {
                "id": 1001,
                "name": "Climb",
                "elapsed_time": 125,
                "moving_time": 120,
                "distance": 800.0,
                "start_date": "2024-06-01T14:05:00Z",
                "average_watts": 250,
                "start_index": 10,
                "end_index": 40,
                "segment": {"id": 55, "name": "Hill Climb", "distance": 800.0},
            }
        ],
    }
    session.get.return_value = resp
    client = StravaClient(_config(tmp_path), session=session)
    efforts = client.list_activity_segment_efforts(10)
    assert len(efforts) == 1
    assert efforts[0].segment_id == 55
    assert efforts[0].name == "Hill Climb"
    assert efforts[0].elapsed_time_s == 125
    assert efforts[0].start_index == 10
    assert efforts[0].end_index == 40
    assert efforts[0].has_stream_indices
    assert "125" in efforts[0].label() or "2:05" in efforts[0].label()
    assert session.get.call_args[0][0].endswith("/activities/10")


def test_slice_route_for_effort_keeps_speed_and_watts() -> None:
    n = 6
    route = RouteStreams(
        lats=[26.0 + i * 0.001 for i in range(n)],
        lons=[-80.0] * n,
        elevs_m=[float(i) for i in range(n)],
        distances_m=[float(i * 100) for i in range(n)],
        times_utc=[datetime(2024, 6, 1, 14, 0, i, tzinfo=timezone.utc) for i in range(n)],
        actual_speeds_mps=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        watts=[100.0, 150.0, 200.0, 250.0, 300.0, 350.0],
    )
    effort = SegmentEffortSummary(
        effort_id=1,
        segment_id=9,
        name="Test",
        distance_m=200.0,
        elapsed_time_s=40,
        moving_time_s=38,
        start_date="2024-06-01T14:00:00Z",
        start_index=1,
        end_index=3,
    )
    sliced = slice_route_for_effort(route, effort)
    assert len(sliced) == 3
    assert sliced.distances_m[0] == 0.0
    assert sliced.distances_m[-1] == 200.0
    assert sliced.actual_speeds_mps == [4.0, 5.0, 6.0]
    assert sliced.watts == [150.0, 200.0, 250.0]
    assert sliced.has_watts
    # Full activity unchanged
    assert route.actual_speeds_mps is not None
    assert route.actual_speeds_mps[0] == 3.0


def test_slice_skips_missing_speed_with_nan() -> None:
    route = RouteStreams(
        lats=[26.0, 26.01, 26.02, 26.03],
        lons=[-80.0] * 4,
        elevs_m=[0.0, 1.0, 2.0, 3.0],
        distances_m=[0.0, 100.0, 200.0, 300.0],
        actual_speeds_mps=[4.0, float("nan"), 5.0, 6.0],
        watts=[180.0, float("nan"), 190.0, 200.0],
    )
    effort = SegmentEffortSummary(
        effort_id=2,
        segment_id=9,
        name="Gap",
        distance_m=200.0,
        elapsed_time_s=50,
        moving_time_s=50,
        start_date="2024-06-01T14:00:00Z",
        start_index=0,
        end_index=2,
    )
    sliced = slice_route_for_effort(route, effort)
    assert sliced.has_actual_speeds
    assert math.isnan(sliced.actual_speeds_mps[1])  # type: ignore[index]
    assert not all(math.isfinite(v) for v in sliced.actual_speeds_mps)  # type: ignore[arg-type]


def test_route_with_effort_actuals_legacy_constant_without_indices() -> None:
    route = RouteStreams(
        lats=[26.0, 26.01, 26.02],
        lons=[-80.0, -80.0, -80.0],
        elevs_m=[1.0, 5.0, 10.0],
        distances_m=[0.0, 100.0, 200.0],
    )
    effort = SegmentEffortSummary(
        effort_id=1,
        segment_id=9,
        name="Test",
        distance_m=200.0,
        elapsed_time_s=40,
        moving_time_s=38,
        start_date="2024-06-01T14:00:00Z",
    )
    enriched = route_with_effort_actuals(route, effort)
    assert enriched.has_actual_speeds
    assert enriched.actual_speeds_mps is not None
    assert enriched.actual_speeds_mps[0] == 5.0  # 200m / 40s
    assert enriched.start_time is not None
    assert enriched.start_time == datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc)
    assert route.actual_speeds_mps is None
