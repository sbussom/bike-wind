"""Stops on the ride clock: omitted Strava pauses become a stretch of 0 mph."""

import math
from datetime import datetime, timedelta, timezone

from bike_wind.ride_clock import MIN_STOP_S, clock_speed_chart, detect_stops, figure_clock_speed


def _times(seconds: list[float]) -> list[datetime]:
    t0 = datetime(2026, 10, 22, 14, 0, tzinfo=timezone.utc)
    return [t0 + timedelta(seconds=s) for s in seconds]


def test_omitted_pause_is_zero_speed_for_its_duration():
    """Strava drops samples while paused: time jumps, distance does not."""
    # 2 min rolling, 4:30 stopped, 2 min rolling. 8 m/s covers 960 m per leg.
    times = _times([0, 60, 120, 120 + 270, 120 + 270 + 60, 120 + 270 + 120])
    dist = [0.0, 480.0, 960.0, 965.0, 1445.0, 1925.0]
    actual = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
    predicted = [7.5, 7.5, 7.5, 7.4, 7.4, 7.4]
    stops = detect_stops(times, dist, actual)
    assert len(stops) == 1
    assert stops[0].duration_s == 270
    assert stops[0].label == "stopped 4:30"

    chart = clock_speed_chart(times, dist, actual, predicted)
    assert chart is not None
    assert chart.stopped_s == 270
    assert chart.elapsed_total_s == 120 + 270 + 120
    assert chart.moving_s == chart.elapsed_total_s - 270
    # The omitted pause is two explicit 0 mph points, 4:30 apart, so the
    # line stays at zero for the whole lapse instead of bridging the gap.
    rows = list(zip(chart.elapsed_s, chart.actual_mph, chart.predicted_mph))
    assert any(
        t0 == 120.0 and a0 == 0.0 and p0 == 0.0 and t1 == 390.0 and a1 == 0.0 and p1 == 0.0
        for (t0, a0, p0), (t1, a1, p1) in zip(rows, rows[1:])
    )
    # After the stop the prediction is rolling again (not stuck at 0).
    resumed = [
        p
        for t, p in zip(chart.elapsed_s, chart.predicted_mph)
        if t >= 120 + 270 and p > 1.0
    ]
    assert resumed
    assert chart.summary().startswith("Elapsed")
    assert "stopped 4:30" in chart.summary()


def test_near_zero_speed_run_is_a_stop_and_a_short_gap_is_not():
    times = _times([0, 5, 15, 25, 50, 55])
    dist = [0.0, 40.0, 40.0, 40.0, 80.0, 120.0]
    # 15s–50s is recorded at 0. Index 2, 3, 4 span 15 to 50 = 35s.
    actual = [8.0, 8.0, 0.0, 0.1, 0.0, 8.0]
    predicted = [7.0] * 6
    stops = detect_stops(times, dist, actual)
    assert len(stops) == 1
    assert stops[0].duration_s >= MIN_STOP_S
    chart = clock_speed_chart(times, dist, actual, predicted)
    assert chart is not None
    held = [
        p
        for t, p in zip(chart.elapsed_s, chart.predicted_mph)
        if stops[0].start_s < t < stops[0].end_s
    ]
    assert held and all(p == 0.0 for p in held)

    # Five seconds of clock with real distance is not a break.
    short_t = _times([0, 5])
    assert detect_stops(short_t, [0.0, 40.0], [8.0, 8.0]) == []


def test_no_chart_without_a_clock():
    assert clock_speed_chart(None, [0.0, 10.0], [8.0, 8.0], [7.0, 7.0]) is None


def test_figure_labels_stop_and_mph_axis():
    times = _times([0, 30, 300, 330])
    chart = clock_speed_chart(
        times,
        [0.0, 240.0, 245.0, 485.0],
        [8.0, 8.0, 8.0, 8.0],
        [7.5, 7.5, 7.5, 7.5],
    )
    assert chart is not None and chart.stops
    fig = figure_clock_speed(chart, dark=True)
    ax = fig.axes[0]
    assert ax.get_ylabel() == "mph"
    assert "Elapsed" in ax.get_xlabel()
    texts = [t.get_text() for t in ax.texts]
    assert any(text.startswith("stopped ") for text in texts)
    assert any(math.isfinite(v) for v in chart.actual_mph)
    fig.clf()
