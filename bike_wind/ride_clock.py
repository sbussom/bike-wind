"""Plot a recorded ride on clock time so a pause is a stretch of 0 mph.

Display only. Nothing here is fed back into ``predict_speeds``.

Strava often drops samples while the ride is paused, so the next point is
minutes later with almost no extra distance. On a distance axis that pause
disappears. On elapsed time it is a flat line at 0 mph for as long as the
stop lasted. The predicted trace sits at 0 across the same gap, then resumes,
so each leg stays lined up with the recording.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from bike_wind.analysis import RECORDED_STOP_MPS, format_duration

MPS_TO_MPH = 2.2369362920544

# Shorter than this is a sample gap or a dab, not a break worth shading.
MIN_STOP_S = 10.0

CAPTION = (
    "Elapsed time from the start of the recording. A stop is a flat line at "
    "0 mph for as long as the clock was stopped or crawling, including pauses "
    "Strava left out of the stream. The prediction sits at 0 mph through each "
    "stop, then continues, so the two lines stay on the same leg."
)


@dataclass(frozen=True)
class StopSpan:
    """One pause on the ride clock, in seconds from the first sample."""

    start_s: float
    end_s: float
    duration_s: float

    @property
    def label(self) -> str:
        return f"stopped {format_duration(self.duration_s)}"


@dataclass(frozen=True)
class ClockSpeedChart:
    """Actual and predicted speed versus elapsed clock time."""

    elapsed_s: list[float]
    actual_mph: list[float]
    predicted_mph: list[float]
    stops: list[StopSpan]
    elapsed_total_s: float
    moving_s: float
    stopped_s: float
    caption: str = CAPTION

    def summary(self) -> str:
        return (
            f"Elapsed {format_duration(self.elapsed_total_s)} · "
            f"moving {format_duration(self.moving_s)} · "
            f"stopped {format_duration(self.stopped_s)}"
        )


def _elapsed(times_utc: list[datetime]) -> list[float]:
    t0 = times_utc[0]
    return [(t - t0).total_seconds() for t in times_utc]


def _merge_spans(spans: list[tuple[float, float]]) -> list[StopSpan]:
    if not spans:
        return []
    spans = sorted((a, b) for a, b in spans if b - a >= MIN_STOP_S)
    if not spans:
        return []
    merged: list[list[float]] = [[spans[0][0], spans[0][1]]]
    for start_s, end_s in spans[1:]:
        if start_s <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], end_s)
        else:
            merged.append([start_s, end_s])
    return [
        StopSpan(start_s=a, end_s=b, duration_s=b - a)
        for a, b in merged
        if b - a >= MIN_STOP_S
    ]


def detect_stops(
    times_utc: list[datetime],
    distances_m: list[float],
    actual_mps: list[float],
) -> list[StopSpan]:
    """Pauses: a time gap with almost no distance, or a run of near-zero speed.

    A gap counts when the clock jumps at least ``MIN_STOP_S`` and the distance
    covered in that jump is slower than ``RECORDED_STOP_MPS`` (about 1.1 mph).
    That is how a Strava pause shows up: the next point is later, not farther.
    Samples already recorded at under that speed for at least ``MIN_STOP_S``
    count too.
    """
    n = min(len(times_utc), len(distances_m), len(actual_mps))
    if n < 2:
        return []
    elapsed = _elapsed(times_utc[:n])
    spans: list[tuple[float, float]] = []
    for i in range(n - 1):
        dt = elapsed[i + 1] - elapsed[i]
        dd = max(0.0, float(distances_m[i + 1]) - float(distances_m[i]))
        if dt < MIN_STOP_S:
            continue
        implied = dd / dt if dt > 0.0 else 0.0
        if implied < RECORDED_STOP_MPS:
            spans.append((elapsed[i], elapsed[i + 1]))

    i = 0
    while i < n:
        speed = actual_mps[i]
        if not (math.isfinite(speed) and speed < RECORDED_STOP_MPS):
            i += 1
            continue
        j = i
        while j + 1 < n:
            nxt = actual_mps[j + 1]
            if not (math.isfinite(nxt) and nxt < RECORDED_STOP_MPS):
                break
            j += 1
        if elapsed[j] - elapsed[i] >= MIN_STOP_S:
            spans.append((elapsed[i], elapsed[j]))
        i = j + 1
    return _merge_spans(spans)


def clock_speed_chart(
    times_utc: list[datetime] | None,
    distances_m: list[float],
    actual_mps: list[float] | None,
    predicted_mps: list[float],
) -> ClockSpeedChart | None:
    """Elapsed-time series with pause gaps filled at 0 mph.

    Predicted speed is 0 across each stop so the trace waits with the rider
    and picks up on the next leg, instead of drawing a slope through the pause.
    """
    if not times_utc or actual_mps is None:
        return None
    n = min(len(times_utc), len(distances_m), len(actual_mps), len(predicted_mps))
    if n < 2:
        return None
    times = list(times_utc[:n])
    dists = [float(d) for d in distances_m[:n]]
    actual = [float(v) for v in actual_mps[:n]]
    predicted = [float(v) for v in predicted_mps[:n]]
    elapsed = _elapsed(times)
    stops = detect_stops(times, dists, actual)
    gap_stop = [False] * (n - 1)
    for i in range(n - 1):
        dt = elapsed[i + 1] - elapsed[i]
        dd = max(0.0, dists[i + 1] - dists[i])
        if dt >= MIN_STOP_S and (dd / dt if dt > 0.0 else 0.0) < RECORDED_STOP_MPS:
            gap_stop[i] = True

    def mph(speed: float, *, force_zero: bool) -> float:
        if force_zero:
            return 0.0
        if not math.isfinite(speed):
            return float("nan")
        return max(0.0, speed) * MPS_TO_MPH

    in_run = [False] * n
    for stop in stops:
        for i, t in enumerate(elapsed):
            if stop.start_s <= t <= stop.end_s and math.isfinite(actual[i]) and actual[i] < RECORDED_STOP_MPS:
                in_run[i] = True

    ts: list[float] = []
    acts: list[float] = []
    preds: list[float] = []
    for i in range(n):
        if i > 0 and gap_stop[i - 1]:
            # Drop to 0 at the last moving sample, hold 0 until the clock resumes.
            ts.append(elapsed[i - 1])
            acts.append(0.0)
            preds.append(0.0)
            ts.append(elapsed[i])
            acts.append(0.0)
            preds.append(0.0)
        ts.append(elapsed[i])
        acts.append(mph(actual[i], force_zero=in_run[i]))
        preds.append(mph(predicted[i], force_zero=in_run[i]))
    elapsed_total = max(0.0, elapsed[-1] - elapsed[0])
    stopped_s = sum(stop.duration_s for stop in stops)
    moving_s = max(0.0, elapsed_total - stopped_s)
    return ClockSpeedChart(
        elapsed_s=ts,
        actual_mph=acts,
        predicted_mph=preds,
        stops=stops,
        elapsed_total_s=elapsed_total,
        moving_s=moving_s,
        stopped_s=stopped_s,
    )


def figure_clock_speed(chart: ClockSpeedChart, *, dark: bool = True):
    """Matplotlib figure. Caller closes it."""
    import matplotlib.pyplot as plt

    if dark:
        bg, fg, accent, spine = "#0c1210", "#e7efea", "#3ecf8e", "#3a4a42"
        actual = "#8fa398"
        shade = "#c45c26"
    else:
        bg, fg, accent, spine = "#f7faf8", "#14201b", "#1a6b52", "#c5d0ca"
        actual = "#5c6f66"
        shade = "#c45c26"
    minutes = [t / 60.0 for t in chart.elapsed_s]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    y_top = max(
        [v for v in chart.actual_mph + chart.predicted_mph if math.isfinite(v)] or [1.0]
    )
    y_top = max(8.0, y_top * 1.08)
    for stop in chart.stops:
        ax.axvspan(
            stop.start_s / 60.0,
            stop.end_s / 60.0,
            color=shade,
            alpha=0.28,
            linewidth=0,
            zorder=0,
        )
        ax.text(
            (stop.start_s + stop.end_s) / 120.0,
            y_top * 0.92,
            stop.label,
            color=fg,
            fontsize=8,
            ha="center",
            va="top",
        )
    ax.plot(minutes, chart.predicted_mph, color=accent, linewidth=2.0, label="Predicted", zorder=2)
    ax.plot(minutes, chart.actual_mph, color=actual, linewidth=1.8, label="Actual", zorder=3)
    ax.set_ylim(0.0, y_top)
    ax.set_title("Speed on the ride clock", color=fg, loc="left", fontsize=12, pad=8)
    ax.set_xlabel("Elapsed time (min)", color=fg)
    ax.set_ylabel("mph", color=fg)
    ax.tick_params(colors=fg)
    for sp in ax.spines.values():
        sp.set_color(spine)
    ax.grid(True, alpha=0.22, color=fg)
    leg = ax.legend(facecolor=bg, edgecolor=spine, fontsize=8)
    for text in leg.get_texts():
        text.set_color(fg)
    fig.tight_layout()
    return fig
