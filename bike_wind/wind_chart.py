"""Wind-along-the-route chart data. Display only; does not change a prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass

from bike_wind.local_wind_map import compass_label
from bike_wind.predict import PredictionResult

MPS_TO_MPH = 2.2369362920544

CAPTION = (
    "Headwind is above the zero line and tailwind is below, in mph at bike height. "
    "This is the sheltered wind the prediction uses (the 10 m forecast brought down "
    "to the rider, then ×0.75). The dotted line is the forecast gust in that same "
    "direction, when the forecast has one. Tags say where the wind is coming from."
)


@dataclass(frozen=True)
class WindAlongChart:
    """Series for the distance wind graph."""

    distance_mi: list[float]
    headwind_mph: list[float]
    gust_headwind_mph: list[float] | None
    from_labels: list[tuple[float, str]]
    caption: str = CAPTION

    @property
    def has_gust(self) -> bool:
        if not self.gust_headwind_mph:
            return False
        return any(math.isfinite(v) for v in self.gust_headwind_mph)


def wind_scale_for_slider(result: PredictionResult, wind_speed_mps: float | None) -> float:
    """Same factor ``what_if_course`` applies to the along-track wind."""
    if wind_speed_mps is None or not result.weather_ok:
        return 1.0
    base = max(0.0, float(result.mean_wind_speed_mps))
    target = max(0.0, float(wind_speed_mps))
    if base > 1e-6:
        return target / base
    if target <= 1e-6:
        return 0.0
    return 1.0


def _headwind_mph(along_tail_mps: float, scale: float) -> float:
    """Model along-track is +tailwind. The chart uses +headwind."""
    return -float(along_tail_mps) * scale * MPS_TO_MPH


def wind_along_chart(
    result: PredictionResult,
    *,
    wind_speed_mps: float | None = None,
) -> WindAlongChart | None:
    """Headwind/tailwind vs distance, or None when there is no forecast wind."""
    if not result.weather_ok or not result.wind_along_mps or not result.distances_m:
        return None
    if len(result.wind_along_mps) != len(result.distances_m):
        return None
    scale = wind_scale_for_slider(result, wind_speed_mps)
    distance_mi = [d / 1609.344 for d in result.distances_m]
    head = [_headwind_mph(a, scale) for a in result.wind_along_mps]
    gust: list[float] | None = None
    if result.gust_along_mps and len(result.gust_along_mps) == len(result.wind_along_mps):
        gust = []
        for g in result.gust_along_mps:
            if g is None or not math.isfinite(g):
                gust.append(float("nan"))
            else:
                gust.append(_headwind_mph(g, scale))
        if not any(math.isfinite(v) for v in gust):
            gust = None
    labels = _from_labels(distance_mi, result.wind_from_deg_samples)
    return WindAlongChart(
        distance_mi=distance_mi,
        headwind_mph=head,
        gust_headwind_mph=gust,
        from_labels=labels,
    )


def _from_labels(
    distance_mi: list[float],
    from_deg: list[float] | None,
) -> list[tuple[float, str]]:
    if not from_deg or len(from_deg) != len(distance_mi) or len(distance_mi) < 1:
        return []
    n = len(distance_mi)
    # A few tags, not one per GPS point.
    idxs = sorted({int(round(t * (n - 1))) for t in (0.08, 0.35, 0.62, 0.9)})
    out: list[tuple[float, str]] = []
    for i in idxs:
        deg = from_deg[i]
        if deg is None or not math.isfinite(deg):
            continue
        out.append((distance_mi[i], compass_label(deg)))
    return out


def figure_wind_along(chart: WindAlongChart, *, dark: bool = True):
    """Matplotlib figure. Caller closes it."""
    import matplotlib.pyplot as plt

    if dark:
        bg, fg, accent, spine = "#0c1210", "#e7efea", "#3ecf8e", "#3a4a42"
        gust_color = "#e0b15a"
        zero = "#8fa398"
    else:
        bg, fg, accent, spine = "#f7faf8", "#14201b", "#1a6b52", "#c5d0ca"
        gust_color = "#8a6a3d"
        zero = "#5c6f66"
    fig, ax = plt.subplots(figsize=(9, 3.0))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axhline(0.0, color=zero, linewidth=1.0, linestyle="-", alpha=0.8, zorder=1)
    ax.plot(
        chart.distance_mi,
        chart.headwind_mph,
        color=accent,
        linewidth=2.0,
        label="Sustained headwind (+) / tailwind (−)",
        zorder=3,
    )
    if chart.has_gust and chart.gust_headwind_mph is not None:
        ax.plot(
            chart.distance_mi,
            chart.gust_headwind_mph,
            color=gust_color,
            linewidth=1.4,
            linestyle=":",
            label="Gust (same direction)",
            zorder=2,
        )
    if chart.distance_mi:
        x0, x1 = chart.distance_mi[0], chart.distance_mi[-1]
        ax.set_xlim(x0, x1)
        ax.axvline(x0, color=spine, linewidth=1.0, linestyle="--", alpha=0.9)
        ax.axvline(x1, color=spine, linewidth=1.0, linestyle="--", alpha=0.9)
        y_top = ax.get_ylim()[1]
        ax.text(x0, y_top, " Start", color=fg, fontsize=8, va="top", ha="left")
        ax.text(x1, y_top, "End ", color=fg, fontsize=8, va="top", ha="right")
    for x, label in chart.from_labels:
        ax.annotate(
            f"from {label}",
            xy=(x, 0.0),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=fg,
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": spine, "lw": 0.6},
        )
    ax.set_title("Wind along the route", color=fg, loc="left", fontsize=12, pad=8)
    ax.set_xlabel("Distance (mi)", color=fg)
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
