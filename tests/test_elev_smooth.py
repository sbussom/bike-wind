"""Elevation smoothing before grade."""

from __future__ import annotations

from bike_wind.gpx.parse import RouteStreams, smooth_elevations


def test_smooth_elevations_kills_blip() -> None:
    # Flat 1 m except a 10 m GPS spike in the middle.
    n = 21
    distances = [float(i * 5) for i in range(n)]
    elevs = [10.0] * n
    elevs[10] = 20.0
    smooth = smooth_elevations(elevs, distances, window_m=100.0)
    # Blip reduced; neighbors pulled up slightly.
    assert smooth[10] < elevs[10]
    assert smooth[10] > 10.0
    # Edge samples share the long window, so they can drift a little.
    assert abs(smooth[0] - 10.0) < 1.5


def test_grades_use_smoothed_elev_by_default() -> None:
    n = 21
    distances = [float(i * 5) for i in range(n)]
    elevs = [10.0] * n
    elevs[10] = 25.0  # sharp spike
    route = RouteStreams(
        lats=[26.0] * n,
        lons=[-80.0] * n,
        elevs_m=elevs,
        distances_m=distances,
    )
    raw = route.grades(elev_smooth_m=0.0)
    smoothed = route.grades()
    # Raw spike produces a large |grade|; smoothed should be milder.
    assert max(abs(g) for g in raw) > max(abs(g) for g in smoothed)
