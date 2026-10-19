"""Tests for nearby Strava explore map helpers."""

from __future__ import annotations

import pytest

from bike_wind.nearby_map import bounds_around, build_nearby_segments_map
from bike_wind.strava.client import ExploreHit


def test_bounds_around_is_symmetric() -> None:
    b = bounds_around(26.7, -80.0, half_span_deg=0.1)
    assert b[0] == pytest.approx(26.6)
    assert b[1] == pytest.approx(-80.1)
    assert b[2] == pytest.approx(26.8)
    assert b[3] == pytest.approx(-79.9)


def test_build_nearby_map_renders_markers() -> None:
    hits = [
        ExploreHit(
            segment_id=1,
            name="Bridge",
            distance_m=1200,
            average_grade=1.5,
            climb_category=0,
            starred=False,
            start_lat=26.70,
            start_lon=-80.01,
            end_lat=26.71,
            end_lon=-80.00,
        )
    ]
    fmap = build_nearby_segments_map(26.7, -80.0, hits, dark=True)
    html = fmap.get_root().render()
    assert "Bridge" in html
    assert "segments/1" in html
