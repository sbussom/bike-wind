"""GPX parse smoke tests."""

from __future__ import annotations

from pathlib import Path

from bike_wind.gpx import parse_gpx

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_route.gpx"


def test_parse_sample_fixture() -> None:
    route = parse_gpx(FIXTURE)
    assert len(route) >= 5
    assert route.distances_m[0] == 0.0
    assert route.distances_m[-1] > 1000.0
    assert len(route.grades()) == len(route)
    assert len(route.bearings_deg()) == len(route)


def test_parse_from_bytes() -> None:
    data = FIXTURE.read_bytes()
    route = parse_gpx(data)
    assert len(route) >= 5


def test_route_midpoint() -> None:
    route = parse_gpx(FIXTURE)
    lat, lon = route.midpoint()
    assert 26.0 < lat < 27.0
    assert -81.0 < lon < -79.0
    # Mid index point should match streams
    i = (len(route) - 1) // 2
    assert lat == route.lats[i]
    assert lon == route.lons[i]
