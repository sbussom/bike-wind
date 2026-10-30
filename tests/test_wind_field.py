"""Tests for the GustyPower wind map grid + HTML (no network)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from bike_wind.wind_field import (
    FLORIDA_GRID,
    HOME_LAT,
    HOME_LON,
    LOCAL_GRID,
    GridSpec,
    WindField,
    WindGrid,
    color_for_mph,
    fetch_grid,
    frame_indices,
    sample_point,
    subset_payload,
    uv_from_speed_dir,
)
from bike_wind.wind_map_html import build_wind_map_html


def test_grids_cover_home_and_stay_small() -> None:
    for g in (FLORIDA_GRID, LOCAL_GRID):
        assert g.lat_min < HOME_LAT < g.lat_max
        assert g.lon_min < HOME_LON < g.lon_max
    # Open-Meteo counts each point as a call; keep refreshes cheap.
    assert FLORIDA_GRID.size + LOCAL_GRID.size <= 250


def test_uv_from_north_wind_blows_south() -> None:
    u, v = uv_from_speed_dir(10.0, 0.0)
    assert abs(u) < 1e-9 and math.isclose(v, -10.0)
    u, v = uv_from_speed_dir(10.0, 90.0)  # from east → toward west
    assert math.isclose(u, -10.0) and abs(v) < 1e-9


def test_color_scale_ends() -> None:
    assert color_for_mph(0)[2] > color_for_mph(0)[0]  # blue-ish at calm
    hot = color_for_mph(45)
    assert hot[0] > hot[1] and hot[0] > hot[2]  # red-ish at 45 mph
    assert color_for_mph(99) == color_for_mph(50)


def _tiny_field() -> WindField:
    spec = GridSpec("t", 27.0, 27.4, -80.6, -80.2, 2, 2)
    g = WindGrid(spec=spec)
    for f in range(3):
        g.speed_mph.append([10.0 + f] * 4)
        g.dir_deg.append([90.0] * 4)
        g.gust_mph.append([15.0] * 4)
    t0 = datetime(2026, 10, 30, tzinfo=timezone.utc)
    return WindField(times_utc=[t0 + timedelta(hours=h) for h in range(3)], grids=[g])


def test_sample_point_and_payload_subset() -> None:
    field = _tiny_field()
    spd, d, gust = sample_point(field.grids[0], 2, 27.2, -80.4)
    assert math.isclose(spd, 12.0, abs_tol=1e-6)
    assert math.isclose(d, 90.0, abs_tol=1e-6)
    assert gust == 15.0
    payload = field.to_payload()
    sub = subset_payload(payload, [0, 2])
    assert sub["times"] == ["2026-10-30T00:00Z", "2026-10-30T02:00Z"]
    assert len(sub["grids"][0]["u"]) == 2


def test_frame_indices_skip_past_hours() -> None:
    field = _tiny_field()
    now = field.times_utc[1] + timedelta(minutes=20)
    assert frame_indices(field.times_utc, now=now) == [1, 2]
    assert frame_indices(field.times_utc, step_hours=3, now=field.times_utc[0]) == [0]


def test_fetch_grid_parses_multi_location() -> None:
    spec = GridSpec("t", 27.0, 27.4, -80.6, -80.2, 2, 2)

    class Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self):
            row = {
                "hourly": {
                    "time": ["2026-10-30T00:00", "2026-10-30T01:00"],
                    "wind_speed_10m": [5.0, 6.0],
                    "wind_direction_10m": [180.0, 190.0],
                    "wind_gusts_10m": [9.0, None],
                }
            }
            return [row] * 4

    class Sess:
        def get(self, *a, **k):
            return Resp()

    times, grid = fetch_grid(spec, session=Sess())  # type: ignore[arg-type]
    assert len(times) == 2 and times[0].tzinfo is not None
    assert grid.speed_mph[1] == [6.0] * 4
    assert grid.gust_mph[1] == [0.0] * 4


def test_html_is_branded_and_has_controls() -> None:
    field = _tiny_field()
    html = build_wind_map_html(
        field.to_payload(),
        center=(HOME_LAT, HOME_LON),
        home=(HOME_LAT, HOME_LON),
        route=[[27.2, -80.47], [27.2, -80.41]],
        route_label="Round trip to pumphouse",
        predictions={"2026-10-30T01:00Z": "Round trip: 23:51"},
        initial_frame=1,
    )
    assert "GustyPower" in html.replace("<span>", "").replace("</span>", "")
    assert "windy" not in html.lower()
    assert 'id="play"' in html and 'type="range"' in html
    assert "__DATA__" not in html and "__CFG__" not in html
