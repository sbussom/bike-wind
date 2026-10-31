"""OSM surface classes and the per-class Crr / Cd·A setup."""

import math

from bike_wind.calibration import cda_defaults
from bike_wind.calibration.eval_corpus import default_params
from bike_wind.physics.solver import solve_speeds
from bike_wind.physics.forces import WindComponents
from bike_wind.rider import resolve_aero, surface_aware_on
from bike_wind.surface import OFF_MAP, PAVED, UNPAVED, classify_tags, route_surface_classes
from bike_wind.surface import osm_surface


def test_classify_tags():
    assert classify_tags({"highway": "secondary"}) == PAVED
    assert classify_tags({"highway": "secondary", "surface": "asphalt"}) == PAVED
    assert classify_tags({"highway": "residential", "surface": "sand"}) == UNPAVED
    assert classify_tags({"highway": "track"}) == UNPAVED
    assert classify_tags({"highway": "track", "tracktype": "grade1"}) == PAVED
    assert classify_tags({"highway": "path", "surface": "concrete"}) == PAVED
    assert classify_tags({"highway": "footway"}) == OFF_MAP
    assert classify_tags({"highway": "footway", "footway": "sidewalk"}) == PAVED


def _way(wid, tags, pts):
    return {"type": "way", "id": wid, "tags": tags, "geometry": [{"lat": a, "lon": b} for a, b in pts]}


def _route():
    # 1.1 km east along lat 27.2 then far from any road.
    lats = [27.2] * 12 + [27.21] * 4
    lons = [-80.40 + 0.001 * i for i in range(12)] + [-80.38 + 0.001 * i for i in range(4)]
    dist = [100.0 * i for i in range(16)]
    return lats, lons, dist


def test_route_classes_with_fake_overpass():
    calls = []

    def fake_fetch(tiles):
        calls.append(list(tiles))
        return [
            _way(1, {"highway": "secondary"}, [(27.2, -80.401), (27.2, -80.3945)]),
            _way(2, {"highway": "track"}, [(27.2, -80.3945), (27.2, -80.388)]),
        ]

    lats, lons, dist = _route()
    out = route_surface_classes(lats, lons, dist, fetch=fake_fetch)
    assert out is not None and len(out) == len(lats)
    assert out[:5] == [PAVED] * 5
    assert out[7:12] == [UNPAVED] * 5
    assert out[12:] == [OFF_MAP] * 4
    # Second call: disk cache, no fetch.
    osm_surface._MEMO.clear()
    out2 = route_surface_classes(lats, lons, dist, fetch=lambda t: (_ for _ in ()).throw(RuntimeError("no")))
    assert out2 == out
    assert len(calls) == 1


def test_offline_without_cache_falls_back_to_none():
    lats, lons, dist = _route()
    assert route_surface_classes(lats, lons, dist) is None


def test_sidewalk_loses_to_road_nearby():
    ways = [
        {"id": 1, "c": PAVED, "hw": "secondary", "g": [[27.2, -80.41], [27.2, -80.39]]},
        {"id": 2, "c": UNPAVED, "hw": "footway", "g": [[27.20005, -80.41], [27.20005, -80.39]]},
    ]
    out = osm_surface.classes_from_ways(ways, [27.20004], [-80.40], [0.0])
    assert out == [PAVED]


def test_surface_aware_flag_only_on_stock_path():
    assert surface_aware_on(resolve_aero("gravel_hoods"), "packed_sand") is cda_defaults.SURFACE_AWARE_ENABLED
    assert surface_aware_on(resolve_aero("gravel_hoods"), "road") is False
    assert surface_aware_on(resolve_aero("gravel_hoods", full_aero_kit=True), "packed_sand") is False


def test_solver_uses_class_setup_and_none_is_unchanged():
    p = default_params().with_updates(power_w=220.0, surface_aware=True)
    n = 200
    grades = [0.0] * n
    winds = [WindComponents()] * n
    dist = [10.0 * i for i in range(n)]
    powers = [220.0] * n
    base = solve_speeds(p, grades, winds, powers_w=powers, distances_m=dist, initial_speed_mps=8.0)
    none = solve_speeds(p, grades, winds, powers_w=powers, distances_m=dist, initial_speed_mps=8.0, surface_classes=None)
    assert base == none
    paved = solve_speeds(p, grades, winds, powers_w=powers, distances_m=dist, initial_speed_mps=8.0, surface_classes=[PAVED] * n)
    crr_p, k_p = cda_defaults.surface_setup(PAVED)
    expect = solve_speeds(
        p.with_updates(crr=crr_p, area_m2=p.area_m2 * k_p, surface_aware=False),
        grades, winds, powers_w=powers, distances_m=dist, initial_speed_mps=8.0,
    )
    assert all(math.isclose(a, b) for a, b in zip(paved, expect))
    # Flag off: classes are ignored.
    off = solve_speeds(p.with_updates(surface_aware=False), grades, winds, powers_w=powers, distances_m=dist,
                       initial_speed_mps=8.0, surface_classes=[PAVED] * n)
    assert off == base
