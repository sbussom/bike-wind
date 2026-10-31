"""Production predictions stay put; experimental setups move them on purpose."""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bike_wind.calibration import cda_defaults
from bike_wind.experimental import (
    AERO_WHEELS_CDA_MULTIPLIER,
    AERO_WHEELS_CRR,
    AERO_WHEELS_PRESET_NAME,
    GRAVEL_SECTION_CRR,
    PRODUCTION_CRR,
    ROAD_SECTION_CRR,
    SurfaceReport,
    apply_manual_setup,
    average_speed_mph,
    build_production_params,
    build_sections,
    compare_with_production,
    crr_for_sections,
    distance_shares,
    lookup_surface,
    resimulate,
    sections_for_route,
)
from bike_wind.gpx.parse import RouteStreams
from bike_wind.physics.forces import WindComponents
from bike_wind.physics.solver import solve_speeds
from bike_wind.predict import PredictionResult, downsample_route, predict_speeds
from bike_wind.surface.osm_surface import OFF_MAP, PAVED, UNPAVED, route_surface_classes
from bike_wind.weather.open_meteo import WeatherFetchError

# Speeds from the production solver on main (55d1e38), before surface classes
# existed. Same rider, power series, grade, and wind as ``_golden_case``.
_GOLDEN_SPEEDS = [
    6.5, 6.626079119214, 6.709708238835, 6.765980612767, 6.804193252512,
    6.830298372117, 6.848203782456, 6.860518306085, 6.869003274722, 5.598488123185,
    6.076315407058, 6.356509922254, 6.532742717549, 6.647655101298, 6.724166244493,
    7.172794098259, 7.481195835234, 7.701906823024, 6.882819767651, 7.28002186394,
    7.557181200466, 7.757274258646, 7.904817913655, 8.01515365849, 8.098474313594,
    8.161835350226, 8.210264526068, 7.358787232024, 7.41226299843, 7.450458094412,
    7.477857858315, 7.497573614532, 7.511791053546, 7.522059412761, 7.529483828089,
    7.534856245475, 6.480065011334, 6.815907114164, 7.036375141965, 7.186304416797,
]


def _golden_case():
    params = build_production_params(rider_kg=69.0, power_w=220.0)
    n = 40
    grades = [0.008 if i < 15 else (-0.004 if i < 28 else 0.0) for i in range(n)]
    winds = [WindComponents(along_track=-1.5, cross_track=0.8)] * n
    powers = [220.0 if i % 9 else 0.0 for i in range(n)]
    dist = [20.0 * i for i in range(n)]
    return params, grades, winds, powers, dist


def _flat_result(params, n: int = 80, power: float = 220.0) -> PredictionResult:
    dist = [15.0 * i for i in range(n)]
    grades = [0.0] * n
    winds = [WindComponents()] * n
    powers = [power] * n
    gap: list[float] = []
    speeds = solve_speeds(
        params,
        grades,
        winds,
        powers_w=powers,
        distances_m=dist,
        gap_dt=gap,
    )
    return PredictionResult(
        distances_m=dist,
        speeds_with_wind_mps=speeds,
        speeds_no_wind_mps=list(speeds),
        grades=grades,
        wind_along_mps=[0.0] * n,
        wind_cross_mps=[0.0] * n,
        winds=winds,
        params=params,
        rho_mean=params.rho,
        weather_ok=True,
        watts=powers,
        gap_dt_with_wind=list(gap),
        gap_dt_no_wind=list(gap),
    )


def _report(classes: list[int], distances: list[float]) -> SurfaceReport:
    paved, unpaved, unmapped = distance_shares(classes, distances)
    return SurfaceReport(
        ok=True,
        classes=tuple(classes),
        error=None,
        paved_share=paved,
        unpaved_share=unpaved,
        unmapped_share=unmapped,
    )


def test_surface_flag_stays_off_and_main_page_does_not_opt_in():
    assert cda_defaults.SURFACE_AWARE_ENABLED is False
    root = Path(__file__).resolve().parents[1]
    app_src = (root / "app.py").read_text()
    assert "surface_aware" not in app_src
    assert "pages/2_Experimental.py" in app_src
    predict_src = inspect.getsource(predict_speeds)
    assert "surface_aware" not in predict_src
    assert "route_surface_classes" not in predict_src
    page = (root / "pages" / "2_Experimental.py").read_text()
    assert "not part of the validated production model" in page
    assert "Experimental" in page
    assert "AERO_WHEELS_PRESET_NAME" in page
    assert AERO_WHEELS_PRESET_NAME == "Aero wheels + road tires (unvalidated guess)"
    assert "unvalidated" in page.lower()


def test_production_solver_matches_pre_surface_golden_speeds():
    params, grades, winds, powers, dist = _golden_case()
    assert params.surface_aware is False
    assert params.crr == pytest.approx(PRODUCTION_CRR)
    speeds = solve_speeds(
        params,
        grades,
        winds,
        powers_w=powers,
        distances_m=dist,
        initial_speed_mps=6.5,
    )
    assert len(speeds) == len(_GOLDEN_SPEEDS)
    assert speeds == pytest.approx(_GOLDEN_SPEEDS, abs=1e-12)
    # Classes are ignored unless the experimental flag on the params is on.
    paved = solve_speeds(
        params,
        grades,
        winds,
        powers_w=powers,
        distances_m=dist,
        initial_speed_mps=6.5,
        surface_classes=[PAVED] * len(grades),
    )
    assert paved == speeds


def test_predict_speeds_does_not_call_osm_and_resimulate_matches(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("production predict_speeds must not look up OSM")

    monkeypatch.setattr("bike_wind.surface.osm_surface.route_surface_classes", boom)
    monkeypatch.setattr(
        "bike_wind.predict.fetch_weather_along_route",
        lambda *_a, **_k: (_ for _ in ()).throw(WeatherFetchError("offline")),
    )
    n = 25
    route = RouteStreams(
        lats=[27.21] * n,
        lons=[-80.45 + i * 0.0002 for i in range(n)],
        elevs_m=[8.0] * n,
        distances_m=[25.0 * i for i in range(n)],
    )
    params = build_production_params(rider_kg=69.0, power_w=200.0)
    result = predict_speeds(
        route,
        params,
        datetime(2026, 10, 31, 15, tzinfo=timezone.utc),
    )
    assert result.params.surface_aware is False
    speeds, eta = resimulate(result, result.params)
    assert speeds == result.speeds_with_wind_mps
    assert eta == result.eta_with_wind_s
    columns = compare_with_production(result)
    assert len(columns) == 1
    assert columns[0].label == "Production"
    assert columns[0].eta_s == result.eta_with_wind_s
    assert columns[0].delta_s == 0.0


def test_manual_setup_moves_time_in_the_expected_direction():
    params = build_production_params(rider_kg=69.0, power_w=220.0)
    result = _flat_result(params)
    faster = compare_with_production(
        result,
        manual_cda_multiplier=AERO_WHEELS_CDA_MULTIPLIER,
        manual_crr=AERO_WHEELS_CRR,
    )
    assert faster[1].label == "Manual setup"
    assert faster[1].eta_s < faster[0].eta_s
    assert faster[1].avg_speed_mph > faster[0].avg_speed_mph
    assert faster[1].delta_s < 0.0
    assert not faster[1].fell_back

    slower = compare_with_production(
        result,
        manual_cda_multiplier=1.05,
        manual_crr=0.008,
    )
    assert slower[1].eta_s > slower[0].eta_s
    assert slower[1].avg_speed_mph < slower[0].avg_speed_mph

    # Identity: multiplier 1 and the production Crr do not move the solve.
    _base_speeds, base_eta = resimulate(result, params)
    assert base_eta == result.eta_with_wind_s
    same = apply_manual_setup(params, cda_multiplier=1.0, crr=params.crr)
    assert same.surface_aware is False
    assert same.clothing_cda_scale == params.clothing_cda_scale
    _speeds, eta = resimulate(result, same)
    assert eta == result.eta_with_wind_s


def test_surface_classes_move_time_and_failure_falls_back():
    params = build_production_params(rider_kg=69.0, power_w=220.0)
    result = _flat_result(params, n=60)
    n = len(result.distances_m)
    paved = compare_with_production(
        result, surface_report=_report([PAVED] * n, result.distances_m)
    )
    assert paved[1].eta_s < paved[0].eta_s
    assert paved[1].avg_speed_mph > paved[0].avg_speed_mph
    assert "paved" in paved[1].note

    unpaved = compare_with_production(
        result, surface_report=_report([UNPAVED] * n, result.distances_m)
    )
    assert unpaved[1].eta_s > unpaved[0].eta_s
    assert unpaved[1].avg_speed_mph < unpaved[0].avg_speed_mph

    off = compare_with_production(
        result, surface_report=_report([OFF_MAP] * n, result.distances_m)
    )
    assert off[1].eta_s == pytest.approx(off[0].eta_s)
    assert off[1].fell_back is False

    failed = SurfaceReport(
        ok=False,
        classes=None,
        error="Overpass timed out",
        paved_share=0.0,
        unpaved_share=0.0,
        unmapped_share=0.0,
    )
    fallback = compare_with_production(result, surface_report=failed)
    assert fallback[1].fell_back is True
    assert fallback[1].eta_s == fallback[0].eta_s
    assert fallback[1].delta_s == 0.0
    assert "fell back" in fallback[1].note.lower()
    assert "Overpass timed out" in fallback[1].note


def test_distance_shares_and_lookup_failure_message():
    shares = distance_shares(
        [PAVED, UNPAVED, OFF_MAP, OFF_MAP], [0.0, 100.0, 150.0, 200.0]
    )
    assert shares[0] == pytest.approx(0.5)  # 100 m paved of 200
    assert shares[1] == pytest.approx(0.25)
    assert shares[2] == pytest.approx(0.25)
    report = lookup_surface([27.2, 27.2], [-80.4, -80.39], [0.0, 100.0])
    assert report.fell_back is True
    assert report.classes is None
    assert report.error


def test_overpass_timeout_is_bounded(monkeypatch):
    captured: dict = {}

    def post(url, data=None, headers=None, timeout=None):
        captured["timeout"] = timeout
        captured["query"] = data["data"]
        raise RuntimeError("no network")

    monkeypatch.delenv("GUSTYPOWER_OSM_OFFLINE", raising=False)
    monkeypatch.setattr("requests.post", post)
    errors: list[str] = []
    out = route_surface_classes(
        [27.2, 27.2],
        [-80.4, -80.39],
        [0.0, 100.0],
        timeout_s=12.0,
        errors=errors,
    )
    assert out is None
    assert errors
    assert captured["timeout"] <= 12.0
    assert "[timeout:60]" not in captured["query"]
    assert "[timeout:" in captured["query"]


def test_bundled_tile_skips_network(monkeypatch):
    from bike_wind.surface import osm_surface

    bundled = Path(osm_surface._bundled_dir())
    bundled.mkdir(parents=True, exist_ok=True)
    tile = osm_surface._tile_of(27.2, -80.4)
    (bundled / f"tile_{tile[0]}_{tile[1]}.json").write_text(
        json.dumps(
            [
                {
                    "id": 7,
                    "c": PAVED,
                    "hw": "secondary",
                    "g": [[27.2, -80.41], [27.2, -80.39]],
                }
            ]
        )
    )

    def boom(_tiles):
        raise AssertionError("bundled tile should not hit the network")

    osm_surface._MEMO.clear()
    out = route_surface_classes(
        [27.2, 27.2],
        [-80.40, -80.399],
        [0.0, 80.0],
        fetch=boom,
    )
    assert out == [PAVED, PAVED]
    mph = average_speed_mph([0.0, 1609.344], 3600.0)
    assert mph == pytest.approx(1.0, abs=0.01)


def test_shipped_tiles_classify_without_network(monkeypatch):
    """Bundled St. Lucie tiles answer offline, including the pumphouse demo."""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("GUSTYPOWER_OSM_BUNDLED", str(root / "bike_wind" / "surface" / "tiles"))
    tile = json.loads((root / "bike_wind/surface/tiles/tile_544_-1609.json").read_text())
    picked = None
    for way in tile:
        if way["c"] != PAVED:
            continue
        inside = [
            (la, lo)
            for la, lo in way["g"]
            if 27.20 <= la < 27.25 and -80.45 <= lo < -80.40
        ]
        if len(inside) >= 2:
            picked = inside[:2]
            break
    assert picked is not None
    (la0, lo0), (la1, lo1) = picked
    out = route_surface_classes([la0, la1], [lo0, lo1], [0.0, 40.0])
    assert out == [PAVED, PAVED]

    seg = json.loads((root / "fixtures" / "segment_41392335.json").read_text())
    # Downsample the same way the page does, by stride, so the test stays small.
    step = max(1, len(seg["latlng"]) // 80)
    pts = seg["latlng"][::step]
    dist = seg["distance"][::step]
    lats = [float(p[0]) for p in pts]
    lons = [float(p[1]) for p in pts]
    classes = route_surface_classes(lats, lons, [float(d) for d in dist])
    assert classes is not None and len(classes) == len(lats)
    # Canal-bank segment: mapped roads are outside the match radius.
    assert classes.count(OFF_MAP) > len(classes) * 0.9


def _east_route(n: int = 40, step_m: float = 100.0):
    lats = [27.2] * n
    lons = [-80.5 + i * 0.001 for i in range(n)]
    dist = [step_m * i for i in range(n)]
    return lats, lons, dist


def test_unmapped_chunks_default_to_road_and_named_unpaved_defaults_to_gravel():
    lats, lons, dist = _east_route()
    plain = build_sections(lats, lons, dist, None)
    assert len(plain) >= 2  # 3.9 km split about every 2 km
    assert all(sec.osm_kind == "unmapped" and not sec.default_gravel for sec in plain)
    covered = [i for sec in plain for i in range(sec.start_idx, sec.end_idx)]
    assert covered == list(range(len(dist)))

    n = 12
    lats, lons, dist = _east_route(n, 50.0)
    paved = {"id": 1, "hw": "secondary", "name": "Okeechobee", "c": PAVED}
    gravel = {"id": 2, "hw": "track", "name": "Canal bank", "c": UNPAVED}
    matches = [paved] * 6 + [gravel] * 6
    parts = build_sections(lats, lons, dist, matches)
    assert len(parts) == 2
    assert parts[0].default_gravel is False
    assert "Okeechobee" in parts[0].title
    assert parts[0].osm_kind == "paved"
    assert parts[1].default_gravel is True
    assert "Canal bank" in parts[1].title
    assert parts[1].osm_kind == "unpaved"


def test_section_crr_road_matches_production_and_gravel_is_slower():
    params = build_production_params(rider_kg=69.0, power_w=220.0)
    result = _flat_result(params, n=60)
    n = len(result.distances_m)
    lats = [27.2] * n
    lons = [-80.5 + i * 0.0004 for i in range(n)]
    sections = build_sections(lats, lons, result.distances_m, None)
    road = crr_for_sections(n, sections, [False] * len(sections))
    assert road == [ROAD_SECTION_CRR] * n
    assert ROAD_SECTION_CRR == pytest.approx(0.0052)
    assert GRAVEL_SECTION_CRR == pytest.approx(0.010)
    road_cols = compare_with_production(result, section_crr=road)
    assert road_cols[-1].label == "By section"
    assert road_cols[-1].eta_s == pytest.approx(road_cols[0].eta_s)
    gravel = crr_for_sections(n, sections, [True] * len(sections))
    gravel_cols = compare_with_production(result, section_crr=gravel)
    assert gravel_cols[-1].eta_s > road_cols[0].eta_s
    assert gravel_cols[-1].avg_speed_mph < road_cols[0].avg_speed_mph
    # First half of the route gravel, rest road: between all-road and all-gravel.
    mixed_crr = [GRAVEL_SECTION_CRR] * (n // 2) + [ROAD_SECTION_CRR] * (n - n // 2)
    mixed = compare_with_production(result, section_crr=mixed_crr)
    assert road_cols[0].eta_s < mixed[-1].eta_s < gravel_cols[-1].eta_s


def test_pumphouse_sections_default_road_and_gravel_is_slower(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("GUSTYPOWER_OSM_BUNDLED", str(root / "bike_wind" / "surface" / "tiles"))
    seg = json.loads((root / "fixtures" / "segment_41392335.json").read_text())
    route = downsample_route(
        RouteStreams(
            lats=[float(p[0]) for p in seg["latlng"]],
            lons=[float(p[1]) for p in seg["latlng"]],
            elevs_m=[float(x) for x in seg["altitude"]],
            distances_m=[float(x) for x in seg["distance"]],
        ),
        max_points=300,
    )
    sections, report = sections_for_route(
        list(route.lats), list(route.lons), list(route.distances_m)
    )
    assert report.ok
    assert report.unmapped_share > 0.9
    assert len(sections) >= 2
    assert all(sec.osm_kind == "unmapped" and not sec.default_gravel for sec in sections)
    page = (root / "pages" / "2_Experimental.py").read_text()
    assert '["Road", "Gravel"]' in page
    assert "By section column updates" in page
    params = build_production_params(rider_kg=69.0, power_w=220.0)
    grades = route.grades()
    n = len(route)
    winds = [WindComponents()] * n
    gap: list[float] = []
    speeds = solve_speeds(
        params, grades, winds, distances_m=list(route.distances_m), gap_dt=gap
    )
    result = PredictionResult(
        distances_m=list(route.distances_m),
        speeds_with_wind_mps=speeds,
        speeds_no_wind_mps=list(speeds),
        grades=grades,
        wind_along_mps=[0.0] * n,
        wind_cross_mps=[0.0] * n,
        winds=winds,
        params=params,
        rho_mean=params.rho,
        weather_ok=True,
        gap_dt_with_wind=list(gap),
        gap_dt_no_wind=list(gap),
    )
    road = compare_with_production(
        result,
        section_crr=crr_for_sections(n, sections, [False] * len(sections)),
    )
    gravel = compare_with_production(
        result,
        section_crr=crr_for_sections(n, sections, [True] * len(sections)),
    )
    assert road[-1].eta_s == pytest.approx(road[0].eta_s, abs=0.05)
    assert gravel[-1].eta_s > road[-1].eta_s + 10.0
