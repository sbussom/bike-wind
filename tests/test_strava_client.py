"""Mocked HTTP tests for Strava client + cache + rate limit."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bike_wind.strava.cache import SegmentCache
from bike_wind.strava.client import (
    StravaAuthError,
    StravaClient,
    StravaConfig,
    StravaError,
    StravaRateLimitError,
)
from bike_wind.strava.rate_limit import LIMIT_15M, RateLimiter
from bike_wind.gpx.parse import RouteStreams


def _config(tmp_path: Path, **kwargs) -> StravaConfig:
    base = dict(
        access_token="test-token",
        client_id="cid",
        client_secret="sec",
        redirect_uri="http://localhost:8501",
        extended_access=False,
        cache_dir=tmp_path,
    )
    base.update(kwargs)
    return StravaConfig(**base)  # type: ignore[arg-type]


def test_resolve_segment_id_no_http(tmp_path: Path) -> None:
    client = StravaClient(_config(tmp_path))
    sid, ref = client.resolve_ref("https://www.strava.com/segments/99")
    assert sid == 99
    assert ref.kind == "segment_id"


def test_resolve_effort_id_fetches_segment(tmp_path: Path) -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"segment": {"id": 555, "name": "Climb"}}
    session.get.return_value = resp

    client = StravaClient(_config(tmp_path), session=session)
    sid, ref = client.resolve_ref(
        "https://www.strava.com/activities/1/segments/777"
    )
    assert sid == 555
    assert ref.kind == "effort_id"
    assert session.get.call_args[0][0].endswith("/segment_efforts/777")


def test_segment_meta_start_latlng(tmp_path: Path) -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "id": 42,
        "name": "Coast",
        "distance": 500.0,
        "average_grade": 0.5,
        "elevation_low": 1.0,
        "elevation_high": 3.0,
        "start_latlng": [26.1, -80.2],
    }
    session.get.return_value = resp
    client = StravaClient(_config(tmp_path), session=session)
    meta = client.get_segment_meta(42, use_cache=False)
    assert meta.start_lat == pytest.approx(26.1)
    assert meta.start_lon == pytest.approx(-80.2)


def test_streams_to_route_and_cache(tmp_path: Path) -> None:
    session = MagicMock()

    def fake_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if url.endswith("/segments/10"):
            resp.json.return_value = {
                "id": 10,
                "name": "Test Seg",
                "distance": 1200.0,
                "average_grade": 2.5,
                "elevation_low": 1.0,
                "elevation_high": 40.0,
            }
        elif "/streams" in url:
            resp.json.return_value = {
                "latlng": {"data": [[26.0, -80.0], [26.01, -80.01], [26.02, -80.02]]},
                "distance": {"data": [0.0, 500.0, 1000.0]},
                "altitude": {"data": [2.0, 10.0, 20.0]},
            }
        else:
            raise AssertionError(url)
        return resp

    session.get.side_effect = fake_get
    client = StravaClient(_config(tmp_path), session=session)
    meta, route = client.load_segment_route(10, max_cache_age_s=0)
    assert meta.name == "Test Seg"
    assert len(route) == 3
    assert route.distances_m[-1] == 1000.0

    # Second load hits cache — no extra HTTP if max_age allows
    session.get.reset_mock()
    meta2, route2 = client.load_segment_route(10)
    assert meta2.from_cache is True
    assert len(route2) == 3
    session.get.assert_not_called()


def test_auth_error_without_token(tmp_path: Path) -> None:
    cfg = _config(tmp_path, access_token=None)
    client = StravaClient(cfg)
    with pytest.raises(StravaAuthError):
        client.get_segment_meta(1, use_cache=False)


def test_list_starred(tmp_path: Path) -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = [
        {"id": 101, "name": "Bridge", "distance": 2500.0, "average_grade": 1.2},
        {"id": 202, "name": "Climb", "distance": 800.0, "average_grade": 6.5},
    ]
    session.get.return_value = resp
    client = StravaClient(_config(tmp_path), session=session)
    starred = client.list_starred()
    assert len(starred) == 2
    assert starred[0].segment_id == 101
    assert starred[0].name == "Bridge"
    assert starred[1].distance_m == 800.0
    assert session.get.call_args[0][0].endswith("/segments/starred")


def test_list_starred_paginates_until_exhausted(tmp_path: Path) -> None:
    session = MagicMock()

    def fake_get(url, **kwargs):
        params = kwargs.get("params") or {}
        page = int(params.get("page") or 1)
        per_page = int(params.get("per_page") or 200)
        resp = MagicMock()
        resp.status_code = 200
        if page == 1:
            resp.json.return_value = [
                {
                    "id": i,
                    "name": f"S{i}",
                    "distance": 1000.0,
                    "average_grade": 1.0,
                }
                for i in range(per_page)
            ]
        elif page == 2:
            resp.json.return_value = [
                {
                    "id": 1000 + i,
                    "name": f"T{i}",
                    "distance": 500.0,
                    "average_grade": 2.0,
                }
                for i in range(17)
            ]
        else:
            resp.json.return_value = []
        return resp

    session.get.side_effect = fake_get
    client = StravaClient(_config(tmp_path), session=session)
    starred = client.list_starred(per_page=30)
    assert len(starred) == 47  # 30 + 17
    assert session.get.call_count == 2
    assert starred[-1].segment_id == 1016


def test_explore_gated_without_flag(tmp_path: Path) -> None:
    client = StravaClient(_config(tmp_path, extended_access=False))
    with pytest.raises(StravaError, match="Extended Access"):
        client.explore([26.0, -80.2, 26.2, -80.0])


def test_explore_top_10_with_flag(tmp_path: Path) -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "segments": [
            {
                "id": i,
                "name": f"S{i}",
                "distance": 1000,
                "avg_grade": 1.0,
                "climb_category": 0,
                "starred": False,
                "start_latlng": [26.1 + i * 0.001, -80.1],
                "end_latlng": [26.11 + i * 0.001, -80.09],
            }
            for i in range(12)
        ]
    }
    session.get.return_value = resp
    client = StravaClient(_config(tmp_path, extended_access=True), session=session)
    hits = client.explore([26.0, -80.2, 26.2, -80.0])
    assert len(hits) == 10
    assert hits[0].start_lat == pytest.approx(26.1)
    assert hits[0].end_lon == pytest.approx(-80.09)


def test_rate_limiter_blocks(tmp_path: Path) -> None:
    limiter = RateLimiter(tmp_path / "rl.sqlite")
    now = 1_000_000.0
    for _ in range(LIMIT_15M):
        limiter.record(now=now)
    status = limiter.status(now=now + 1)
    assert status.limited is True
    assert status.remaining_15m == 0


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = SegmentCache(tmp_path / "seg.sqlite")
    route = RouteStreams(
        lats=[1.0, 2.0],
        lons=[3.0, 4.0],
        elevs_m=[5.0, 6.0],
        distances_m=[0.0, 100.0],
    )
    cache.put(
        segment_id=1,
        name="N",
        distance_m=100.0,
        average_grade=1.0,
        elev_low=5.0,
        elev_high=6.0,
        route=route,
    )
    got = cache.get(1)
    assert got is not None
    assert got.name == "N"
    assert got.route.lats == [1.0, 2.0]
