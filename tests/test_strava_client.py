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
        refresh_token="refresh-tok",
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
    cfg = _config(tmp_path, access_token=None, refresh_token=None)
    client = StravaClient(cfg)
    with pytest.raises(StravaAuthError, match="settings/api"):
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


def test_list_recent_activities(tmp_path: Path) -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = [
        {
            "id": 55,
            "name": "Sunset",
            "sport_type": "Ride",
            "start_date": "2024-01-02T18:00:00Z",
            "distance": 40000,
            "moving_time": 5400,
            "elapsed_time": 5600,
            "average_watts": 195,
            "start_latlng": [26.1, -80.1],
        }
    ]
    session.get.return_value = resp
    client = StravaClient(_config(tmp_path), session=session)
    acts = client.list_recent_activities()
    assert len(acts) == 1
    assert acts[0].activity_id == 55
    assert acts[0].has_power is True
    assert session.get.call_args[0][0].endswith("/athlete/activities")


def test_list_activities_on_date(tmp_path: Path) -> None:
    from datetime import date

    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = [
        {
            "id": 77,
            "name": "Morning",
            "sport_type": "Ride",
            "start_date": "2024-06-01T12:00:00Z",
            "distance": 20000,
            "moving_time": 3000,
            "elapsed_time": 3100,
            "average_watts": None,
            "start_latlng": [26.1, -80.1],
        }
    ]
    session.get.return_value = resp
    client = StravaClient(_config(tmp_path), session=session)
    acts = client.list_activities_on_date(date(2024, 6, 1))
    assert len(acts) == 1
    assert acts[0].activity_id == 77
    params = session.get.call_args.kwargs["params"]
    assert "after" in params and "before" in params
    assert params["before"] > params["after"]


def test_explore_top_10(tmp_path: Path) -> None:
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
    client = StravaClient(_config(tmp_path), session=session)
    hits = client.explore([26.0, -80.2, 26.2, -80.0])
    assert len(hits) == 10
    assert hits[0].start_lat == pytest.approx(26.1)
    assert hits[0].end_lon == pytest.approx(-80.09)


def test_explore_403_mentions_extended_access_tier(tmp_path: Path) -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 403
    session.get.return_value = resp
    client = StravaClient(_config(tmp_path), session=session)
    with pytest.raises(StravaAuthError, match="Extended Access"):
        client.explore([26.0, -80.2, 26.2, -80.0])


def test_segment_403_mentions_read_scope(tmp_path: Path) -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 403
    session.get.return_value = resp
    client = StravaClient(_config(tmp_path), session=session)
    with pytest.raises(StravaAuthError, match="read") as ei:
        client.get_segment_meta(1, use_cache=False)
    assert ei.value.kind == "forbidden_scope"


def test_401_refreshes_and_retries(tmp_path: Path) -> None:
    session = MagicMock()
    unauthorized = MagicMock()
    unauthorized.status_code = 401
    ok = MagicMock()
    ok.status_code = 200
    ok.json.return_value = {
        "id": 42,
        "name": "Fresh",
        "distance": 1000.0,
        "average_grade": 2.0,
    }
    # refresh uses a fresh requests.Session — patch at oauth layer via client path
    refresh_resp = MagicMock()
    refresh_resp.status_code = 200
    refresh_resp.json.return_value = {
        "access_token": "new-access-token-value",
        "refresh_token": "new-refresh",
        "expires_at": 9999999999,
        "scope": "read",
    }
    session.get.side_effect = [unauthorized, ok]

    seen: list[str] = []

    def on_refreshed(tokens):
        seen.append(tokens.access_token)

    with pytest.MonkeyPatch.context() as mp:
        from bike_wind.strava import client as client_mod

        def fake_refresh(**kwargs):
            from bike_wind.strava.oauth import OAuthTokens

            return OAuthTokens(
                access_token="new-access-token-value",
                refresh_token="new-refresh",
                expires_at=9999999999,
                athlete_id=None,
                scope="read",
            )

        mp.setattr(client_mod, "refresh_access_token", fake_refresh)
        client = StravaClient(
            _config(tmp_path, access_token="expired"),
            session=session,
            on_tokens_refreshed=on_refreshed,
        )
        meta = client.get_segment_meta(42, use_cache=False)

    assert meta.name == "Fresh"
    assert client.access_token == "new-access-token-value"
    assert client.refresh_token == "new-refresh"
    assert seen == ["new-access-token-value"]
    assert session.get.call_count == 2
    # Second GET must use the refreshed bearer, not the expired secrets token.
    second_headers = session.get.call_args_list[1].kwargs["headers"]
    assert second_headers["Authorization"] == "Bearer new-access-token-value"


def test_explore_401_after_refresh_with_valid_athlete_is_tier(tmp_path: Path) -> None:
    """Explore can 401 even when the refreshed token is otherwise valid."""
    session = MagicMock()
    unauthorized = MagicMock()
    unauthorized.status_code = 401
    unauthorized.json.return_value = {"message": "Authorization Error"}
    explore_still_401 = MagicMock()
    explore_still_401.status_code = 401
    explore_still_401.json.return_value = {
        "message": "Authorization Error",
        "errors": [{"resource": "Application", "code": "unauthorized"}],
    }
    athlete_ok = MagicMock()
    athlete_ok.status_code = 200
    athlete_ok.json.return_value = {"id": 1}
    session.get.side_effect = [unauthorized, explore_still_401, athlete_ok]

    from bike_wind.strava import client as client_mod
    from bike_wind.strava.oauth import OAuthTokens

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            client_mod,
            "refresh_access_token",
            lambda **kwargs: OAuthTokens(
                access_token="fresh-access-token-xxxxx",
                refresh_token="r2",
                expires_at=9999999999,
                athlete_id=None,
                scope="read",
            ),
        )
        client = StravaClient(_config(tmp_path, access_token="expired"), session=session)
        with pytest.raises(StravaAuthError, match="explore blocked") as ei:
            client.explore([26.0, -80.2, 26.2, -80.0])
    assert ei.value.kind == "forbidden_explore_tier"
    assert session.get.call_args_list[1].kwargs["headers"]["Authorization"].startswith(
        "Bearer fresh-access"
    )


def test_401_refresh_failure_surfaces_strava_detail(tmp_path: Path) -> None:
    session = MagicMock()
    unauthorized = MagicMock()
    unauthorized.status_code = 401
    session.get.return_value = unauthorized
    bad_refresh = MagicMock()
    bad_refresh.status_code = 400
    bad_refresh.json.return_value = {
        "message": "Bad Request",
        "errors": [{"resource": "RefreshToken", "code": "invalid"}],
    }
    session.post.return_value = bad_refresh

    client = StravaClient(_config(tmp_path, access_token="expired"), session=session)
    with pytest.raises(StravaAuthError, match=r"\[refresh failed\].*invalid") as ei:
        client.get_segment_meta(1, use_cache=False)
    assert ei.value.kind == "refresh_failed"


def test_401_without_refresh_token_is_actionable(tmp_path: Path) -> None:
    session = MagicMock()
    unauthorized = MagicMock()
    unauthorized.status_code = 401
    session.get.return_value = unauthorized
    client = StravaClient(
        _config(tmp_path, access_token="expired", refresh_token=None),
        session=session,
    )
    with pytest.raises(StravaAuthError, match=r"\[refresh failed\]"):
        client.get_segment_meta(1, use_cache=False)


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


def test_load_activity_route_absolute_times(tmp_path: Path) -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "latlng": {"data": [[26.1, -80.1], [26.11, -80.1], [26.12, -80.1]]},
        "distance": {"data": [0.0, 100.0, 200.0]},
        "altitude": {"data": [2.0, 4.0, 6.0]},
        "time": {"data": [0, 30, 60]},
        "velocity_smooth": {"data": [3.2, 3.4, 3.3]},
        "watts": {"data": [180, 190, 200]},
    }
    session.get.return_value = resp
    client = StravaClient(_config(tmp_path), session=session)
    route = client.load_activity_route(
        99, start_date="2024-06-01T14:00:00Z"
    )
    assert len(route) == 3
    assert route.has_actual_speeds
    assert route.start_time is not None
    assert route.start_time.year == 2024
    assert route.start_time.hour == 14
    assert session.get.call_args[0][0].endswith("/activities/99/streams")
    keys = session.get.call_args.kwargs["params"]["keys"]
    assert "velocity_smooth" in keys
    assert "watts" in keys
    assert "time" in keys
