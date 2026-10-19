"""Tests for Strava OAuth exchange / refresh helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bike_wind.strava.oauth import exchange_code, refresh_access_token


def test_refresh_access_token_success() -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "access_token": "a2",
        "refresh_token": "r2",
        "expires_at": 123,
    }
    session.post.return_value = resp
    tokens = refresh_access_token(
        client_id="cid",
        client_secret="sec",
        refresh_token="r1",
        session=session,
    )
    assert tokens.access_token == "a2"
    assert tokens.refresh_token == "r2"
    assert tokens.expires_at == 123
    posted = session.post.call_args.kwargs["data"]
    assert posted["grant_type"] == "refresh_token"
    assert posted["refresh_token"] == "r1"


def test_refresh_access_token_http_error() -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 401
    session.post.return_value = resp
    with pytest.raises(RuntimeError, match="refresh failed"):
        refresh_access_token(
            client_id="cid",
            client_secret="sec",
            refresh_token="dead",
            session=session,
        )


def test_exchange_code_success() -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "access_token": "a1",
        "refresh_token": "r1",
        "expires_at": 99,
        "athlete": {"id": 7},
    }
    session.post.return_value = resp
    tokens = exchange_code(
        client_id="cid",
        client_secret="sec",
        code="auth-code",
        session=session,
    )
    assert tokens.access_token == "a1"
    assert tokens.athlete_id == 7
