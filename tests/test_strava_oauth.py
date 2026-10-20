"""Tests for Strava OAuth exchange / refresh helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bike_wind.strava.oauth import (
    OAuthTokenError,
    exchange_code,
    refresh_access_token,
    safe_oauth_error_detail,
)


def test_refresh_access_token_success() -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "access_token": "a2",
        "refresh_token": "r2",
        "expires_at": 123,
        "scope": "read",
    }
    session.post.return_value = resp
    tokens = refresh_access_token(
        client_id="12345",
        client_secret="sec",
        refresh_token="r1",
        session=session,
    )
    assert tokens.access_token == "a2"
    assert tokens.refresh_token == "r2"
    assert tokens.expires_at == 123
    assert tokens.scope == "read"
    posted = session.post.call_args.kwargs["data"]
    assert posted["grant_type"] == "refresh_token"
    assert posted["refresh_token"] == "r1"
    assert posted["client_id"] == 12345  # coerced to int
    assert session.post.call_args.args[0] == "https://www.strava.com/oauth/token"


def test_refresh_rejects_missing_access_token() -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"refresh_token": "r2"}
    session.post.return_value = resp
    with pytest.raises(OAuthTokenError, match="missing access_token"):
        refresh_access_token(
            client_id="cid",
            client_secret="sec",
            refresh_token="r1",
            session=session,
        )


def test_refresh_access_token_surfaces_strava_error() -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 400
    resp.json.return_value = {
        "message": "Bad Request",
        "errors": [{"resource": "RefreshToken", "field": "refresh_token", "code": "invalid"}],
    }
    session.post.return_value = resp
    with pytest.raises(OAuthTokenError) as ei:
        refresh_access_token(
            client_id="cid",
            client_secret="sec",
            refresh_token="dead",
            session=session,
        )
    assert ei.value.status == 400
    assert "invalid" in ei.value.detail
    assert "dead" not in ei.value.detail


def test_safe_oauth_error_redacts_tokenish() -> None:
    resp = MagicMock()
    resp.status_code = 401
    resp.json.return_value = {
        "error": "invalid_client",
        "error_description": "client auth failed token=abcdefghijklmnopqrstuvwxyz012345",
    }
    detail = safe_oauth_error_detail(resp)
    assert "invalid_client" in detail
    assert "abcdefghijklmnopqrstuvwxyz012345" not in detail
    assert "[redacted]" in detail


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
