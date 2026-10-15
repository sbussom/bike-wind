"""Strava OAuth helpers (read scope). Tokens stay in session — never logged."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import requests

AUTHORIZE_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL = "https://www.strava.com/oauth/token"
READ_SCOPE = "read"


@dataclass(frozen=True)
class OAuthTokens:
    access_token: str
    refresh_token: str | None
    expires_at: int | None
    athlete_id: int | None


def oauth_configured(client_id: str | None, client_secret: str | None, redirect_uri: str | None) -> bool:
    return bool(client_id and client_secret and redirect_uri)


def build_authorize_url(client_id: str, redirect_uri: str, *, state: str = "bike_wind") -> str:
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "approval_prompt": "auto",
        "scope": READ_SCOPE,
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def exchange_code(
    *,
    client_id: str,
    client_secret: str,
    code: str,
    session: requests.Session | None = None,
    timeout_s: float = 20,
) -> OAuthTokens:
    """Exchange authorization code for tokens. Does not log response bodies."""
    http = session or requests.Session()
    resp = http.post(
        TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
        },
        timeout=timeout_s,
    )
    if resp.status_code >= 400:
        # Avoid dumping body (may include tokens)
        raise RuntimeError(f"Strava token exchange failed (HTTP {resp.status_code})")
    data: dict[str, Any] = resp.json()
    athlete = data.get("athlete") or {}
    return OAuthTokens(
        access_token=str(data["access_token"]),
        refresh_token=data.get("refresh_token"),
        expires_at=data.get("expires_at"),
        athlete_id=athlete.get("id"),
    )


def refresh_access_token(
    *,
    client_id: str,
    client_secret: str,
    refresh_token: str,
    session: requests.Session | None = None,
    timeout_s: float = 20,
) -> OAuthTokens:
    http = session or requests.Session()
    resp = http.post(
        TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=timeout_s,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Strava token refresh failed (HTTP {resp.status_code})")
    data: dict[str, Any] = resp.json()
    return OAuthTokens(
        access_token=str(data["access_token"]),
        refresh_token=data.get("refresh_token", refresh_token),
        expires_at=data.get("expires_at"),
        athlete_id=None,
    )
