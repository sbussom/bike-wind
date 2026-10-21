"""Strava OAuth helpers. Tokens stay in session — never logged."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import requests

AUTHORIZE_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL = "https://www.strava.com/oauth/token"
# read = public segment GPS; activity:read = athlete recent rides / streams
READ_SCOPE = "read,activity:read"
ACTIVITY_READ_SCOPE = "activity:read"
DEFAULT_REDIRECT_URI = "https://gustypower.streamlit.app"
# Shown when the browser session token lacks activity:read (secrets stay read-only).
ACTIVITY_SCOPE_HINT = "Connect again and allow access to activities."

# Long hex/base64-ish blobs that look like tokens — never show in UI.
_TOKENISH = re.compile(r"[A-Za-z0-9_\-]{20,}")


@dataclass(frozen=True)
class OAuthTokens:
    access_token: str
    refresh_token: str | None
    expires_at: int | None
    athlete_id: int | None
    scope: str | None = None


class OAuthTokenError(RuntimeError):
    """Refresh/exchange failure with a safe, user-visible detail string."""

    def __init__(self, message: str, *, status: int | None = None, detail: str = "") -> None:
        self.status = status
        self.detail = detail
        super().__init__(message)


def oauth_configured(client_id: str | None, client_secret: str | None, redirect_uri: str | None) -> bool:
    return bool(client_id and client_secret and redirect_uri)


def build_authorize_url(client_id: str, redirect_uri: str, *, state: str = "bike_wind") -> str:
    """Authorize URL that always re-prompts so activity:read can be granted."""
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        # force: prior "read"-only approvals must show the activity permission again
        "approval_prompt": "force",
        "scope": READ_SCOPE,
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def scope_has_activity_read(scope: str | None) -> bool:
    """True when OAuth scope includes activity:read (or activity:read_all)."""
    if not scope:
        return False
    parts = {p.strip().lower() for p in str(scope).replace(",", " ").split() if p.strip()}
    return ACTIVITY_READ_SCOPE in parts or "activity:read_all" in parts


def _coerce_client_id(client_id: str) -> str | int:
    """Strava accepts numeric client_id as int; keep string if non-numeric."""
    text = str(client_id).strip()
    if text.isdigit():
        return int(text)
    return text


def _redact_tokenish(text: str) -> str:
    return _TOKENISH.sub("[redacted]", text)


def safe_oauth_error_detail(resp: requests.Response) -> str:
    """Extract Strava error/message fields; never return raw tokens."""
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return f"HTTP {resp.status_code}"
    if not isinstance(data, dict):
        return f"HTTP {resp.status_code}"

    parts: list[str] = []
    for key in ("message", "error", "error_description"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(f"{key}={_redact_tokenish(val.strip())}")
    errors = data.get("errors")
    if isinstance(errors, list):
        for item in errors[:5]:
            if isinstance(item, dict):
                bits = [
                    item.get("resource"),
                    item.get("field"),
                    item.get("code"),
                    item.get("message"),
                ]
                bit = " ".join(str(x) for x in bits if x).strip()
                if bit:
                    parts.append(_redact_tokenish(bit))
            elif isinstance(item, str) and item.strip():
                parts.append(_redact_tokenish(item.strip()))
    if not parts:
        return f"HTTP {resp.status_code}"
    return f"HTTP {resp.status_code}: " + "; ".join(parts)


def _parse_token_payload(data: dict[str, Any], *, fallback_refresh: str | None = None) -> OAuthTokens:
    raw = data.get("access_token")
    if raw is None or not str(raw).strip():
        raise OAuthTokenError(
            "Strava token response missing access_token",
            detail="missing access_token in JSON",
        )
    refresh = data.get("refresh_token", fallback_refresh)
    scope = data.get("scope")
    if isinstance(scope, list):
        scope = " ".join(str(s) for s in scope)
    elif scope is not None:
        scope = str(scope)
    athlete = data.get("athlete") or {}
    expires_at = data.get("expires_at")
    try:
        expires_at_i = int(expires_at) if expires_at is not None else None
    except (TypeError, ValueError):
        expires_at_i = None
    return OAuthTokens(
        access_token=str(raw).strip(),
        refresh_token=str(refresh).strip() if refresh else None,
        expires_at=expires_at_i,
        athlete_id=athlete.get("id") if isinstance(athlete, dict) else None,
        scope=scope,
    )


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
            "client_id": _coerce_client_id(client_id),
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
        },
        timeout=timeout_s,
    )
    if resp.status_code >= 400:
        detail = safe_oauth_error_detail(resp)
        raise OAuthTokenError(
            f"Strava token exchange failed ({detail})",
            status=resp.status_code,
            detail=detail,
        )
    return _parse_token_payload(resp.json())


def refresh_access_token(
    *,
    client_id: str,
    client_secret: str,
    refresh_token: str,
    session: requests.Session | None = None,
    timeout_s: float = 20,
) -> OAuthTokens:
    """POST refresh_token grant. Surfaces safe Strava error detail (no tokens)."""
    http = session or requests.Session()
    resp = http.post(
        TOKEN_URL,
        data={
            "client_id": _coerce_client_id(client_id),
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=timeout_s,
    )
    if resp.status_code >= 400:
        detail = safe_oauth_error_detail(resp)
        raise OAuthTokenError(
            f"Strava token refresh failed ({detail})",
            status=resp.status_code,
            detail=detail,
        )
    return _parse_token_payload(resp.json(), fallback_refresh=refresh_token)
