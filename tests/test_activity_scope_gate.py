"""Activity APIs must not use the owner secrets (read-only) token."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bike_wind.strava.oauth import ACTIVITY_SCOPE_HINT


def test_session_has_activity_read_requires_scope(monkeypatch) -> None:
    state = {
        "strava_oauth_connected": True,
        "strava_access_token": "tok",
        "strava_token_scope": "read",
    }

    class _SS(dict):
        def get(self, key, default=None):
            return super().get(key, default)

        def __getattr__(self, name: str):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    ss = _SS(state)
    monkeypatch.setattr("app.st.session_state", ss)
    from app import _session_has_activity_read

    assert _session_has_activity_read() is False
    ss["strava_token_scope"] = "read,activity:read"
    assert _session_has_activity_read() is True


def test_run_activity_call_never_uses_secrets(monkeypatch, tmp_path: Path) -> None:
    state = {
        "strava_oauth_connected": False,
        "strava_access_token": None,
        "strava_token_scope": None,
    }

    class _SS(dict):
        def get(self, key, default=None):
            return super().get(key, default)

        def __getattr__(self, name: str):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    ss = _SS(state)
    monkeypatch.setattr("app.st.session_state", ss)

    from app import _run_activity_call
    from bike_wind.strava.client import StravaAuthError, StravaConfig

    config = StravaConfig(
        access_token="secrets-tok",
        client_id="1",
        client_secret="s",
        redirect_uri="https://gustypower.streamlit.app",
        refresh_token="r",
        cache_dir=tmp_path,
    )
    called = MagicMock()
    with pytest.raises(StravaAuthError, match="allow access to activities") as ei:
        _run_activity_call(config, called)
    assert ei.value.kind == "forbidden_scope"
    assert ei.value.args[0] == ACTIVITY_SCOPE_HINT
    called.assert_not_called()
