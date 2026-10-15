"""Tests for env-first / st.secrets-fallback settings resolution."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from bike_wind.settings import get_bool_setting, get_setting
from bike_wind.strava.client import load_strava_config


def test_get_setting_prefers_env_over_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRAVA_CLIENT_ID", "from-env")
    with patch("bike_wind.settings._secret_get", return_value="from-secrets"):
        assert get_setting("STRAVA_CLIENT_ID") == "from-env"


def test_get_setting_falls_back_to_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRAVA_CLIENT_ID", raising=False)
    with patch("bike_wind.settings._secret_get", return_value="from-secrets"):
        assert get_setting("STRAVA_CLIENT_ID") == "from-secrets"


def test_get_setting_missing_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRAVA_CLIENT_SECRET", raising=False)
    with patch("bike_wind.settings._secret_get", return_value=None):
        assert get_setting("STRAVA_CLIENT_SECRET") is None


def test_get_bool_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRAVA_EXTENDED_ACCESS", "true")
    assert get_bool_setting("STRAVA_EXTENDED_ACCESS") is True
    monkeypatch.setenv("STRAVA_EXTENDED_ACCESS", "false")
    assert get_bool_setting("STRAVA_EXTENDED_ACCESS") is False


def test_load_strava_config_uses_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("STRAVA_CLIENT_ID", "cid")
    monkeypatch.setenv("STRAVA_CLIENT_SECRET", "sec")
    monkeypatch.setenv("STRAVA_REDIRECT_URI", "https://example.com")
    monkeypatch.setenv("STRAVA_EXTENDED_ACCESS", "1")
    with patch("bike_wind.settings._secret_get", return_value=None):
        cfg = load_strava_config(cache_dir=tmp_path)
    assert cfg.oauth_ready is True
    assert cfg.client_id == "cid"
    assert cfg.extended_access is True


def test_load_strava_config_falls_back_to_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    for key in (
        "STRAVA_CLIENT_ID",
        "STRAVA_CLIENT_SECRET",
        "STRAVA_REDIRECT_URI",
        "STRAVA_ACCESS_TOKEN",
        "STRAVA_EXTENDED_ACCESS",
    ):
        monkeypatch.delenv(key, raising=False)

    secrets = {
        "STRAVA_CLIENT_ID": "secret-cid",
        "STRAVA_CLIENT_SECRET": "secret-sec",
        "STRAVA_REDIRECT_URI": "https://bike-wind.streamlit.app",
    }

    def fake_secret(name: str):
        return secrets.get(name)

    with patch("bike_wind.settings._secret_get", side_effect=fake_secret):
        cfg = load_strava_config(cache_dir=tmp_path)
    assert cfg.oauth_ready is True
    assert cfg.client_id == "secret-cid"
    assert cfg.redirect_uri == "https://bike-wind.streamlit.app"


def test_secret_get_graceful_when_streamlit_missing() -> None:
    import bike_wind.settings as settings

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "streamlit":
            raise ImportError("no streamlit")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        assert settings._secret_get("STRAVA_CLIENT_ID") is None
