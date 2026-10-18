"""Tests for env-first / st.secrets-fallback settings resolution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bike_wind.settings import (
    _lookup_mapping,
    _secret_get,
    get_bool_setting,
    get_setting,
    strava_secrets_presence,
)
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


def test_lookup_mapping_uses_get_and_subscript() -> None:
    class BrokenContains(dict):
        def __contains__(self, key):  # type: ignore[override]
            raise RuntimeError("in broken")

    d = BrokenContains(STRAVA_CLIENT_ID="cid-via-get")
    assert _lookup_mapping(d, "STRAVA_CLIENT_ID") == "cid-via-get"

    class SubscriptOnly:
        def __getitem__(self, key):
            if key == "STRAVA_ACCESS_TOKEN":
                return "tok"
            raise KeyError(key)

    assert _lookup_mapping(SubscriptOnly(), "STRAVA_ACCESS_TOKEN") == "tok"
    assert _lookup_mapping(SubscriptOnly(), "MISSING") is None


def test_secret_get_flat_via_mocked_st_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRAVA_CLIENT_ID", raising=False)
    fake_st = SimpleNamespace(secrets={"STRAVA_CLIENT_ID": "flat-cid"})
    with patch.dict("sys.modules", {"streamlit": fake_st}):
        # Force re-import path inside _secret_get
        assert _secret_get("STRAVA_CLIENT_ID") == "flat-cid"


def test_secret_get_nested_full_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRAVA_CLIENT_ID", raising=False)
    monkeypatch.delenv("STRAVA_CLIENT_SECRET", raising=False)
    fake_st = SimpleNamespace(
        secrets={
            "strava": {
                "STRAVA_CLIENT_ID": "nested-cid",
                "STRAVA_CLIENT_SECRET": "nested-sec",
            }
        }
    )
    with patch.dict("sys.modules", {"streamlit": fake_st}):
        assert _secret_get("STRAVA_CLIENT_ID") == "nested-cid"
        assert _secret_get("STRAVA_CLIENT_SECRET") == "nested-sec"


def test_secret_get_nested_short_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "STRAVA_CLIENT_ID",
        "STRAVA_CLIENT_SECRET",
        "STRAVA_REDIRECT_URI",
        "STRAVA_ACCESS_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)

    fake_st = SimpleNamespace(
        secrets={
            "strava": {
                "CLIENT_ID": "short-cid",
                "CLIENT_SECRET": "short-sec",
                "REDIRECT_URI": "https://example.streamlit.app",
                "ACCESS_TOKEN": "short-tok",
            }
        }
    )
    with patch.dict("sys.modules", {"streamlit": fake_st}):
        assert _secret_get("STRAVA_CLIENT_ID") == "short-cid"
        assert _secret_get("STRAVA_CLIENT_SECRET") == "short-sec"
        assert _secret_get("STRAVA_REDIRECT_URI") == "https://example.streamlit.app"
        assert _secret_get("STRAVA_ACCESS_TOKEN") == "short-tok"


def test_secret_get_attrdict_like_without_reliable_contains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate Streamlit AttrDict where ``in`` misbehaves but subscript works."""
    monkeypatch.delenv("STRAVA_REDIRECT_URI", raising=False)

    class AttrDictLike:
        def __init__(self, data):
            self._data = data

        def __contains__(self, key):
            return False  # broken / always false

        def __getitem__(self, key):
            return self._data[key]

        def get(self, key, default=None):
            return self._data.get(key, default)

    fake_st = SimpleNamespace(
        secrets=AttrDictLike({"STRAVA_REDIRECT_URI": "https://app.example"})
    )
    with patch.dict("sys.modules", {"streamlit": fake_st}):
        assert _secret_get("STRAVA_REDIRECT_URI") == "https://app.example"


def test_strava_secrets_presence_booleans_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRAVA_CLIENT_ID", "cid")
    monkeypatch.delenv("STRAVA_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("STRAVA_REDIRECT_URI", raising=False)
    monkeypatch.delenv("STRAVA_ACCESS_TOKEN", raising=False)
    with patch("bike_wind.settings._secret_get", return_value=None):
        presence = strava_secrets_presence()
    assert presence == {
        "CLIENT_ID": True,
        "CLIENT_SECRET": False,
        "REDIRECT_URI": False,
        "ACCESS_TOKEN": False,
    }
    # Ensure no secret values leaked into the dict
    assert all(isinstance(v, bool) for v in presence.values())


def test_load_strava_config_nested_short_aliases(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    for key in (
        "STRAVA_CLIENT_ID",
        "STRAVA_CLIENT_SECRET",
        "STRAVA_REDIRECT_URI",
        "STRAVA_ACCESS_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)

    fake_st = SimpleNamespace(
        secrets={
            "strava": {
                "CLIENT_ID": "n-cid",
                "CLIENT_SECRET": "n-sec",
                "REDIRECT_URI": "https://nested.example",
                "ACCESS_TOKEN": "n-tok",
            }
        }
    )
    with patch.dict("sys.modules", {"streamlit": fake_st}):
        cfg = load_strava_config(cache_dir=tmp_path)
    assert cfg.oauth_ready is True
    assert cfg.token_ready is True
    assert cfg.client_id == "n-cid"
    assert cfg.access_token == "n-tok"
