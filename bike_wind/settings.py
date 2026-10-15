"""Resolve config from environment first, then Streamlit secrets.

Streamlit Cloud injects secrets via ``st.secrets``, which are often *not*
mirrored into ``os.environ``. Never log resolved secret values.
"""

from __future__ import annotations

import os
from typing import Any


def get_setting(name: str, default: str | None = None) -> str | None:
    """Return stripped setting: env var wins, then ``st.secrets[name]``.

    Graceful when Streamlit is not installed or secrets are unavailable.
    """
    env_val = os.environ.get(name)
    if env_val is not None and str(env_val).strip():
        return str(env_val).strip()

    secret_val = _secret_get(name)
    if secret_val is not None and str(secret_val).strip():
        return str(secret_val).strip()

    if default is not None and str(default).strip():
        return str(default).strip()
    return None


def get_bool_setting(name: str, default: bool = False) -> bool:
    raw = get_setting(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _secret_get(name: str) -> Any | None:
    try:
        import streamlit as st
    except Exception:  # noqa: BLE001 — streamlit optional at import time
        return None
    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001 — no secrets.toml / not in Streamlit runtime
        return None
    try:
        if name in secrets:
            return secrets[name]
    except Exception:  # noqa: BLE001
        return None
    # Also support nested [strava] tables if present
    try:
        if "strava" in secrets and name in secrets["strava"]:
            return secrets["strava"][name]
    except Exception:  # noqa: BLE001
        return None
    return None
