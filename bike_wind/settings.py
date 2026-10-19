"""Resolve config from environment first, then Streamlit secrets.

Streamlit Cloud injects secrets via ``st.secrets``, which are often *not*
mirrored into ``os.environ``. Never log resolved secret values.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

# Flat env/secret names → short keys under a nested [strava] table.
# Include lowercase (Streamlit TOML / common paste style: client_id).
_STRAVA_NESTED_ALIASES: dict[str, tuple[str, ...]] = {
    "STRAVA_CLIENT_ID": (
        "STRAVA_CLIENT_ID",
        "CLIENT_ID",
        "client_id",
    ),
    "STRAVA_CLIENT_SECRET": (
        "STRAVA_CLIENT_SECRET",
        "CLIENT_SECRET",
        "client_secret",
    ),
    "STRAVA_REDIRECT_URI": (
        "STRAVA_REDIRECT_URI",
        "REDIRECT_URI",
        "redirect_uri",
    ),
    "STRAVA_ACCESS_TOKEN": (
        "STRAVA_ACCESS_TOKEN",
        "ACCESS_TOKEN",
        "access_token",
    ),
    "STRAVA_EXTENDED_ACCESS": (
        "STRAVA_EXTENDED_ACCESS",
        "EXTENDED_ACCESS",
        "extended_access",
    ),
    # Optional future refresh path (not required for access-token usage).
    "STRAVA_REFRESH_TOKEN": (
        "STRAVA_REFRESH_TOKEN",
        "REFRESH_TOKEN",
        "refresh_token",
    ),
}


def get_setting(name: str, default: str | None = None) -> str | None:
    """Return stripped setting: env var wins, then ``st.secrets``.

    Graceful when Streamlit is not installed or secrets are unavailable.
    Never logs the resolved value.
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


def strava_secrets_presence() -> dict[str, bool]:
    """Booleans only — safe to show in UI / screenshots. Never returns values."""
    return {
        "CLIENT_ID": bool(get_setting("STRAVA_CLIENT_ID")),
        "CLIENT_SECRET": bool(get_setting("STRAVA_CLIENT_SECRET")),
        "REDIRECT_URI": bool(get_setting("STRAVA_REDIRECT_URI")),
        "ACCESS_TOKEN": bool(get_setting("STRAVA_ACCESS_TOKEN")),
    }


def strava_secrets_debug() -> dict[str, Any]:
    """Safe secrets shape for UI diagnostics — never includes values.

    Returns:
      available: whether ``st.secrets`` could be opened
      keys: top-level key names, plus ``strava.<child>`` for nested tables
      nested_strava: True when any ``strava.*`` key name was found
    """
    available, keys = _secret_key_names()
    nested = any(k.lower().startswith("strava.") for k in keys)
    return {"available": available, "keys": keys, "nested_strava": nested}


def _secret_key_names() -> tuple[bool, list[str]]:
    """List secret key names only (no values)."""
    try:
        import streamlit as st
    except Exception:  # noqa: BLE001
        return False, []
    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001
        return False, []

    names: list[str] = []
    top = _iter_keys(secrets)
    for key in top:
        names.append(str(key))
        if str(key).lower() == "strava":
            section = _lookup_mapping(secrets, str(key))
            for child in _iter_keys(section):
                names.append(f"strava.{child}")
    # Stable, unique, short
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return True, out


def _iter_keys(container: Any) -> list[str]:
    """Best-effort key listing for dict / AttrDict / Secrets."""
    if container is None:
        return []
    try:
        keys = list(container.keys())  # type: ignore[attr-defined]
        return [str(k) for k in keys]
    except Exception:  # noqa: BLE001
        pass
    try:
        return [str(k) for k in list(container)]
    except Exception:  # noqa: BLE001
        return []


def _secret_get(name: str) -> Any | None:
    """Read one key from ``st.secrets`` without relying on ``key in secrets``.

    Streamlit's AttrDict/`Secrets` sometimes mishandles ``in`` / iteration.
    Prefer try/subscript and ``.get`` when available. Also accept nested
    ``[strava]`` tables with ``STRAVA_*``, ``CLIENT_ID``, or lowercase
    ``client_id`` / ``access_token`` names (AttrDict attribute access too).
    """
    try:
        import streamlit as st
    except Exception:  # noqa: BLE001 — streamlit optional at import time
        return None
    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001 — no secrets.toml / not in Streamlit runtime
        return None

    found = _lookup_mapping(secrets, name)
    if found is not None:
        return found

    # Nested [strava] / [STRAVA] tables (case variants)
    for section_name in ("strava", "STRAVA", "Strava"):
        section = _lookup_mapping(secrets, section_name)
        if section is None:
            continue
        aliases = _STRAVA_NESTED_ALIASES.get(name, (name,))
        for alias in aliases:
            nested = _lookup_mapping(section, alias)
            if nested is not None:
                return nested
    return None


def _lookup_mapping(container: Any, key: str) -> Any | None:
    """Fetch ``key`` from a secrets-like mapping; never raise; never log values."""
    if container is None:
        return None

    # Prefer .get when present (AttrDict / dict)
    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            val = getter(key)
            if val is not None and not _is_empty(val):
                return val
        except Exception:  # noqa: BLE001
            pass

    # Direct subscript — works even when ``in`` is unreliable
    try:
        val = container[key]
        if val is not None and not _is_empty(val):
            return val
    except Exception:  # noqa: BLE001 — KeyError / TypeError / StreamlitAPIException
        pass

    # AttrDict attribute access: secrets.strava.access_token
    if key.isidentifier():
        try:
            val = getattr(container, key)
            # Avoid methods / callables on the container
            if val is not None and not callable(val) and not _is_empty(val):
                return val
        except Exception:  # noqa: BLE001
            pass

    # Mapping-style fallback
    if isinstance(container, Mapping):
        try:
            if key in container:
                val = container[key]
                if val is not None and not _is_empty(val):
                    return val
        except Exception:  # noqa: BLE001
            pass

    # Case-insensitive match (TOML / Cloud often uses client_id)
    try:
        for existing in _iter_keys(container):
            if existing.lower() == key.lower():
                if existing == key:
                    continue
                return _lookup_mapping_exact(container, existing)
    except Exception:  # noqa: BLE001
        pass

    return None


def _lookup_mapping_exact(container: Any, key: str) -> Any | None:
    """Exact-key lookup without case-insensitive recursion."""
    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            val = getter(key)
            if val is not None and not _is_empty(val):
                return val
        except Exception:  # noqa: BLE001
            pass
    try:
        val = container[key]
        if val is not None and not _is_empty(val):
            return val
    except Exception:  # noqa: BLE001
        pass
    if key.isidentifier():
        try:
            val = getattr(container, key)
            if val is not None and not callable(val) and not _is_empty(val):
                return val
        except Exception:  # noqa: BLE001
            pass
    return None


def _is_empty(val: Any) -> bool:
    if isinstance(val, str):
        return not val.strip()
    # Nested tables (dict-like) are not "empty values" for leaf lookups
    if isinstance(val, Mapping):
        return False
    return False
