"""Persist non-secret rider/model settings to a local TOML file."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from bike_wind.clothing import DEFAULT_CLOTHING_KEY


def default_config_path() -> Path:
    """Prefer XDG config, else ``.streamlit/rider.toml`` under cwd when present."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "bike-wind" / "rider.toml"
    return Path.home() / ".config" / "bike-wind" / "rider.toml"


def _resolve_read_path(path: Path | None) -> Path | None:
    candidates: list[Path] = []
    if path is not None:
        candidates.append(path)
    else:
        candidates.append(default_config_path())
        candidates.append(Path(".streamlit") / "rider.toml")
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def load_rider_config(path: Path | None = None) -> dict[str, Any]:
    """Load ``[rider]`` settings; missing file → empty dict.

    Parses a tiny TOML subset (booleans, numbers, quoted strings) so we do not
    depend on tomllib / tomli.
    """
    resolved = _resolve_read_path(path)
    if resolved is None:
        return {}
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError:
        return {}
    in_rider = False
    out: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            in_rider = line == "[rider]"
            continue
        if not in_rider or "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        val = rest.strip()
        if val in {"true", "false"}:
            out[key] = val == "true"
        elif re.fullmatch(r"-?\d+", val):
            out[key] = int(val)
        elif re.fullmatch(r"-?\d+\.\d+", val):
            out[key] = float(val)
        elif len(val) >= 2 and val[0] == val[-1] and val[0] in {'"', "'"}:
            out[key] = val[1:-1]
        else:
            out[key] = val
    return out


def save_rider_config(updates: dict[str, Any], path: Path | None = None) -> Path:
    """Merge ``updates`` into ``[rider]`` and write TOML (no secrets)."""
    if path is None:
        streamlit_dir = Path(".streamlit")
        path = (
            streamlit_dir / "rider.toml"
            if streamlit_dir.is_dir()
            else default_config_path()
        )

    existing = load_rider_config(path)
    existing.update({k: v for k, v in updates.items() if v is not None})
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Bike Wind rider / model settings (not secrets).", "[rider]"]
    for key in sorted(existing):
        val = existing[key]
        if isinstance(val, bool):
            lines.append(f"{key} = {'true' if val else 'false'}")
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            lines.append(f"{key} = {val}")
        else:
            escaped = str(val).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{escaped}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def clothing_key_from_config(path: Path | None = None) -> str:
    cfg = load_rider_config(path)
    key = cfg.get("clothing_key")
    if isinstance(key, str) and key.strip():
        return key.strip()
    return DEFAULT_CLOTHING_KEY
