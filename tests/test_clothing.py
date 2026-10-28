"""Clothing Cd·A multiplier and rider.toml persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from bike_wind.clothing import (
    DEFAULT_CLOTHING_KEY,
    LOOSE_JERSEY_CDA_MULT,
    clothing_key_for_label,
    resolve_clothing,
)
from bike_wind.physics.forces import RiderBikeParams, WindComponents, aero_force
from bike_wind.user_config import clothing_key_from_config, load_rider_config, save_rider_config


def test_default_clothing_is_skin_tight() -> None:
    assert DEFAULT_CLOTHING_KEY == "skin_tight_aero"
    assert resolve_clothing().cda_multiplier == 1.0


def test_loose_jersey_multiplier() -> None:
    loose = resolve_clothing("loose_jersey")
    assert loose.cda_multiplier == pytest.approx(LOOSE_JERSEY_CDA_MULT)
    assert 1.05 <= loose.cda_multiplier <= 1.10
    assert clothing_key_for_label(loose.label) == "loose_jersey"


def test_aero_force_scales_with_clothing() -> None:
    base = RiderBikeParams(rider_mass_kg=70.0, power_w=200.0, clothing_cda_scale=1.0)
    loose = base.with_updates(clothing_cda_scale=1.08)
    f0 = aero_force(10.0, base)
    f1 = aero_force(10.0, loose)
    assert f1 == pytest.approx(f0 * 1.08, rel=1e-9)
    # Calm zero crosswind — clothing still applies.
    assert aero_force(10.0, loose, wind=WindComponents()) == pytest.approx(f0 * 1.08)


def test_rider_toml_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "rider.toml"
    save_rider_config({"clothing_key": "loose_jersey"}, path=path)
    assert clothing_key_from_config(path) == "loose_jersey"
    cfg = load_rider_config(path)
    assert cfg["clothing_key"] == "loose_jersey"
