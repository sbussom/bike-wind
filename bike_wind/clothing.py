"""Clothing presets that scale Cd·A (manual override; Strava has no clothing field)."""

from __future__ import annotations

from dataclasses import dataclass

# Default leaves predictions unchanged vs prior releases.
DEFAULT_CLOTHING_KEY = "skin_tight_aero"

# Loose jersey band was 1.05–1.10; pick the midpoint.
LOOSE_JERSEY_CDA_MULT = 1.08


@dataclass(frozen=True)
class ClothingPreset:
    """Named clothing choice with a Cd·A multiplier on the base aero."""

    key: str
    label: str
    cda_multiplier: float
    note: str | None = None


CLOTHING_PRESETS: dict[str, ClothingPreset] = {
    "skin_tight_aero": ClothingPreset(
        key="skin_tight_aero",
        label="Full skin-tight aero",
        cda_multiplier=1.0,
        note="Baseline Cd·A (in use 0.380 for gravel hoods).",
    ),
    "loose_jersey": ClothingPreset(
        key="loose_jersey",
        label="Loose-fitting biking jersey",
        cda_multiplier=LOOSE_JERSEY_CDA_MULT,
        note="About +8% Cd·A vs skin-tight (band 1.05–1.10).",
    ),
}


def resolve_clothing(clothing_key: str = DEFAULT_CLOTHING_KEY) -> ClothingPreset:
    return CLOTHING_PRESETS.get(clothing_key) or CLOTHING_PRESETS[DEFAULT_CLOTHING_KEY]


def clothing_labels() -> list[str]:
    return [p.label for p in CLOTHING_PRESETS.values()]


def clothing_key_for_label(label: str) -> str:
    for key, preset in CLOTHING_PRESETS.items():
        if preset.label == label:
            return key
    return DEFAULT_CLOTHING_KEY
