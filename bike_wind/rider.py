"""Rider position presets, aero kit toggles, and unit helpers."""

from __future__ import annotations

from dataclasses import dataclass

from bike_wind.physics.constants import BIKE_MASS_ROAD_KG, BIKE_MASS_TT_KG, RIDER_HEIGHT_M

KG_PER_LB = 0.45359237
CM_PER_IN = 2.54

# Cd / A deltas applied on top of the selected position preset
AERO_HELMET_KIT_CD = -0.05
AERO_HELMET_KIT_A = -0.02
NON_AERO_CD = 0.03


@dataclass(frozen=True)
class PositionPreset:
    """Named riding position with Cd, frontal area, and default bike mass."""

    key: str
    label: str
    cd: float
    area_m2: float
    bike_mass_kg: float


POSITION_PRESETS: dict[str, PositionPreset] = {
    "tt_aero": PositionPreset(
        key="tt_aero",
        label="TT aero bars",
        cd=0.6,
        area_m2=0.32,
        bike_mass_kg=BIKE_MASS_TT_KG,
    ),
    "drops": PositionPreset(
        key="drops",
        label="Road drops",
        cd=0.85,
        area_m2=0.38,
        bike_mass_kg=BIKE_MASS_ROAD_KG,
    ),
    "hoods": PositionPreset(
        key="hoods",
        label="Hoods",
        cd=0.95,
        area_m2=0.42,
        bike_mass_kg=BIKE_MASS_ROAD_KG,
    ),
    "upright": PositionPreset(
        key="upright",
        label="Upright",
        cd=1.1,
        area_m2=0.50,
        bike_mass_kg=BIKE_MASS_ROAD_KG,
    ),
}

DEFAULT_POSITION_KEY = "drops"
RIDER_WEIGHT_PRESETS_KG: tuple[float, ...] = (65.0, 70.0, 75.0, 80.0, 85.0, 90.0)


@dataclass(frozen=True)
class ResolvedAero:
    """Effective Cd, area, and bike mass after position + kit toggles."""

    position_key: str
    cd: float
    area_m2: float
    bike_mass_kg: float
    full_aero_kit: bool
    non_aero: bool


def kg_to_lb(kg: float) -> float:
    return kg / KG_PER_LB


def lb_to_kg(lb: float) -> float:
    return lb * KG_PER_LB


def smart_default_height_cm(weight_kg: float) -> float:
    """Optional stature default from weight; 170 cm at 75 kg, linear nearby."""
    return round(170.0 + (weight_kg - 75.0) * 0.4, 1)


def standing_height_to_wind_z_m(height_cm: float) -> float:
    """Map standing height to approximate riding wind height (~1.2 m @ 170 cm).

    Effect on results is small; used only for log-profile wind correction.
    """
    if height_cm <= 0:
        return RIDER_HEIGHT_M
    z = (height_cm / 100.0) * 0.70
    return max(0.8, min(1.5, z))


def resolve_aero(
    position_key: str = DEFAULT_POSITION_KEY,
    *,
    full_aero_kit: bool = False,
    non_aero: bool = False,
    bike_mass_kg: float | None = None,
) -> ResolvedAero:
    """Combine position preset with optional aero kit toggles.

    If both kit toggles are set, aero-kit wins (mutually exclusive intent).
    """
    preset = POSITION_PRESETS.get(position_key) or POSITION_PRESETS[DEFAULT_POSITION_KEY]
    cd = preset.cd
    area = preset.area_m2

    use_aero = full_aero_kit and not non_aero
    use_non = non_aero and not full_aero_kit
    # If both True, prefer full aero kit
    if full_aero_kit and non_aero:
        use_aero, use_non = True, False

    if use_aero:
        cd += AERO_HELMET_KIT_CD
        area += AERO_HELMET_KIT_A
    elif use_non:
        cd += NON_AERO_CD

    mass = preset.bike_mass_kg if bike_mass_kg is None else bike_mass_kg
    return ResolvedAero(
        position_key=preset.key,
        cd=cd,
        area_m2=max(area, 0.15),
        bike_mass_kg=mass,
        full_aero_kit=use_aero,
        non_aero=use_non,
    )


def position_labels() -> list[str]:
    return [p.label for p in POSITION_PRESETS.values()]


def position_key_for_label(label: str) -> str:
    for key, preset in POSITION_PRESETS.items():
        if preset.label == label:
            return key
    return DEFAULT_POSITION_KEY
