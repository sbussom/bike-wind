"""Logarithmic wind-profile correction from 10 m to rider height."""

from __future__ import annotations

import math

from bike_wind.physics.constants import RIDER_HEIGHT_M, WIND_REF_HEIGHT_M, Z0_OPEN_GRASS


def correct_wind_speed(
    speed_at_ref_mps: float,
    *,
    z_ref_m: float = WIND_REF_HEIGHT_M,
    z_m: float = RIDER_HEIGHT_M,
    z0_m: float = Z0_OPEN_GRASS,
) -> float:
    """Scale wind speed from reference height to rider height via log profile.

    u(z) = u(z_ref) * ln(z/z0) / ln(z_ref/z0)
    """
    if speed_at_ref_mps == 0.0:
        return 0.0
    if z_m <= z0_m or z_ref_m <= z0_m:
        raise ValueError("heights must be greater than roughness length z0")
    return speed_at_ref_mps * math.log(z_m / z0_m) / math.log(z_ref_m / z0_m)


def wind_uv_at_rider_height(
    u10: float,
    v10: float,
    *,
    z_m: float = RIDER_HEIGHT_M,
    z0_m: float = Z0_OPEN_GRASS,
) -> tuple[float, float]:
    """Correct U/V components from 10 m to rider height (same scale factor)."""
    # Magnitude scale; apply to each component so direction is preserved.
    speed10 = math.hypot(u10, v10)
    if speed10 == 0.0:
        return 0.0, 0.0
    speed_z = correct_wind_speed(speed10, z_m=z_m, z0_m=z0_m)
    scale = speed_z / speed10
    return u10 * scale, v10 * scale
