"""Cycling physics: forces and iterative velocity solver.

Import leaf modules (``constants``, ``forces``, ``solver``) directly when
possible. This package ``__init__`` only re-exports and loads ``solver``
lazily so Streamlit Cloud reloads do not deadlock while ``physics`` is
partially initialized.
"""

from __future__ import annotations

from typing import Any

from bike_wind.physics.constants import (
    BIKE_MASS_GRAVEL_KG,
    BIKE_MASS_ROAD_KG,
    BIKE_MASS_TT_KG,
    G,
    RHO_DEFAULT,
)
from bike_wind.physics.forces import (
    RiderBikeParams,
    WindComponents,
    aero_force,
    air_density,
    gravity_force,
    rolling_force,
    total_force,
)

__all__ = [
    "BIKE_MASS_GRAVEL_KG",
    "BIKE_MASS_ROAD_KG",
    "BIKE_MASS_TT_KG",
    "G",
    "RHO_DEFAULT",
    "RiderBikeParams",
    "WindComponents",
    "aero_force",
    "air_density",
    "gravity_force",
    "rolling_force",
    "total_force",
    "solve_speed",
    "solve_speeds",
]


def __getattr__(name: str) -> Any:
    """Lazy-load solver symbols (avoids import-time cycle on reload)."""
    if name == "solve_speed":
        from bike_wind.physics.solver import solve_speed

        return solve_speed
    if name == "solve_speeds":
        from bike_wind.physics.solver import solve_speeds

        return solve_speeds
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
