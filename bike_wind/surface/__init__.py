"""Road-surface lookup (OpenStreetMap) for the surface-aware rolling model."""

from bike_wind.surface.osm_surface import (
    OFF_MAP,
    PAVED,
    UNPAVED,
    classify_tags,
    route_surface_classes,
)

__all__ = [
    "OFF_MAP",
    "PAVED",
    "UNPAVED",
    "classify_tags",
    "route_surface_classes",
]
