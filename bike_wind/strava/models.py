"""Shared Strava data models (safe to import without the HTTP client)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActivitySummary:
    """One row from GET /athlete/activities."""

    activity_id: int
    name: str
    sport_type: str
    start_date: str  # ISO UTC from Strava
    distance_m: float
    moving_time_s: int
    elapsed_time_s: int
    average_watts: float | None
    start_lat: float | None = None
    start_lon: float | None = None

    @property
    def has_power(self) -> bool:
        return self.average_watts is not None and self.average_watts > 0
