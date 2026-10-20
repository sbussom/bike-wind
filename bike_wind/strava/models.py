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


@dataclass(frozen=True)
class SegmentEffortSummary:
    """One segment effort from GET /activities/{id} ``segment_efforts``."""

    effort_id: int
    segment_id: int
    name: str
    distance_m: float
    elapsed_time_s: int
    moving_time_s: int
    start_date: str  # ISO UTC
    average_watts: float | None = None

    @property
    def average_speed_mps(self) -> float | None:
        """Ground speed from effort distance / elapsed (or moving) time."""
        t = self.elapsed_time_s or self.moving_time_s
        if t <= 0 or self.distance_m <= 0:
            return None
        return self.distance_m / float(t)

    def label(self) -> str:
        km = self.distance_m / 1000.0
        t = self.elapsed_time_s or self.moving_time_s
        mins, secs = divmod(max(0, int(t)), 60)
        return f"{self.name} · {km:.1f} km · {mins}:{secs:02d}"
