"""Shared Strava data models (safe to import without the HTTP client)."""

from __future__ import annotations

from dataclasses import dataclass


def device_watts_flag(payload: dict) -> bool | None:
    """Strava ``device_watts``. Missing means the watts are not a power meter."""
    if not isinstance(payload, dict) or "device_watts" not in payload:
        return None
    value = payload.get("device_watts")
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


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
    # Strava ``trainer`` flag. Indoor rides often also lack a GPS track.
    trainer: bool = False
    # True only when the summary has a start point and a non-empty map polyline.
    has_gps: bool = False
    # Strava ``device_watts``. True is a power meter. False is estimated power.
    # None means the payload did not say, which is not counted as a meter.
    device_watts: bool | None = None

    @property
    def has_power(self) -> bool:
        """Any positive average watts, including Strava estimated power."""
        return self.average_watts is not None and self.average_watts > 0

    @property
    def has_meter_power(self) -> bool:
        """Power-meter watts only. Estimated Strava power does not count."""
        return self.device_watts is True and self.has_power


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
    start_index: int | None = None
    end_index: int | None = None
    # Same Strava flag as the parent activity, when the effort payload has it.
    device_watts: bool | None = None

    @property
    def has_meter_power(self) -> bool:
        return self.device_watts is True and self.average_watts is not None and self.average_watts > 0

    @property
    def average_speed_mps(self) -> float | None:
        """Ground speed from effort distance / elapsed (or moving) time."""
        t = self.elapsed_time_s or self.moving_time_s
        if t <= 0 or self.distance_m <= 0:
            return None
        return self.distance_m / float(t)

    @property
    def has_stream_indices(self) -> bool:
        return (
            self.start_index is not None
            and self.end_index is not None
            and self.end_index > self.start_index
        )

    def label(self) -> str:
        km = self.distance_m / 1000.0
        t = self.elapsed_time_s or self.moving_time_s
        mins, secs = divmod(max(0, int(t)), 60)
        return f"{self.name} · {km:.1f} km · {mins}:{secs:02d}"
