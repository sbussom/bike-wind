"""Strava segment fetch, OAuth, cache, and rate limiting."""

from bike_wind.strava.client import (
    ActivitySummary,
    ExploreHit,
    SegmentMeta,
    StarredSegment,
    StravaAuthError,
    StravaClient,
    StravaConfig,
    StravaError,
    StravaRateLimitError,
    load_strava_config,
)
from bike_wind.strava.urls import ParsedStravaRef, parse_strava_segment_ref

__all__ = [
    "ActivitySummary",
    "ExploreHit",
    "ParsedStravaRef",
    "SegmentMeta",
    "StarredSegment",
    "StravaAuthError",
    "StravaClient",
    "StravaConfig",
    "StravaError",
    "StravaRateLimitError",
    "load_strava_config",
    "parse_strava_segment_ref",
]
