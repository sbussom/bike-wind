"""Parse Strava segment / activity-effort URLs and raw IDs."""

from __future__ import annotations

import re
from dataclasses import dataclass

# https://www.strava.com/segments/229781  (optional www / http)
SEGMENT_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?strava\.com/segments/(\d+)",
    re.IGNORECASE,
)
# https://www.strava.com/activities/123456/segments/789012  → effort id
ACTIVITY_EFFORT_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?strava\.com/activities/(\d+)/segments/(\d+)",
    re.IGNORECASE,
)
RAW_ID_RE = re.compile(r"^\d+$")


@dataclass(frozen=True)
class ParsedStravaRef:
    """Resolved reference before API calls."""

    kind: str  # "segment_id" | "effort_id"
    value: int
    activity_id: int | None = None
    raw: str = ""


def parse_strava_segment_ref(text: str) -> ParsedStravaRef:
    """Parse a segment URL, activity-effort URL, or raw numeric ID.

    Activity-effort URLs yield ``kind="effort_id"`` — caller must resolve via
    ``GET /segment_efforts/{id}`` to obtain the real segment id.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Enter a Strava segment URL or numeric ID")

    m = ACTIVITY_EFFORT_URL_RE.search(raw)
    if m:
        return ParsedStravaRef(
            kind="effort_id",
            value=int(m.group(2)),
            activity_id=int(m.group(1)),
            raw=raw,
        )

    m = SEGMENT_URL_RE.search(raw)
    if m:
        return ParsedStravaRef(kind="segment_id", value=int(m.group(1)), raw=raw)

    if RAW_ID_RE.fullmatch(raw):
        return ParsedStravaRef(kind="segment_id", value=int(raw), raw=raw)

    raise ValueError(
        "Unrecognized Strava reference. Use a segment URL "
        "(strava.com/segments/…), an activity effort URL "
        "(strava.com/activities/…/segments/…), or a numeric ID."
    )
