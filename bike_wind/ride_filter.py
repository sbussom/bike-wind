"""Which Strava activities belong in the ride lists.

Outdoor prediction needs a bike on a real road with a GPS track. Zwift,
trainer rides, e-bikes, runs, and activities with no map are left out.
This does not change the fitted model. The calibration corpus is a folder
of stream files and does not apply this filter.

``accuracy_skip_reason`` is the extra gate for forecast and post-ride
scores: those same indoor rides, plus any ride that is not a power meter
(Strava ``device_watts``). Ride lists still show outdoor rides that only
have estimated watts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from bike_wind.strava.models import ActivitySummary, SegmentEffortSummary

# Strava sport_type values, compared after lowercasing and stripping spaces/hyphens.
OUTDOOR_SPORTS = frozenset({"ride", "gravelride", "mountainbikeride"})
EBIKE_SPORTS = frozenset({"ebikeride", "emountainbikeride"})
# Counted in the "hiding N indoor/virtual rides" note.
_HIDDEN_REASONS = frozenset({"virtual", "trainer", "ebike", "no_gps"})


def sport_key(sport: str | None) -> str:
    return (sport or "").lower().replace(" ", "").replace("-", "").replace("_", "")


def exclusion_reason(act: ActivitySummary) -> str | None:
    """Why this activity is left off the list, or None if it is shown.

    ``virtual`` / ``trainer`` / ``ebike`` / ``no_gps`` are indoor-style rides.
    ``not_a_ride`` is a run, swim, walk, or anything else that is not a bike.
    """
    key = sport_key(act.sport_type)
    if key == "virtualride" or key.startswith("virtual"):
        return "virtual"
    if act.trainer:
        return "trainer"
    if key in EBIKE_SPORTS:
        return "ebike"
    if key not in OUTDOOR_SPORTS:
        return "not_a_ride"
    if not act.has_gps:
        return "no_gps"
    return None


def is_outdoor_ride(act: ActivitySummary) -> bool:
    return exclusion_reason(act) is None


def _named_zwift(act: ActivitySummary) -> bool:
    return "zwift" in f"{act.sport_type} {act.name}".lower()


def accuracy_skip_reason(
    act: ActivitySummary | None,
    effort: SegmentEffortSummary | None = None,
) -> str | None:
    """Why this ride must not enter an accuracy tally, or None if it counts.

    Virtual, Zwift, and trainer rides are out. So is any ride whose watts are
    not from a power meter. Average watts above 0 is not enough: Strava
    estimated power sets that without ``device_watts``.
    """
    if act is None and effort is None:
        return "no_power_meter"
    if act is not None:
        if _named_zwift(act):
            return "virtual"
        reason = exclusion_reason(act)
        if reason is not None:
            return reason
        if act.device_watts is not True:
            if act.average_watts is not None and act.average_watts > 0:
                return "estimated_power"
            return "no_power_meter"
        if effort is not None and effort.device_watts is False:
            return "estimated_power"
        return None
    assert effort is not None
    if effort.device_watts is not True:
        if effort.average_watts is not None and effort.average_watts > 0:
            return "estimated_power"
        return "no_power_meter"
    return None


def split_outdoor(acts: Sequence[ActivitySummary]) -> tuple[list[ActivitySummary], int]:
    """Return ``(kept, hidden_indoor_count)``.

    Runs and other non-rides are dropped and are not included in the count.
    """
    kept: list[ActivitySummary] = []
    hidden = 0
    for act in acts:
        reason = exclusion_reason(act)
        if reason is None:
            kept.append(act)
        elif reason in _HIDDEN_REASONS:
            hidden += 1
    return kept, hidden


def hidden_rides_note(count: int) -> str | None:
    if count <= 0:
        return None
    noun = "ride" if count == 1 else "rides"
    return f"Hiding {count} indoor/virtual {noun}."


def collect_outdoor_rides(
    fetch_page: Callable[[int], Sequence[ActivitySummary]],
    *,
    per_page: int,
    want: int | None,
    max_pages: int = 8,
) -> tuple[list[ActivitySummary], int]:
    """Keep asking Strava for the next page until enough outdoor rides are in hand.

    A page that is mostly trainer rides must not leave the list empty when a
    later page has a real ride. Stop when ``want`` outdoor rides are kept, when
    a page comes back shorter than ``per_page``, or after ``max_pages``.
    ``want`` None keeps every outdoor ride in the pages that were fetched.
    """
    if per_page < 1:
        raise ValueError("per_page must be at least 1")
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    kept: list[ActivitySummary] = []
    hidden = 0
    for page in range(1, max_pages + 1):
        batch = list(fetch_page(page))
        good, hide = split_outdoor(batch)
        kept.extend(good)
        hidden += hide
        if (want is not None and len(kept) >= want) or len(batch) < per_page:
            break
    if want is not None:
        kept = kept[:want]
    return kept, hidden
