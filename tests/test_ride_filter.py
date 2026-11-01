"""Outdoor-ride filter: lists skip virtual, trainer, e-bike, and no-GPS activities."""

from __future__ import annotations

from pathlib import Path

import pytest

from bike_wind.choice_list import page_window
from bike_wind.ride_filter import (
    collect_outdoor_rides,
    exclusion_reason,
    hidden_rides_note,
    is_outdoor_ride,
    split_outdoor,
)
from bike_wind.strava.client import StravaClient
from bike_wind.strava.models import ActivitySummary


def _act(
    activity_id: int = 1,
    *,
    sport_type: str = "Ride",
    trainer: bool = False,
    has_gps: bool = True,
    name: str = "Ride",
) -> ActivitySummary:
    return ActivitySummary(
        activity_id=activity_id,
        name=name,
        sport_type=sport_type,
        start_date="2024-06-01T12:00:00Z",
        distance_m=20000.0,
        moving_time_s=3600,
        elapsed_time_s=3700,
        average_watts=180.0,
        start_lat=26.1 if has_gps else None,
        start_lon=-80.1 if has_gps else None,
        trainer=trainer,
        has_gps=has_gps,
    )


def test_keeps_outdoor_bike_sports() -> None:
    for sport in ("Ride", "GravelRide", "MountainBikeRide", "Gravel Ride", "Mountain-Bike-Ride"):
        assert is_outdoor_ride(_act(sport_type=sport)) is True
        assert exclusion_reason(_act(sport_type=sport)) is None


@pytest.mark.parametrize(
    ("sport", "reason"),
    [
        ("VirtualRide", "virtual"),
        ("Virtual Ride", "virtual"),
        ("virtualride", "virtual"),
        ("EBikeRide", "ebike"),
        ("E-Bike Ride", "ebike"),
        ("EMountainBikeRide", "ebike"),
        ("E-Mountain-Bike-Ride", "ebike"),
        ("Run", "not_a_ride"),
        ("Swim", "not_a_ride"),
        ("Walk", "not_a_ride"),
        ("Hike", "not_a_ride"),
    ],
)
def test_excludes_virtual_ebike_and_non_rides(sport: str, reason: str) -> None:
    assert exclusion_reason(_act(sport_type=sport)) == reason


def test_trainer_flag_excludes_even_a_gps_ride() -> None:
    act = _act(sport_type="Ride", trainer=True, has_gps=True)
    assert exclusion_reason(act) == "trainer"
    assert is_outdoor_ride(act) is False


def test_virtual_wins_over_trainer_flag() -> None:
    act = _act(sport_type="VirtualRide", trainer=True, has_gps=True)
    assert exclusion_reason(act) == "virtual"


def test_missing_gps_excludes_a_real_ride() -> None:
    assert exclusion_reason(_act(has_gps=False)) == "no_gps"


def test_split_counts_indoor_not_runs() -> None:
    acts = [
        _act(1, sport_type="Ride"),
        _act(2, sport_type="VirtualRide"),
        _act(3, sport_type="Ride", trainer=True),
        _act(4, sport_type="EBikeRide"),
        _act(5, sport_type="EMountainBikeRide"),
        _act(6, sport_type="Ride", has_gps=False),
        _act(7, sport_type="Run"),
        _act(8, sport_type="Swim"),
        _act(9, sport_type="GravelRide"),
        _act(10, sport_type="MountainBikeRide"),
    ]
    kept, hidden = split_outdoor(acts)
    assert [a.activity_id for a in kept] == [1, 9, 10]
    # virtual, trainer, two e-bikes, no GPS. Runs and swims are dropped quietly.
    assert hidden == 5


def test_hidden_note_wording() -> None:
    assert hidden_rides_note(0) is None
    assert hidden_rides_note(1) == "Hiding 1 indoor/virtual ride."
    assert hidden_rides_note(3) == "Hiding 3 indoor/virtual rides."


def test_collect_fetches_another_page_when_the_first_is_indoor() -> None:
    """A full page of trainer rides must not leave the list empty."""
    pages = {
        1: [_act(i, sport_type="VirtualRide", name="Zwift") for i in range(1, 7)],
        2: [_act(100 + i, sport_type="Ride", name=f"Road {i}") for i in range(1, 7)],
    }
    seen: list[int] = []

    def fetch(page: int) -> list[ActivitySummary]:
        seen.append(page)
        return pages.get(page, [])

    kept, hidden = collect_outdoor_rides(fetch, per_page=6, want=6)
    assert seen == [1, 2]
    assert hidden == 6
    assert [a.activity_id for a in kept] == [101, 102, 103, 104, 105, 106]
    window, index, pages_n = page_window([a.name for a in kept], page=0, page_size=6)
    assert index == 0
    assert pages_n == 1
    assert window == [a.name for a in kept]
    assert window  # the visible page is the outdoor rides, not a blank batch


def test_collect_pages_a_long_outdoor_list_six_at_a_time() -> None:
    """Newer/Older still walks the filtered list, six rides per page."""
    outdoor = [_act(i, name=f"Outdoor {i}") for i in range(1, 9)]
    trainers = [_act(100 + i, trainer=True, name="Trainer") for i in range(4)]
    # First API page mixes trainers in front of real rides; second page finishes the set.
    pages = {1: trainers + outdoor[:2], 2: outdoor[2:]}

    def fetch(page: int) -> list[ActivitySummary]:
        return pages.get(page, [])

    kept, hidden = collect_outdoor_rides(fetch, per_page=6, want=8)
    assert hidden == 4
    assert [a.activity_id for a in kept] == list(range(1, 9))
    first, _, pages_n = page_window([a.name for a in kept], page=0, page_size=6)
    second, index, _ = page_window([a.name for a in kept], page=1, page_size=6)
    assert pages_n == 2
    assert first == [f"Outdoor {i}" for i in range(1, 7)]
    assert index == 1
    assert second == ["Outdoor 7", "Outdoor 8"]


def test_collect_stops_on_a_short_page() -> None:
    seen: list[int] = []

    def fetch(page: int) -> list[ActivitySummary]:
        seen.append(page)
        if page == 1:
            return [_act(1, sport_type="VirtualRide"), _act(2)]
        raise AssertionError("should not fetch past a short page")

    kept, hidden = collect_outdoor_rides(fetch, per_page=6, want=10)
    assert seen == [1]
    assert [a.activity_id for a in kept] == [2]
    assert hidden == 1


def test_collect_want_none_keeps_every_outdoor_ride_until_history_ends() -> None:
    pages = {
        1: [_act(1, sport_type="Run"), _act(2), _act(3, sport_type="VirtualRide")],
        2: [_act(4, sport_type="EBikeRide")],  # short page
    }

    def fetch(page: int) -> list[ActivitySummary]:
        return pages.get(page, [])

    kept, hidden = collect_outdoor_rides(fetch, per_page=3, want=None)
    assert [a.activity_id for a in kept] == [2]
    assert hidden == 2  # virtual + e-bike; the run is not in the note


def test_collect_respects_max_pages() -> None:
    def fetch(page: int) -> list[ActivitySummary]:
        return [_act(page, sport_type="VirtualRide")]

    kept, hidden = collect_outdoor_rides(fetch, per_page=1, want=5, max_pages=3)
    assert kept == []
    assert hidden == 3


def test_activity_summary_marks_gps_from_start_and_polyline() -> None:
    with_track = StravaClient._activity_from_item(
        {
            "id": 9,
            "name": "Canal",
            "sport_type": "Ride",
            "start_date": "2024-06-01T12:00:00Z",
            "distance": 1000,
            "moving_time": 200,
            "elapsed_time": 200,
            "average_watts": 150,
            "trainer": False,
            "start_latlng": [26.1, -80.1],
            "map": {"summary_polyline": "abc"},
        }
    )
    assert with_track.has_gps is True
    assert with_track.trainer is False
    assert is_outdoor_ride(with_track) is True

    trainer = StravaClient._activity_from_item(
        {
            "id": 10,
            "name": "Trainer",
            "type": "Ride",
            "start_date": "2024-06-01T12:00:00Z",
            "distance": 1000,
            "moving_time": 200,
            "elapsed_time": 200,
            "trainer": True,
            "start_latlng": [26.1, -80.1],
            "map": {"polyline": "abc"},
        }
    )
    assert trainer.sport_type == "Ride"
    assert trainer.trainer is True
    assert trainer.has_gps is True
    assert exclusion_reason(trainer) == "trainer"

    no_map = StravaClient._activity_from_item(
        {
            "id": 11,
            "name": "No GPS",
            "sport_type": "Ride",
            "start_date": "2024-06-01T12:00:00Z",
            "distance": 1000,
            "moving_time": 200,
            "elapsed_time": 200,
            "start_latlng": [26.1, -80.1],
            "map": {"summary_polyline": "   "},
        }
    )
    assert no_map.has_gps is False
    assert exclusion_reason(no_map) == "no_gps"

    virtual = StravaClient._activity_from_item(
        {
            "id": 12,
            "name": "Zwift",
            "type": "VirtualRide",
            "start_date": "2024-06-01T12:00:00Z",
            "distance": 1000,
            "moving_time": 200,
            "elapsed_time": 200,
            "start_latlng": None,
            "map": {},
        }
    )
    assert virtual.sport_type == "VirtualRide"
    assert virtual.has_gps is False
    assert exclusion_reason(virtual) == "virtual"


def test_lists_in_the_app_go_through_the_outdoor_filter() -> None:
    root = Path(__file__).resolve().parents[1]
    app = (root / "app.py").read_text()
    assert app.count("collect_outdoor_rides(") == 2
    assert "hidden_rides_note" in app
    assert "_filter_cycling" not in app
    assert "virtualride" not in app.lower()
    experimental = (root / "pages" / "2_Experimental.py").read_text()
    assert "list_recent_activities" not in experimental
