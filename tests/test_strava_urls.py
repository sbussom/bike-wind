"""Unit tests for Strava URL / ID parsing (no network)."""

from __future__ import annotations

import pytest

from bike_wind.strava.urls import parse_strava_segment_ref


def test_parse_segment_url() -> None:
    ref = parse_strava_segment_ref("https://www.strava.com/segments/229781")
    assert ref.kind == "segment_id"
    assert ref.value == 229781


def test_parse_segment_url_no_www() -> None:
    ref = parse_strava_segment_ref("http://strava.com/segments/42")
    assert ref.kind == "segment_id"
    assert ref.value == 42


def test_parse_activity_effort_url() -> None:
    ref = parse_strava_segment_ref(
        "https://www.strava.com/activities/1234567890/segments/987654321"
    )
    assert ref.kind == "effort_id"
    assert ref.value == 987654321
    assert ref.activity_id == 1234567890


def test_parse_raw_id() -> None:
    ref = parse_strava_segment_ref("229781")
    assert ref.kind == "segment_id"
    assert ref.value == 229781


def test_parse_rejects_empty() -> None:
    with pytest.raises(ValueError):
        parse_strava_segment_ref("  ")


def test_parse_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_strava_segment_ref("https://example.com/segments/1")
