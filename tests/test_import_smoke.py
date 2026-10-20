"""Startup import smoke tests — catch broken app imports before Cloud redeploy."""

from __future__ import annotations


def test_past_ride_imports() -> None:
    from bike_wind.past_ride import compare_activity, ride_activity_label
    from bike_wind.strava import ActivitySummary, StravaClient

    assert callable(compare_activity)
    assert callable(ride_activity_label)
    assert ActivitySummary is not None
    assert StravaClient is not None


def test_app_module_imports() -> None:
    import app as app_mod

    assert callable(app_mod.main)
