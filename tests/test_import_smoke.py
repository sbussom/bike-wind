"""Startup import smoke tests — catch broken app imports before Cloud redeploy."""

from __future__ import annotations


def test_past_ride_imports() -> None:
    from bike_wind.past_ride import compare_activity, ride_activity_label
    from bike_wind.strava.client import StravaClient
    from bike_wind.strava.models import ActivitySummary

    assert callable(compare_activity)
    assert callable(ride_activity_label)
    assert ActivitySummary is not None
    assert StravaClient is not None


def test_activity_summary_defined_in_models() -> None:
    import inspect

    import bike_wind.strava.models as models

    assert hasattr(models, "ActivitySummary")
    src = inspect.getsource(models.ActivitySummary)
    assert "activity_id" in src


def test_app_module_imports() -> None:
    import app as app_mod

    assert callable(app_mod.main)
