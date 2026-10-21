"""Startup import smoke tests — catch broken app imports before Cloud redeploy."""

from __future__ import annotations

import subprocess
import sys


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


def test_app_imports_twice_without_deadlock() -> None:
    """Fresh process: import app twice (Streamlit-style reload must not hang)."""
    code = """
import app
assert callable(app.main)
import app as app2
assert callable(app2.main)
from bike_wind.predict import PredictionResult
assert hasattr(PredictionResult, "distances_mi")
assert hasattr(PredictionResult, "speeds_with_wind_mph")
assert hasattr(PredictionResult, "speeds_no_wind_mph")
assert hasattr(PredictionResult, "has_watts")
print("ok")
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=".",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "ok" in proc.stdout


def test_physics_package_lazy_solver() -> None:
    """solver is not required until solve_speed is accessed."""
    import bike_wind.physics as physics

    assert physics.RiderBikeParams is not None
    assert callable(physics.solve_speed)
    assert callable(physics.solve_speeds)
