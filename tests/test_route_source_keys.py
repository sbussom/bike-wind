"""route_source must stay a non-widget session key (Streamlit widget order)."""

from __future__ import annotations

from bike_wind.strava.models import ActivitySummary


def test_select_helpers_only_set_route_source_not_widget_keys(monkeypatch) -> None:
    """Mimic session_state: selecting a ride must not write use_strava_* keys.

    Those names were widget keys and caused StreamlitWidgetAlreadyInstantiatedError
    when _load_ride_track ran after the segment checkbox.
    """
    state: dict = {"route_source": None, "power_w": 220}

    class _SS(dict):
        def __getattr__(self, name: str):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name: str, value) -> None:
            self[name] = value

    ss = _SS(state)
    monkeypatch.setattr("app.st.session_state", ss)

    from app import _select_strava_activity, _select_strava_segment

    act = ActivitySummary(
        activity_id=1,
        name="R",
        sport_type="Ride",
        start_date="2024-01-01T12:00:00Z",
        distance_m=1000,
        moving_time_s=100,
        elapsed_time_s=100,
        average_watts=200.0,
    )
    _select_strava_activity(act)
    assert ss.route_source == "activity"
    assert "use_strava_segment" not in ss
    assert "use_strava_activity" not in ss

    _select_strava_segment(9, name="Seg", distance_m=500.0)
    assert ss.route_source == "segment"
    assert "use_strava_segment" not in ss
    assert "use_strava_activity" not in ss
