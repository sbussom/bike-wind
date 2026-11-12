"""Forecast logging, matching, power-meter filter, and segment-time scores."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bike_wind.calibration.metrics import direction_error_deg, score_segment_time
from bike_wind.forecast_log import (
    ForecastLog,
    default_forecast_log_path,
    match_forecasts,
    summarize_forecast,
    summarize_post_rides,
)
from bike_wind.past_ride import PastRideComparison, grade_comparison
from bike_wind.physics.forces import RiderBikeParams, WindComponents
from bike_wind.predict import PredictionResult
from bike_wind.ride_filter import accuracy_skip_reason
from bike_wind.strava.models import ActivitySummary, SegmentEffortSummary
from bike_wind.weather.cache import default_weather_cache_path


def _act(
    activity_id: int,
    *,
    name: str = "Ride",
    sport_type: str = "Ride",
    start: str = "2026-11-12T18:00:00+00:00",
    moving: int = 1000,
    elapsed: int = 1100,
    watts: float | None = 220.0,
    device_watts: bool | None = True,
    trainer: bool = False,
    lat: float | None = 27.20,
    lon: float | None = -80.45,
    has_gps: bool = True,
) -> ActivitySummary:
    return ActivitySummary(
        activity_id=activity_id,
        name=name,
        sport_type=sport_type,
        start_date=start,
        distance_m=20000.0,
        moving_time_s=moving,
        elapsed_time_s=elapsed,
        average_watts=watts,
        start_lat=lat,
        start_lon=lon,
        trainer=trainer,
        has_gps=has_gps,
        device_watts=device_watts,
    )


def _effort(
    effort_id: int,
    segment_id: int,
    *,
    elapsed: int = 500,
    moving: int = 480,
    start: str = "2026-11-12T18:05:00+00:00",
    watts: float | None = 250.0,
    device_watts: bool | None = True,
    name: str = "Bank",
) -> SegmentEffortSummary:
    return SegmentEffortSummary(
        effort_id=effort_id,
        segment_id=segment_id,
        name=name,
        distance_m=5000.0,
        elapsed_time_s=elapsed,
        moving_time_s=moving,
        start_date=start,
        average_watts=watts,
        device_watts=device_watts,
    )


def _log(store: ForecastLog, **overrides) -> int:
    issued = datetime(2026, 11, 12, 15, 0, tzinfo=timezone.utc)
    target = datetime(2026, 11, 12, 18, 0, tzinfo=timezone.utc)
    fields = dict(
        issued_at=issued,
        target_time=target,
        lat=27.20,
        lon=-80.45,
        wind_speed_mps=5.0,
        wind_from_deg=90.0,
        gust_speed_mps=8.0,
        forecast_source="open-meteo-15min",
        segment_id=41,
        segment_name="Bank",
        target_power_w=250.0,
        predicted_time_s=500.0,
        kind="plan",
    )
    fields.update(overrides)
    row_id = store.log_forecast(**fields)
    assert row_id is not None
    return row_id


def _result(**overrides) -> PredictionResult:
    fields = dict(
        distances_m=[0.0, 1000.0],
        speeds_with_wind_mps=[10.0, 10.0],
        speeds_no_wind_mps=[10.0, 10.0],
        grades=[0.0, 0.0],
        wind_along_mps=[0.0, 0.0],
        wind_cross_mps=[0.0, 0.0],
        winds=[WindComponents(), WindComponents()],
        params=RiderBikeParams(rider_mass_kg=75, power_w=220, bike_mass_kg=8),
        rho_mean=1.2,
        weather_ok=True,
        lats=[27.20, 27.21],
        lons=[-80.45, -80.44],
        weather_source="open-meteo-15min",
        mean_wind_speed_mps=4.0,
        mean_wind_from_deg=90.0,
        mean_gust_speed_mps=6.5,
        used_ride_start_time=False,
    )
    fields.update(overrides)
    return PredictionResult(**fields)


class _SS(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


def test_segment_time_gate_and_wind_direction() -> None:
    inside = score_segment_time(105.0, 100.0)
    outside = score_segment_time(106.0, 100.0)
    assert inside is not None and inside.passed and inside.error_pct == pytest.approx(5.0)
    assert outside is not None and not outside.passed
    assert score_segment_time(100.0, 0.0) is None
    # Shortest turn from 350° to 10° is +20°, not −340°.
    assert direction_error_deg(10.0, 350.0) == pytest.approx(20.0)


def test_log_roundtrip_and_dedupe(tmp_path: Path) -> None:
    store = ForecastLog(tmp_path / "log.sqlite")
    row_id = _log(store)
    saved = store.list_forecasts()[0]
    assert saved.id == row_id
    assert saved.wind_speed_mps == pytest.approx(5.0)
    assert saved.wind_from_deg == pytest.approx(90.0)
    assert saved.gust_speed_mps == pytest.approx(8.0)
    assert saved.forecast_source == "open-meteo-15min"
    assert saved.issued_at.startswith("2026-11-12T15:00:00")
    assert saved.target_time.startswith("2026-11-12T18:00:00")
    assert saved.lat == pytest.approx(27.20)
    assert saved.segment_id == 41
    assert saved.segment_name == "Bank"
    assert saved.target_power_w == pytest.approx(250.0)
    assert saved.predicted_time_s == pytest.approx(500.0)
    assert saved.actual_time_s is None

    again = store.log_forecast(
        issued_at=datetime(2026, 11, 12, 15, 5, tzinfo=timezone.utc),
        target_time=datetime(2026, 11, 12, 18, 0, tzinfo=timezone.utc),
        lat=27.20,
        lon=-80.45,
        wind_speed_mps=5.0,
        wind_from_deg=90.0,
        gust_speed_mps=8.0,
        forecast_source="open-meteo-15min",
        segment_id=41,
        segment_name="Bank",
        target_power_w=250.0,
        predicted_time_s=500.0,
        kind="plan",
    )
    assert again is None
    assert len(store.list_forecasts()) == 1

    upgraded = store.log_forecast(
        issued_at=datetime(2026, 11, 12, 15, 6, tzinfo=timezone.utc),
        target_time=datetime(2026, 11, 12, 18, 0, tzinfo=timezone.utc),
        lat=27.20,
        lon=-80.45,
        wind_speed_mps=5.0,
        wind_from_deg=90.0,
        gust_speed_mps=8.0,
        forecast_source="open-meteo-15min",
        segment_id=41,
        segment_name="Bank",
        target_power_w=250.0,
        predicted_time_s=500.0,
        kind="kom",
    )
    assert upgraded == row_id
    assert store.list_forecasts()[0].kind == "kom"


def test_match_segment_time_and_wind_error(tmp_path: Path) -> None:
    store = ForecastLog(tmp_path / "log.sqlite")
    _log(store, predicted_time_s=510.0)
    # Elapsed is the segment clock. Moving time is different on purpose.
    effort = _effort(9, 41, elapsed=500, moving=450)
    act = _act(3)
    wind = lambda lat, lon, when: (4.0, 100.0)
    updated = match_forecasts(store.pending_forecasts(), [act], {3: [effort]}, wind)
    assert updated[0].matched_kind == "segment"
    assert updated[0].effort_id == 9
    assert updated[0].actual_time_s == pytest.approx(500.0)
    assert updated[0].time_error_s == pytest.approx(10.0)
    assert updated[0].time_error_pct == pytest.approx(2.0)
    assert updated[0].passed is True
    assert updated[0].wind_speed_error_mps == pytest.approx(1.0)
    assert updated[0].wind_dir_error_deg == pytest.approx(-10.0)
    store.update_forecast(updated[0])
    summary = summarize_forecast(store.list_forecasts())
    assert summary.source == "forecast"
    assert summary.n == 1 and summary.n_pass == 1
    assert summary.mean_abs_wind_mps == pytest.approx(1.0)
    assert summary.mean_abs_wind_dir_deg == pytest.approx(10.0)


def test_segment_time_fail_and_wind_optional(tmp_path: Path) -> None:
    store = ForecastLog(tmp_path / "log.sqlite")
    _log(store, predicted_time_s=600.0)
    effort = _effort(9, 41, elapsed=500, moving=500)
    updated = match_forecasts(
        store.pending_forecasts(),
        [_act(3)],
        {3: [effort]},
        lambda lat, lon, when: None,
    )
    assert updated[0].passed is False
    assert updated[0].time_error_pct == pytest.approx(20.0)
    assert updated[0].wind_speed_error_mps is None
    summary = summarize_forecast(updated)
    assert summary.n_pass == 0
    assert summary.mean_abs_wind_mps is None


def test_power_meter_filter_skips_estimated_and_missing_flag(tmp_path: Path) -> None:
    store = ForecastLog(tmp_path / "log.sqlite")
    _log(store)
    estimated = _act(1, device_watts=False, watts=220.0)
    missing = _act(2, device_watts=None, watts=220.0, start="2026-11-12T18:10:00+00:00")
    assert estimated.has_power is True
    assert estimated.has_meter_power is False
    assert missing.has_power is True
    assert accuracy_skip_reason(estimated) == "estimated_power"
    assert accuracy_skip_reason(missing) == "estimated_power"
    effort = _effort(9, 41)
    updated = match_forecasts(
        store.pending_forecasts(),
        [estimated, missing],
        {1: [effort], 2: [effort]},
        None,
    )
    assert updated[0].actual_time_s is None
    assert updated[0].passed is None

    # A later meter ride in the same window still counts.
    meter = _act(4, start="2026-11-12T19:00:00+00:00")
    scored = match_forecasts(updated, [estimated, meter], {1: [effort], 4: [effort]}, None)
    assert scored[0].activity_id == 4
    assert scored[0].passed is True


def test_effort_device_watts_false_overrides_activity(tmp_path: Path) -> None:
    store = ForecastLog(tmp_path / "log.sqlite")
    _log(store)
    act = _act(3)
    effort = _effort(9, 41, device_watts=False)
    assert accuracy_skip_reason(act, effort) == "estimated_power"
    updated = match_forecasts(store.pending_forecasts(), [act], {3: [effort]}, None)
    assert updated[0].actual_time_s is None


def test_virtual_trainer_and_zwift_are_not_scored(tmp_path: Path) -> None:
    store = ForecastLog(tmp_path / "log.sqlite")
    _log(store)
    virtual = _act(1, sport_type="VirtualRide", name="Zwift", start="2026-11-12T18:01:00+00:00")
    trainer = _act(2, trainer=True, start="2026-11-12T18:02:00+00:00")
    zwift = _act(3, name="Zwift race", start="2026-11-12T18:03:00+00:00")
    assert accuracy_skip_reason(virtual) == "virtual"
    assert accuracy_skip_reason(trainer) == "trainer"
    assert accuracy_skip_reason(zwift) == "virtual"
    effort = _effort(9, 41)
    efforts = {1: [effort], 2: [effort], 3: [effort]}
    updated = match_forecasts(store.pending_forecasts(), [virtual, trainer, zwift], efforts, None)
    assert updated[0].actual_time_s is None

    # The real ride is later, and it wins over the closer indoor rides.
    real = _act(8, start="2026-11-12T20:00:00+00:00")
    scored = match_forecasts(
        store.pending_forecasts(),
        [virtual, trainer, zwift, real],
        {**efforts, 8: [effort]},
        None,
    )
    assert scored[0].activity_id == 8


def test_whole_ride_uses_moving_time_and_location(tmp_path: Path) -> None:
    store = ForecastLog(tmp_path / "log.sqlite")
    _log(store, segment_id=None, segment_name="Out and back", predicted_time_s=3600.0)
    near = _act(5, moving=3700, elapsed=4000, lat=27.21, lon=-80.44)
    far = _act(6, moving=3600, start="2026-11-12T18:01:00+00:00", lat=40.0, lon=-74.0)
    updated = match_forecasts(store.pending_forecasts(), [far, near], {}, None)
    assert updated[0].matched_kind == "ride"
    assert updated[0].activity_id == 5
    assert updated[0].effort_id is None
    assert updated[0].actual_time_s == pytest.approx(3700.0)
    # 3600 vs 3700 is about −2.7%, inside the gate.
    assert updated[0].passed is True

    only_far = match_forecasts(
        store.pending_forecasts(),
        [far],
        {},
        None,
    )
    assert only_far[0].actual_time_s is None


def test_forecast_and_post_ride_stay_separate(tmp_path: Path) -> None:
    store = ForecastLog(tmp_path / "log.sqlite")
    _log(store, predicted_time_s=500.0)
    matched = match_forecasts(
        store.pending_forecasts(),
        [_act(3)],
        {3: [_effort(9, 41, elapsed=500)]},
        lambda lat, lon, when: (5.0, 90.0),
    )
    store.update_forecast(matched[0])
    store.upsert_post_ride(
        activity_id=3,
        segment_id=41,
        segment_name="Bank",
        predicted_time_s=480.0,
        actual_time_s=500.0,
        time_error_s=-20.0,
        time_error_pct=-4.0,
        passed=True,
        power_w=240.0,
        checked_at=datetime(2026, 11, 12, 20, tzinfo=timezone.utc),
    )
    # A second check for the same ride replaces the row.
    store.upsert_post_ride(
        activity_id=3,
        segment_id=41,
        segment_name="Bank",
        predicted_time_s=490.0,
        actual_time_s=500.0,
        time_error_s=-10.0,
        time_error_pct=-2.0,
        passed=True,
        power_w=245.0,
        checked_at=datetime(2026, 11, 12, 21, tzinfo=timezone.utc),
    )
    forecast = summarize_forecast(store.list_forecasts())
    post = summarize_post_rides(store.list_post_rides())
    assert forecast.source == "forecast" and forecast.n == 1
    assert forecast.mean_error_pct == pytest.approx(0.0)
    assert forecast.mean_abs_wind_mps == pytest.approx(0.0)
    assert post.source == "post_ride" and post.n == 1
    assert post.mean_error_pct == pytest.approx(-2.0)
    assert post.mean_abs_wind_mps is None
    assert len(store.list_post_rides()) == 1
    assert store.list_post_rides()[0].power_w == pytest.approx(245.0)


def test_grade_comparison_uses_same_gate_and_filter() -> None:
    def comparison(act: ActivitySummary) -> PastRideComparison:
        return PastRideComparison(
            activity=act,
            predicted_s=100.0,
            actual_moving_s=100.0,
            actual_elapsed_s=110.0,
            power_w=200.0,
            power_from_activity=True,
            weather_ok=True,
            weather_source="open-meteo-archive",
            delta_moving_s=0.0,
            note="",
        )

    meter = grade_comparison(comparison(_act(1)))
    assert meter is not None and meter.passed
    estimated = _act(2, device_watts=False)
    assert grade_comparison(comparison(estimated)) is None
    assert grade_comparison(comparison(_act(3, sport_type="VirtualRide"))) is None


def test_log_path_sits_beside_weather_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIKE_WIND_CACHE", str(tmp_path))
    assert default_forecast_log_path().parent == default_weather_cache_path().parent
    assert default_forecast_log_path().name == "forecast_log.sqlite"


def test_strava_device_watts_flag_is_parsed(tmp_path: Path) -> None:
    from bike_wind.strava.client import StravaClient, StravaConfig

    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = [
        {
            "id": 1,
            "name": "Outdoor",
            "sport_type": "Ride",
            "start_date": "2026-11-12T18:00:00Z",
            "distance": 20000,
            "moving_time": 3600,
            "elapsed_time": 3700,
            "average_watts": 210,
            "device_watts": True,
            "start_latlng": [27.2, -80.4],
            "map": {"summary_polyline": "abc"},
        },
        {
            "id": 2,
            "name": "Estimated",
            "sport_type": "Ride",
            "start_date": "2026-11-12T12:00:00Z",
            "distance": 20000,
            "moving_time": 3600,
            "elapsed_time": 3700,
            "average_watts": 180,
            "device_watts": False,
            "start_latlng": [27.2, -80.4],
            "map": {"summary_polyline": "abc"},
        },
    ]
    session.get.return_value = resp
    client = StravaClient(
        StravaConfig(
            access_token="t",
            client_id="1",
            client_secret="s",
            redirect_uri=None,
            refresh_token=None,
            cache_dir=tmp_path,
        ),
        session=session,
    )
    acts = client.list_recent_activities()
    assert acts[0].has_meter_power is True
    assert acts[0].has_power is True
    assert acts[1].has_power is True
    assert acts[1].has_meter_power is False
    assert accuracy_skip_reason(acts[1]) == "estimated_power"

    detail = MagicMock()
    detail.status_code = 200
    detail.json.return_value = {
        "segment_efforts": [
            {
                "id": 9,
                "elapsed_time": 500,
                "moving_time": 480,
                "start_date": "2026-11-12T18:05:00Z",
                "distance": 5000,
                "average_watts": 250,
                "device_watts": True,
                "segment": {"id": 41, "name": "Bank"},
            }
        ]
    }
    session.get.return_value = detail
    efforts = client.list_activity_segment_efforts(1)
    assert efforts[0].device_watts is True
    assert efforts[0].has_meter_power is True


def test_capture_logs_a_plan_and_scores_only_meter_rides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BIKE_WIND_CACHE", str(tmp_path))
    ss = _SS(
        route_source="segment",
        strava_meta={"segment_id": 41, "name": "Bank"},
    )
    monkeypatch.setattr("app.st.session_state", ss)
    from app import _capture_run_snapshot, _log_planned_forecast

    when = datetime.now(timezone.utc) + timedelta(hours=6)
    _capture_run_snapshot(
        _result(),
        when_dt=when,
        power_w=220.0,
        source_label="Strava: Bank",
    )
    store = ForecastLog()
    rows = store.list_forecasts()
    assert len(rows) == 1
    assert rows[0].kind == "plan"
    assert rows[0].segment_id == 41
    assert rows[0].segment_name == "Bank"
    assert rows[0].target_power_w == pytest.approx(220.0)
    assert rows[0].wind_speed_mps == pytest.approx(4.0)
    assert rows[0].gust_speed_mps == pytest.approx(6.5)
    assert rows[0].forecast_source == "open-meteo-15min"
    assert rows[0].predicted_time_s == pytest.approx(100.0)
    assert store.list_post_rides() == []
    assert "Logged the plan forecast" in ss._forecast_log_note

    # A past activity with a meter is a post-ride score, not another forecast.
    ss.route_source = "activity"
    ss.selected_activity = _act(7, moving=100, elapsed=120)
    _capture_run_snapshot(
        _result(used_ride_start_time=True),
        when_dt=datetime(2026, 11, 1, tzinfo=timezone.utc),
        power_w=210.0,
        source_label="Strava ride",
    )
    assert len(store.list_forecasts()) == 1
    posts = store.list_post_rides()
    assert len(posts) == 1
    assert posts[0].activity_id == 7
    assert posts[0].segment_id == 0
    assert posts[0].actual_time_s == pytest.approx(100.0)
    assert posts[0].passed is True

    # Estimated power is not a post-ride score. has_power would still be true.
    ss.selected_activity = _act(8, device_watts=False, watts=200.0, moving=100)
    _capture_run_snapshot(
        _result(used_ride_start_time=True),
        when_dt=datetime(2026, 11, 1, tzinfo=timezone.utc),
        power_w=200.0,
        source_label="Strava ride",
    )
    assert len(store.list_post_rides()) == 1
    assert ss.selected_activity.has_power is True

    # Segment effort uses elapsed time, and a virtual ride is ignored.
    ss.route_source = "effort"
    ss.selected_activity = _act(9, sport_type="VirtualRide", name="Zwift")
    ss.selected_effort = _effort(4, 41, elapsed=110, moving=90)
    _capture_run_snapshot(
        _result(used_ride_start_time=True),
        when_dt=datetime(2026, 11, 1, tzinfo=timezone.utc),
        power_w=250.0,
        source_label="effort",
    )
    assert len(store.list_post_rides()) == 1

    ss.selected_activity = _act(10)
    ss.selected_effort = _effort(4, 41, elapsed=110, moving=90)
    _capture_run_snapshot(
        _result(used_ride_start_time=True),
        when_dt=datetime(2026, 11, 1, tzinfo=timezone.utc),
        power_w=250.0,
        source_label="effort",
    )
    effort_row = next(row for row in store.list_post_rides() if row.activity_id == 10)
    assert effort_row.actual_time_s == pytest.approx(110.0)
    assert effort_row.segment_id == 41
    assert effort_row.passed is False

    # Plan / Use KOM logs only while the sliders are still the forecast.
    ss.route_source = "segment"
    before = len(store.list_forecasts())
    _log_planned_forecast(
        _result(),
        power_w=230.0,
        wind_speed_mps=4.0,
        grade_scale=1.0,
        when_utc=when.isoformat(),
        predicted_s=95.0,
        raw_target="1:23",
        kom_s=83.0,
        source_label="Strava: Bank",
    )
    kom_rows = [row for row in store.list_forecasts() if row.kind == "kom"]
    assert len(store.list_forecasts()) == before + 1
    assert len(kom_rows) == 1
    assert kom_rows[0].target_power_w == pytest.approx(230.0)
    assert kom_rows[0].predicted_time_s == pytest.approx(95.0)

    _log_planned_forecast(
        _result(),
        power_w=230.0,
        wind_speed_mps=12.0,
        grade_scale=1.0,
        when_utc=when.isoformat(),
        predicted_s=140.0,
        raw_target="2:00",
        kom_s=None,
        source_label="Strava: Bank",
    )
    assert len(store.list_forecasts()) == before + 1
