"""Pre-ride forecast log and the segment-time scores that come back later.

A plan or a start stores the forecast as it stood then, plus the prediction
made from it. After the ride, each row is matched to a Strava activity or
segment effort and graded with ``score_segment_time`` — the same gate as a
post-ride check. Forecast rows and post-ride rows stay in separate tables so
a wrong forecast is not mixed into the model error.

Storage is SQLite next to the weather cache (``BIKE_WIND_CACHE``, otherwise
``.cache/weather/forecast_log.sqlite``). On Streamlit Community Cloud that
disk is ephemeral: it survives reruns on the same container and is wiped on
reboot or redeploy, and replicas do not share it. The file holds no tokens.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bike_wind.analysis import format_duration
from bike_wind.calibration.metrics import (
    SEGMENT_TIME_PASS_PCT,
    direction_error_deg,
    score_segment_time,
)
from bike_wind.ride_filter import accuracy_skip_reason
from bike_wind.strava.models import ActivitySummary, SegmentEffortSummary
from bike_wind.weather.cache import default_weather_cache_path

# Activity start must fall within this of the planned start.
MATCH_WINDOW_S = 12 * 3600
# Whole-ride match (no segment id) also has to start near the logged pin.
LOCATION_KM = 30.0
# A second click with the same snapshot does not insert another row.
DEDUPE_S = 15 * 60

PERSISTENCE_NOTE = (
    "Saved in SQLite next to the weather cache. On Streamlit Community Cloud "
    "that disk is wiped when the app reboots or redeploys, and it is not shared "
    "across replicas. Set BIKE_WIND_CACHE to a persistent folder if you have one. "
    "The log stores no Strava tokens."
)


def default_forecast_log_path() -> Path:
    """Sibling of the weather SQLite file, same directory and cache root."""
    return default_weather_cache_path().with_name("forecast_log.sqlite")


@dataclass(frozen=True)
class ForecastEntry:
    id: int
    issued_at: str
    target_time: str
    lat: float
    lon: float
    wind_speed_mps: float
    wind_from_deg: float
    gust_speed_mps: float | None
    forecast_source: str
    segment_id: int | None
    segment_name: str
    target_power_w: float
    predicted_time_s: float
    kind: str  # "ride" | "plan" | "kom"
    activity_id: int | None = None
    effort_id: int | None = None
    actual_time_s: float | None = None
    time_error_s: float | None = None
    time_error_pct: float | None = None
    passed: bool | None = None
    actual_wind_speed_mps: float | None = None
    actual_wind_from_deg: float | None = None
    wind_speed_error_mps: float | None = None
    wind_dir_error_deg: float | None = None
    matched_kind: str | None = None  # "segment" | "ride"
    scored_at: str | None = None


@dataclass(frozen=True)
class PostRideEntry:
    id: int
    activity_id: int
    segment_id: int
    segment_name: str
    checked_at: str
    predicted_time_s: float
    actual_time_s: float
    time_error_s: float
    time_error_pct: float
    passed: bool
    power_w: float | None = None


@dataclass(frozen=True)
class AccuracySummary:
    """One error source. Forecast and post-ride summaries are never combined."""

    source: str  # "forecast" | "post_ride"
    n: int
    n_pass: int
    mean_error_pct: float
    mae_pct: float
    mean_abs_wind_mps: float | None = None
    mean_abs_wind_dir_deg: float | None = None


def _parse_utc(text: str) -> datetime:
    raw = (text or "").strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(when: datetime) -> str:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    else:
        when = when.astimezone(timezone.utc)
    return when.replace(microsecond=0).isoformat()


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def _activity_start(act: ActivitySummary) -> datetime | None:
    if not act.start_date:
        return None
    try:
        return _parse_utc(act.start_date)
    except ValueError:
        return None


def _effort_start(effort: SegmentEffortSummary, act: ActivitySummary) -> datetime | None:
    if effort.start_date:
        try:
            return _parse_utc(effort.start_date)
        except ValueError:
            return _activity_start(act)
    return _activity_start(act)


class ForecastLog:
    """SQLite log of forecast snapshots and post-ride segment-time scores."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else default_forecast_log_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    issued_at TEXT NOT NULL,
                    target_time TEXT NOT NULL,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    wind_speed_mps REAL NOT NULL,
                    wind_from_deg REAL NOT NULL,
                    gust_speed_mps REAL,
                    forecast_source TEXT NOT NULL,
                    segment_id INTEGER,
                    segment_name TEXT NOT NULL,
                    target_power_w REAL NOT NULL,
                    predicted_time_s REAL NOT NULL,
                    kind TEXT NOT NULL,
                    activity_id INTEGER,
                    effort_id INTEGER,
                    actual_time_s REAL,
                    time_error_s REAL,
                    time_error_pct REAL,
                    passed INTEGER,
                    actual_wind_speed_mps REAL,
                    actual_wind_from_deg REAL,
                    wind_speed_error_mps REAL,
                    wind_dir_error_deg REAL,
                    matched_kind TEXT,
                    scored_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS post_ride_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    activity_id INTEGER NOT NULL,
                    segment_id INTEGER NOT NULL,
                    segment_name TEXT NOT NULL,
                    checked_at TEXT NOT NULL,
                    predicted_time_s REAL NOT NULL,
                    actual_time_s REAL NOT NULL,
                    time_error_s REAL NOT NULL,
                    time_error_pct REAL NOT NULL,
                    passed INTEGER NOT NULL,
                    power_w REAL,
                    UNIQUE(activity_id, segment_id)
                )
                """
            )

    def log_forecast(
        self,
        *,
        issued_at: datetime,
        target_time: datetime,
        lat: float,
        lon: float,
        wind_speed_mps: float,
        wind_from_deg: float,
        gust_speed_mps: float | None,
        forecast_source: str,
        segment_id: int | None,
        segment_name: str,
        target_power_w: float,
        predicted_time_s: float,
        kind: str,
    ) -> int | None:
        """Insert one snapshot. Returns the row id, or None when it is a duplicate.

        A Kom click soon after the same snapshot updates ``kind`` to ``kom``
        and returns that existing id.
        """
        if kind not in {"ride", "plan", "kom"}:
            raise ValueError(f"unknown forecast kind {kind!r}")
        speed = _finite(wind_speed_mps)
        direction = _finite(wind_from_deg)
        power = _finite(target_power_w)
        predicted = _finite(predicted_time_s)
        lat_f = _finite(lat)
        lon_f = _finite(lon)
        if None in (speed, direction, power, predicted, lat_f, lon_f):
            return None
        assert speed is not None and direction is not None
        assert power is not None and predicted is not None
        assert lat_f is not None and lon_f is not None
        if predicted <= 0.0:
            return None
        issued = issued_at if issued_at.tzinfo else issued_at.replace(tzinfo=timezone.utc)
        target = target_time if target_time.tzinfo else target_time.replace(tzinfo=timezone.utc)
        for row in self.list_forecasts():
            try:
                age = abs((_parse_utc(row.issued_at) - issued).total_seconds())
            except ValueError:
                continue
            if age > DEDUPE_S:
                continue
            if not _same_snapshot(
                row,
                target_time=target,
                segment_id=segment_id,
                power_w=power,
                predicted_s=predicted,
                lat=lat_f,
                lon=lon_f,
            ):
                continue
            if kind == "kom" and row.kind != "kom":
                self.update_forecast(replace(row, kind="kom"))
                return row.id
            return None
        gust = _finite(gust_speed_mps)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO forecast_logs (
                    issued_at, target_time, lat, lon,
                    wind_speed_mps, wind_from_deg, gust_speed_mps, forecast_source,
                    segment_id, segment_name, target_power_w, predicted_time_s, kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _iso(issued),
                    _iso(target),
                    lat_f,
                    lon_f,
                    speed,
                    direction,
                    gust,
                    forecast_source,
                    segment_id,
                    segment_name,
                    power,
                    predicted,
                    kind,
                ),
            )
            return int(cur.lastrowid or 0)

    def list_forecasts(self) -> list[ForecastEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM forecast_logs ORDER BY issued_at DESC, id DESC"
            ).fetchall()
        return [_forecast_from_row(row) for row in rows]

    def pending_forecasts(self) -> list[ForecastEntry]:
        return [row for row in self.list_forecasts() if row.actual_time_s is None]

    def update_forecast(self, row: ForecastEntry) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE forecast_logs SET
                    kind=?,
                    activity_id=?,
                    effort_id=?,
                    actual_time_s=?,
                    time_error_s=?,
                    time_error_pct=?,
                    passed=?,
                    actual_wind_speed_mps=?,
                    actual_wind_from_deg=?,
                    wind_speed_error_mps=?,
                    wind_dir_error_deg=?,
                    matched_kind=?,
                    scored_at=?
                WHERE id=?
                """,
                (
                    row.kind,
                    row.activity_id,
                    row.effort_id,
                    row.actual_time_s,
                    row.time_error_s,
                    row.time_error_pct,
                    None if row.passed is None else int(row.passed),
                    row.actual_wind_speed_mps,
                    row.actual_wind_from_deg,
                    row.wind_speed_error_mps,
                    row.wind_dir_error_deg,
                    row.matched_kind,
                    row.scored_at,
                    row.id,
                ),
            )

    def upsert_post_ride(
        self,
        *,
        activity_id: int,
        segment_id: int,
        segment_name: str,
        predicted_time_s: float,
        actual_time_s: float,
        time_error_s: float,
        time_error_pct: float,
        passed: bool,
        power_w: float | None = None,
        checked_at: datetime | None = None,
    ) -> None:
        """One post-ride score per activity and segment. A rerun replaces it."""
        when = checked_at or datetime.now(timezone.utc)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO post_ride_checks (
                    activity_id, segment_id, segment_name, checked_at,
                    predicted_time_s, actual_time_s, time_error_s, time_error_pct,
                    passed, power_w
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(activity_id, segment_id) DO UPDATE SET
                    segment_name=excluded.segment_name,
                    checked_at=excluded.checked_at,
                    predicted_time_s=excluded.predicted_time_s,
                    actual_time_s=excluded.actual_time_s,
                    time_error_s=excluded.time_error_s,
                    time_error_pct=excluded.time_error_pct,
                    passed=excluded.passed,
                    power_w=excluded.power_w
                """,
                (
                    int(activity_id),
                    int(segment_id),
                    segment_name,
                    _iso(when),
                    float(predicted_time_s),
                    float(actual_time_s),
                    float(time_error_s),
                    float(time_error_pct),
                    int(passed),
                    _finite(power_w),
                ),
            )

    def list_post_rides(self) -> list[PostRideEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM post_ride_checks ORDER BY checked_at DESC, id DESC"
            ).fetchall()
        return [_post_from_row(row) for row in rows]


def _same_snapshot(
    row: ForecastEntry,
    *,
    target_time: datetime,
    segment_id: int | None,
    power_w: float,
    predicted_s: float,
    lat: float,
    lon: float,
) -> bool:
    if row.segment_id != segment_id:
        return False
    if abs(row.target_power_w - power_w) > 1.0:
        return False
    if abs(row.predicted_time_s - predicted_s) > 1.0:
        return False
    try:
        if abs((_parse_utc(row.target_time) - target_time).total_seconds()) > 60:
            return False
    except ValueError:
        return False
    return _km(row.lat, row.lon, lat, lon) <= 1.0


def _opt_float(row: sqlite3.Row, key: str) -> float | None:
    value = row[key]
    return None if value is None else float(value)


def _opt_int(row: sqlite3.Row, key: str) -> int | None:
    value = row[key]
    return None if value is None else int(value)


def _forecast_from_row(row: sqlite3.Row) -> ForecastEntry:
    passed = row["passed"]
    return ForecastEntry(
        id=int(row["id"]),
        issued_at=str(row["issued_at"]),
        target_time=str(row["target_time"]),
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        wind_speed_mps=float(row["wind_speed_mps"]),
        wind_from_deg=float(row["wind_from_deg"]),
        gust_speed_mps=_opt_float(row, "gust_speed_mps"),
        forecast_source=str(row["forecast_source"]),
        segment_id=_opt_int(row, "segment_id"),
        segment_name=str(row["segment_name"]),
        target_power_w=float(row["target_power_w"]),
        predicted_time_s=float(row["predicted_time_s"]),
        kind=str(row["kind"]),
        activity_id=_opt_int(row, "activity_id"),
        effort_id=_opt_int(row, "effort_id"),
        actual_time_s=_opt_float(row, "actual_time_s"),
        time_error_s=_opt_float(row, "time_error_s"),
        time_error_pct=_opt_float(row, "time_error_pct"),
        passed=None if passed is None else bool(passed),
        actual_wind_speed_mps=_opt_float(row, "actual_wind_speed_mps"),
        actual_wind_from_deg=_opt_float(row, "actual_wind_from_deg"),
        wind_speed_error_mps=_opt_float(row, "wind_speed_error_mps"),
        wind_dir_error_deg=_opt_float(row, "wind_dir_error_deg"),
        matched_kind=None if row["matched_kind"] is None else str(row["matched_kind"]),
        scored_at=None if row["scored_at"] is None else str(row["scored_at"]),
    )


def _post_from_row(row: sqlite3.Row) -> PostRideEntry:
    return PostRideEntry(
        id=int(row["id"]),
        activity_id=int(row["activity_id"]),
        segment_id=int(row["segment_id"]),
        segment_name=str(row["segment_name"]),
        checked_at=str(row["checked_at"]),
        predicted_time_s=float(row["predicted_time_s"]),
        actual_time_s=float(row["actual_time_s"]),
        time_error_s=float(row["time_error_s"]),
        time_error_pct=float(row["time_error_pct"]),
        passed=bool(row["passed"]),
        power_w=_opt_float(row, "power_w"),
    )


WindLookup = Callable[[float, float, datetime], tuple[float, float] | None]


def match_forecasts(
    entries: Sequence[ForecastEntry],
    activities: Sequence[ActivitySummary],
    efforts_by_activity: Mapping[int, Sequence[SegmentEffortSummary]],
    actual_wind: WindLookup | None = None,
) -> list[ForecastEntry]:
    """Attach actual segment time and wind error where a power-meter ride matches.

    Already scored rows are returned as they are. Virtual, Zwift, trainer, and
    estimated-power rides are not matches, and a row with no eligible ride
    stays unscored so a later power-meter ride can still match.
    """
    matched: list[ForecastEntry] = []
    for entry in entries:
        if entry.actual_time_s is not None:
            matched.append(entry)
            continue
        matched.append(_match_one(entry, activities, efforts_by_activity, actual_wind))
    return matched


def _match_one(
    entry: ForecastEntry,
    activities: Sequence[ActivitySummary],
    efforts_by_activity: Mapping[int, Sequence[SegmentEffortSummary]],
    actual_wind: WindLookup | None,
) -> ForecastEntry:
    try:
        target = _parse_utc(entry.target_time)
    except ValueError:
        return entry
    if entry.segment_id is not None:
        hits: list[tuple[float, ActivitySummary, SegmentEffortSummary]] = []
        for act in activities:
            for effort in efforts_by_activity.get(act.activity_id, ()):
                if int(effort.segment_id) != int(entry.segment_id):
                    continue
                start = _effort_start(effort, act)
                if start is None:
                    continue
                gap = abs((start - target).total_seconds())
                if gap > MATCH_WINDOW_S:
                    continue
                hits.append((gap, act, effort))
        hits.sort(key=lambda item: item[0])
        for _gap, act, effort in hits:
            if accuracy_skip_reason(act, effort) is not None:
                continue
            start = _effort_start(effort, act)
            assert start is not None
            scored = _score(entry, act, effort, start, actual_wind, matched_kind="segment")
            if scored.actual_time_s is not None:
                return scored
        return entry

    ride_hits: list[tuple[float, ActivitySummary]] = []
    for act in activities:
        start = _activity_start(act)
        if start is None or act.start_lat is None or act.start_lon is None:
            continue
        gap = abs((start - target).total_seconds())
        if gap > MATCH_WINDOW_S:
            continue
        if _km(act.start_lat, act.start_lon, entry.lat, entry.lon) > LOCATION_KM:
            continue
        ride_hits.append((gap, act))
    ride_hits.sort(key=lambda item: item[0])
    for _gap, act in ride_hits:
        if accuracy_skip_reason(act) is not None:
            continue
        start = _activity_start(act)
        assert start is not None
        scored = _score(entry, act, None, start, actual_wind, matched_kind="ride")
        if scored.actual_time_s is not None:
            return scored
    return entry


def _score(
    entry: ForecastEntry,
    act: ActivitySummary,
    effort: SegmentEffortSummary | None,
    start: datetime,
    actual_wind: WindLookup | None,
    *,
    matched_kind: str,
) -> ForecastEntry:
    if effort is not None:
        actual = float(effort.elapsed_time_s or effort.moving_time_s)
    else:
        actual = float(act.moving_time_s)
    grade = score_segment_time(entry.predicted_time_s, actual)
    if grade is None:
        return entry
    actual_speed = None
    actual_dir = None
    speed_error = None
    dir_error = None
    if actual_wind is not None:
        observed = actual_wind(entry.lat, entry.lon, start)
        if observed is not None:
            actual_speed, actual_dir = observed
            speed_error = float(entry.wind_speed_mps) - float(actual_speed)
            dir_error = direction_error_deg(entry.wind_from_deg, actual_dir)
    return replace(
        entry,
        activity_id=act.activity_id,
        effort_id=None if effort is None else effort.effort_id,
        actual_time_s=grade.actual_s,
        time_error_s=grade.error_s,
        time_error_pct=grade.error_pct,
        passed=grade.passed,
        actual_wind_speed_mps=actual_speed,
        actual_wind_from_deg=actual_dir,
        wind_speed_error_mps=speed_error,
        wind_dir_error_deg=dir_error,
        matched_kind=matched_kind,
        scored_at=_iso(datetime.now(timezone.utc)),
    )


def target_dates(entries: Sequence[ForecastEntry]) -> list:
    """UTC dates to ask Strava for, including the 12-hour match window edges."""
    days = set()
    for entry in entries:
        try:
            target = _parse_utc(entry.target_time)
        except ValueError:
            continue
        for shift in (-12, 0, 12):
            days.add((target + timedelta(hours=shift)).date())
    return sorted(days)


def summarize_forecast(rows: Sequence[ForecastEntry]) -> AccuracySummary:
    scored = [
        (row.time_error_pct, bool(row.passed), row.wind_speed_error_mps, row.wind_dir_error_deg)
        for row in rows
        if row.time_error_pct is not None and row.passed is not None
    ]
    return _summarize("forecast", scored)


def summarize_post_rides(rows: Sequence[PostRideEntry]) -> AccuracySummary:
    scored = [(row.time_error_pct, row.passed, None, None) for row in rows]
    return _summarize("post_ride", scored)


def _summarize(
    source: str,
    items: Sequence[tuple[float, bool, float | None, float | None]],
) -> AccuracySummary:
    if not items:
        return AccuracySummary(source, 0, 0, 0.0, 0.0, None, None)
    errors = [item[0] for item in items]
    winds = [abs(item[2]) for item in items if item[2] is not None]
    directions = [abs(item[3]) for item in items if item[3] is not None]
    return AccuracySummary(
        source=source,
        n=len(items),
        n_pass=sum(1 for item in items if item[1]),
        mean_error_pct=sum(errors) / len(errors),
        mae_pct=sum(abs(err) for err in errors) / len(errors),
        mean_abs_wind_mps=(sum(winds) / len(winds)) if winds else None,
        mean_abs_wind_dir_deg=(sum(directions) / len(directions)) if directions else None,
    )


def summary_line(summary: AccuracySummary) -> str:
    label = "Forecast" if summary.source == "forecast" else "Post-ride"
    if summary.n == 0:
        return f"{label}: no scored rides yet."
    line = (
        f"{label}: {summary.n_pass} of {summary.n} within "
        f"±{SEGMENT_TIME_PASS_PCT:.0f}% segment time. "
        f"Mean error {summary.mean_error_pct:+.1f}% (MAE {summary.mae_pct:.1f}%)."
    )
    if summary.mean_abs_wind_mps is not None:
        mph = summary.mean_abs_wind_mps * 2.236936
        line += f" Mean |wind| error {mph:.1f} mph"
        if summary.mean_abs_wind_dir_deg is not None:
            line += f", {summary.mean_abs_wind_dir_deg:.0f}°"
        line += "."
    return line


def forecast_row_line(row: ForecastEntry) -> str:
    when = row.target_time[:16].replace("T", " ")
    base = (
        f"{row.segment_name} · {row.kind} · {when} UTC · {row.target_power_w:.0f} W · "
        f"predicted {format_duration(row.predicted_time_s)}"
    )
    if row.actual_time_s is None or row.time_error_pct is None:
        return base + " · waiting for a power-meter ride"
    flag = "pass" if row.passed else "fail"
    wind = ""
    if row.wind_speed_error_mps is not None:
        wind = f" · wind {row.wind_speed_error_mps * 2.236936:+.1f} mph"
        if row.wind_dir_error_deg is not None:
            wind += f", {row.wind_dir_error_deg:+.0f}°"
    return (
        f"{base} · actual {format_duration(row.actual_time_s)} · "
        f"{row.time_error_pct:+.1f}% {flag}{wind}"
    )


def post_ride_row_line(row: PostRideEntry) -> str:
    flag = "pass" if row.passed else "fail"
    return (
        f"{row.segment_name} · predicted {format_duration(row.predicted_time_s)} · "
        f"actual {format_duration(row.actual_time_s)} · {row.time_error_pct:+.1f}% {flag}"
    )


def fetch_actual_wind(
    lat: float,
    lon: float,
    when: datetime,
) -> tuple[float, float] | None:
    """Archive wind at one point, scaled the same way the logged forecast is.

    Returns ``(speed_mps, from_deg)`` at rider height after ``WIND_SCALE``,
    matching ``PredictionResult.mean_wind_speed_mps``. None when the archive
    cannot be read. Does not change the prediction.
    """
    try:
        from bike_wind.calibration.cda_defaults import WIND_SCALE
        from bike_wind.local_wind_map import wind_from_uv
        from bike_wind.weather.fetch import fetch_weather_along_route

        wx = fetch_weather_along_route(
            [lat],
            [lon],
            [0.0],
            when,
            use_archive=True,
            max_samples=1,
            prefer_hrrr=False,
        )
    except Exception:
        return None
    if not wx.points:
        return None
    point = wx.points[0]
    speed, direction = wind_from_uv(
        point.u_rider_mps * WIND_SCALE,
        point.v_rider_mps * WIND_SCALE,
    )
    if not (math.isfinite(speed) and math.isfinite(direction)):
        return None
    return speed, direction
