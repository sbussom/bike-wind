"""SQLite cache for forecast points keyed by (lat, lon, forecast_ts, source)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from bike_wind.settings import get_setting

REPO_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "weather"


@dataclass(frozen=True)
class CachedForecastRaw:
    """Raw 10 m wind + thermo at a point/hour (before log-profile / bearing)."""

    lat: float
    lon: float
    forecast_ts: str  # ISO hour UTC
    source: str
    temperature_c: float
    pressure_hpa: float
    u10_mps: float
    v10_mps: float


def default_weather_cache_path() -> Path:
    raw = get_setting("BIKE_WIND_CACHE")
    base = Path(raw or REPO_CACHE)
    # If BIKE_WIND_CACHE points at strava dir parent, keep weather sibling
    if base.name == "strava":
        return base.parent / "weather" / "forecasts.sqlite"
    return base / "forecasts.sqlite" if base.suffix != ".sqlite" else base


class ForecastCache:
    """Cache Open-Meteo / HRRR hourly samples. Reuses BIKE_WIND_CACHE root."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else default_weather_cache_path()
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
                CREATE TABLE IF NOT EXISTS forecasts (
                    lat_key REAL NOT NULL,
                    lon_key REAL NOT NULL,
                    forecast_ts TEXT NOT NULL,
                    source TEXT NOT NULL,
                    temperature_c REAL NOT NULL,
                    pressure_hpa REAL NOT NULL,
                    u10_mps REAL NOT NULL,
                    v10_mps REAL NOT NULL,
                    fetched_at REAL NOT NULL,
                    PRIMARY KEY (lat_key, lon_key, forecast_ts, source)
                )
                """
            )

    @staticmethod
    def round_coord(value: float, ndigits: int = 3) -> float:
        """~100 m grid — enough for wind cache without exploding keys."""
        return round(float(value), ndigits)

    @staticmethod
    def hour_key(when: datetime) -> str:
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        else:
            when = when.astimezone(timezone.utc)
        return when.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")

    @staticmethod
    def quarter_key(when: datetime) -> str:
        """Round down to :00 / :15 / :30 / :45 UTC for 15-min cache slots."""
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        else:
            when = when.astimezone(timezone.utc)
        minute = (when.minute // 15) * 15
        return when.replace(minute=minute, second=0, microsecond=0).strftime(
            "%Y-%m-%dT%H:%M"
        )

    def get(
        self,
        lat: float,
        lon: float,
        forecast_ts: str,
        source: str,
    ) -> CachedForecastRaw | None:
        lat_k = self.round_coord(lat)
        lon_k = self.round_coord(lon)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM forecasts
                WHERE lat_key=? AND lon_key=? AND forecast_ts=? AND source=?
                """,
                (lat_k, lon_k, forecast_ts, source),
            ).fetchone()
        if row is None:
            return None
        return CachedForecastRaw(
            lat=lat_k,
            lon=lon_k,
            forecast_ts=str(row["forecast_ts"]),
            source=str(row["source"]),
            temperature_c=float(row["temperature_c"]),
            pressure_hpa=float(row["pressure_hpa"]),
            u10_mps=float(row["u10_mps"]),
            v10_mps=float(row["v10_mps"]),
        )

    def put(self, raw: CachedForecastRaw, *, fetched_at: float | None = None) -> None:
        import time

        fetched_at = time.time() if fetched_at is None else fetched_at
        lat_k = self.round_coord(raw.lat)
        lon_k = self.round_coord(raw.lon)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO forecasts (
                    lat_key, lon_key, forecast_ts, source,
                    temperature_c, pressure_hpa, u10_mps, v10_mps, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(lat_key, lon_key, forecast_ts, source) DO UPDATE SET
                    temperature_c=excluded.temperature_c,
                    pressure_hpa=excluded.pressure_hpa,
                    u10_mps=excluded.u10_mps,
                    v10_mps=excluded.v10_mps,
                    fetched_at=excluded.fetched_at
                """,
                (
                    lat_k,
                    lon_k,
                    raw.forecast_ts,
                    raw.source,
                    raw.temperature_c,
                    raw.pressure_hpa,
                    raw.u10_mps,
                    raw.v10_mps,
                    fetched_at,
                ),
            )
