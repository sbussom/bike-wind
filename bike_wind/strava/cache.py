"""SQLite cache for Strava segment details + streams (no LLM logging)."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bike_wind.gpx.parse import RouteStreams


@dataclass
class CachedSegment:
    segment_id: int
    name: str
    distance_m: float
    average_grade: float
    elev_low: float | None
    elev_high: float | None
    fetched_at: float
    route: RouteStreams


class SegmentCache:
    """Cache keyed by segment ID with fetch timestamp."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
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
                CREATE TABLE IF NOT EXISTS segments (
                    segment_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    distance_m REAL NOT NULL,
                    average_grade REAL NOT NULL,
                    elev_low REAL,
                    elev_high REAL,
                    fetched_at REAL NOT NULL,
                    lats_json TEXT NOT NULL,
                    lons_json TEXT NOT NULL,
                    elevs_json TEXT NOT NULL,
                    distances_json TEXT NOT NULL
                )
                """
            )

    def get(self, segment_id: int, *, max_age_s: float | None = None) -> CachedSegment | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM segments WHERE segment_id = ?",
                (segment_id,),
            ).fetchone()
        if row is None:
            return None
        if max_age_s is not None and (time.time() - float(row["fetched_at"])) > max_age_s:
            return None
        route = RouteStreams(
            lats=json.loads(row["lats_json"]),
            lons=json.loads(row["lons_json"]),
            elevs_m=json.loads(row["elevs_json"]),
            distances_m=json.loads(row["distances_json"]),
        )
        return CachedSegment(
            segment_id=int(row["segment_id"]),
            name=str(row["name"]),
            distance_m=float(row["distance_m"]),
            average_grade=float(row["average_grade"]),
            elev_low=row["elev_low"],
            elev_high=row["elev_high"],
            fetched_at=float(row["fetched_at"]),
            route=route,
        )

    def put(
        self,
        *,
        segment_id: int,
        name: str,
        distance_m: float,
        average_grade: float,
        elev_low: float | None,
        elev_high: float | None,
        route: RouteStreams,
        fetched_at: float | None = None,
    ) -> None:
        fetched_at = time.time() if fetched_at is None else fetched_at
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO segments (
                    segment_id, name, distance_m, average_grade,
                    elev_low, elev_high, fetched_at,
                    lats_json, lons_json, elevs_json, distances_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(segment_id) DO UPDATE SET
                    name=excluded.name,
                    distance_m=excluded.distance_m,
                    average_grade=excluded.average_grade,
                    elev_low=excluded.elev_low,
                    elev_high=excluded.elev_high,
                    fetched_at=excluded.fetched_at,
                    lats_json=excluded.lats_json,
                    lons_json=excluded.lons_json,
                    elevs_json=excluded.elevs_json,
                    distances_json=excluded.distances_json
                """,
                (
                    segment_id,
                    name,
                    distance_m,
                    average_grade,
                    elev_low,
                    elev_high,
                    fetched_at,
                    json.dumps(route.lats),
                    json.dumps(route.lons),
                    json.dumps(route.elevs_m),
                    json.dumps(route.distances_m),
                ),
            )

    def meta_dict(self, cached: CachedSegment) -> dict[str, Any]:
        """Safe summary for UI — no raw stream dumps."""
        return {
            "segment_id": cached.segment_id,
            "name": cached.name,
            "distance_m": cached.distance_m,
            "average_grade": cached.average_grade,
            "points": len(cached.route),
            "fetched_at": cached.fetched_at,
        }
