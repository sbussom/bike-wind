"""Local Strava rate-limit ledger (200/15 min, 2000/day)."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

WINDOW_15M_S = 15 * 60
WINDOW_DAY_S = 24 * 60 * 60
LIMIT_15M = 200
LIMIT_DAY = 2000


@dataclass(frozen=True)
class RateLimitStatus:
    used_15m: int
    used_day: int
    remaining_15m: int
    remaining_day: int
    limited: bool
    retry_after_s: float
    message: str | None


class RateLimiter:
    """Track API calls in SQLite; never stores response payloads."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_calls_ts ON api_calls(ts)")

    def status(self, *, now: float | None = None) -> RateLimitStatus:
        now = time.time() if now is None else now
        with self._connect() as conn:
            used_15m = conn.execute(
                "SELECT COUNT(*) FROM api_calls WHERE ts >= ?",
                (now - WINDOW_15M_S,),
            ).fetchone()[0]
            used_day = conn.execute(
                "SELECT COUNT(*) FROM api_calls WHERE ts >= ?",
                (now - WINDOW_DAY_S,),
            ).fetchone()[0]
            # Prune older than 1 day
            conn.execute("DELETE FROM api_calls WHERE ts < ?", (now - WINDOW_DAY_S,))

        rem_15 = max(0, LIMIT_15M - used_15m)
        rem_day = max(0, LIMIT_DAY - used_day)
        limited = rem_15 <= 0 or rem_day <= 0
        retry = 0.0
        msg = None
        if rem_15 <= 0:
            with self._connect() as conn:
                oldest = conn.execute(
                    "SELECT MIN(ts) FROM api_calls WHERE ts >= ?",
                    (now - WINDOW_15M_S,),
                ).fetchone()[0]
            if oldest is not None:
                retry = max(retry, oldest + WINDOW_15M_S - now)
            msg = (
                f"Strava 15-minute rate limit reached ({LIMIT_15M}/15 min). "
                f"Retry in ~{int(retry)}s."
            )
        elif rem_day <= 0:
            with self._connect() as conn:
                oldest = conn.execute(
                    "SELECT MIN(ts) FROM api_calls WHERE ts >= ?",
                    (now - WINDOW_DAY_S,),
                ).fetchone()[0]
            if oldest is not None:
                retry = max(retry, oldest + WINDOW_DAY_S - now)
            msg = (
                f"Strava daily rate limit reached ({LIMIT_DAY}/day). "
                f"Retry in ~{int(retry // 60)} min."
            )
        return RateLimitStatus(
            used_15m=used_15m,
            used_day=used_day,
            remaining_15m=rem_15,
            remaining_day=rem_day,
            limited=limited,
            retry_after_s=retry,
            message=msg,
        )

    def check(self, *, now: float | None = None) -> RateLimitStatus:
        st = self.status(now=now)
        if st.limited:
            raise RateLimitExceeded(st)
        return st

    def record(self, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._connect() as conn:
            conn.execute("INSERT INTO api_calls (ts) VALUES (?)", (now,))


class RateLimitExceeded(RuntimeError):
    def __init__(self, status: RateLimitStatus) -> None:
        self.status = status
        super().__init__(status.message or "Strava rate limit exceeded")
