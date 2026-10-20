"""Strava API client: segments, streams, starred, explore."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import requests

from bike_wind.gpx.parse import RouteStreams
from bike_wind.settings import get_setting
from bike_wind.strava.cache import CachedSegment, SegmentCache
from bike_wind.strava import errors as strava_errors
from bike_wind.strava.models import ActivitySummary
from bike_wind.strava.oauth import OAuthTokenError, OAuthTokens, refresh_access_token
from bike_wind.strava.rate_limit import RateLimitExceeded, RateLimiter, RateLimitStatus
from bike_wind.strava.urls import ParsedStravaRef, parse_strava_segment_ref

API_BASE = "https://www.strava.com/api/v3"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "bike_wind"
# Prefer project-local cache when running from repo
REPO_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "strava"

# Back-compat alias for callers/tests that imported the old hint.
TOKEN_RECOVERY_HINT = strava_errors.REFRESH_FAILED


@dataclass(frozen=True)
class StravaConfig:
    access_token: str | None
    client_id: str | None
    client_secret: str | None
    redirect_uri: str | None
    refresh_token: str | None
    cache_dir: Path

    @property
    def oauth_ready(self) -> bool:
        return bool(self.client_id and self.client_secret and self.redirect_uri)

    @property
    def token_ready(self) -> bool:
        return bool(self.access_token)

    @property
    def refresh_ready(self) -> bool:
        return bool(self.client_id and self.client_secret and self.refresh_token)


@dataclass(frozen=True)
class SegmentMeta:
    segment_id: int
    name: str
    distance_m: float
    average_grade: float
    elev_low: float | None = None
    elev_high: float | None = None
    from_cache: bool = False
    start_lat: float | None = None
    start_lon: float | None = None


@dataclass(frozen=True)
class StarredSegment:
    segment_id: int
    name: str
    distance_m: float
    average_grade: float


@dataclass(frozen=True)
class ExploreHit:
    segment_id: int
    name: str
    distance_m: float
    average_grade: float
    climb_category: int
    starred: bool
    start_lat: float | None = None
    start_lon: float | None = None
    end_lat: float | None = None
    end_lon: float | None = None


class StravaError(RuntimeError):
    pass


class StravaAuthError(StravaError):
    """Auth/permission failure with a stable ``kind`` for UI / screenshots."""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "auth",
        diag: dict[str, Any] | None = None,
    ) -> None:
        self.kind = kind
        self.diag = diag or {}
        super().__init__(message)


class StravaRateLimitError(StravaError):
    def __init__(self, status: RateLimitStatus) -> None:
        self.status = status
        msg = status.message or strava_errors.RATE_LIMIT
        super().__init__(msg)


def load_strava_config(*, cache_dir: Path | None = None) -> StravaConfig:
    """Load Strava settings from env, then Streamlit secrets (same key names)."""
    cache_raw = get_setting("BIKE_WIND_CACHE")
    cache = cache_dir or Path(cache_raw or REPO_CACHE)
    return StravaConfig(
        access_token=get_setting("STRAVA_ACCESS_TOKEN"),
        client_id=get_setting("STRAVA_CLIENT_ID"),
        client_secret=get_setting("STRAVA_CLIENT_SECRET"),
        redirect_uri=get_setting("STRAVA_REDIRECT_URI"),
        refresh_token=get_setting("STRAVA_REFRESH_TOKEN"),
        cache_dir=cache,
    )


class StravaClient:
    """Authenticated Strava client with cache + rate limits.

    Never logs response bodies or stream payloads (privacy).
    On HTTP 401, refreshes via OAuth refresh_token (when configured) and retries once.
    """

    def __init__(
        self,
        config: StravaConfig,
        *,
        access_token: str | None = None,
        refresh_token: str | None = None,
        on_tokens_refreshed: Callable[[OAuthTokens], None] | None = None,
        session: requests.Session | None = None,
        timeout_s: float = 25,
    ) -> None:
        self.config = config
        raw_access = access_token if access_token is not None else config.access_token
        raw_refresh = refresh_token if refresh_token is not None else config.refresh_token
        self.access_token = str(raw_access).strip() if raw_access else None
        self.refresh_token = str(raw_refresh).strip() if raw_refresh else None
        self.on_tokens_refreshed = on_tokens_refreshed
        # Dedicated HTTP session; never stash Authorization on session.headers —
        # each request sets Bearer from self.access_token explicitly.
        self.session = session or requests.Session()
        self.timeout_s = timeout_s
        self.cache = SegmentCache(config.cache_dir / "segments.sqlite")
        self.limiter = RateLimiter(config.cache_dir / "rate_limit.sqlite")
        self.token_scope: str | None = None
        self.token_expires_at: int | None = None
        self.last_diag: dict[str, Any] = {}

    def _can_refresh(self) -> bool:
        return bool(
            self.config.client_id
            and self.config.client_secret
            and self.refresh_token
        )

    def _auth_headers(self) -> dict[str, str]:
        if not self.access_token:
            raise StravaAuthError(strava_errors.MISSING_SECRETS, kind="missing_secrets")
        return {"Authorization": f"Bearer {self.access_token}"}

    def _refresh_tokens(self) -> None:
        """Exchange refresh_token for a new access_token. Raises StravaAuthError."""
        if not (self.config.client_id and self.config.client_secret):
            raise StravaAuthError(strava_errors.MISSING_SECRETS, kind="missing_secrets")
        if not self.refresh_token:
            raise StravaAuthError(strava_errors.REFRESH_FAILED, kind="refresh_failed")
        try:
            # Use a clean Session for the token POST so no leftover API headers apply.
            token_http = requests.Session()
            tokens = refresh_access_token(
                client_id=str(self.config.client_id),
                client_secret=str(self.config.client_secret),
                refresh_token=str(self.refresh_token),
                session=token_http,
                timeout_s=self.timeout_s,
            )
        except OAuthTokenError as exc:
            detail = exc.detail or str(exc)
            raise StravaAuthError(
                strava_errors.REFRESH_FAILED_DETAIL.format(detail=detail),
                kind="refresh_failed",
            ) from exc
        except Exception as exc:  # noqa: BLE001 — never include token/body details
            raise StravaAuthError(strava_errors.REFRESH_FAILED, kind="refresh_failed") from exc

        import time as _time

        if tokens.expires_at is not None and tokens.expires_at < int(_time.time()) - 60:
            raise StravaAuthError(
                strava_errors.REFRESH_FAILED_DETAIL.format(
                    detail=(
                        f"refresh returned already-expired access_token "
                        f"(expires_at={tokens.expires_at})"
                    )
                ),
                kind="refresh_failed",
            )

        # Critical: replace bearer immediately — do not fall back to secrets token.
        self.access_token = tokens.access_token
        if tokens.refresh_token:
            self.refresh_token = tokens.refresh_token
        self.token_scope = tokens.scope
        self.token_expires_at = tokens.expires_at
        self.last_diag = {
            "refresh": "ok",
            "scope": tokens.scope,
            "expires_at": tokens.expires_at,
            "access_token_len": len(tokens.access_token),
            "access_token_prefix": tokens.access_token[:4] + "…",
        }
        if self.on_tokens_refreshed is not None:
            self.on_tokens_refreshed(tokens)

    def _probe_athlete(self) -> tuple[bool, int, str]:
        """Return (ok, status, safe_detail) for GET /athlete with current bearer."""
        try:
            resp = self.session.get(
                f"{API_BASE}/athlete",
                headers=self._auth_headers(),
                timeout=self.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            return False, 0, f"probe error: {exc.__class__.__name__}"
        from bike_wind.strava.oauth import safe_oauth_error_detail

        detail = safe_oauth_error_detail(resp) if resp.status_code >= 400 else "ok"
        return resp.status_code == 200, resp.status_code, detail

    def rate_status(self) -> RateLimitStatus:
        return self.limiter.status()

    def resolve_ref(self, text: str) -> tuple[int, ParsedStravaRef]:
        """Parse URL/ID; resolve effort → segment id when needed."""
        ref = parse_strava_segment_ref(text)
        if ref.kind == "segment_id":
            return ref.value, ref
        # effort → segment
        effort = self._get(f"/segment_efforts/{ref.value}")
        segment = effort.get("segment") or {}
        sid = segment.get("id")
        if sid is None:
            raise StravaError("Segment effort did not include a segment id")
        return int(sid), ref

    def get_segment_meta(self, segment_id: int, *, use_cache: bool = True) -> SegmentMeta:
        if use_cache:
            cached = self.cache.get(segment_id)
            if cached is not None:
                return SegmentMeta(
                    segment_id=cached.segment_id,
                    name=cached.name,
                    distance_m=cached.distance_m,
                    average_grade=cached.average_grade,
                    elev_low=cached.elev_low,
                    elev_high=cached.elev_high,
                    from_cache=True,
                )
        data = self._get(f"/segments/{segment_id}")
        start = data.get("start_latlng") or []
        start_lat = float(start[0]) if len(start) >= 2 else None
        start_lon = float(start[1]) if len(start) >= 2 else None
        return SegmentMeta(
            segment_id=int(data["id"]),
            name=str(data.get("name") or f"Segment {segment_id}"),
            distance_m=float(data.get("distance") or 0.0),
            average_grade=float(data.get("average_grade") or 0.0),
            elev_low=_opt_float(data.get("elevation_low")),
            elev_high=_opt_float(data.get("elevation_high")),
            from_cache=False,
            start_lat=start_lat,
            start_lon=start_lon,
        )

    def load_segment_route(
        self,
        segment_id: int,
        *,
        max_cache_age_s: float | None = 7 * 24 * 3600,
    ) -> tuple[SegmentMeta, RouteStreams]:
        """Return meta + RouteStreams (from cache or API)."""
        cached = self.cache.get(segment_id, max_age_s=max_cache_age_s)
        if cached is not None and len(cached.route) >= 2:
            meta = SegmentMeta(
                segment_id=cached.segment_id,
                name=cached.name,
                distance_m=cached.distance_m,
                average_grade=cached.average_grade,
                elev_low=cached.elev_low,
                elev_high=cached.elev_high,
                from_cache=True,
            )
            return meta, cached.route

        meta = self.get_segment_meta(segment_id, use_cache=False)
        route = self._fetch_streams_as_route(segment_id)
        self.cache.put(
            segment_id=meta.segment_id,
            name=meta.name,
            distance_m=meta.distance_m,
            average_grade=meta.average_grade,
            elev_low=meta.elev_low,
            elev_high=meta.elev_high,
            route=route,
        )
        return meta, route

    def list_starred(
        self,
        *,
        page: int | None = None,
        per_page: int = 200,
        max_pages: int = 50,
    ) -> list[StarredSegment]:
        """Return starred segments.

        When ``page`` is set, fetch that page only. Otherwise paginate until a
        short page (or empty) so the UI is not silently truncated at 30.
        """
        if page is not None:
            return self._list_starred_page(page=page, per_page=per_page)

        out: list[StarredSegment] = []
        for p in range(1, max_pages + 1):
            batch = self._list_starred_page(page=p, per_page=per_page)
            out.extend(batch)
            if len(batch) < per_page:
                break
        return out

    def _list_starred_page(self, *, page: int, per_page: int) -> list[StarredSegment]:
        data = self._get(
            "/segments/starred",
            params={"page": page, "per_page": per_page},
        )
        if not isinstance(data, list):
            raise StravaError("Unexpected starred segments response")
        out: list[StarredSegment] = []
        for item in data:
            out.append(
                StarredSegment(
                    segment_id=int(item["id"]),
                    name=str(item.get("name") or item["id"]),
                    distance_m=float(item.get("distance") or 0.0),
                    average_grade=float(item.get("average_grade") or 0.0),
                )
            )
        return out

    def explore(
        self,
        bounds: Sequence[float],
        *,
        activity_type: str = "riding",
    ) -> list[ExploreHit]:
        """Explore segments in bbox (GET /segments/explore). Needs a normal token."""
        if len(bounds) != 4:
            raise ValueError("bounds must be [sw_lat, sw_lng, ne_lat, ne_lng]")
        data = self._get(
            "/segments/explore",
            params={
                "bounds": ",".join(str(b) for b in bounds),
                "activity_type": activity_type,
            },
        )
        segs = data.get("segments") if isinstance(data, dict) else data
        if not isinstance(segs, list):
            raise StravaError("Unexpected explore response")
        hits: list[ExploreHit] = []
        for item in segs[:10]:
            start = item.get("start_latlng") or []
            end = item.get("end_latlng") or []
            hits.append(
                ExploreHit(
                    segment_id=int(item["id"]),
                    name=str(item.get("name") or item["id"]),
                    distance_m=float(item.get("distance") or 0.0),
                    average_grade=float(
                        item.get("avg_grade") or item.get("average_grade") or 0.0
                    ),
                    climb_category=int(item.get("climb_category") or 0),
                    starred=bool(item.get("starred")),
                    start_lat=float(start[0]) if len(start) >= 2 else None,
                    start_lon=float(start[1]) if len(start) >= 2 else None,
                    end_lat=float(end[0]) if len(end) >= 2 else None,
                    end_lon=float(end[1]) if len(end) >= 2 else None,
                )
            )
        return hits

    def list_recent_activities(
        self,
        *,
        per_page: int = 30,
        page: int = 1,
    ) -> list[ActivitySummary]:
        """List the authenticated athlete's recent activities (needs activity:read)."""
        data = self._get(
            "/athlete/activities",
            params={"per_page": per_page, "page": page},
        )
        if not isinstance(data, list):
            raise StravaError("Unexpected activities response")
        out: list[ActivitySummary] = []
        for item in data:
            start = item.get("start_latlng") or []
            watts = item.get("average_watts")
            if watts is None:
                watts = item.get("weighted_average_watts")
            sport = str(item.get("sport_type") or item.get("type") or "Ride")
            out.append(
                ActivitySummary(
                    activity_id=int(item["id"]),
                    name=str(item.get("name") or item["id"]),
                    sport_type=sport,
                    start_date=str(item.get("start_date") or ""),
                    distance_m=float(item.get("distance") or 0.0),
                    moving_time_s=int(item.get("moving_time") or 0),
                    elapsed_time_s=int(item.get("elapsed_time") or 0),
                    average_watts=float(watts) if watts is not None else None,
                    start_lat=float(start[0]) if len(start) >= 2 else None,
                    start_lon=float(start[1]) if len(start) >= 2 else None,
                )
            )
        return out

    def load_activity_route(self, activity_id: int) -> RouteStreams:
        """Activity latlng/distance/altitude streams as a RouteStreams."""
        return self._fetch_activity_streams_as_route(activity_id)

    def _fetch_activity_streams_as_route(self, activity_id: int) -> RouteStreams:
        data = self._get(
            f"/activities/{activity_id}/streams",
            params={
                "keys": "latlng,distance,altitude,time,velocity_smooth",
                "key_by_type": "true",
            },
        )
        return self._streams_payload_to_route(data)

    def _fetch_streams_as_route(self, segment_id: int) -> RouteStreams:
        # Segment streams are usually geometry-only; ask for time/velocity when present.
        data = self._get(
            f"/segments/{segment_id}/streams",
            params={
                "keys": "latlng,distance,altitude,time,velocity_smooth",
                "key_by_type": "true",
            },
        )
        return self._streams_payload_to_route(data)

    def _streams_payload_to_route(self, data: Any) -> RouteStreams:
        from datetime import datetime, timezone

        from bike_wind.gpx.parse import speeds_from_times

        # key_by_type → {"latlng": {"data": [[lat,lng],...]}, ...}
        if isinstance(data, dict) and "latlng" in data:
            latlng = data["latlng"]["data"]
            distances = data.get("distance", {}).get("data") or []
            altitudes = data.get("altitude", {}).get("data") or []
            times_raw = data.get("time", {}).get("data") or []
            vel_raw = data.get("velocity_smooth", {}).get("data") or []
        elif isinstance(data, list):
            by_type = {item.get("type"): item.get("data") for item in data}
            latlng = by_type.get("latlng") or []
            distances = by_type.get("distance") or []
            altitudes = by_type.get("altitude") or []
            times_raw = by_type.get("time") or []
            vel_raw = by_type.get("velocity_smooth") or []
        else:
            raise StravaError("Unexpected streams response shape")

        if len(latlng) < 2:
            raise StravaError("Streams had fewer than 2 points")

        lats = [float(p[0]) for p in latlng]
        lons = [float(p[1]) for p in latlng]
        n = len(lats)
        if len(distances) == n:
            dists = [float(d) for d in distances]
        else:
            dists = _cumulative_haversine(lats, lons)
        if len(altitudes) == n:
            elevs = [float(a) for a in altitudes]
        else:
            elevs = [0.0] * n

        # Absolute wall-clock times are not on bare segment streams; relative
        # time/velocity only feed actual_speeds for the validation chart.
        actual_speeds = None
        if len(vel_raw) == n:
            actual_speeds = [max(0.0, float(v)) for v in vel_raw]
        elif len(times_raw) == n:
            from datetime import timedelta

            base = datetime(1970, 1, 1, tzinfo=timezone.utc)
            rel_times = [base + timedelta(seconds=float(t)) for t in times_raw]
            actual_speeds = speeds_from_times(dists, rel_times)

        return RouteStreams(
            lats=lats,
            lons=lons,
            elevs_m=elevs,
            distances_m=dists,
            times_utc=None,
            actual_speeds_mps=actual_speeds,
        )

    def _format_diag(self, **extra: Any) -> str:
        blob = {
            "scope": self.token_scope,
            "expires_at": self.token_expires_at,
            "bearer_len": len(self.access_token or ""),
            **self.last_diag,
            **extra,
        }
        # Compact, secret-free key=value list for UI / screenshots
        parts = [f"{k}={v}" for k, v in blob.items() if v is not None and v != ""]
        return ", ".join(parts) if parts else "(none)"

    def _raise_after_failed_auth(
        self,
        path: str,
        resp: requests.Response,
        *,
        after_refresh: bool,
    ) -> None:
        from bike_wind.strava.oauth import safe_oauth_error_detail

        detail = safe_oauth_error_detail(resp)
        is_explore = path.rstrip("/").endswith("/segments/explore")
        diag_base = {
            "endpoint": path,
            "http_status": resp.status_code,
            "after_refresh": after_refresh,
            "strava_detail": detail,
        }

        if after_refresh and is_explore:
            athlete_ok, athlete_status, athlete_detail = self._probe_athlete()
            diag_base.update(
                {
                    "athlete_probe_ok": athlete_ok,
                    "athlete_probe_status": athlete_status,
                    "athlete_probe_detail": athlete_detail,
                }
            )
            diag = self._format_diag(**diag_base)
            if athlete_ok:
                raise StravaAuthError(
                    strava_errors.EXPLORE_UNAUTHORIZED_TOKEN_OK.format(
                        scope=self.token_scope or "(unknown)",
                        path=path,
                        status=resp.status_code,
                        detail=detail,
                        diag=diag,
                    ),
                    kind="forbidden_explore_tier",
                    diag=diag_base,
                )
            raise StravaAuthError(
                strava_errors.UNAUTHORIZED_AFTER_REFRESH.format(
                    expires_at=self.token_expires_at,
                    scope=self.token_scope or "(unknown)",
                    path=path,
                    status=resp.status_code,
                    detail=detail,
                    diag=diag,
                ),
                kind="expired_access",
                diag=diag_base,
            )

        if after_refresh:
            diag = self._format_diag(**diag_base)
            raise StravaAuthError(
                strava_errors.UNAUTHORIZED_AFTER_REFRESH.format(
                    expires_at=self.token_expires_at,
                    scope=self.token_scope or "(unknown)",
                    path=path,
                    status=resp.status_code,
                    detail=detail,
                    diag=diag,
                ),
                kind="expired_access",
                diag=diag_base,
            )

        if is_explore and resp.status_code == 403:
            diag = self._format_diag(**diag_base)
            raise StravaAuthError(
                strava_errors.FORBIDDEN_EXPLORE_TIER.format(diag=diag),
                kind="forbidden_explore_tier",
                diag=diag_base,
            )

        if resp.status_code == 403:
            diag = self._format_diag(**diag_base)
            raise StravaAuthError(
                strava_errors.FORBIDDEN_SCOPE.format(path=path, diag=diag),
                kind="forbidden_scope",
                diag=diag_base,
            )

        raise StravaAuthError(
            strava_errors.REFRESH_FAILED_DETAIL.format(detail=detail),
            kind="expired_access",
            diag=diag_base,
        )

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        _retried: bool = False,
    ) -> Any:
        if not self.access_token:
            if self._can_refresh():
                self._refresh_tokens()
            elif self.config.client_id or self.config.client_secret or self.refresh_token:
                raise StravaAuthError(strava_errors.REFRESH_FAILED, kind="refresh_failed")
            else:
                raise StravaAuthError(strava_errors.MISSING_SECRETS, kind="missing_secrets")
        try:
            self.limiter.check()
        except RateLimitExceeded as exc:
            raise StravaRateLimitError(exc.status) from exc

        url = f"{API_BASE}{path}"
        # Always build Authorization from the current self.access_token (post-refresh).
        headers = self._auth_headers()
        resp = self.session.get(
            url,
            headers=headers,
            params=params,
            timeout=self.timeout_s,
        )
        self.limiter.record()
        self.last_diag = {
            **self.last_diag,
            "last_path": path,
            "last_status": resp.status_code,
            "bearer_len": len(self.access_token or ""),
        }

        if resp.status_code == 401:
            if not _retried and self._can_refresh():
                self._refresh_tokens()
                # Retry must use the refreshed bearer (self.access_token just updated).
                return self._get(path, params, _retried=True)
            self._raise_after_failed_auth(path, resp, after_refresh=_retried)
        if resp.status_code == 403:
            self._raise_after_failed_auth(path, resp, after_refresh=_retried)
        if resp.status_code == 429:
            status = self.limiter.status()
            raise StravaRateLimitError(
                RateLimitStatus(
                    used_15m=status.used_15m,
                    used_day=status.used_day,
                    remaining_15m=0,
                    remaining_day=status.remaining_day,
                    limited=True,
                    retry_after_s=float(resp.headers.get("Retry-After") or 60),
                    message=strava_errors.RATE_LIMIT,
                )
            )
        if resp.status_code >= 400:
            # Do not include response body (may contain athlete data)
            raise StravaError(f"Strava API error HTTP {resp.status_code} for {path}")
        return resp.json()


def _opt_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _cumulative_haversine(lats: list[float], lons: list[float]) -> list[float]:
    import math

    dists = [0.0]
    r = 6_371_000.0
    for i in range(1, len(lats)):
        p1, p2 = math.radians(lats[i - 1]), math.radians(lats[i])
        dp = math.radians(lats[i] - lats[i - 1])
        dl = math.radians(lons[i] - lons[i - 1])
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        dists.append(dists[-1] + 2 * r * math.asin(math.sqrt(a)))
    return dists
