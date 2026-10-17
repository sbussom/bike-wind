"""Strava API client: segments, streams, starred, explore (gated)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import requests

from bike_wind.gpx.parse import RouteStreams
from bike_wind.settings import get_bool_setting, get_setting
from bike_wind.strava.cache import CachedSegment, SegmentCache
from bike_wind.strava.rate_limit import RateLimitExceeded, RateLimiter, RateLimitStatus
from bike_wind.strava.urls import ParsedStravaRef, parse_strava_segment_ref

API_BASE = "https://www.strava.com/api/v3"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "bike_wind"
# Prefer project-local cache when running from repo
REPO_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "strava"


@dataclass(frozen=True)
class StravaConfig:
    access_token: str | None
    client_id: str | None
    client_secret: str | None
    redirect_uri: str | None
    extended_access: bool
    cache_dir: Path

    @property
    def oauth_ready(self) -> bool:
        return bool(self.client_id and self.client_secret and self.redirect_uri)

    @property
    def token_ready(self) -> bool:
        return bool(self.access_token)


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


class StravaError(RuntimeError):
    pass


class StravaAuthError(StravaError):
    pass


class StravaRateLimitError(StravaError):
    def __init__(self, status: RateLimitStatus) -> None:
        self.status = status
        super().__init__(status.message or "Strava rate limit exceeded")


def load_strava_config(*, cache_dir: Path | None = None) -> StravaConfig:
    """Load Strava settings from env, then Streamlit secrets (same key names)."""
    extended = get_bool_setting("STRAVA_EXTENDED_ACCESS", default=False)
    cache_raw = get_setting("BIKE_WIND_CACHE")
    cache = cache_dir or Path(cache_raw or REPO_CACHE)
    return StravaConfig(
        access_token=get_setting("STRAVA_ACCESS_TOKEN"),
        client_id=get_setting("STRAVA_CLIENT_ID"),
        client_secret=get_setting("STRAVA_CLIENT_SECRET"),
        redirect_uri=get_setting("STRAVA_REDIRECT_URI"),
        extended_access=extended,
        cache_dir=cache,
    )


class StravaClient:
    """Authenticated Strava client with cache + rate limits.

    Never logs response bodies or stream payloads (privacy).
    """

    def __init__(
        self,
        config: StravaConfig,
        *,
        access_token: str | None = None,
        session: requests.Session | None = None,
        timeout_s: float = 25,
    ) -> None:
        self.config = config
        self.access_token = access_token or config.access_token
        self.session = session or requests.Session()
        self.timeout_s = timeout_s
        self.cache = SegmentCache(config.cache_dir / "segments.sqlite")
        self.limiter = RateLimiter(config.cache_dir / "rate_limit.sqlite")

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

    def list_starred(self, *, page: int = 1, per_page: int = 30) -> list[StarredSegment]:
        data = self._get("/segments/starred", params={"page": page, "per_page": per_page})
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
        """Explore segments in bbox. Requires STRAVA_EXTENDED_ACCESS=true."""
        if not self.config.extended_access:
            raise StravaError(
                "Explore Segments requires Extended Access "
                "(set STRAVA_EXTENDED_ACCESS=true when approved)."
            )
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
            hits.append(
                ExploreHit(
                    segment_id=int(item["id"]),
                    name=str(item.get("name") or item["id"]),
                    distance_m=float(item.get("distance") or 0.0),
                    average_grade=float(item.get("avg_grade") or item.get("average_grade") or 0.0),
                    climb_category=int(item.get("climb_category") or 0),
                    starred=bool(item.get("starred")),
                )
            )
        return hits

    def _fetch_streams_as_route(self, segment_id: int) -> RouteStreams:
        data = self._get(
            f"/segments/{segment_id}/streams",
            params={"keys": "latlng,distance,altitude", "key_by_type": "true"},
        )
        # key_by_type → {"latlng": {"data": [[lat,lng],...]}, ...}
        if isinstance(data, dict) and "latlng" in data:
            latlng = data["latlng"]["data"]
            distances = data.get("distance", {}).get("data") or []
            altitudes = data.get("altitude", {}).get("data") or []
        elif isinstance(data, list):
            by_type = {item.get("type"): item.get("data") for item in data}
            latlng = by_type.get("latlng") or []
            distances = by_type.get("distance") or []
            altitudes = by_type.get("altitude") or []
        else:
            raise StravaError("Unexpected streams response shape")

        if len(latlng) < 2:
            raise StravaError("Segment streams had fewer than 2 points")

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
        return RouteStreams(lats=lats, lons=lons, elevs_m=elevs, distances_m=dists)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.access_token:
            raise StravaAuthError(
                "No Strava access token. Set STRAVA_ACCESS_TOKEN or complete OAuth."
            )
        try:
            self.limiter.check()
        except RateLimitExceeded as exc:
            raise StravaRateLimitError(exc.status) from exc

        url = f"{API_BASE}{path}"
        resp = self.session.get(
            url,
            headers={"Authorization": f"Bearer {self.access_token}"},
            params=params,
            timeout=self.timeout_s,
        )
        self.limiter.record()

        if resp.status_code == 401:
            raise StravaAuthError("Strava unauthorized — token missing or expired")
        if resp.status_code == 403:
            raise StravaAuthError("Strava forbidden — check scopes / Extended Access")
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
                    message="Strava HTTP 429 — rate limited by API.",
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
