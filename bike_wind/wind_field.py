"""Gridded Open-Meteo wind forecast for the animated GustyPower wind map.

Two nested grids are sampled: a coarse one over the Florida peninsula and a
fine one around the rider's home area. The browser draws whichever grid
covers each pixel (fine wins), so zooming in sharpens the local field without
asking Open-Meteo for hundreds of extra points.

Nothing here touches the prediction model; it is display data only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
MPS_PER_MPH = 0.44704

# Round trip to pumphouse (Strava 41392335), Fort Pierce / Okeechobee Rd area.
HOME_LAT = 27.206
HOME_LON = -80.45

# Display scale: 0–50 mph, blue → green → yellow → red.
COLOR_STOPS_MPH: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0, (40, 70, 170)),
    (5.0, (35, 125, 205)),
    (10.0, (30, 170, 175)),
    (15.0, (60, 180, 80)),
    (20.0, (150, 200, 50)),
    (25.0, (235, 215, 45)),
    (32.0, (245, 150, 35)),
    (40.0, (225, 60, 40)),
    (50.0, (150, 20, 60)),
)


@dataclass(frozen=True)
class GridSpec:
    """Regular lat/lon grid (inclusive bounds)."""

    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    n_lat: int
    n_lon: int

    def lats(self) -> list[float]:
        return _linspace(self.lat_min, self.lat_max, self.n_lat)

    def lons(self) -> list[float]:
        return _linspace(self.lon_min, self.lon_max, self.n_lon)

    def points(self) -> list[tuple[float, float]]:
        """Row-major (lat ascending, then lon ascending)."""
        return [(la, lo) for la in self.lats() for lo in self.lons()]

    @property
    def size(self) -> int:
        return self.n_lat * self.n_lon


# Open-Meteo counts every grid point as an API call (free tier: 600/min,
# 10k/day), so the grids stay small: ~200 points per refresh in total.
# ~0.8° spacing over the peninsula + panhandle.
FLORIDA_GRID = GridSpec("florida", 24.4, 31.0, -87.6, -78.8, 9, 11)
# ~0.13° spacing (≈14 km) around home.
LOCAL_GRID = GridSpec(
    "local",
    HOME_LAT - 0.55,
    HOME_LAT + 0.55,
    HOME_LON - 0.65,
    HOME_LON + 0.65,
    9,
    11,
)


@dataclass
class WindGrid:
    """Speed (mph) and meteorological from-direction per frame and grid point."""

    spec: GridSpec
    speed_mph: list[list[float]] = field(default_factory=list)  # [frame][point]
    dir_deg: list[list[float]] = field(default_factory=list)
    gust_mph: list[list[float]] = field(default_factory=list)


@dataclass
class WindField:
    times_utc: list[datetime]
    grids: list[WindGrid]

    def to_payload(self) -> dict[str, Any]:
        """JSON-able dict for the browser (u/v in mph, east/north positive)."""
        grids = []
        for g in self.grids:
            u_frames: list[list[float]] = []
            v_frames: list[list[float]] = []
            for spd_row, dir_row in zip(g.speed_mph, g.dir_deg):
                us, vs = [], []
                for spd, d in zip(spd_row, dir_row):
                    u, v = uv_from_speed_dir(spd, d)
                    us.append(round(u, 1))
                    vs.append(round(v, 1))
                u_frames.append(us)
                v_frames.append(vs)
            grids.append(
                {
                    "name": g.spec.name,
                    "latMin": g.spec.lat_min,
                    "latMax": g.spec.lat_max,
                    "lonMin": g.spec.lon_min,
                    "lonMax": g.spec.lon_max,
                    "nLat": g.spec.n_lat,
                    "nLon": g.spec.n_lon,
                    "u": u_frames,
                    "v": v_frames,
                }
            )
        return {
            "times": [t.strftime("%Y-%m-%dT%H:%MZ") for t in self.times_utc],
            "grids": grids,
        }


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n < 2:
        return [a]
    step = (b - a) / (n - 1)
    return [round(a + i * step, 4) for i in range(n)]


def uv_from_speed_dir(speed: float, from_deg: float) -> tuple[float, float]:
    """Vector the air moves toward (u east, v north) from a wind-from bearing."""
    rad = math.radians(from_deg)
    return -speed * math.sin(rad), -speed * math.cos(rad)


def color_for_mph(mph: float) -> tuple[int, int, int]:
    """Piecewise-linear scale used by the map legend (clamped 0–50 mph)."""
    stops = COLOR_STOPS_MPH
    if mph <= stops[0][0]:
        return stops[0][1]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if mph <= b:
            t = (mph - a) / (b - a)
            return tuple(int(round(x + (y - x) * t)) for x, y in zip(ca, cb))  # type: ignore[return-value]
    return stops[-1][1]


def fetch_grid(
    spec: GridSpec,
    *,
    forecast_days: int = 4,
    session: requests.Session | None = None,
    timeout_s: float = 30,
    chunk: int = 100,
) -> tuple[list[datetime], WindGrid]:
    """Hourly 10 m wind for every grid point from Open-Meteo (multi-location)."""
    http = session or requests.Session()
    pts = spec.points()
    per_point: list[dict[str, Any]] = []
    for start in range(0, len(pts), chunk):
        part = pts[start : start + chunk]
        resp = _get_with_retry(
            http,
            params={
                "latitude": ",".join(f"{p[0]:.4f}" for p in part),
                "longitude": ",".join(f"{p[1]:.4f}" for p in part),
                "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
                "wind_speed_unit": "mph",
                "timezone": "UTC",
                "forecast_days": forecast_days,
            },
            timeout_s=timeout_s,
        )
        payload = resp.json()
        if isinstance(payload, dict):
            payload = [payload]
        per_point.extend(payload)
    if len(per_point) != len(pts):
        raise ValueError(f"Open-Meteo returned {len(per_point)} of {len(pts)} points")
    times_raw: Sequence[str] = per_point[0]["hourly"]["time"]
    times = [
        datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in times_raw
    ]
    n_t = len(times)
    grid = WindGrid(spec=spec)
    for ti in range(n_t):
        spd_row, dir_row, gust_row = [], [], []
        for p in per_point:
            h = p["hourly"]
            spd_row.append(_num(h["wind_speed_10m"], ti))
            dir_row.append(_num(h["wind_direction_10m"], ti))
            gust_row.append(_num(h.get("wind_gusts_10m") or [], ti))
        grid.speed_mph.append(spd_row)
        grid.dir_deg.append(dir_row)
        grid.gust_mph.append(gust_row)
    return times, grid


def _get_with_retry(
    http: requests.Session,
    *,
    params: dict[str, Any],
    timeout_s: float,
    retries: int = 2,
    backoff_s: float = 8.0,
) -> requests.Response:
    """GET Open-Meteo; back off briefly on 429 / 5xx before giving up."""
    import time

    last: requests.Response | None = None
    for attempt in range(retries + 1):
        resp = http.get(OPEN_METEO_URL, params=params, timeout=timeout_s)
        if resp.status_code != 429 and resp.status_code < 500:
            resp.raise_for_status()
            return resp
        last = resp
        if attempt < retries:
            time.sleep(backoff_s * (attempt + 1))
    assert last is not None
    last.raise_for_status()
    return last


def _num(series: Sequence[Any], i: int) -> float:
    try:
        x = float(series[i])
    except (TypeError, ValueError, IndexError):
        return 0.0
    return x if math.isfinite(x) else 0.0


def fetch_wind_field(
    grids: Sequence[GridSpec] = (FLORIDA_GRID, LOCAL_GRID),
    *,
    forecast_days: int = 4,
    session: requests.Session | None = None,
) -> WindField:
    """Fetch every grid and align them on the shared hourly time axis."""
    times: list[datetime] | None = None
    out: list[WindGrid] = []
    for spec in grids:
        t, g = fetch_grid(spec, forecast_days=forecast_days, session=session)
        if times is None:
            times = t
        elif t != times:
            n = min(len(times), len(t))
            times = times[:n]
        out.append(g)
    assert times is not None
    n = len(times)
    for g in out:
        g.speed_mph = g.speed_mph[:n]
        g.dir_deg = g.dir_deg[:n]
        g.gust_mph = g.gust_mph[:n]
    return WindField(times_utc=times, grids=out)


def frame_indices(
    times_utc: Sequence[datetime],
    *,
    step_hours: int = 1,
    now: datetime | None = None,
) -> list[int]:
    """Frames from the current hour onward, every ``step_hours``."""
    now = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    first = next((i for i, t in enumerate(times_utc) if t >= now), 0)
    return list(range(first, len(times_utc), max(1, step_hours)))


def subset_payload(payload: dict[str, Any], idxs: Sequence[int]) -> dict[str, Any]:
    """Keep only the chosen frames (keeps the browser payload small)."""
    out = {"times": [payload["times"][i] for i in idxs], "grids": []}
    for g in payload["grids"]:
        g2 = dict(g)
        for key in ("u", "v"):
            g2[key] = [g[key][i] for i in idxs]
        out["grids"].append(g2)
    return out


def sample_point(grid: WindGrid, frame: int, lat: float, lon: float) -> tuple[float, float, float]:
    """Bilinear (speed_mph, from_deg, gust_mph) at a point inside ``grid``."""
    s = grid.spec
    fy = (lat - s.lat_min) / (s.lat_max - s.lat_min) * (s.n_lat - 1)
    fx = (lon - s.lon_min) / (s.lon_max - s.lon_min) * (s.n_lon - 1)
    fy = min(max(fy, 0.0), s.n_lat - 1.0)
    fx = min(max(fx, 0.0), s.n_lon - 1.0)
    y0, x0 = int(fy), int(fx)
    y1, x1 = min(y0 + 1, s.n_lat - 1), min(x0 + 1, s.n_lon - 1)
    ty, tx = fy - y0, fx - x0
    u_acc = v_acc = g_acc = 0.0
    for yy, wy in ((y0, 1 - ty), (y1, ty)):
        for xx, wx in ((x0, 1 - tx), (x1, tx)):
            k = yy * s.n_lon + xx
            u, v = uv_from_speed_dir(grid.speed_mph[frame][k], grid.dir_deg[frame][k])
            w = wy * wx
            u_acc += u * w
            v_acc += v * w
            g_acc += grid.gust_mph[frame][k] * w
    speed = math.hypot(u_acc, v_acc)
    from_deg = (math.degrees(math.atan2(-u_acc, -v_acc)) + 360.0) % 360.0
    return speed, from_deg, g_acc
