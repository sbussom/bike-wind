"""Parse GPX into lat/lon/elevation/distance streams (optional times/speeds)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Sequence, TextIO, Union

GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/1"}
# Some files omit the default namespace or use 1.0
GPX_NS_CANDIDATES = (
    {"gpx": "http://www.topografix.com/GPX/1/1"},
    {"gpx": "http://www.topografix.com/GPX/1/0"},
    {},
)

Source = Union[str, Path, bytes, BinaryIO, TextIO]

# Distance window for elevation smoothing before grade (GPS blip filter).
# 100 m (was 35 m): held-out 2024+ moving MAE improved on device-power rides;
# short windows still left ~3 mph predicted swings under steady power from
# residual GPS grade spikes.
ELEV_SMOOTH_WINDOW_M = 100.0


def smooth_elevations(
    elevs_m: Sequence[float],
    distances_m: Sequence[float],
    *,
    window_m: float = ELEV_SMOOTH_WINDOW_M,
) -> list[float]:
    """Centered distance-box mean of elevation (meters).

    A ~100 m window kills GPS altitude spikes that otherwise swing predicted
    speed by multiple mph under steady power, without flattening real climbs.
    ``window_m <= 0`` returns a copy unchanged.
    """
    n = len(elevs_m)
    if n != len(distances_m):
        raise ValueError("elevs_m and distances_m must have the same length")
    if n == 0 or window_m <= 0.0:
        return [float(e) for e in elevs_m]

    half = float(window_m) * 0.5
    out = [0.0] * n
    j_lo = 0
    j_hi = 0
    total = 0.0
    for i in range(n):
        d0 = distances_m[i]
        while j_hi < n and distances_m[j_hi] <= d0 + half:
            total += float(elevs_m[j_hi])
            j_hi += 1
        while j_lo < j_hi and distances_m[j_lo] < d0 - half:
            total -= float(elevs_m[j_lo])
            j_lo += 1
        count = j_hi - j_lo
        out[i] = total / count if count > 0 else float(elevs_m[i])
    return out


@dataclass(frozen=True)
class RoutePoint:
    lat: float
    lon: float
    elev_m: float
    distance_m: float


@dataclass
class RouteStreams:
    """Aligned route streams suitable for wind sampling and physics."""

    lats: list[float]
    lons: list[float]
    elevs_m: list[float]
    distances_m: list[float]
    times_utc: list[datetime] | None = None
    actual_speeds_mps: list[float] | None = None
    watts: list[float] | None = None

    def __len__(self) -> int:
        return len(self.lats)

    def midpoint(self) -> tuple[float, float]:
        """Return (lat, lon) at the middle track point."""
        n = len(self.lats)
        if n == 0:
            raise ValueError("empty route")
        i = (n - 1) // 2
        return float(self.lats[i]), float(self.lons[i])

    @property
    def start_time(self) -> datetime | None:
        """First point timestamp when the ride recorded times."""
        if not self.times_utc:
            return None
        return self.times_utc[0]

    @property
    def has_actual_speeds(self) -> bool:
        """True when a speed sample exists for every point (NaN gaps allowed)."""
        if not self.actual_speeds_mps or len(self.actual_speeds_mps) != len(self):
            return False
        return any(math.isfinite(v) for v in self.actual_speeds_mps)

    @property
    def has_watts(self) -> bool:
        """True when power-meter watts align with the route (NaN gaps allowed)."""
        if not self.watts or len(self.watts) != len(self):
            return False
        return any(math.isfinite(w) and w > 0.0 for w in self.watts)

    @property
    def points(self) -> list[RoutePoint]:
        return [
            RoutePoint(lat, lon, elev, dist)
            for lat, lon, elev, dist in zip(
                self.lats, self.lons, self.elevs_m, self.distances_m
            )
        ]

    def grades(self, *, elev_smooth_m: float = ELEV_SMOOTH_WINDOW_M) -> list[float]:
        """Segment grades (rise/run) between consecutive points; last repeats prior.

        Elevations are smoothed over ``elev_smooth_m`` before differencing so a
        GPS altitude blip does not spike predicted speed. Pass ``0`` to disable.
        """
        n = len(self)
        if n < 2:
            return [0.0] * n
        elevs = smooth_elevations(
            self.elevs_m, self.distances_m, window_m=elev_smooth_m
        )
        out: list[float] = []
        for i in range(n - 1):
            dd = self.distances_m[i + 1] - self.distances_m[i]
            de = elevs[i + 1] - elevs[i]
            if dd <= 0.0:
                out.append(0.0)
            else:
                out.append(de / dd)
        out.append(out[-1])
        return out

    def mean_grade_pct(self) -> float:
        """Distance-weighted mean |grade| as percent (for UI notes)."""
        n = len(self)
        if n < 2:
            return 0.0
        grades = self.grades()
        num = 0.0
        den = 0.0
        for i in range(n - 1):
            dd = max(0.0, self.distances_m[i + 1] - self.distances_m[i])
            num += abs(grades[i]) * 100.0 * dd
            den += dd
        return num / den if den > 0 else 0.0

    def bearings_deg(self) -> list[float]:
        """Forward bearing (degrees from north, clockwise) at each point."""
        n = len(self)
        if n < 2:
            return [0.0] * n
        out: list[float] = []
        for i in range(n - 1):
            out.append(
                _bearing_deg(self.lats[i], self.lons[i], self.lats[i + 1], self.lons[i + 1])
            )
        out.append(out[-1])
        return out


def parse_gpx(source: Source) -> RouteStreams:
    """Parse a GPX track or route into distance-aligned streams.

    When track points include ``<time>``, also returns UTC times and derived
    actual ground speeds (m/s). Points without times yield geometry only.
    """
    root = _load_root(source)
    coords = _extract_points(root)
    if len(coords) < 2:
        raise ValueError("GPX must contain at least two track/route points")

    lats: list[float] = []
    lons: list[float] = []
    elevs: list[float] = []
    dists: list[float] = []
    times: list[datetime] = []
    cumulative = 0.0
    timed = 0

    for i, (lat, lon, elev, t) in enumerate(coords):
        if i > 0:
            cumulative += _haversine_m(lats[-1], lons[-1], lat, lon)
        lats.append(lat)
        lons.append(lon)
        elevs.append(elev)
        dists.append(cumulative)
        if t is not None:
            times.append(t)
            timed += 1
        else:
            times.append(datetime.fromtimestamp(0, tz=timezone.utc))  # placeholder

    times_utc: list[datetime] | None = times if timed == len(coords) else None
    speeds = speeds_from_times(dists, times_utc) if times_utc is not None else None
    return RouteStreams(
        lats=lats,
        lons=lons,
        elevs_m=elevs,
        distances_m=dists,
        times_utc=times_utc,
        actual_speeds_mps=speeds,
    )


def speeds_from_times(
    distances_m: list[float],
    times_utc: list[datetime],
) -> list[float]:
    """Ground speed (m/s) per point from consecutive distance/time deltas."""
    n = len(distances_m)
    if n != len(times_utc) or n < 2:
        return [0.0] * n
    out: list[float] = []
    for i in range(n - 1):
        dd = distances_m[i + 1] - distances_m[i]
        dt = (times_utc[i + 1] - times_utc[i]).total_seconds()
        if dt > 0.05 and dd >= 0.0:
            out.append(dd / dt)
        else:
            out.append(out[-1] if out else 0.0)
    out.append(out[-1] if out else 0.0)
    return out


def _load_root(source: Source) -> ET.Element:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists():
            tree = ET.parse(path)
            return tree.getroot()
        # Treat as raw XML string
        return ET.fromstring(source if isinstance(source, str) else source.decode())
    if isinstance(source, bytes):
        return ET.fromstring(source)
    data = source.read()
    if isinstance(data, bytes):
        return ET.fromstring(data)
    return ET.fromstring(data)


def _extract_points(root: ET.Element) -> list[tuple[float, float, float, datetime | None]]:
    for ns in GPX_NS_CANDIDATES:
        points = _find_points(root, ns)
        if points:
            return points
    points: list[tuple[float, float, float, datetime | None]] = []
    for el in root.iter():
        tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if tag in ("trkpt", "rtept"):
            points.append(_point_from_el(el))
    return points


def _find_points(
    root: ET.Element, ns: dict[str, str]
) -> list[tuple[float, float, float, datetime | None]]:
    if ns:
        trk = root.findall(".//gpx:trkpt", ns)
        if trk:
            return [_point_from_el(el) for el in trk]
        rte = root.findall(".//gpx:rtept", ns)
        if rte:
            return [_point_from_el(el) for el in rte]
        return []
    trk = root.findall(".//trkpt")
    if trk:
        return [_point_from_el(el) for el in trk]
    rte = root.findall(".//rtept")
    return [_point_from_el(el) for el in rte]


def _point_from_el(el: ET.Element) -> tuple[float, float, float, datetime | None]:
    lat = float(el.attrib["lat"])
    lon = float(el.attrib["lon"])
    elev = 0.0
    t: datetime | None = None
    for child in el:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "ele" and child.text:
            elev = float(child.text)
        elif tag == "time" and child.text:
            t = _parse_gpx_time(child.text.strip())
    return lat, lon, elev, t


def _parse_gpx_time(raw: str) -> datetime | None:
    text = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
