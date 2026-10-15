"""Parse GPX into lat/lon/elevation/distance streams."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO, Union

GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/1"}
# Some files omit the default namespace or use 1.0
GPX_NS_CANDIDATES = (
    {"gpx": "http://www.topografix.com/GPX/1/1"},
    {"gpx": "http://www.topografix.com/GPX/1/0"},
    {},
)

Source = Union[str, Path, bytes, BinaryIO, TextIO]


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

    def __len__(self) -> int:
        return len(self.lats)

    @property
    def points(self) -> list[RoutePoint]:
        return [
            RoutePoint(lat, lon, elev, dist)
            for lat, lon, elev, dist in zip(
                self.lats, self.lons, self.elevs_m, self.distances_m
            )
        ]

    def grades(self) -> list[float]:
        """Segment grades (rise/run) between consecutive points; last repeats prior."""
        n = len(self)
        if n < 2:
            return [0.0] * n
        out: list[float] = []
        for i in range(n - 1):
            dd = self.distances_m[i + 1] - self.distances_m[i]
            de = self.elevs_m[i + 1] - self.elevs_m[i]
            if dd <= 0.0:
                out.append(0.0)
            else:
                out.append(de / dd)
        out.append(out[-1])
        return out

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
    """Parse a GPX track or route into distance-aligned streams."""
    root = _load_root(source)
    coords = _extract_points(root)
    if len(coords) < 2:
        raise ValueError("GPX must contain at least two track/route points")

    lats: list[float] = []
    lons: list[float] = []
    elevs: list[float] = []
    dists: list[float] = []
    cumulative = 0.0

    for i, (lat, lon, elev) in enumerate(coords):
        if i > 0:
            cumulative += _haversine_m(lats[-1], lons[-1], lat, lon)
        lats.append(lat)
        lons.append(lon)
        elevs.append(elev)
        dists.append(cumulative)

    return RouteStreams(lats=lats, lons=lons, elevs_m=elevs, distances_m=dists)


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


def _extract_points(root: ET.Element) -> list[tuple[float, float, float]]:
    # Try namespaced and un-namespaced trkpt / rtept
    for ns in GPX_NS_CANDIDATES:
        points = _find_points(root, ns)
        if points:
            return points
    # Fallback: any element named trkpt or rtept
    points: list[tuple[float, float, float]] = []
    for el in root.iter():
        tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if tag in ("trkpt", "rtept"):
            points.append(_point_from_el(el))
    return points


def _find_points(root: ET.Element, ns: dict[str, str]) -> list[tuple[float, float, float]]:
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


def _point_from_el(el: ET.Element) -> tuple[float, float, float]:
    lat = float(el.attrib["lat"])
    lon = float(el.attrib["lon"])
    elev = 0.0
    for child in el:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "ele" and child.text:
            elev = float(child.text)
            break
    return lat, lon, elev


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
