"""Paved / unpaved / off-map class for each route point from OpenStreetMap.

Ways tagged ``highway=*`` are fetched from the Overpass API one 0.05° tile
at a time (only the tiles the route touches) and cached on disk, so a route
ridden before needs no network. Each point is matched to the nearest road
within ``MATCH_RADIUS_M`` (roads beat a parallel sidewalk within
``ROAD_PREFERENCE_M``), then a ±``SMOOTH_HALF_M`` majority filter removes
flicker at junctions.

Classes:

* ``PAVED`` (1): ``surface`` asphalt / concrete / paved ..., or no surface tag
  on a road type that is paved by default (primary ... residential, service,
  cycleway).
* ``UNPAVED`` (0): ``surface`` sand / gravel / dirt / compacted ..., or
  ``highway`` track / path / bridleway with no tag.
* ``OFF_MAP`` (-1): no mapped road nearby (canal banks, levees, fields, trails
  OSM does not have). A failed lookup returns ``None`` instead of a class list
  so the caller keeps the production setup everywhere.

Any network or parse failure returns ``None`` and the caller keeps the
single-setup model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

PAVED = 1
UNPAVED = 0
OFF_MAP = -1

TILE_DEG = 0.05
MATCH_RADIUS_M = 20.0
ROAD_PREFERENCE_M = 15.0
SMOOTH_HALF_M = 40.0
OVERPASS_ENDPOINTS: tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
# Interactive pages should fail fast and fall back to the production setup.
# A union query that runs for the old 45–60 s Overpass budget felt hung.
OVERPASS_DEADLINE_S = 12.0
USER_AGENT = "GustyPower/0.6.28 (bike wind model; surface lookup)"

PAVED_SURFACES = frozenset(
    {
        "asphalt", "paved", "concrete", "concrete:plates", "concrete:lanes",
        "paving_stones", "sett", "chipseal", "metal", "bricks", "brick",
        "cobblestone", "unhewn_cobblestone", "wood", "rubber", "tartan",
    }
)
UNPAVED_SURFACES = frozenset(
    {
        "unpaved", "gravel", "fine_gravel", "compacted", "dirt", "earth",
        "ground", "sand", "grass", "mud", "pebblestone", "rock", "shells",
        "woodchips", "grass_paver", "stone", "soil", "limerock",
    }
)
PAVED_HIGHWAYS = frozenset(
    {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "motorway_link", "trunk_link", "primary_link", "secondary_link",
        "tertiary_link", "residential", "living_street", "service",
        "unclassified", "cycleway",
    }
)
UNPAVED_HIGHWAYS = frozenset({"track", "path", "bridleway"})
SIDEWALK_HIGHWAYS = frozenset({"footway", "pedestrian", "steps", "crossing"})
SKIP_HIGHWAYS = frozenset(
    {"proposed", "construction", "steps", "elevator", "platform", "corridor", "bus_stop"}
)
_GRID_DEG = 0.0005  # ~55 m spatial hash for segment lookup


def classify_tags(tags: dict) -> int:
    """Surface class of one OSM way from its tags."""
    s = str(tags.get("surface") or "").lower()
    if s in PAVED_SURFACES:
        return PAVED
    if s in UNPAVED_SURFACES:
        return UNPAVED
    tt = tags.get("tracktype")
    if tt == "grade1":
        return PAVED
    if tt in ("grade2", "grade3", "grade4", "grade5"):
        return UNPAVED
    hw = tags.get("highway", "")
    if hw in PAVED_HIGHWAYS:
        return PAVED
    if hw in UNPAVED_HIGHWAYS:
        return UNPAVED
    if hw == "footway" and tags.get("footway") == "sidewalk":
        return PAVED
    return OFF_MAP


def _cache_dir() -> Path:
    root = os.environ.get("GUSTYPOWER_OSM_CACHE")
    if root:
        p = Path(root)
    else:
        p = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "gustypower" / "osm_surface"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _tile_of(lat: float, lon: float) -> tuple[int, int]:
    return (int(math.floor(lat / TILE_DEG)), int(math.floor(lon / TILE_DEG)))


def _tile_path(tile: tuple[int, int]) -> Path:
    return _cache_dir() / f"tile_{tile[0]}_{tile[1]}.json"


def _bundled_dir() -> Path:
    """Shipped tiles, or ``GUSTYPOWER_OSM_BUNDLED`` when tests relocate them."""
    override = os.environ.get("GUSTYPOWER_OSM_BUNDLED")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "tiles"


def _bundled_tile_path(tile: tuple[int, int]) -> Path:
    return _bundled_dir() / f"tile_{tile[0]}_{tile[1]}.json"


def _fetch_tiles(
    tiles: Sequence[tuple[int, int]],
    timeout_s: float | None = None,
) -> list[dict]:
    """One Overpass union query for ``tiles``; raises on failure.

    ``timeout_s`` is the whole budget across endpoints (default
    ``OVERPASS_DEADLINE_S``). The Overpass ``[timeout:]`` is a couple of
    seconds shorter so the server answers before the client gives up.
    """
    import requests

    budget = OVERPASS_DEADLINE_S if timeout_s is None else float(timeout_s)
    budget = max(3.0, budget)
    server_timeout = max(1, int(budget) - 2)
    parts = []
    for a, b in tiles:
        s, w, n, e = a * TILE_DEG, b * TILE_DEG, (a + 1) * TILE_DEG, (b + 1) * TILE_DEG
        parts.append(f'way["highway"]({s:.4f},{w:.4f},{n:.4f},{e:.4f});')
    query = f"[out:json][timeout:{server_timeout}];({''.join(parts)});out tags geom;"
    deadline = time.monotonic() + budget
    last: Exception | None = None
    for ep in OVERPASS_ENDPOINTS:
        left = deadline - time.monotonic()
        if left <= 1.0:
            break
        try:
            r = requests.post(
                ep,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=left,
            )
            r.raise_for_status()
            return list(r.json().get("elements", []))
        except Exception as exc:  # noqa: BLE001 - any failure -> next endpoint
            last = exc
    raise RuntimeError(f"Overpass unavailable: {last!r}")


def _slim(el: dict) -> dict | None:
    geom = el.get("geometry")
    tags = el.get("tags") or {}
    if not geom or tags.get("highway") in SKIP_HIGHWAYS:
        return None
    return {
        "id": el.get("id"),
        "c": classify_tags(tags),
        "hw": tags.get("highway", ""),
        "g": [[round(p["lat"], 7), round(p["lon"], 7)] for p in geom],
    }


def _read_slim_tile(path: Path) -> list[dict] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, list):
        return None
    return data


def load_ways(
    tiles: Iterable[tuple[int, int]],
    *,
    fetch=None,
    timeout_s: float | None = None,
) -> list[dict]:
    """Slim ways for ``tiles``.

    Order: shipped tile, disk cache, then one Overpass call for the rest.
    ``fetch``, when passed, replaces the network call (tests). It is only
    invoked for tiles that are neither bundled nor cached.
    """
    tiles = sorted(set(tiles))
    ways: dict[int, dict] = {}
    missing: list[tuple[int, int]] = []
    for t in tiles:
        bundled = _read_slim_tile(_bundled_tile_path(t))
        if bundled is not None:
            for w in bundled:
                ways[w["id"]] = w
            continue
        cached = _read_slim_tile(_tile_path(t))
        if cached is not None:
            for w in cached:
                ways[w["id"]] = w
            continue
        missing.append(t)
    if missing:
        if fetch is None:
            def fetch(missing_tiles, _timeout=timeout_s):  # noqa: ANN001
                return _fetch_tiles(missing_tiles, timeout_s=_timeout)

        raw = fetch(missing)
        per_tile: dict[tuple[int, int], list[dict]] = {t: [] for t in missing}
        for el in raw:
            w = _slim(el)
            if w is None:
                continue
            ways[w["id"]] = w
            touched = {_tile_of(la, lo) for la, lo in w["g"]}
            for t in touched & set(missing):
                per_tile[t].append(w)
        for t, lst in per_tile.items():
            try:
                _tile_path(t).write_text(json.dumps(lst))
            except OSError:
                pass
    return list(ways.values())


class _SegmentIndex:
    def __init__(self, ways: Sequence[dict]):
        self.ways = list(ways)
        self.grid: dict[tuple[int, int], list] = defaultdict(list)
        g = _GRID_DEG
        for wi, w in enumerate(self.ways):
            pts = w["g"]
            for (la0, lo0), (la1, lo1) in zip(pts[:-1], pts[1:]):
                for i in range(int(math.floor(min(la0, la1) / g)) - 1, int(math.floor(max(la0, la1) / g)) + 2):
                    for j in range(int(math.floor(min(lo0, lo1) / g)) - 1, int(math.floor(max(lo0, lo1) / g)) + 2):
                        self.grid[(i, j)].append((wi, la0, lo0, la1, lo1))

    def lookup(self, lat: float, lon: float) -> int:
        cell = (int(math.floor(lat / _GRID_DEG)), int(math.floor(lon / _GRID_DEG)))
        kx = 111320.0 * math.cos(math.radians(lat))
        ky = 110540.0
        best_d, best_w = 1e18, -1
        road_d, road_w = 1e18, -1
        for wi, la0, lo0, la1, lo1 in self.grid.get(cell, ()):
            ax, ay = (lo0 - lon) * kx, (la0 - lat) * ky
            bx, by = (lo1 - lon) * kx, (la1 - lat) * ky
            dx, dy = bx - ax, by - ay
            ll = dx * dx + dy * dy
            t = 0.0 if ll == 0.0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / ll))
            px, py = ax + t * dx, ay + t * dy
            dd = px * px + py * py
            if dd < best_d:
                best_d, best_w = dd, wi
            if self.ways[wi]["hw"] not in SIDEWALK_HIGHWAYS and dd < road_d:
                road_d, road_w = dd, wi
        if road_w >= 0 and road_d <= ROAD_PREFERENCE_M**2:
            best_d, best_w = road_d, road_w
        if best_w < 0 or best_d > MATCH_RADIUS_M**2:
            return OFF_MAP
        return int(self.ways[best_w]["c"])


def smooth_classes(classes: Sequence[int], distances_m: Sequence[float], half_m: float = SMOOTH_HALF_M) -> list[int]:
    """Majority of paved vs unpaved within ±``half_m``; off-map if neither."""
    n = len(classes)
    cp = [0] * (n + 1)
    cu = [0] * (n + 1)
    for i, c in enumerate(classes):
        cp[i + 1] = cp[i] + (c == PAVED)
        cu[i + 1] = cu[i] + (c == UNPAVED)
    out = [OFF_MAP] * n
    lo = hi = 0
    for i in range(n):
        d = distances_m[i]
        while lo < n and distances_m[lo] < d - half_m:
            lo += 1
        if hi < i:
            hi = i
        while hi + 1 < n and distances_m[hi + 1] <= d + half_m:
            hi += 1
        p = cp[hi + 1] - cp[lo]
        u = cu[hi + 1] - cu[lo]
        if p + u:
            out[i] = PAVED if p >= u else UNPAVED
    return out


def classes_from_ways(ways, lats, lons, distances_m) -> list[int]:
    idx = _SegmentIndex(ways)
    raw = [
        idx.lookup(la, lo) if (math.isfinite(la) and math.isfinite(lo)) else OFF_MAP
        for la, lo in zip(lats, lons)
    ]
    dist = [float(d) if math.isfinite(d) else 0.0 for d in distances_m]
    for k in range(1, len(dist)):
        if dist[k] < dist[k - 1]:
            dist[k] = dist[k - 1]
    return smooth_classes(raw, dist)


_MEMO: dict[str, list[int]] = {}


def route_surface_classes(
    lats: Sequence[float],
    lons: Sequence[float],
    distances_m: Sequence[float],
    *,
    fetch=None,
    timeout_s: float | None = None,
    errors: list[str] | None = None,
) -> list[int] | None:
    """Per-point surface class, or ``None`` when OSM cannot be reached.

    ``None`` means "use the single gravel setup everywhere" (the production
    model). Set ``GUSTYPOWER_OSM_OFFLINE=1`` to use only bundled tiles and
    the disk cache. ``errors`` receives one short reason when the result is
    ``None`` because the lookup failed (not when the input is empty).
    """
    if not lats or len(lats) != len(lons) or len(lats) != len(distances_m):
        return None
    key = hashlib.sha1(
        json.dumps([round(float(x), 5) for x in list(lats) + list(lons)]).encode()
    ).hexdigest()
    if key in _MEMO:
        return list(_MEMO[key])
    tiles = {_tile_of(la, lo) for la, lo in zip(lats, lons) if math.isfinite(la) and math.isfinite(lo)}
    if not tiles:
        return None
    try:
        if fetch is None and os.environ.get("GUSTYPOWER_OSM_OFFLINE") == "1":
            def fetch(_missing):  # noqa: ANN001
                raise RuntimeError("offline")
        ways = load_ways(tiles, fetch=fetch, timeout_s=timeout_s)
        out = classes_from_ways(ways, lats, lons, distances_m)
    except Exception as exc:  # noqa: BLE001 - fallback to the single-setup model
        if errors is not None:
            errors.append(str(exc) or exc.__class__.__name__)
        return None
    _MEMO[key] = out
    return list(out)
