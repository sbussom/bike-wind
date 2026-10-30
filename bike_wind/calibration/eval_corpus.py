"""Evaluate the speed model on a local corpus of Strava stream JSON files.

Expected layout (not in git)::

    <corpus_dir>/
      ride_<activity_id>_streams.json   # Strava streams dict
      sample_manifest.json              # optional metadata

Each streams file needs ``distance``, ``altitude``, ``latlng``, ``time``,
``velocity_smooth``, and ``watts``. Rides without watts are skipped.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from bike_wind.calibration.metrics import RideSpeedError, speed_errors_mph
from bike_wind.gpx.parse import RouteStreams
from bike_wind.physics.constants import DRIVETRAIN_EFFICIENCY
from bike_wind.physics.forces import RiderBikeParams
from bike_wind.physics.solver import solve_speeds
from bike_wind.predict import power_series_3s
from bike_wind.rider import DEFAULT_RIDER_KG, resolve_aero, resolve_surface


@dataclass(frozen=True)
class CorpusRideResult:
    activity_id: int
    error: RideSpeedError
    median_lat: float | None
    region: str  # "florida" | "travel" | "unknown"


@dataclass(frozen=True)
class CorpusEvalSummary:
    n_rides: int
    mean_signed_mph: float
    mae_mph: float
    rides: list[CorpusRideResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rides": self.n_rides,
            "mean_signed_mph": self.mean_signed_mph,
            "mae_mph": self.mae_mph,
            "rides": [
                {
                    "activity_id": r.activity_id,
                    "median_lat": r.median_lat,
                    "region": r.region,
                    **asdict(r.error),
                }
                for r in self.rides
            ],
        }


def _florida(lat: float | None) -> str:
    if lat is None or not math.isfinite(lat):
        return "unknown"
    return "florida" if 24.5 <= lat <= 31.0 else "travel"


def iter_stream_files(corpus_dir: Path) -> Iterator[Path]:
    yield from sorted(corpus_dir.glob("ride_*_streams.json"))


def load_route_from_streams(path: Path) -> tuple[int, RouteStreams, float | None] | None:
    raw = json.loads(path.read_text())
    if "watts" not in raw or "velocity_smooth" not in raw:
        return None
    if not any(w for w in raw["watts"] if w):
        return None
    stem = path.stem  # ride_<id>_streams
    try:
        activity_id = int(stem.split("_")[1])
    except (IndexError, ValueError):
        activity_id = 0
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    times = [t0 + timedelta(seconds=int(t)) for t in raw["time"]]
    lats = [ll[0] for ll in raw["latlng"]]
    med = sorted(lats)[len(lats) // 2] if lats else None
    route = RouteStreams(
        lats=lats,
        lons=[ll[1] for ll in raw["latlng"]],
        elevs_m=[float(x) for x in raw["altitude"]],
        distances_m=[float(x) for x in raw["distance"]],
        times_utc=times,
        actual_speeds_mps=[
            float(x) if x is not None else float("nan") for x in raw["velocity_smooth"]
        ],
        watts=[float(x) if x is not None else float("nan") for x in raw["watts"]],
    )
    return activity_id, route, med


def default_params() -> RiderBikeParams:
    from bike_wind.calibration.cda_defaults import YAW_CDA_ENABLED

    aero = resolve_aero("gravel_hoods")
    surface = resolve_surface("packed_sand")
    return RiderBikeParams(
        rider_mass_kg=DEFAULT_RIDER_KG,
        power_w=200.0,
        bike_mass_kg=aero.bike_mass_kg,
        cd=aero.cd,
        area_m2=aero.area_m2,
        crr=surface.crr,
        drivetrain_efficiency=DRIVETRAIN_EFFICIENCY,
        yaw_cda=YAW_CDA_ENABLED,
        power_cda=aero.power_cda,
    )


def evaluate_corpus(
    corpus_dir: Path | str,
    params: RiderBikeParams | None = None,
    *,
    min_points: int = 50,
) -> CorpusEvalSummary:
    """Run inertia-limited prediction on every power ride in ``corpus_dir``."""
    params = params or default_params()
    root = Path(corpus_dir)
    rides: list[CorpusRideResult] = []
    for path in iter_stream_files(root):
        loaded = load_route_from_streams(path)
        if loaded is None:
            continue
        activity_id, route, med = loaded
        powers, _ = power_series_3s(route.watts, route.times_utc)
        v0 = None
        if route.actual_speeds_mps:
            for v in route.actual_speeds_mps:
                if math.isfinite(v) and v >= 0.0:
                    v0 = float(v)
                    break
        pred = solve_speeds(
            params,
            route.grades(),
            powers_w=powers,
            distances_m=route.distances_m,
            initial_speed_mps=v0,
        )
        err = speed_errors_mph(pred, route.actual_speeds_mps or [])
        if err is None or err.n < min_points:
            continue
        rides.append(
            CorpusRideResult(
                activity_id=activity_id,
                error=err,
                median_lat=med,
                region=_florida(med),
            )
        )
    if not rides:
        return CorpusEvalSummary(0, 0.0, 0.0, [])
    return CorpusEvalSummary(
        n_rides=len(rides),
        mean_signed_mph=sum(r.error.mean_signed_mph for r in rides) / len(rides),
        mae_mph=sum(r.error.mae_mph for r in rides) / len(rides),
        rides=rides,
    )
