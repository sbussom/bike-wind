"""Experimental setups compared with the validated production prediction.

The prediction page never imports this module. Production parameters are built
the same way as ``app.py`` (no ``surface_aware``), then optional experiments
are re-solved on the weather that prediction already fetched.

Surface-aware rolling uses OpenStreetMap per point:

* paved → Crr 0.0052 and Cd·A × 0.95
* unpaved → Crr 0.010 and Cd·A × 1
* unmapped → production Crr 0.0052 and Cd·A × 1

A failed lookup does not invent a surface. The caller keeps the production
solve and says so.

The manual Cd·A multiplier and Crr are an unvalidated bike-setup override
(aero wheels, road tires, and similar). They are not a fitted model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bike_wind.analysis import eta_seconds
from bike_wind.calibration.cda_defaults import (
    SURFACE_OFFMAP_CDA_SCALE,
    SURFACE_OFFMAP_CRR,
    SURFACE_PAVED_CDA_SCALE,
    SURFACE_PAVED_CRR,
    SURFACE_UNPAVED_CDA_SCALE,
    SURFACE_UNPAVED_CRR,
)
from bike_wind.clothing import DEFAULT_CLOTHING_KEY, resolve_clothing
from bike_wind.physics.constants import CRR_PACKED_SAND, DRIVETRAIN_EFFICIENCY
from bike_wind.physics.forces import RiderBikeParams, WindComponents
from bike_wind.physics.solver import solve_speeds
from bike_wind.predict import PredictionResult
from bike_wind.rider import (
    DEFAULT_POSITION_KEY,
    DEFAULT_SURFACE_KEY,
    resolve_aero,
    resolve_surface,
)
from bike_wind.surface.osm_surface import OFF_MAP, PAVED, UNPAVED, route_surface_classes

# Interactive Overpass budget. Longer calls stalled the page and then fell
# back anyway; the page caches a success so the next run skips the network.
OSM_TIMEOUT_S = 12.0

MPS_TO_MPH = 2.2369362920544

# Unvalidated guess for 55 mm aero wheels and 28 mm road tires on smooth tarmac.
AERO_WHEELS_PRESET_NAME = "Aero wheels + road tires (unvalidated guess)"
AERO_WHEELS_CDA_MULTIPLIER = 0.95
AERO_WHEELS_CRR = 0.0045
CDA_MULTIPLIER_DEFAULT = 1.0
CDA_MULTIPLIER_MIN = 0.85
CDA_MULTIPLIER_MAX = 1.10
PRODUCTION_CRR = CRR_PACKED_SAND


@dataclass(frozen=True)
class SurfaceReport:
    """Outcome of one route's OSM surface lookup."""

    ok: bool
    classes: tuple[int, ...] | None
    error: str | None
    paved_share: float
    unpaved_share: float
    unmapped_share: float

    @property
    def fell_back(self) -> bool:
        """True when the lookup failed and the model must stay on production."""
        return not self.ok


@dataclass(frozen=True)
class SetupPrediction:
    """One column in the experimental comparison."""

    label: str
    eta_s: float
    avg_speed_mph: float
    delta_s: float
    delta_mph: float
    note: str = ""
    fell_back: bool = False


def build_production_params(
    *,
    rider_kg: float,
    power_w: float,
    position_key: str = DEFAULT_POSITION_KEY,
    surface_key: str = DEFAULT_SURFACE_KEY,
    clothing_key: str = DEFAULT_CLOTHING_KEY,
    bike_mass_kg: float | None = None,
    full_aero_kit: bool = False,
    non_aero: bool = False,
) -> RiderBikeParams:
    """Parameters the prediction page sends to ``predict_speeds``.

    ``surface_aware`` is left at its default (False). Do not set it here.
    """
    from bike_wind.calibration.cda_defaults import YAW_CDA_ENABLED

    aero = resolve_aero(
        position_key,
        full_aero_kit=full_aero_kit,
        non_aero=non_aero,
        bike_mass_kg=bike_mass_kg,
    )
    surface = resolve_surface(surface_key)
    clothing = resolve_clothing(clothing_key)
    return RiderBikeParams(
        rider_mass_kg=float(rider_kg),
        power_w=float(power_w),
        bike_mass_kg=aero.bike_mass_kg,
        cd=aero.cd,
        area_m2=aero.area_m2,
        crr=surface.crr,
        drivetrain_efficiency=DRIVETRAIN_EFFICIENCY,
        yaw_cda=YAW_CDA_ENABLED,
        clothing_cda_scale=clothing.cda_multiplier,
        power_cda=aero.power_cda,
    )


def apply_manual_setup(
    params: RiderBikeParams,
    *,
    cda_multiplier: float,
    crr: float,
) -> RiderBikeParams:
    """Scale Cd·A and replace Crr. Unvalidated. Does not turn on OSM surfaces."""
    mult = float(cda_multiplier)
    if not math.isfinite(mult) or mult <= 0.0:
        mult = CDA_MULTIPLIER_DEFAULT
    crr_v = float(crr)
    if not math.isfinite(crr_v) or crr_v <= 0.0:
        crr_v = float(params.crr)
    return params.with_updates(
        area_m2=params.area_m2 * mult,
        crr=crr_v,
        surface_aware=False,
    )


def apply_surface_setup(params: RiderBikeParams) -> RiderBikeParams:
    """Ask the solver to use per-point OSM Crr and Cd·A scale."""
    return params.with_updates(surface_aware=True)


def distance_shares(
    classes: list[int] | tuple[int, ...],
    distances_m: list[float] | tuple[float, ...],
) -> tuple[float, float, float]:
    """(paved, unpaved, unmapped) fractions of route distance.

    Each gap is charged to the class at its start. Returns zeros when the
    route has no length.
    """
    paved = unpaved = unmapped = 0.0
    n = min(len(classes), len(distances_m))
    for i in range(n - 1):
        dd = float(distances_m[i + 1]) - float(distances_m[i])
        if dd <= 0.0 or not math.isfinite(dd):
            continue
        kind = int(classes[i])
        if kind == PAVED:
            paved += dd
        elif kind == UNPAVED:
            unpaved += dd
        else:
            unmapped += dd
    total = paved + unpaved + unmapped
    if total <= 0.0:
        return 0.0, 0.0, 0.0
    return paved / total, unpaved / total, unmapped / total


def format_surface_shares(report: SurfaceReport) -> str:
    """One line for the page: share of the route in each class."""
    return (
        f"{100.0 * report.paved_share:.1f}% paved · "
        f"{100.0 * report.unpaved_share:.1f}% unpaved · "
        f"{100.0 * report.unmapped_share:.1f}% unmapped"
    )


def lookup_surface(
    lats: list[float],
    lons: list[float],
    distances_m: list[float],
    *,
    timeout_s: float = OSM_TIMEOUT_S,
    fetch=None,
) -> SurfaceReport:
    """Classify the route, or report a failure so the UI can fall back."""
    errors: list[str] = []
    classes = route_surface_classes(
        lats,
        lons,
        distances_m,
        fetch=fetch,
        timeout_s=timeout_s,
        errors=errors,
    )
    if classes is None or len(classes) != len(lats):
        detail = errors[0] if errors else "OpenStreetMap returned no surface classes"
        return SurfaceReport(
            ok=False,
            classes=None,
            error=detail,
            paved_share=0.0,
            unpaved_share=0.0,
            unmapped_share=0.0,
        )
    paved, unpaved, unmapped = distance_shares(classes, distances_m)
    return SurfaceReport(
        ok=True,
        classes=tuple(int(c) for c in classes),
        error=None,
        paved_share=paved,
        unpaved_share=unpaved,
        unmapped_share=unmapped,
    )


def average_speed_mph(distances_m: list[float], eta_s: float) -> float:
    """Distance / predicted time, in mph. Zero when the clock is empty."""
    if eta_s <= 0.0 or not math.isfinite(eta_s) or len(distances_m) < 2:
        return 0.0
    dist = float(distances_m[-1]) - float(distances_m[0])
    if dist <= 0.0 or not math.isfinite(dist):
        return 0.0
    return (dist / eta_s) * MPS_TO_MPH


def _initial_speed(result: PredictionResult) -> float | None:
    speeds = result.actual_speeds_mps
    if not speeds:
        return None
    for v in speeds:
        if math.isfinite(v) and v >= 0.0:
            return float(v)
    return None


def resimulate(
    result: PredictionResult,
    params: RiderBikeParams,
    surface_classes: list[int] | tuple[int, ...] | None = None,
) -> tuple[list[float], float]:
    """Re-solve on the production weather. Returns (speeds m/s, eta seconds).

    Uses the same grades, wind, power series, and initial speed as the
    production result. Passing the production parameters back (and no surface
    classes) reproduces that result.
    """
    n = len(result.distances_m)
    powers = None
    if result.watts is not None and len(result.watts) == n:
        powers = list(result.watts)
    if result.weather_ok and len(result.winds) == n:
        winds: list[WindComponents] = list(result.winds)
    else:
        winds = [WindComponents()] * n
    classes = None
    if (
        params.surface_aware
        and surface_classes is not None
        and len(surface_classes) == n
    ):
        classes = [int(c) for c in surface_classes]
    tuned = params.with_updates(rho=result.params.rho, power_w=result.params.power_w)
    gap: list[float] = []
    speeds = solve_speeds(
        tuned,
        result.grades,
        winds,
        powers_w=powers,
        distances_m=result.distances_m,
        initial_speed_mps=_initial_speed(result),
        gap_dt=gap,
        surface_classes=classes,
    )
    eta = eta_seconds(
        result.distances_m,
        speeds,
        gap_dt=gap,
        actual_speeds_mps=result.actual_speeds_mps,
    )
    return speeds, eta


def _column(
    result: PredictionResult,
    *,
    label: str,
    eta_s: float,
    production_eta: float,
    production_mph: float,
    note: str = "",
    fell_back: bool = False,
) -> SetupPrediction:
    mph = average_speed_mph(result.distances_m, eta_s)
    return SetupPrediction(
        label=label,
        eta_s=eta_s,
        avg_speed_mph=mph,
        delta_s=eta_s - production_eta,
        delta_mph=mph - production_mph,
        note=note,
        fell_back=fell_back,
    )


def compare_with_production(
    result: PredictionResult,
    *,
    surface_report: SurfaceReport | None = None,
    manual_cda_multiplier: float | None = None,
    manual_crr: float | None = None,
) -> list[SetupPrediction]:
    """Production column, then each experimental column that was requested.

    ``surface_report`` is set when the surface-aware option is on. A failed
    report still adds a column, marked as a fallback to production.
    ``manual_cda_multiplier`` / ``manual_crr`` together add the manual column.
    """
    prod_eta = result.eta_with_wind_s
    prod_mph = average_speed_mph(result.distances_m, prod_eta)
    columns = [
        _column(
            result,
            label="Production",
            eta_s=prod_eta,
            production_eta=prod_eta,
            production_mph=prod_mph,
            note="Validated model. Same solve as the prediction page.",
        )
    ]
    if surface_report is not None:
        if surface_report.fell_back or surface_report.classes is None:
            columns.append(
                _column(
                    result,
                    label="Surface-aware",
                    eta_s=prod_eta,
                    production_eta=prod_eta,
                    production_mph=prod_mph,
                    note=(
                        "Surface lookup failed. Fell back to the production model. "
                        + (surface_report.error or "")
                    ).strip(),
                    fell_back=True,
                )
            )
        else:
            params = apply_surface_setup(result.params)
            _speeds, eta = resimulate(result, params, surface_report.classes)
            columns.append(
                _column(
                    result,
                    label="Surface-aware",
                    eta_s=eta,
                    production_eta=prod_eta,
                    production_mph=prod_mph,
                    note=(
                        f"Unvalidated OSM setup. {format_surface_shares(surface_report)} "
                        f"(by distance). Paved Crr {SURFACE_PAVED_CRR:.4f}, "
                        f"Cd·A ×{SURFACE_PAVED_CDA_SCALE:.2f}; "
                        f"unpaved Crr {SURFACE_UNPAVED_CRR:.3f}, "
                        f"Cd·A ×{SURFACE_UNPAVED_CDA_SCALE:.2f}; "
                        f"unmapped Crr {SURFACE_OFFMAP_CRR:.4f}, "
                        f"Cd·A ×{SURFACE_OFFMAP_CDA_SCALE:.2f}."
                    ),
                )
            )
    if manual_cda_multiplier is not None and manual_crr is not None:
        params = apply_manual_setup(
            result.params,
            cda_multiplier=manual_cda_multiplier,
            crr=manual_crr,
        )
        _speeds, eta = resimulate(result, params)
        columns.append(
            _column(
                result,
                label="Manual setup",
                eta_s=eta,
                production_eta=prod_eta,
                production_mph=prod_mph,
                note=(
                    f"Unvalidated. Cd·A ×{float(manual_cda_multiplier):.2f}, "
                    f"Crr {float(manual_crr):.4f}, same on every point."
                ),
            )
        )
    return columns


def surface_rule_blurb() -> str:
    """Short rule shown next to the surface-aware option."""
    return (
        f"OpenStreetMap paved → Crr {SURFACE_PAVED_CRR:.4f} and Cd·A ×{SURFACE_PAVED_CDA_SCALE:.2f}. "
        f"OSM unpaved → Crr {SURFACE_UNPAVED_CRR:.3f}. "
        f"No mapped way → production values (Crr {SURFACE_OFFMAP_CRR:.4f}, Cd·A ×{SURFACE_OFFMAP_CDA_SCALE:.2f}). "
        "Not part of the validated model."
    )
