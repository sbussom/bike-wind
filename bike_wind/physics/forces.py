"""Force components for cycling power models."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from bike_wind.physics.constants import (
    AREA_DEFAULT_M2,
    BIKE_MASS_ROAD_KG,
    CD_DEFAULT,
    CRR_DEFAULT,
    DRIVETRAIN_EFFICIENCY,
    G,
    R_DRY_AIR,
    RHO_DEFAULT,
    TUCK_AREA_FACTOR,
)

# Extra CdA growth with yaw angle (radians). Crosswind already raises v_rel;
# this multiplies Cd·A on top. Kept off unless hold-out MAE drops.
YAW_CDA_K = 1.5
YAW_CDA_FACTOR_CAP = 1.25


@dataclass(frozen=True)
class RiderBikeParams:
    """Rider + bike configuration for force calculations."""

    rider_mass_kg: float
    power_w: float
    bike_mass_kg: float = BIKE_MASS_ROAD_KG
    cd: float = CD_DEFAULT
    area_m2: float = AREA_DEFAULT_M2
    crr: float = CRR_DEFAULT
    rho: float = RHO_DEFAULT
    tuck_on_descent: bool = True
    drivetrain_efficiency: float = DRIVETRAIN_EFFICIENCY
    # When True, multiply Cd·A by a yaw factor after v_rel is formed.
    yaw_cda: bool = False
    # Clothing Cd·A scale (1.0 = skin-tight aero; ~1.08 = loose jersey).
    clothing_cda_scale: float = 1.0

    @property
    def total_mass_kg(self) -> float:
        return self.rider_mass_kg + self.bike_mass_kg

    @property
    def cda_m2(self) -> float:
        """Base Cd·A before yaw (includes clothing scale)."""
        return self.cd * self.area_m2 * max(0.0, float(self.clothing_cda_scale))

    def with_updates(self, **kwargs: object) -> RiderBikeParams:
        """Return a copy with selected fields replaced (preserves yaw_cda etc.)."""
        return replace(self, **kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True)
class WindComponents:
    """Wind relative to the rider's direction of travel (m/s).

    along_track: positive = tailwind (helps).
    cross_track: absolute magnitude used; always adds drag.
    """

    along_track: float = 0.0
    cross_track: float = 0.0


def air_density(temperature_c: float, pressure_hpa: float) -> float:
    """Dry-air density from temperature (°C) and pressure (hPa)."""
    temperature_k = temperature_c + 273.15
    pressure_pa = pressure_hpa * 100.0
    if temperature_k <= 0.0 or pressure_pa <= 0.0:
        return RHO_DEFAULT
    return pressure_pa / (R_DRY_AIR * temperature_k)


def effective_area(params: RiderBikeParams, grade: float) -> float:
    """Frontal area, optionally tucked (−15%) on descents."""
    if params.tuck_on_descent and grade < 0.0:
        return params.area_m2 * TUCK_AREA_FACTOR
    return params.area_m2


def yaw_cda_factor(airspeed_along: float, cross_track: float, *, k: float = YAW_CDA_K) -> float:
    """Cd·A multiplier from yaw angle; 1.0 with no crosswind, capped at 1.25."""
    yaw = math.atan2(abs(cross_track), abs(airspeed_along) + 1e-6)
    return min(1.0 + k * yaw**2, YAW_CDA_FACTOR_CAP)


def aero_force(
    ground_speed_mps: float,
    params: RiderBikeParams,
    wind: WindComponents | None = None,
    grade: float = 0.0,
) -> float:
    """Aerodynamic drag force opposing motion (N).

    airspeed along-track = ground − along-track wind (tailwind reduces airspeed).
    Crosswind increases relative airspeed magnitude and thus drag; it never
    contributes free forward speed.

    Clothing multiplies Cd·A via ``params.clothing_cda_scale`` (default 1.0).
    When ``params.yaw_cda`` is True, Cd·A is also scaled by ``yaw_cda_factor``
    (extra yaw penalty on top of the crosswind term in ``v_rel``).
    """
    wind = wind or WindComponents()
    area = effective_area(params, grade)
    airspeed_along = ground_speed_mps - wind.along_track
    # Relative airspeed magnitude includes crosswind; along component sets sign
    # of the force along the travel direction.
    v_rel = (airspeed_along**2 + wind.cross_track**2) ** 0.5
    if v_rel == 0.0:
        return 0.0
    cda = params.cd * area * max(0.0, float(params.clothing_cda_scale))
    if params.yaw_cda:
        cda *= yaw_cda_factor(airspeed_along, wind.cross_track)
    # Force along track: ½ρ CdA |v_rel| * airspeed_along
    return 0.5 * params.rho * cda * v_rel * airspeed_along


def rolling_force(params: RiderBikeParams, grade: float) -> float:
    """Rolling resistance (N). grade is rise/run (tan θ ≈ sin θ for small grades)."""
    theta = _grade_to_theta(grade)
    return params.crr * params.total_mass_kg * G * _cos(theta)


def gravity_force(params: RiderBikeParams, grade: float) -> float:
    """Gravitational force along the road (N); positive on climbs."""
    theta = _grade_to_theta(grade)
    return params.total_mass_kg * G * _sin(theta)


def total_force(
    ground_speed_mps: float,
    params: RiderBikeParams,
    grade: float = 0.0,
    wind: WindComponents | None = None,
) -> float:
    """Sum of aero + rolling + gravity opposing/aiding motion (N)."""
    return (
        aero_force(ground_speed_mps, params, wind=wind, grade=grade)
        + rolling_force(params, grade)
        + gravity_force(params, grade)
    )


def power_required(
    ground_speed_mps: float,
    params: RiderBikeParams,
    grade: float = 0.0,
    wind: WindComponents | None = None,
) -> float:
    """Crank power (W) needed for this ground speed.

    Wheel mechanical power is F·v; crank power divides by drivetrain
    efficiency (~3% loss) so meter watts compare to this value.
    """
    if ground_speed_mps <= 0.0:
        return 0.0
    mech = total_force(ground_speed_mps, params, grade=grade, wind=wind) * ground_speed_mps
    eta = params.drivetrain_efficiency
    if eta <= 0.0:
        return mech
    return mech / eta


def _grade_to_theta(grade: float) -> float:
    """Convert grade (rise/run) to road angle θ in radians."""
    # grade = tan(θ); use atan for correctness on steep pitches
    return math.atan(grade)


def _sin(theta: float) -> float:
    return math.sin(theta)


def _cos(theta: float) -> float:
    return math.cos(theta)
