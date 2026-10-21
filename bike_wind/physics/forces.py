"""Force components for cycling power models."""

from __future__ import annotations

from dataclasses import dataclass

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

    @property
    def total_mass_kg(self) -> float:
        return self.rider_mass_kg + self.bike_mass_kg


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
    """
    wind = wind or WindComponents()
    area = effective_area(params, grade)
    airspeed_along = ground_speed_mps - wind.along_track
    # Relative airspeed magnitude includes crosswind; along component sets sign
    # of the force along the travel direction.
    v_rel = (airspeed_along**2 + wind.cross_track**2) ** 0.5
    if v_rel == 0.0:
        return 0.0
    # Force along track: ½ρCdA |v_rel| * airspeed_along
    return 0.5 * params.rho * params.cd * area * v_rel * airspeed_along


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
    import math

    return math.atan(grade)


def _sin(theta: float) -> float:
    import math

    return math.sin(theta)


def _cos(theta: float) -> float:
    import math

    return math.cos(theta)
