"""Ride-filled physics equations using the same force/solver functions as predict."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from bike_wind.analysis import eta_seconds, format_duration
from bike_wind.local_wind_map import format_wind_speed_dir
from bike_wind.physics.forces import (
    RiderBikeParams,
    WindComponents,
    aero_force,
    gravity_force,
    power_required,
    rolling_force,
    total_force,
)
from bike_wind.physics.solver import solve_speed
from bike_wind.predict import MPS_TO_MPH, PredictionResult, WhatIfResult

MPS_TO_MPH_LOCAL = MPS_TO_MPH


@dataclass(frozen=True)
class EquationLine:
    """One equation with symbols, plugged-in values, and a plain-language line."""

    title: str
    latex_or_text: str
    plain: str


@dataclass(frozen=True)
class PhysicsExplain:
    """Snapshot of the model at a representative sample on the loaded effort."""

    sample_index: int
    distance_mi: float
    lines: list[EquationLine]


def _mid_moving_index(actual: Sequence[float] | None, n: int) -> int:
    if n <= 0:
        return 0
    if not actual or len(actual) != n:
        return n // 2
    moving = [i for i, v in enumerate(actual) if math.isfinite(v) and v * MPS_TO_MPH > 3.0]
    if not moving:
        return n // 2
    return moving[len(moving) // 2]


def explain_prediction(
    result: PredictionResult,
    whatif: WhatIfResult,
    *,
    grade_scale: float = 1.0,
) -> PhysicsExplain:
    """Build equations for one sample using ``aero_force`` / ``solve_speed`` etc.

    Values come from the active what-if (recorded series or flat slider power,
    scaled wind/grade). Units in the plain text are mph, miles, watts.
    """
    n = len(result.distances_m)
    i = _mid_moving_index(result.actual_speeds_mps, n)
    dist_mi = result.distances_m[i] / 1609.344
    grade = float(result.grades[i]) * max(0.0, float(grade_scale))
    # Wind at this sample (sustained); gust if band exists
    if result.weather_ok and len(result.winds) == n:
        # Reconstruct what-if wind scale from mean speeds
        base = max(1e-9, float(result.mean_wind_speed_mps))
        scale = max(0.0, float(whatif.wind_speed_mps)) / base if result.weather_ok else 0.0
        w0 = result.winds[i]
        wind = WindComponents(
            along_track=w0.along_track * scale,
            cross_track=w0.cross_track * scale,
        )
    else:
        wind = WindComponents()

    if whatif.used_recorded_power and whatif.powers_w is not None and len(whatif.powers_w) == n:
        power_w = float(whatif.powers_w[i]) if math.isfinite(whatif.powers_w[i]) else float(whatif.power_w)
        power_note = f"recorded series at this point ({power_w:.0f} W)"
    else:
        power_w = float(whatif.power_w)
        power_note = f"flat slider ({power_w:.0f} W)"

    params = RiderBikeParams(
        rider_mass_kg=result.params.rider_mass_kg,
        power_w=max(0.0, power_w),
        bike_mass_kg=result.params.bike_mass_kg,
        cd=result.params.cd,
        area_m2=result.params.area_m2,
        crr=result.params.crr,
        rho=result.params.rho,
        tuck_on_descent=result.params.tuck_on_descent,
        drivetrain_efficiency=result.params.drivetrain_efficiency,
    )
    cda = params.cd * params.area_m2
    mass = params.total_mass_kg
    eta = params.drivetrain_efficiency
    rho = params.rho

    v_pred = solve_speed(params, grade=grade, wind=wind)
    v_mph = v_pred * MPS_TO_MPH
    f_aero = aero_force(v_pred, params, wind=wind, grade=grade)
    f_roll = rolling_force(params, grade)
    f_grav = gravity_force(params, grade)
    f_tot = total_force(v_pred, params, grade=grade, wind=wind)
    p_req = power_required(v_pred, params, grade=grade, wind=wind)

    wind_label = format_wind_speed_dir(whatif.wind_speed_mps, whatif.wind_from_deg)
    gust_note = ""
    if whatif.has_gust_band and result.mean_gust_speed_mps:
        gust_note = (
            f" Gust product ~{result.mean_gust_speed_mps * MPS_TO_MPH:.1f} mph "
            "shows as a band around the sustained-wind line, not a second spiked path."
        )

    eta_s = whatif.eta_with_wind_s
    dist_tot_mi = (
        (result.distances_m[-1] - result.distances_m[0]) / 1609.344
        if n >= 2
        else 0.0
    )

    lines = [
        EquationLine(
            title="Inputs at this point",
            latex_or_text=(
                f"P = {power_w:.0f} W ({power_note}); "
                f"m = {params.rider_mass_kg:.1f}+{params.bike_mass_kg:.1f} = {mass:.1f} kg; "
                f"Cd·A = {params.cd:.2f}×{params.area_m2:.3f} = {cda:.3f} m²; "
                f"Crr = {params.crr:.4f}; η = {eta:.2f}; ρ = {rho:.3f} kg/m³"
            ),
            plain="These are the same rider/bike numbers the speed solver uses on this effort.",
        ),
        EquationLine(
            title="Grade and wind",
            latex_or_text=(
                f"grade g = {grade * 100:.2f}% (rise/run); "
                f"wind {wind_label}; "
                f"along-track {wind.along_track * MPS_TO_MPH:+.1f} mph "
                f"(+tail / −head), cross {abs(wind.cross_track) * MPS_TO_MPH:.1f} mph."
                f"{gust_note}"
            ),
            plain="Sustained wind sets the predicted line and ETA; gusts only widen the band.",
        ),
        EquationLine(
            title="Forces at predicted speed",
            latex_or_text=(
                f"F_aero = ½ρ Cd A |v_rel| v_along = {f_aero:.1f} N; "
                f"F_roll = Crr m g cosθ = {f_roll:.1f} N; "
                f"F_grav = m g sinθ = {f_grav:.1f} N; "
                f"F_total = {f_tot:.1f} N"
            ),
            plain="Aero, rolling, and gravity are summed exactly as in total_force().",
        ),
        EquationLine(
            title="Power balance → speed",
            latex_or_text=(
                f"P_crank ≈ F_total · v / η → at v = {v_mph:.1f} mph, "
                f"required crank power ≈ {p_req:.0f} W "
                f"(available {power_w:.0f} W). "
                f"solve_speed finds v where required power matches P."
            ),
            plain="Predicted speed is the steady speed whose required watts match your power here.",
        ),
        EquationLine(
            title="Route time",
            latex_or_text=(
                f"ETA = ∫ ds/v ≈ {format_duration(eta_s)} "
                f"over {dist_tot_mi:.2f} mi (inertia-limited along the path)."
            ),
            plain="Time adds distance over predicted speed at each sample — same ETA as the results row.",
        ),
    ]
    return PhysicsExplain(sample_index=i, distance_mi=dist_mi, lines=lines)
