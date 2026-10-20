"""Compare model prediction to a past Strava activity under archive weather."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from bike_wind.analysis import format_duration
from bike_wind.physics.forces import RiderBikeParams
from bike_wind.predict import downsample_route, predict_speeds
from bike_wind.rider import resolve_aero, standing_height_to_wind_z_m
# Public package exports (avoids brittle deep imports if client layout shifts).
from bike_wind.strava import ActivitySummary, StravaClient


@dataclass(frozen=True)
class PastRideComparison:
    activity: ActivitySummary
    predicted_s: float
    actual_moving_s: float
    actual_elapsed_s: float
    power_w: float
    power_from_activity: bool
    weather_ok: bool
    weather_source: str
    delta_moving_s: float
    note: str

    @property
    def predicted_label(self) -> str:
        return format_duration(self.predicted_s)

    @property
    def actual_moving_label(self) -> str:
        return format_duration(float(self.actual_moving_s))

    @property
    def delta_moving_label(self) -> str:
        sign = "+" if self.delta_moving_s >= 0 else "−"
        return f"{sign}{format_duration(abs(self.delta_moving_s))}"


def parse_activity_start(iso: str) -> datetime:
    text = (iso or "").strip()
    if not text:
        raise ValueError("Activity has no start_date")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def compare_activity(
    client: StravaClient,
    activity: ActivitySummary,
    *,
    rider_mass_kg: float,
    bike_mass_kg: float,
    position_key: str,
    fallback_power_w: float,
    height_cm: float,
    full_aero_kit: bool = False,
    non_aero: bool = False,
) -> PastRideComparison:
    """Load activity geometry, archive weather, predict ETA, compare to moving time."""
    start = parse_activity_start(activity.start_date)
    route = client.load_activity_route(activity.activity_id)
    route = downsample_route(route, max_points=300)

    power_from_activity = activity.has_power
    power_w = float(activity.average_watts) if power_from_activity else float(fallback_power_w)

    aero = resolve_aero(
        position_key,
        full_aero_kit=full_aero_kit,
        non_aero=non_aero,
        bike_mass_kg=bike_mass_kg,
    )
    params = RiderBikeParams(
        rider_mass_kg=rider_mass_kg,
        power_w=power_w,
        bike_mass_kg=aero.bike_mass_kg,
        cd=aero.cd,
        area_m2=aero.area_m2,
    )
    wind_z = standing_height_to_wind_z_m(height_cm)
    result = predict_speeds(
        route,
        params,
        start,
        wind_height_m=wind_z,
        use_archive=True,
        prefer_hrrr=False,
    )
    predicted = result.eta_with_wind_s
    actual = float(activity.moving_time_s)
    delta = predicted - actual
    note_bits = [
        "Model assumes steady power along the route.",
        "Stops, coasting, drafting, and gusts are not modeled.",
        "Archive wind is hourly reanalysis near the start time.",
    ]
    if not power_from_activity:
        note_bits.insert(0, "No activity power — used your rider power setting.")
    return PastRideComparison(
        activity=activity,
        predicted_s=predicted,
        actual_moving_s=activity.moving_time_s,
        actual_elapsed_s=activity.elapsed_time_s,
        power_w=power_w,
        power_from_activity=power_from_activity,
        weather_ok=result.weather_ok,
        weather_source=result.weather_source,
        delta_moving_s=delta,
        note=" ".join(note_bits),
    )


def ride_activity_label(act: ActivitySummary) -> str:
    when = (act.start_date or "")[:10]
    km = act.distance_m / 1000.0
    watts = f"{act.average_watts:.0f} W" if act.has_power else "no power"
    return f"{when} · {act.name} · {km:.1f} km · {watts}"
