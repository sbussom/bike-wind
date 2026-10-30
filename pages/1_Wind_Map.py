"""GustyPower wind map: animated forecast wind with a time slider.

Wind comes from Open-Meteo on two nested grids (Florida + home area), cached
for three hours. The optional segment prediction calls the same
``predict_speeds`` the main page uses, with the rider settings from that page.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as components

from bike_wind.analysis import format_duration
from bike_wind.clothing import DEFAULT_CLOTHING_KEY, resolve_clothing
from bike_wind.gpx.parse import RouteStreams, parse_gpx
from bike_wind.physics.constants import DRIVETRAIN_EFFICIENCY
from bike_wind.physics.forces import RiderBikeParams
from bike_wind.predict import downsample_route, predict_speeds
from bike_wind.rider import (
    DEFAULT_POSITION_KEY,
    DEFAULT_RIDER_KG,
    DEFAULT_SURFACE_KEY,
    POSITION_PRESETS,
    resolve_aero,
    resolve_surface,
    smart_default_height_cm,
    standing_height_to_wind_z_m,
)
from bike_wind.weather import hrrr_enabled_by_env
from bike_wind.wind_field import (
    FLORIDA_GRID,
    HOME_LAT,
    HOME_LON,
    LOCAL_GRID,
    fetch_wind_field,
    frame_indices,
    sample_point,
    subset_payload,
)
from bike_wind.wind_map_html import build_wind_map_html

ROOT = Path(__file__).resolve().parents[1]
SEGMENT_FIXTURE = ROOT / "fixtures" / "segment_41392335.json"
TZ = ZoneInfo("America/New_York")
MAP_HEIGHT_PX = 520

st.set_page_config(
    page_title="GustyPower wind map",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def load_wind(forecast_days: int = 4) -> dict:
    """Gridded forecast + the home-spot series. Cached 3 h (Open-Meteo limits)."""
    field = fetch_wind_field((FLORIDA_GRID, LOCAL_GRID), forecast_days=forecast_days)
    local = field.grids[-1]
    home = [
        [round(x, 1) for x in sample_point(local, i, HOME_LAT, HOME_LON)]
        for i in range(len(field.times_utc))
    ]
    return {
        "payload": field.to_payload(),
        "times": [t.isoformat() for t in field.times_utc],
        "home": home,
        "fetched": datetime.now(timezone.utc).isoformat(),
    }


@st.cache_data(show_spinner=False)
def load_segment_fixture() -> dict:
    return json.loads(SEGMENT_FIXTURE.read_text())


def fixture_route(seg: dict) -> RouteStreams:
    lats = [float(p[0]) for p in seg["latlng"]]
    lons = [float(p[1]) for p in seg["latlng"]]
    return RouteStreams(
        lats=lats,
        lons=lons,
        elevs_m=[float(x) for x in seg["altitude"]],
        distances_m=[float(x) for x in seg["distance"]],
    )


def rider_params(power_w: float) -> tuple[RiderBikeParams, float]:
    """Same parameter build as the main page, from its session settings."""
    ss = st.session_state
    rider_kg = float(ss.get("rider_kg", DEFAULT_RIDER_KG))
    position_key = ss.get("position_key", DEFAULT_POSITION_KEY)
    bike_kg = float(ss.get("bike_kg", POSITION_PRESETS[DEFAULT_POSITION_KEY].bike_mass_kg))
    height_cm = float(ss.get("height_cm", smart_default_height_cm(DEFAULT_RIDER_KG)))
    aero = resolve_aero(
        position_key,
        full_aero_kit=bool(ss.get("full_aero_kit", False)),
        non_aero=bool(ss.get("non_aero", False)),
        bike_mass_kg=bike_kg,
    )
    surface = resolve_surface(ss.get("surface_key", DEFAULT_SURFACE_KEY))
    clothing = resolve_clothing(ss.get("clothing_key", DEFAULT_CLOTHING_KEY))
    from bike_wind.calibration.cda_defaults import YAW_CDA_ENABLED

    params = RiderBikeParams(
        rider_mass_kg=rider_kg,
        power_w=float(power_w),
        bike_mass_kg=aero.bike_mass_kg,
        cd=aero.cd,
        area_m2=aero.area_m2,
        crr=surface.crr,
        drivetrain_efficiency=DRIVETRAIN_EFFICIENCY,
        yaw_cda=YAW_CDA_ENABLED,
        clothing_cda_scale=clothing.cda_multiplier,
    )
    return params, standing_height_to_wind_z_m(height_cm)


def predict_at(route: RouteStreams, power_w: float, when_utc: datetime) -> tuple[float, float]:
    """(with-wind ETA s, calm ETA s) from the unchanged model."""
    params, wind_z = rider_params(power_w)
    result = predict_speeds(
        downsample_route(route, max_points=300),
        params,
        when_utc,
        wind_height_m=wind_z,
        prefer_hrrr=hrrr_enabled_by_env(),
    )
    return result.eta_with_wind_s, result.eta_no_wind_s


def when_label(t: datetime) -> str:
    local = t.astimezone(TZ)
    return local.strftime("%a %b %-d, %-I %p ") + local.tzname()


def main() -> None:
    st.html(
        """
        <div style="margin:-0.5rem 0 0.25rem 0">
          <p style="font-size:1.6rem;font-weight:800;margin:0">
            <span style="color:#3fb950">Gusty</span><span style="color:#58a6ff">Power</span> wind map
          </p>
          <p style="margin:0;color:#8b949e">Forecast wind around your rides. Press play or drag the slider.</p>
        </div>
        """
    )

    try:
        with st.spinner("Loading wind forecast grid…"):
            wind = load_wind()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the Open-Meteo wind grid: {exc}")
        return

    times_all = [datetime.fromisoformat(t) for t in wind["times"]]
    step = st.radio(
        "Slider step",
        options=[1, 3],
        format_func=lambda h: f"{h}-hourly",
        horizontal=True,
        key="wm_step",
    )
    idxs = frame_indices(times_all, step_hours=int(step))
    if not idxs:
        st.warning("No forecast hours left in the cached grid; refresh in a bit.")
        return
    times = [times_all[i] for i in idxs]
    payload = subset_payload(wind["payload"], idxs)
    home_series = [wind["home"][i] for i in idxs]

    # ---- segment + prediction controls ----
    seg = load_segment_fixture()
    route_choice = st.radio(
        "Route overlay",
        ["Round trip to pumphouse", "Upload GPX", "None"],
        horizontal=True,
        key="wm_route_choice",
    )
    route: RouteStreams | None = None
    route_name = ""
    if route_choice == "Round trip to pumphouse":
        route = fixture_route(seg)
        route_name = seg["name"]
    elif route_choice == "Upload GPX":
        up = st.file_uploader("GPX file", type=["gpx"], key="wm_gpx")
        if up is not None:
            try:
                route = parse_gpx(up.getvalue())
                route_name = up.name
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read GPX: {exc}")

    labels = [when_label(t) for t in times]
    sel = 0
    power_w = float(st.session_state.get("power_w", 220))
    if route is not None:
        c1, c2 = st.columns([3, 1])
        with c1:
            sel_label = st.select_slider(
                "Predict my time at",
                options=labels,
                value=labels[0],
                key=f"wm_when_{step}",
            )
            sel = labels.index(sel_label)
        with c2:
            power_w = float(
                st.number_input("Power (W)", 80, 450, int(power_w), 5, key="wm_power")
            )

    preds: dict[str, dict] = st.session_state.setdefault("wm_preds", {})
    pred_key = None
    if route is not None:
        when = times[sel]
        pred_key = f"{route_name}|{power_w:.0f}|{when.isoformat()}"
        if pred_key not in preds:
            try:
                with st.spinner("Solving GustyPower prediction…"):
                    eta, calm = predict_at(route, power_w, when)
                preds[pred_key] = {"eta": eta, "calm": calm}
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Prediction unavailable: {exc}")

    # Map labels for every prediction made this session on this route/power.
    map_preds: dict[str, str] = {}
    for t in times:
        key = f"{route_name}|{power_w:.0f}|{t.isoformat()}"
        if key in preds:
            p = preds[key]
            delta = p["eta"] - p["calm"]
            sign = "+" if delta >= 0 else "−"
            map_preds[t.strftime("%Y-%m-%dT%H:%MZ")] = (
                f"{route_name}: {format_duration(p['eta'])} @ {power_w:.0f} W "
                f"(wind {sign}{abs(delta):.0f}s)"
            )

    if route is not None:
        c_lat, c_lon = route.midpoint()
        center = (c_lat, c_lon)
        route_ll = [[round(a, 5), round(b, 5)] for a, b in zip(route.lats[::4], route.lons[::4])]
        route_ll.append([route.lats[-1], route.lons[-1]])
    else:
        center = (HOME_LAT, HOME_LON)
        route_ll = []

    html = build_wind_map_html(
        payload,
        center=center,
        zoom=11,
        home=(HOME_LAT, HOME_LON),
        home_label="Your spot",
        home_series=home_series,
        route=route_ll,
        route_label=route_name,
        predictions=map_preds,
        initial_frame=sel,
        height_px=MAP_HEIGHT_PX,
    )
    components.html(html, height=MAP_HEIGHT_PX + 6, scrolling=False)

    if route is not None and pred_key in preds:
        p = preds[pred_key]
        delta = p["eta"] - p["calm"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted time", format_duration(p["eta"]))
        m2.metric("Calm-air time", format_duration(p["calm"]))
        m3.metric("Wind effect", f"{'+' if delta >= 0 else '−'}{abs(delta):.0f} s")
        st.caption(
            f"{route_name} at {power_w:.0f} W, starting {labels[sel]}. Same GustyPower "
            "model and rider settings as the main page. The map's own slider "
            "scrubs the wind only; pick a time above to predict it."
        )

    fetched = datetime.fromisoformat(wind["fetched"]).astimezone(TZ)
    st.caption(
        f"10 m wind from Open-Meteo, {FLORIDA_GRID.size + LOCAL_GRID.size} grid points "
        f"(finer near home), fetched {fetched.strftime('%-I:%M %p')} {fetched.tzname()}; "
        "refreshes every 3 h. Zoom out to see the Florida peninsula."
    )


main()
