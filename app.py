"""Streamlit app: GPX → wind → speed chart with rider / aero controls."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from streamlit_folium import st_folium

import streamlit.components.v1 as components

from bike_wind.analysis import format_duration, parse_target_time
from bike_wind.gpx import parse_gpx
from bike_wind.local_wind_map import (
    DEFAULT_LAT,
    DEFAULT_LON,
    WINDY_EMBED_HEIGHT_PX,
    compass_label,
    fetch_pin_wind,
    windy_iframe_html,
)
from bike_wind.map_view import MAP_MIN_HEIGHT_PX, build_wind_route_map
from bike_wind.physics import RiderBikeParams
from bike_wind.predict import downsample_route, predict_speeds
from bike_wind.rider import (
    DEFAULT_POSITION_KEY,
    POSITION_PRESETS,
    RIDER_WEIGHT_PRESETS_KG,
    kg_to_lb,
    lb_to_kg,
    position_key_for_label,
    position_labels,
    resolve_aero,
    smart_default_height_cm,
    standing_height_to_wind_z_m,
)
try:
    from bike_wind.settings import strava_secrets_presence
except ImportError:  # pragma: no cover — stale site-packages on Cloud

    def strava_secrets_presence():
        return {
            "CLIENT_ID": False,
            "CLIENT_SECRET": False,
            "REDIRECT_URI": False,
            "ACCESS_TOKEN": False,
        }


from bike_wind.strava import (
    StravaAuthError,
    StravaClient,
    StravaRateLimitError,
    load_strava_config,
)
from bike_wind.strava.oauth import build_authorize_url, exchange_code
from bike_wind.weather import hrrr_enabled_by_env

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_route.gpx"

st.set_page_config(
    page_title="Bike Wind Speed",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def _apply_theme(dark: bool) -> None:
    if dark:
        st.markdown(
            """
            <style>
            .stApp { background-color: #0e1117; color: #fafafa; }
            [data-testid="stHeader"] { background: rgba(14,17,23,0.8); }
            div[data-testid="stMetricValue"] { color: #fafafa; }
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: #161b22; border: 1px solid #30363d; border-radius: 10px;
                padding: 0.4rem 0.6rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp { background-color: #f7f8fa; color: #111; }
            [data-testid="stHeader"] { background: rgba(247,248,250,0.9); }
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px;
                padding: 0.4rem 0.6rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


def _init_state() -> None:
    if "rider_kg" not in st.session_state:
        st.session_state.rider_kg = 75.0
    if "bike_kg" not in st.session_state:
        st.session_state.bike_kg = POSITION_PRESETS[DEFAULT_POSITION_KEY].bike_mass_kg
    if "bike_mass_manual" not in st.session_state:
        st.session_state.bike_mass_manual = False
    if "height_cm" not in st.session_state:
        st.session_state.height_cm = smart_default_height_cm(75.0)
    if "height_manual" not in st.session_state:
        st.session_state.height_manual = False
    if "position_key" not in st.session_state:
        st.session_state.position_key = DEFAULT_POSITION_KEY
    if "power_w" not in st.session_state:
        st.session_state.power_w = 220
    if "strava_meta" not in st.session_state:
        st.session_state.strava_meta = None
    if "strava_segment_id" not in st.session_state:
        st.session_state.strava_segment_id = None
    if "strava_access_token" not in st.session_state:
        st.session_state.strava_access_token = None
    if "starred_segments" not in st.session_state:
        st.session_state.starred_segments = None
    if "use_strava_segment" not in st.session_state:
        st.session_state.use_strava_segment = False
    if "map_lat" not in st.session_state:
        st.session_state.map_lat = DEFAULT_LAT
    if "map_lon" not in st.session_state:
        st.session_state.map_lon = DEFAULT_LON
    if "_last_run" not in st.session_state:
        st.session_state._last_run = None

    # Widget-backed keys (Streamlit owns these after first render)
    if "rider_kg_input" not in st.session_state:
        st.session_state.rider_kg_input = float(st.session_state.rider_kg)
    if "rider_lb_input" not in st.session_state:
        st.session_state.rider_lb_input = round(kg_to_lb(st.session_state.rider_kg), 1)
    if "bike_kg_input" not in st.session_state:
        st.session_state.bike_kg_input = float(st.session_state.bike_kg)
    if "bike_lb_input" not in st.session_state:
        st.session_state.bike_lb_input = round(kg_to_lb(st.session_state.bike_kg), 1)
    if "height_cm_input" not in st.session_state:
        st.session_state.height_cm_input = float(st.session_state.height_cm)
    if "map_lat_input" not in st.session_state:
        st.session_state.map_lat_input = float(st.session_state.map_lat)
    if "map_lon_input" not in st.session_state:
        st.session_state.map_lon_input = float(st.session_state.map_lon)


def _effective_strava_token(config) -> str | None:
    """Prefer session OAuth token, then STRAVA_ACCESS_TOKEN from env/secrets."""
    return st.session_state.strava_access_token or config.access_token


def _handle_oauth_callback(config) -> None:
    """Exchange ?code= from Strava redirect; never log tokens or payloads."""
    params = st.query_params
    code = params.get("code")
    if not code or st.session_state.get("_oauth_handled_code") == code:
        return
    if not config.oauth_ready:
        return
    try:
        tokens = exchange_code(
            client_id=config.client_id,  # type: ignore[arg-type]
            client_secret=config.client_secret,  # type: ignore[arg-type]
            code=str(code),
        )
        st.session_state.strava_access_token = tokens.access_token
        st.session_state._oauth_handled_code = code
        try:
            del st.query_params["code"]
            if "scope" in st.query_params:
                del st.query_params["scope"]
            if "state" in st.query_params:
                del st.query_params["state"]
        except Exception:  # noqa: BLE001
            pass
        st.success("Strava connected (read scope).")
    except Exception as exc:  # noqa: BLE001 — no response body in message
        st.error(f"Strava OAuth failed: {exc}")


def _on_reset_map_location() -> None:
    st.session_state.map_lat = DEFAULT_LAT
    st.session_state.map_lon = DEFAULT_LON
    st.session_state.map_lat_input = DEFAULT_LAT
    st.session_state.map_lon_input = DEFAULT_LON
    st.session_state._pin_auto_centered = False


def _queue_map_center(lat: float, lon: float) -> None:
    """Queue a pin update for the next render (before lat/lon widgets exist)."""
    st.session_state._pending_map_center = (round(float(lat), 4), round(float(lon), 4))


def _apply_pending_map_center() -> bool:
    """Apply queued pin center before map widgets instantiate. Returns True if applied."""
    pending = st.session_state.pop("_pending_map_center", None)
    if pending is None:
        return False
    lat, lon = pending
    st.session_state.map_lat = lat
    st.session_state.map_lon = lon
    st.session_state.map_lat_input = lat
    st.session_state.map_lon_input = lon
    st.session_state._pin_auto_centered = True
    return True


def _pin_needs_recenter(lat: float, lon: float, *, threshold_deg: float = 0.02) -> bool:
    cur_lat = float(st.session_state.get("map_lat", DEFAULT_LAT))
    cur_lon = float(st.session_state.get("map_lon", DEFAULT_LON))
    return abs(cur_lat - lat) > threshold_deg or abs(cur_lon - lon) > threshold_deg


def _select_strava_segment(
    segment_id: int,
    *,
    name: str,
    distance_m: float = 0.0,
    average_grade: float = 0.0,
    from_cache: bool = False,
    start_lat: float | None = None,
    start_lon: float | None = None,
) -> None:
    """Set active Strava segment and auto-enable 'use instead of GPX'."""
    st.session_state.strava_segment_id = segment_id
    st.session_state.strava_meta = {
        "segment_id": segment_id,
        "name": name,
        "distance_m": distance_m,
        "average_grade": average_grade,
        "from_cache": from_cache,
    }
    st.session_state.use_strava_segment = True
    if start_lat is not None and start_lon is not None and _pin_needs_recenter(start_lat, start_lon):
        _queue_map_center(start_lat, start_lon)


def _apply_rider_kg(kg: float, *, reset_height: bool = False) -> None:
    """Update canonical rider mass + dual-unit widget keys.

    Safe only from ``on_change`` / ``on_click`` callbacks (before widgets render).
    """
    kg = round(float(kg), 2)
    st.session_state.rider_kg = kg
    st.session_state.rider_kg_input = float(kg)
    st.session_state.rider_lb_input = round(kg_to_lb(kg), 1)
    if reset_height:
        st.session_state.height_manual = False
    if not st.session_state.height_manual:
        height = smart_default_height_cm(kg)
        st.session_state.height_cm = height
        st.session_state.height_cm_input = float(height)


def _apply_bike_kg(kg: float, *, manual: bool | None = None) -> None:
    """Update canonical bike mass + dual-unit widget keys (callback-only)."""
    kg = round(float(kg), 2)
    st.session_state.bike_kg = kg
    st.session_state.bike_kg_input = float(kg)
    st.session_state.bike_lb_input = round(kg_to_lb(kg), 1)
    if manual is not None:
        st.session_state.bike_mass_manual = manual


def _on_rider_kg() -> None:
    _apply_rider_kg(st.session_state.rider_kg_input)


def _on_rider_lb() -> None:
    _apply_rider_kg(lb_to_kg(st.session_state.rider_lb_input))


def _on_bike_kg() -> None:
    _apply_bike_kg(st.session_state.bike_kg_input, manual=True)


def _on_bike_lb() -> None:
    _apply_bike_kg(lb_to_kg(st.session_state.bike_lb_input), manual=True)


def _on_height_cm() -> None:
    st.session_state.height_cm = float(st.session_state.height_cm_input)
    st.session_state.height_manual = True


def _on_weight_preset(kg: float) -> None:
    """Button ``on_click`` — runs before the next script run's widgets."""
    _apply_rider_kg(kg, reset_height=True)


def _on_reset_bike_mass() -> None:
    preset_kg = POSITION_PRESETS[st.session_state.position_key].bike_mass_kg
    _apply_bike_kg(preset_kg, manual=False)


def main() -> None:
    _init_state()
    _apply_pending_map_center()
    dark = st.toggle("Dark theme", value=True)
    _apply_theme(dark)

    st.title("Bike wind speed")
    st.caption(
        "Local wind overview + GPX route prediction under forecast wind. "
        "Climbs are noisier when GPX elevation is coarse."
    )

    # --- Card 0: Local wind map (top) ---
    with st.container(border=True):
        st.subheader("Local wind")
        if st.session_state.get("_pin_auto_centered"):
            st.caption(
                "Windy.com wind overlay — pin centered on the loaded route midpoint. "
                "Tweak below or reset to SE Florida."
            )
        else:
            st.caption(
                "Windy.com wind overlay (speed colors + streamlines). "
                "Default: South Florida — pin auto-centers when you load a route."
            )
        loc1, loc2, loc3 = st.columns([1, 1, 1])
        with loc1:
            st.number_input(
                "Lat",
                min_value=-85.0,
                max_value=85.0,
                step=0.1,
                format="%.4f",
                key="map_lat_input",
            )
        with loc2:
            st.number_input(
                "Lon",
                min_value=-180.0,
                max_value=180.0,
                step=0.1,
                format="%.4f",
                key="map_lon_input",
            )
        with loc3:
            st.write("")
            st.write("")
            st.button(
                "SE Florida default",
                on_click=_on_reset_map_location,
                use_container_width=True,
            )

        map_lat = float(st.session_state.map_lat_input)
        map_lon = float(st.session_state.map_lon_input)
        st.session_state.map_lat = map_lat
        st.session_state.map_lon = map_lon

        pin = fetch_pin_wind(map_lat, map_lon)
        if pin is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric("Wind at pin", f"{pin.speed_mph:.1f} mph")
            m2.metric("From", f"{compass_label(pin.direction_deg)} ({pin.direction_deg:.0f}°)")
            if pin.temperature_c is not None:
                m3.metric("Temp", f"{pin.temperature_c * 9 / 5 + 32:.0f} °F")
            else:
                m3.metric("Temp", "—")
        else:
            st.caption("Pin wind sample unavailable (network). Map still loads.")

        components.html(
            windy_iframe_html(map_lat, map_lon, height_px=WINDY_EMBED_HEIGHT_PX),
            height=WINDY_EMBED_HEIGHT_PX + 8,
            scrolling=False,
        )
        st.caption("Embed © Windy.com — overlay=wind · metricWind=mph")

    # --- Card 1: Rider / aero ---
    with st.container(border=True):
        st.subheader("Rider & aero")

        labels = position_labels()
        current_label = POSITION_PRESETS[st.session_state.position_key].label
        position_label = st.radio(
            "Position",
            options=labels,
            index=labels.index(current_label),
            horizontal=True,
        )
        position_key = position_key_for_label(position_label)
        if position_key != st.session_state.position_key:
            st.session_state.position_key = position_key
            if not st.session_state.bike_mass_manual:
                _apply_bike_kg(POSITION_PRESETS[position_key].bike_mass_kg, manual=False)

        preset = POSITION_PRESETS[st.session_state.position_key]
        st.caption(
            f"Preset Cd={preset.cd:.2f}, A={preset.area_m2:.2f} m² · "
            f"default bike {preset.bike_mass_kg:.1f} kg"
        )

        st.markdown("**Rider weight**")
        rw1, rw2 = st.columns(2)
        with rw1:
            st.number_input(
                "kg",
                min_value=40.0,
                max_value=150.0,
                step=0.5,
                key="rider_kg_input",
                on_change=_on_rider_kg,
            )
        with rw2:
            st.number_input(
                "lbs",
                min_value=88.0,
                max_value=330.0,
                step=1.0,
                key="rider_lb_input",
                on_change=_on_rider_lb,
            )
        st.session_state.rider_kg = float(st.session_state.rider_kg_input)

        preset_cols = st.columns(len(RIDER_WEIGHT_PRESETS_KG))
        for col, kg in zip(preset_cols, RIDER_WEIGHT_PRESETS_KG):
            with col:
                st.button(
                    f"{int(kg)}",
                    key=f"wpreset_{int(kg)}",
                    use_container_width=True,
                    on_click=_on_weight_preset,
                    args=(kg,),
                )

        st.markdown("**Bike weight**")
        bw1, bw2 = st.columns(2)
        with bw1:
            st.number_input(
                "Bike kg",
                min_value=4.0,
                max_value=20.0,
                step=0.1,
                key="bike_kg_input",
                on_change=_on_bike_kg,
            )
        with bw2:
            st.number_input(
                "Bike lbs",
                min_value=9.0,
                max_value=44.0,
                step=0.2,
                key="bike_lb_input",
                on_change=_on_bike_lb,
            )
        st.session_state.bike_kg = float(st.session_state.bike_kg_input)
        if st.session_state.bike_mass_manual:
            st.button(
                "Reset bike weight to position default",
                on_click=_on_reset_bike_mass,
            )

        power_w = st.number_input(
            "Sustained power (W)",
            min_value=50,
            max_value=600,
            step=10,
            key="power_w",
        )

        a1, a2 = st.columns(2)
        with a1:
            full_aero = st.checkbox(
                "Full aero helmet + kit (−0.05 Cd, −0.02 A)",
                key="full_aero_kit",
            )
        with a2:
            non_aero = st.checkbox(
                "Non-aero / no helmet (+0.03 Cd)",
                key="non_aero",
            )
        if full_aero and non_aero:
            st.caption("Both kit toggles on — applying full aero kit only.")

        with st.expander("Advanced", expanded=False):
            st.number_input(
                "Rider height (cm)",
                min_value=140.0,
                max_value=210.0,
                step=0.5,
                key="height_cm_input",
                on_change=_on_height_cm,
                help="Maps to ~riding wind height for log-profile correction; effect is small.",
            )
            st.caption(
                f"Wind height ≈ {standing_height_to_wind_z_m(st.session_state.height_cm):.2f} m "
                f"(from {st.session_state.height_cm:.0f} cm stature)."
            )

        aero = resolve_aero(
            st.session_state.position_key,
            full_aero_kit=bool(full_aero),
            non_aero=bool(non_aero),
            bike_mass_kg=float(st.session_state.bike_kg),
        )
        st.caption(
            f"Effective Cd={aero.cd:.2f}, A={aero.area_m2:.2f} m² · "
            f"total mass {st.session_state.rider_kg + aero.bike_mass_kg:.1f} kg"
        )

    strava_cfg = load_strava_config()
    _handle_oauth_callback(strava_cfg)
    token = _effective_strava_token(strava_cfg)

    # --- Card 2: Route / time (GPX primary) ---
    with st.container(border=True):
        st.subheader("Route & time")
        default_dt = datetime.now(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        ) + timedelta(hours=3)
        when = st.datetime_input("Ride datetime (UTC)", value=default_dt)
        target_raw = st.text_input(
            "Optional KOM / target time (seconds or mm:ss)",
            value="",
            placeholder="e.g. 12:30 or 750",
            help="Leave blank to skip required-power vs sustainable comparison.",
        )
        prefer_hrrr = st.toggle(
            "Prefer NOAA HRRR wind (CONUS, ~3 km)",
            value=hrrr_enabled_by_env(),
            help=(
                "Uses Open-Meteo's gfs_hrrr model when the route is in CONUS. "
                "Falls back to Open-Meteo otherwise. Default off."
            ),
        )

        use_sample = st.checkbox("Use sample South Florida GPX", value=True)
        uploaded = st.file_uploader("Or upload a GPX file", type=["gpx"])

        use_strava = False
        strava_ref_text = ""
        with st.expander("Strava (optional)", expanded=False):
            st.caption(
                "Paste a segment URL/ID or pick a starred segment. "
                "GPX remains the default route source."
            )

            if token:
                st.success("Strava token available (secrets or Connect Strava).")
            elif strava_cfg.oauth_ready:
                auth_url = build_authorize_url(
                    strava_cfg.client_id,  # type: ignore[arg-type]
                    strava_cfg.redirect_uri,  # type: ignore[arg-type]
                )
                st.markdown(f"[Connect Strava (read scope)]({auth_url})")
            else:
                st.caption(
                    "No token yet — set `STRAVA_ACCESS_TOKEN` in secrets, or "
                    "`STRAVA_CLIENT_ID` / `SECRET` / `REDIRECT_URI` for Connect Strava. "
                    "GPX still works."
                )
                presence = strava_secrets_presence()
                st.caption(
                    "Secrets seen by app (set/missing only — reboot after saving):  \n"
                    f"`CLIENT_ID`: {'set' if presence['CLIENT_ID'] else 'missing'} · "
                    f"`CLIENT_SECRET`: {'set' if presence['CLIENT_SECRET'] else 'missing'} · "
                    f"`REDIRECT_URI`: {'set' if presence['REDIRECT_URI'] else 'missing'} · "
                    f"`ACCESS_TOKEN`: {'set' if presence['ACCESS_TOKEN'] else 'missing'}"
                )

            strava_ref_text = st.text_input(
                "Segment URL or ID",
                placeholder="https://www.strava.com/segments/229781",
                key="strava_ref_input",
            )
            do_resolve = st.button(
                "Resolve segment",
                disabled=not token or not strava_ref_text.strip(),
                key="strava_resolve_btn",
            )
            if do_resolve and token and strava_ref_text.strip():
                try:
                    client = StravaClient(strava_cfg, access_token=token)
                    sid, _ref = client.resolve_ref(strava_ref_text)
                    meta = client.get_segment_meta(sid)
                    _select_strava_segment(
                        sid,
                        name=meta.name,
                        distance_m=meta.distance_m,
                        average_grade=meta.average_grade,
                        from_cache=meta.from_cache,
                        start_lat=meta.start_lat,
                        start_lon=meta.start_lon,
                    )
                    if st.session_state.get("_pending_map_center") is not None:
                        st.rerun()
                except StravaRateLimitError as exc:
                    st.error(exc.status.message or str(exc))
                except StravaAuthError as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Resolve failed: {exc}")

            # Starred / saved segments
            if token:
                st.markdown("**Starred segments**")
                if st.button("Refresh starred", key="refresh_starred"):
                    try:
                        client = StravaClient(strava_cfg, access_token=token)
                        st.session_state.starred_segments = [
                            {
                                "id": s.segment_id,
                                "label": (
                                    f"{s.name} ({s.distance_m/1000:.1f} km, "
                                    f"{s.average_grade:.1f}%)"
                                ),
                                "distance_m": s.distance_m,
                                "average_grade": s.average_grade,
                                "name": s.name,
                            }
                            for s in client.list_starred()
                        ]
                    except StravaRateLimitError as exc:
                        st.error(exc.status.message or str(exc))
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Starred fetch failed: {exc}")

                starred = st.session_state.starred_segments
                if starred is None:
                    st.caption("Click **Refresh starred** to load saved segments.")
                elif not starred:
                    st.caption("No starred segments returned.")
                else:
                    labels = [s["label"] for s in starred]
                    picked = st.radio(
                        "Select a starred segment",
                        options=["(none)"] + labels,
                        key="starred_pick",
                    )
                    if picked == "(none)":
                        if st.session_state.get("use_strava_segment") and st.session_state.get(
                            "strava_segment_id"
                        ):
                            # Only clear if the active id is from this list
                            starred_ids = {s["id"] for s in starred}
                            if st.session_state.strava_segment_id in starred_ids:
                                st.session_state.use_strava_segment = False
                    else:
                        chosen = next(s for s in starred if s["label"] == picked)
                        _select_strava_segment(
                            int(chosen["id"]),
                            name=str(chosen.get("name") or picked.split(" (")[0]),
                            distance_m=float(chosen.get("distance_m") or 0.0),
                            average_grade=float(chosen.get("average_grade") or 0.0),
                        )

            # Explore stays gated
            with st.expander("Explore Segments", expanded=False):
                if not strava_cfg.extended_access:
                    st.caption(
                        "Requires Extended Access — set `STRAVA_EXTENDED_ACCESS=true` "
                        "when approved. Currently disabled."
                    )
                elif not token:
                    st.caption("Needs a Strava token to explore.")
                else:
                    st.caption("Top 10 ride segments in a bounding box.")
                    b1, b2, b3, b4 = st.columns(4)
                    sw_lat = b1.number_input("SW lat", value=26.05, format="%.4f")
                    sw_lon = b2.number_input("SW lon", value=-80.20, format="%.4f")
                    ne_lat = b3.number_input("NE lat", value=26.25, format="%.4f")
                    ne_lon = b4.number_input("NE lon", value=-80.05, format="%.4f")
                    if st.button("Explore top 10", key="explore_btn"):
                        try:
                            client = StravaClient(strava_cfg, access_token=token)
                            hits = client.explore([sw_lat, sw_lon, ne_lat, ne_lon])
                            st.session_state.explore_hits = [
                                {
                                    "id": h.segment_id,
                                    "label": (
                                        f"{h.name} ({h.distance_m/1000:.1f} km, "
                                        f"{h.average_grade:.1f}%)"
                                    ),
                                    "name": h.name,
                                    "distance_m": h.distance_m,
                                    "average_grade": h.average_grade,
                                }
                                for h in hits
                            ]
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Explore failed: {exc}")
                    explore = st.session_state.get("explore_hits") or []
                    if explore:
                        elabels = [h["label"] for h in explore]
                        epick = st.radio(
                            "Select explored segment",
                            options=["(none)"] + elabels,
                            key="explore_pick",
                        )
                        if epick != "(none)":
                            chosen = next(h for h in explore if h["label"] == epick)
                            _select_strava_segment(
                                int(chosen["id"]),
                                name=str(chosen.get("name") or epick.split(" (")[0]),
                                distance_m=float(chosen.get("distance_m") or 0.0),
                                average_grade=float(chosen.get("average_grade") or 0.0),
                            )

            if st.session_state.strava_meta and st.session_state.strava_segment_id:
                m = st.session_state.strava_meta
                st.info(
                    f"**{m['name']}** · id {m['segment_id']}"
                    + (
                        f" · {m['distance_m']/1000:.2f} km · avg grade {m['average_grade']:.1f}%"
                        if m.get("distance_m")
                        else ""
                    )
                )
                use_strava = st.checkbox(
                    "Use this Strava segment instead of GPX",
                    key="use_strava_segment",
                )

        # Preview-center Windy when a custom GPX is uploaded (before Run)
        if uploaded is not None and not use_sample and not use_strava:
            gpx_bytes = uploaded.getvalue()
            digest = hashlib.sha1(gpx_bytes).hexdigest()
            if st.session_state.get("_gpx_center_digest") != digest:
                try:
                    preview = parse_gpx(gpx_bytes)
                    mid_lat, mid_lon = preview.midpoint()
                    st.session_state._gpx_center_digest = digest
                    if _pin_needs_recenter(mid_lat, mid_lon):
                        _queue_map_center(mid_lat, mid_lon)
                        st.rerun()
                except Exception:  # noqa: BLE001 — keep form usable if GPX is bad
                    st.session_state._gpx_center_digest = digest

        run = st.button("Run prediction", type="primary")

    last_run = st.session_state.get("_last_run")
    if not run and not last_run:
        st.info("Adjust the wind pin, rider setup, and GPX — then run prediction.")
        return

    if run:
        target_time_s: float | None = None
        try:
            target_time_s = parse_target_time(target_raw)
        except ValueError as exc:
            st.error(f"Target time: {exc}")
            return

        try:
            if use_strava and st.session_state.strava_segment_id and token:
                client = StravaClient(strava_cfg, access_token=token)
                meta, route = client.load_segment_route(int(st.session_state.strava_segment_id))
                st.session_state.strava_meta = {
                    "segment_id": meta.segment_id,
                    "name": meta.name,
                    "distance_m": meta.distance_m,
                    "average_grade": meta.average_grade,
                    "from_cache": meta.from_cache,
                }
                source_label = f"Strava: {meta.name} ({meta.segment_id})"
            elif uploaded is not None and not use_sample:
                route = parse_gpx(uploaded.getvalue())
                source_label = uploaded.name
            else:
                route = parse_gpx(FIXTURE)
                source_label = FIXTURE.name
        except StravaRateLimitError as exc:
            st.error(exc.status.message or str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load route: {exc}")
            return

        route = downsample_route(route, max_points=300)
        params = RiderBikeParams(
            rider_mass_kg=float(st.session_state.rider_kg),
            power_w=float(power_w),
            bike_mass_kg=aero.bike_mass_kg,
            cd=aero.cd,
            area_m2=aero.area_m2,
        )
        wind_z = standing_height_to_wind_z_m(float(st.session_state.height_cm))

        with st.spinner("Fetching forecast and solving speeds…"):
            when_dt = when if isinstance(when, datetime) else datetime.combine(when, datetime.min.time())
            if when_dt.tzinfo is None:
                when_dt = when_dt.replace(tzinfo=timezone.utc)
            result = predict_speeds(
                route,
                params,
                when_dt,
                wind_height_m=wind_z,
                target_time_s=target_time_s,
                prefer_hrrr=prefer_hrrr,
            )

        mid_lat, mid_lon = route.midpoint()
        st.session_state._last_run = {
            "result": result,
            "source_label": source_label,
            "route_km": route.distances_m[-1] / 1000.0,
            "route_n": len(route),
            "position_key": aero.position_key,
            "cda": aero.cd * aero.area_m2,
        }
        if _pin_needs_recenter(mid_lat, mid_lon):
            _queue_map_center(mid_lat, mid_lon)
            st.rerun()

        source_label_out = source_label
        route_km = route.distances_m[-1] / 1000.0
        route_n = len(route)
        position_key_out = aero.position_key
        cda_out = aero.cd * aero.area_m2
    else:
        result = last_run["result"]
        source_label_out = last_run["source_label"]
        route_km = last_run["route_km"]
        route_n = last_run["route_n"]
        position_key_out = last_run["position_key"]
        cda_out = last_run["cda"]

    # --- Card 3: Route wind-colored map ---
    with st.container(border=True):
        st.subheader("Route map")
        st.caption(
            "Route colored teal (tailwind) → red (headwind). "
            f"{source_label_out} · {route_km:.1f} km"
        )
        try:
            fmap = build_wind_route_map(
                result.lats,
                result.lons,
                result.wind_along_mps,
                dark=dark,
                height_px=MAP_MIN_HEIGHT_PX,
                weather_ok=result.weather_ok,
            )
            st_folium(
                fmap,
                width=None,
                height=max(MAP_MIN_HEIGHT_PX, 360),
                returned_objects=[],
                use_container_width=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Map unavailable: {exc}")

    # --- Card 4: Results ---
    with st.container(border=True):
        st.subheader("Results")
        st.caption(
            f"{POSITION_PRESETS[position_key_out].label} · Cd·A={cda_out:.3f} · "
            f"{route_n} points"
        )
        st.info(
            "Decision aid, not a fortune teller. Forecast accuracy degrades past day 5; "
            "climbs are noisier when GPX elevation/grade is coarse."
        )
        with st.expander("Reliability & back-test notes", expanded=False):
            st.markdown(
                r"""
- **Flat roads:** expect roughly ±5% on predicted time vs known sustained-power files when
  wind and mass are well specified.
- **Climbs:** noisier — GPX grade noise and elevation smoothing dominate.
- **Assumptions (conservative):** open-grass roughness \(z_0=0.03\), tuck −15% area on
  descents only, crosswind adds drag (never free speed).
- **Wind sources:** Open-Meteo primary (cached). Optional HRRR via toggle / `BIKE_WIND_HRRR`.
                """
            )

        if result.weather_message and not result.weather_ok:
            st.warning(result.weather_message)
        elif result.weather_ok:
            src = "HRRR" if result.weather_source == "hrrr" else "Open-Meteo"
            st.success(
                f"{src} wind applied (ρ ≈ {result.rho_mean:.3f} kg/m³). "
                f"Mean along-track {result.mean_wind_along_mps:+.1f} m/s "
                f"(head {result.mean_headwind_mps:.1f} · tail {result.mean_tailwind_mps:.1f})."
            )
            if result.weather_message:
                st.caption(result.weather_message)

        band = result.time_band
        eta = result.eta_with_wind_s
        if band is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "ETA with wind",
                format_duration(band.eta_s),
                delta=f"band {format_duration(band.eta_low_s)}–{format_duration(band.eta_high_s)}",
                delta_color="off",
            )
            m2.metric("ETA calm air", format_duration(result.eta_no_wind_s))
            avg = (
                sum(result.speeds_with_wind_kmh) / len(result.speeds_with_wind_kmh)
                if result.speeds_with_wind_kmh
                else 0.0
            )
            m3.metric("Avg speed (wind)", f"{avg:.1f} km/h")
            m4.metric("Horizon", f"{band.horizon_days:.1f} d")
            st.caption(band.note)
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("ETA with wind", format_duration(eta))
            m2.metric("ETA calm air", format_duration(result.eta_no_wind_s))
            avg = (
                sum(result.speeds_with_wind_kmh) / len(result.speeds_with_wind_kmh)
                if result.speeds_with_wind_kmh
                else 0.0
            )
            m3.metric("Avg speed (wind)", f"{avg:.1f} km/h")

        if result.target_power is not None:
            tp = result.target_power
            st.markdown("**KOM / target power**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Required power", f"{tp.required_power_w:.0f} W")
            c2.metric("Your sustainable", f"{tp.sustainable_power_w:.0f} W")
            c3.metric("Ratio", f"{tp.ratio:.2f}×")
            if tp.implausible:
                st.error(tp.message)
            elif tp.achievable:
                st.success(tp.message)
            else:
                st.warning(tp.message)

        fig, ax = plt.subplots(figsize=(9, 4.2))
        bg = "#0e1117" if dark else "#f7f8fa"
        fg = "#fafafa" if dark else "#111111"
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.plot(
            result.distances_km,
            result.speeds_no_wind_kmh,
            label="No wind",
            color="#8b9cb3",
            linewidth=1.6,
            linestyle="--",
        )
        ax.plot(
            result.distances_km,
            result.speeds_with_wind_kmh,
            label="With wind" if result.weather_ok else "With wind (unavailable)",
            color="#3ecf8e" if dark else "#0b6e4f",
            linewidth=2.0,
        )
        ax.set_xlabel("Distance (km)", color=fg)
        ax.set_ylabel("Ground speed (km/h)", color=fg)
        ax.set_title("Speed vs distance", color=fg)
        ax.tick_params(colors=fg)
        for spine in ax.spines.values():
            spine.set_color("#444" if dark else "#ccc")
        legend = ax.legend(facecolor=bg, edgecolor="#444" if dark else "#ccc")
        for text in legend.get_texts():
            text.set_color(fg)
        ax.grid(True, alpha=0.25, color=fg)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if len(result.bins) > 1:
            st.markdown("**Distance bins**")
            rows = []
            for b in result.bins:
                flag = "⚠" if b.implausible else ""
                rows.append(
                    {
                        "Bin": b.index,
                        "km": f"{b.start_km:.1f}–{b.end_km:.1f}",
                        "Grade %": f"{b.avg_grade_pct:.1f}",
                        "Wind": b.wind_label,
                        "km/h wind": f"{b.speed_with_wind_kmh:.1f}",
                        "km/h calm": f"{b.speed_no_wind_kmh:.1f}",
                        "Time": format_duration(b.time_with_wind_s),
                        "P@pace W": f"{b.required_power_w:.0f}" if b.required_power_w else "—",
                        "Flag": flag,
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)
        elif len(result.bins) == 1:
            b = result.bins[0]
            st.caption(
                f"Whole-route summary: {b.wind_label}, grade {b.avg_grade_pct:.1f}%, "
                f"{b.speed_with_wind_kmh:.1f} km/h with wind vs {b.speed_no_wind_kmh:.1f} calm."
            )


if __name__ == "__main__":
    main()
