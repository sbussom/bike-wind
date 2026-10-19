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
    from bike_wind.settings import strava_secrets_debug, strava_secrets_presence
except ImportError:  # pragma: no cover — stale site-packages on Cloud

    def strava_secrets_presence():
        return {
            "CLIENT_ID": False,
            "CLIENT_SECRET": False,
            "REDIRECT_URI": False,
            "ACCESS_TOKEN": False,
        }

    def strava_secrets_debug():
        return {"available": False, "keys": [], "nested_strava": False}


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
    page_title="Bike Wind",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def _apply_theme(dark: bool) -> None:
    """Calm athletic theme: Syne + Figtree, teal forest accent, soft gradient.

    Use ``st.html`` (not ``st.markdown``) so ``<style>`` is injected. Recent
    Streamlit versions sanitize markdown HTML and strip style tags, which made
    the stylesheet print as visible plain text on the page.
    """
    if dark:
        vars_block = """
        :root {
          --bw-bg0: #0c1210;
          --bw-bg1: #14201b;
          --bw-surface: rgba(22, 32, 28, 0.72);
          --bw-text: #e7efea;
          --bw-muted: #8fa398;
          --bw-accent: #3ecf8e;
          --bw-accent-dim: #1f6b4a;
          --bw-line: rgba(62, 207, 142, 0.18);
          --bw-metric: #f3faf6;
        }
        """
    else:
        vars_block = """
        :root {
          --bw-bg0: #eef3f0;
          --bw-bg1: #f7faf8;
          --bw-surface: rgba(255, 255, 255, 0.82);
          --bw-text: #14201b;
          --bw-muted: #5c6f66;
          --bw-accent: #1a6b52;
          --bw-accent-dim: #0f4334;
          --bw-line: rgba(26, 107, 82, 0.16);
          --bw-metric: #0f241c;
        }
        """
    # st.html keeps <style>; markdown would strip tags and leak CSS as text.
    st.html(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600&family=Syne:wght@600;700;800&display=swap');
        {vars_block}
        .stApp {{
          background:
            radial-gradient(1200px 600px at 12% -10%, rgba(62, 207, 142, 0.10), transparent 55%),
            radial-gradient(900px 500px at 90% 0%, rgba(26, 107, 82, 0.12), transparent 50%),
            linear-gradient(165deg, var(--bw-bg0), var(--bw-bg1) 55%, var(--bw-bg0));
          color: var(--bw-text);
          font-family: "Figtree", sans-serif;
        }}
        [data-testid="stHeader"] {{
          background: transparent;
        }}
        h1, h2, h3, .bw-brand {{
          font-family: "Syne", sans-serif !important;
          letter-spacing: -0.02em;
          font-weight: 700 !important;
        }}
        .bw-hero {{
          padding: 0.4rem 0 1.1rem 0;
          border-bottom: 1px solid var(--bw-line);
          margin-bottom: 1.25rem;
        }}
        .bw-brand {{
          font-size: clamp(2.4rem, 7vw, 3.4rem);
          line-height: 1.05;
          margin: 0;
          color: var(--bw-text);
        }}
        .bw-tagline {{
          margin: 0.55rem 0 0 0;
          color: var(--bw-muted);
          font-size: 1.05rem;
          max-width: 34rem;
          line-height: 1.45;
        }}
        .bw-section-label {{
          font-family: "Syne", sans-serif;
          font-size: 0.78rem;
          text-transform: uppercase;
          letter-spacing: 0.14em;
          color: var(--bw-accent);
          margin: 0 0 0.35rem 0;
        }}
        div[data-testid="stMetricValue"] {{ color: var(--bw-metric); }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
          background: var(--bw-surface);
          border: 1px solid var(--bw-line);
          border-radius: 14px;
          padding: 0.85rem 1rem 1rem;
          backdrop-filter: blur(8px);
        }}
        div.stButton > button {{
          min-height: 44px;
          border-radius: 10px;
          font-family: "Figtree", sans-serif;
          font-weight: 600;
        }}
        div.stButton > button[kind="primary"] {{
          background: var(--bw-accent) !important;
          border-color: var(--bw-accent) !important;
          color: #04140e !important;
        }}
        .bw-run-wrap {{
          position: sticky;
          bottom: 0.5rem;
          z-index: 20;
          padding: 0.65rem 0 0.25rem;
          background: linear-gradient(180deg, transparent, var(--bw-bg1) 35%);
        }}
        @media (max-width: 640px) {{
          .bw-brand {{ font-size: 2.35rem; }}
          .bw-tagline {{ font-size: 0.98rem; }}
        }}
        </style>
        """
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
    if "strava_refresh_token" not in st.session_state:
        st.session_state.strava_refresh_token = None
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
    """Prefer this visitor's session OAuth token, then deployer secrets token."""
    return st.session_state.strava_access_token or config.access_token


def _token_source(config) -> str:
    if st.session_state.strava_access_token:
        return "session"
    if config.access_token:
        return "secrets"
    return "none"


def _handle_oauth_callback(config) -> None:
    """Exchange ?code= from Strava redirect into session_state (never secrets)."""
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
        st.session_state.strava_refresh_token = tokens.refresh_token
        st.session_state.starred_segments = None  # reload under the new account
        st.session_state._oauth_handled_code = code
        try:
            del st.query_params["code"]
            if "scope" in st.query_params:
                del st.query_params["scope"]
            if "state" in st.query_params:
                del st.query_params["state"]
        except Exception:  # noqa: BLE001
            pass
        st.success("Strava connected for this browser session.")
    except Exception as exc:  # noqa: BLE001 — no response body in message
        st.error(f"Strava OAuth failed: {exc}")


def _on_disconnect_strava() -> None:
    st.session_state.strava_access_token = None
    st.session_state.strava_refresh_token = None
    st.session_state.starred_segments = None
    st.session_state.use_strava_segment = False


def _load_starred_into_state(client: StravaClient) -> int:
    """Fetch all starred pages into session_state. Returns count."""
    starred = client.list_starred()
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
        for s in starred
    ]
    return len(st.session_state.starred_segments)


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

    top_l, top_r = st.columns([4, 1])
    with top_r:
        dark = st.toggle("Dark", value=True, help="Dark / light theme")
    _apply_theme(dark)

    with top_l:
        st.html(
            """
            <div class="bw-hero">
              <p class="bw-brand">Bike Wind</p>
              <p class="bw-tagline">
                Predict ground speed on your route under forecast wind —
                set when and where, dial rider power, run.
              </p>
            </div>
            """
        )

    strava_cfg = load_strava_config()
    _handle_oauth_callback(strava_cfg)
    token = _effective_strava_token(strava_cfg)
    tok_src = _token_source(strava_cfg)

    default_dt = datetime.now(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    ) + timedelta(hours=3)

    map_lat = float(st.session_state.map_lat_input)
    map_lon = float(st.session_state.map_lon_input)
    st.session_state.map_lat = map_lat
    st.session_state.map_lon = map_lon

    use_strava = False
    use_sample = True
    uploaded = None
    power_w = st.session_state.power_w
    full_aero = st.session_state.get("full_aero_kit", False)
    non_aero = st.session_state.get("non_aero", False)
    prefer_hrrr = hrrr_enabled_by_env()
    target_raw = ""
    when = default_dt

    # Desktop: map left · segment + rider right. Mobile stacks left→right.
    left_col, right_col = st.columns([1.35, 1], gap="large")

    with left_col:
        with st.container(border=True):
            st.html('<p class="bw-section-label">Wind map</p>')
            when = st.datetime_input("Ride datetime (UTC)", value=default_dt)

            pin = fetch_pin_wind(map_lat, map_lon)
            if pin is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("Wind", f"{pin.speed_mph:.1f} mph")
                m2.metric("From", f"{compass_label(pin.direction_deg)}")
                if pin.temperature_c is not None:
                    m3.metric("Temp", f"{pin.temperature_c * 9 / 5 + 32:.0f} °F")
                else:
                    m3.metric("Temp", "—")
            else:
                st.caption("Pin wind sample unavailable — map still loads.")

            windy_h = max(WINDY_EMBED_HEIGHT_PX, 380)
            components.html(
                windy_iframe_html(map_lat, map_lon, height_px=windy_h),
                height=windy_h + 8,
                scrolling=False,
            )

            with st.expander("Map pin", expanded=False):
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
                        "SE Florida",
                        on_click=_on_reset_map_location,
                        use_container_width=True,
                    )
                st.caption("Pin auto-centers on the loaded route after you run.")

    with right_col:
        # --- Enter segment (top of right column) ---
        with st.container(border=True):
            st.html('<p class="bw-section-label">Enter segment</p>')
            st.subheader("Route source")

            if tok_src == "session":
                st.caption("Using **your** Strava account (this browser session).")
                st.button(
                    "Disconnect Strava",
                    on_click=_on_disconnect_strava,
                    key="disconnect_strava",
                )
            elif tok_src == "secrets":
                st.caption(
                    "Using the **app owner's** token — Connect Strava to load "
                    "**your** starred segments instead."
                )
            else:
                st.caption("Connect Strava or paste a segment ID. GPX still works.")

            if strava_cfg.oauth_ready and tok_src != "session":
                auth_url = build_authorize_url(
                    strava_cfg.client_id,  # type: ignore[arg-type]
                    strava_cfg.redirect_uri,  # type: ignore[arg-type]
                )
                st.link_button(
                    "Connect with Strava",
                    auth_url,
                    use_container_width=True,
                )
                st.caption(
                    "Authorization Callback Domain / redirect URI must be "
                    f"`{strava_cfg.redirect_uri}` in your Strava API app."
                )
            elif not strava_cfg.oauth_ready and tok_src == "none":
                st.caption(
                    "Owner: set `client_id` / `client_secret` / `redirect_uri` "
                    "in secrets to enable Connect for visitors."
                )

            if token:
                if st.button("Load starred segments", key="refresh_starred"):
                    try:
                        client = StravaClient(strava_cfg, access_token=token)
                        n = _load_starred_into_state(client)
                        st.success(f"Loaded {n} starred segment{'s' if n != 1 else ''}.")
                    except StravaRateLimitError as exc:
                        st.error(exc.status.message or str(exc))
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Starred fetch failed: {exc}")

                starred = st.session_state.starred_segments
                if starred:
                    st.caption(f"{len(starred)} starred")
                    labels_s = [s["label"] for s in starred]
                    picked = st.selectbox(
                        "Starred segment",
                        options=["(none)"] + labels_s,
                        key="starred_pick",
                    )
                    if picked == "(none)":
                        if st.session_state.get("use_strava_segment") and st.session_state.get(
                            "strava_segment_id"
                        ):
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
                elif starred is None:
                    st.caption("Load starred to pick one (all pages fetched).")
                else:
                    st.caption("No starred segments returned.")

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

            if st.session_state.strava_meta and st.session_state.strava_segment_id:
                m = st.session_state.strava_meta
                st.caption(
                    f"Selected: **{m['name']}** · id {m['segment_id']}"
                    + (
                        f" · {m['distance_m']/1000:.2f} km"
                        if m.get("distance_m")
                        else ""
                    )
                )
                use_strava = st.checkbox(
                    "Use Strava instead of GPX",
                    key="use_strava_segment",
                )

            use_sample = st.checkbox("Use sample South Florida GPX", value=True)
            uploaded = st.file_uploader(
                "Upload GPX",
                type=["gpx"],
                help=(
                    "Use a GPX when you have a full ride/route export, or the "
                    "segment isn’t in Strava/starred. Prefer starred or a "
                    "segment ID when available."
                ),
            )
            st.caption(
                "GPX: full ride or route export, or when the segment isn’t on "
                "Strava. Starred / segment ID preferred when you have them."
            )

        # --- Rider (below segment on the right) ---
        with st.container(border=True):
            st.html('<p class="bw-section-label">Rider</p>')
            st.subheader("You")

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
                    _apply_bike_kg(
                        POSITION_PRESETS[position_key].bike_mass_kg, manual=False
                    )

            rw1, rw2 = st.columns(2)
            with rw1:
                st.number_input(
                    "Rider kg",
                    min_value=40.0,
                    max_value=150.0,
                    step=0.5,
                    key="rider_kg_input",
                    on_change=_on_rider_kg,
                )
            with rw2:
                st.number_input(
                    "Rider lbs",
                    min_value=88.0,
                    max_value=330.0,
                    step=1.0,
                    key="rider_lb_input",
                    on_change=_on_rider_lb,
                )
            st.session_state.rider_kg = float(st.session_state.rider_kg_input)

            power_w = st.number_input(
                "Sustained power (W)",
                min_value=50,
                max_value=600,
                step=10,
                key="power_w",
            )

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

            with st.expander("Advanced / troubleshooting", expanded=False):
                st.markdown("**Bike & aero**")
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
                        "Reset bike to position default",
                        on_click=_on_reset_bike_mass,
                    )

                a1, a2 = st.columns(2)
                with a1:
                    full_aero = st.checkbox(
                        "Full aero kit (−Cd / −A)",
                        key="full_aero_kit",
                    )
                with a2:
                    non_aero = st.checkbox(
                        "Non-aero / no helmet (+Cd)",
                        key="non_aero",
                    )

                st.number_input(
                    "Rider height (cm)",
                    min_value=140.0,
                    max_value=210.0,
                    step=0.5,
                    key="height_cm_input",
                    on_change=_on_height_cm,
                )
                target_raw = st.text_input(
                    "KOM / target time (mm:ss or seconds)",
                    value="",
                    placeholder="12:30",
                    key="target_time_input",
                )
                prefer_hrrr = st.toggle(
                    "Prefer NOAA HRRR (CONUS)",
                    value=hrrr_enabled_by_env(),
                    key="prefer_hrrr_toggle",
                )

                with st.expander("Explore Segments", expanded=False):
                    if not strava_cfg.extended_access:
                        st.caption("Requires `STRAVA_EXTENDED_ACCESS=true`.")
                    elif not token:
                        st.caption("Needs a Strava token.")
                    else:
                        b1, b2, b3, b4 = st.columns(4)
                        sw_lat = b1.number_input("SW lat", value=26.05, format="%.4f")
                        sw_lon = b2.number_input("SW lon", value=-80.20, format="%.4f")
                        ne_lat = b3.number_input("NE lat", value=26.25, format="%.4f")
                        ne_lon = b4.number_input("NE lon", value=-80.05, format="%.4f")
                        if st.button("Explore top 10", key="explore_btn"):
                            try:
                                client = StravaClient(strava_cfg, access_token=token)
                                hits = client.explore(
                                    [sw_lat, sw_lon, ne_lat, ne_lon]
                                )
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
                                "Explored segment",
                                options=["(none)"] + elabels,
                                key="explore_pick",
                            )
                            if epick != "(none)":
                                chosen = next(
                                    h for h in explore if h["label"] == epick
                                )
                                _select_strava_segment(
                                    int(chosen["id"]),
                                    name=str(
                                        chosen.get("name") or epick.split(" (")[0]
                                    ),
                                    distance_m=float(
                                        chosen.get("distance_m") or 0.0
                                    ),
                                    average_grade=float(
                                        chosen.get("average_grade") or 0.0
                                    ),
                                )

                if tok_src == "none" and not strava_cfg.oauth_ready:
                    with st.expander("Secrets diagnostic", expanded=False):
                        presence = strava_secrets_presence()
                        dbg = strava_secrets_debug()
                        key_list = (
                            ", ".join(f"`{k}`" for k in dbg["keys"])
                            if dbg["keys"]
                            else "(none)"
                        )
                        nested_note = (
                            " · Found nested `strava.*` keys"
                            if dbg.get("nested_strava")
                            else ""
                        )
                        st.caption(
                            f"`st.secrets` available: "
                            f"{'yes' if dbg['available'] else 'no'}"
                            f"{nested_note} · keys: {key_list}  \n"
                            f"`CLIENT_ID`: "
                            f"{'set' if presence['CLIENT_ID'] else 'missing'} · "
                            f"`CLIENT_SECRET`: "
                            f"{'set' if presence['CLIENT_SECRET'] else 'missing'} · "
                            f"`REDIRECT_URI`: "
                            f"{'set' if presence['REDIRECT_URI'] else 'missing'} · "
                            f"`ACCESS_TOKEN`: "
                            f"{'set' if presence['ACCESS_TOKEN'] else 'missing'}"
                        )

            aero = resolve_aero(
                st.session_state.position_key,
                full_aero_kit=bool(full_aero),
                non_aero=bool(non_aero),
                bike_mass_kg=float(st.session_state.bike_kg),
            )
            st.caption(
                f"{POSITION_PRESETS[aero.position_key].label} · "
                f"Cd·A={aero.cd * aero.area_m2:.3f} · "
                f"total {st.session_state.rider_kg + aero.bike_mass_kg:.1f} kg"
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
                    except Exception:  # noqa: BLE001
                        st.session_state._gpx_center_digest = digest

            run = st.button("Run prediction", type="primary", use_container_width=True)

    last_run = st.session_state.get("_last_run")
    if not run and not last_run:
        st.caption("Set wind time and rider setup, then run prediction.")
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

    # --- 3. Results ---
    with st.container(border=True):
        st.html('<p class="bw-section-label">03 · Results</p>')
        st.subheader("Predicted time")
        st.caption(
            f"{source_label_out} · {route_km:.1f} km · "
            f"{POSITION_PRESETS[position_key_out].label} · Cd·A={cda_out:.3f}"
        )

        if result.weather_message and not result.weather_ok:
            st.warning(result.weather_message)
        elif result.weather_ok:
            src = "HRRR" if result.weather_source == "hrrr" else "Open-Meteo"
            st.caption(
                f"{src} · along-track {result.mean_wind_along_mps:+.1f} m/s "
                f"(head {result.mean_headwind_mps:.1f} · tail {result.mean_tailwind_mps:.1f})"
            )

        band = result.time_band
        eta = result.eta_with_wind_s
        if band is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "ETA with wind",
                format_duration(band.eta_s),
                delta=f"{format_duration(band.eta_low_s)}–{format_duration(band.eta_high_s)}",
                delta_color="off",
            )
            m2.metric("ETA calm", format_duration(result.eta_no_wind_s))
            avg = (
                sum(result.speeds_with_wind_kmh) / len(result.speeds_with_wind_kmh)
                if result.speeds_with_wind_kmh
                else 0.0
            )
            m3.metric("Avg speed", f"{avg:.1f} km/h")
            st.caption(band.note)
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("ETA with wind", format_duration(eta))
            m2.metric("ETA calm", format_duration(result.eta_no_wind_s))
            avg = (
                sum(result.speeds_with_wind_kmh) / len(result.speeds_with_wind_kmh)
                if result.speeds_with_wind_kmh
                else 0.0
            )
            m3.metric("Avg speed", f"{avg:.1f} km/h")

        if result.target_power is not None:
            tp = result.target_power
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

        st.markdown("**Route map**")
        st.caption("Teal = tailwind · red = headwind")
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
                height=max(MAP_MIN_HEIGHT_PX, 320),
                returned_objects=[],
                use_container_width=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Map unavailable: {exc}")

        fig, ax = plt.subplots(figsize=(9, 3.8))
        bg = "#0c1210" if dark else "#f7faf8"
        fg = "#e7efea" if dark else "#14201b"
        accent = "#3ecf8e" if dark else "#1a6b52"
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.plot(
            result.distances_km,
            result.speeds_no_wind_kmh,
            label="No wind",
            color="#8fa398",
            linewidth=1.5,
            linestyle="--",
        )
        ax.plot(
            result.distances_km,
            result.speeds_with_wind_kmh,
            label="With wind" if result.weather_ok else "With wind (unavailable)",
            color=accent,
            linewidth=2.0,
        )
        ax.set_xlabel("Distance (km)", color=fg)
        ax.set_ylabel("Ground speed (km/h)", color=fg)
        ax.tick_params(colors=fg)
        for spine in ax.spines.values():
            spine.set_color("#3a4a42" if dark else "#c5d0ca")
        legend = ax.legend(facecolor=bg, edgecolor="#3a4a42" if dark else "#c5d0ca")
        for text in legend.get_texts():
            text.set_color(fg)
        ax.grid(True, alpha=0.22, color=fg)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if len(result.bins) > 1:
            with st.expander("Distance bins", expanded=False):
                rows = []
                for b in result.bins:
                    rows.append(
                        {
                            "Bin": b.index,
                            "km": f"{b.start_km:.1f}–{b.end_km:.1f}",
                            "Grade %": f"{b.avg_grade_pct:.1f}",
                            "Wind": b.wind_label,
                            "km/h wind": f"{b.speed_with_wind_kmh:.1f}",
                            "km/h calm": f"{b.speed_no_wind_kmh:.1f}",
                            "Time": format_duration(b.time_with_wind_s),
                            "P@pace W": (
                                f"{b.required_power_w:.0f}" if b.required_power_w else "—"
                            ),
                        }
                    )
                st.dataframe(rows, use_container_width=True, hide_index=True)
        elif len(result.bins) == 1:
            b = result.bins[0]
            st.caption(
                f"{b.wind_label} · grade {b.avg_grade_pct:.1f}% · "
                f"{b.speed_with_wind_kmh:.1f} km/h with wind"
            )

        with st.expander("Reliability notes", expanded=False):
            st.markdown(
                r"""
- **Flat roads:** roughly ±5% on predicted time when wind and mass are well specified.
- **Climbs:** noisier — GPX grade noise dominates.
- **Assumptions:** \(z_0=0.03\), tuck −15% A on descents, crosswind adds drag only.
- **Wind:** Open-Meteo primary (cached); optional HRRR in Advanced.
                """
            )


if __name__ == "__main__":
    main()
