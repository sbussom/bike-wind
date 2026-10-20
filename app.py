"""Streamlit app: GPX → wind → speed chart with rider / aero controls."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from streamlit_folium import st_folium

from bike_wind.analysis import format_duration
from bike_wind.gpx import parse_gpx
from bike_wind.local_wind_map import (
    DEFAULT_LAT,
    DEFAULT_LON,
    compass_label,
    fetch_pin_wind,
)
from bike_wind.map_view import MAP_MIN_HEIGHT_PX, build_wind_route_map
from bike_wind.nearby_map import NEARBY_MAP_HEIGHT_PX, build_nearby_segments_map
from bike_wind.past_ride import compare_activity, ride_activity_label
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
            "REFRESH_TOKEN": False,
        }

    def strava_secrets_debug():
        return {"available": False, "keys": [], "nested_strava": False}


from bike_wind.strava import (
    ActivitySummary,
    StravaAuthError,
    StravaClient,
    StravaRateLimitError,
    load_strava_config,
)
from bike_wind.strava.oauth import exchange_code
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
    if "strava_oauth_connected" not in st.session_state:
        st.session_state.strava_oauth_connected = False
    if "strava_refreshed_access_token" not in st.session_state:
        st.session_state.strava_refreshed_access_token = None
    if "strava_refreshed_refresh_token" not in st.session_state:
        st.session_state.strava_refreshed_refresh_token = None
    if "strava_last_error" not in st.session_state:
        st.session_state.strava_last_error = None
    if "use_strava_segment" not in st.session_state:
        st.session_state.use_strava_segment = False
    if "explore_hits" not in st.session_state:
        st.session_state.explore_hits = []
    if "explore_cache" not in st.session_state:
        st.session_state.explore_cache = {}
    if "location_source" not in st.session_state:
        st.session_state.location_source = "default"
    if "map_lat" not in st.session_state:
        st.session_state.map_lat = DEFAULT_LAT
    if "map_lon" not in st.session_state:
        st.session_state.map_lon = DEFAULT_LON
    if "_last_run" not in st.session_state:
        st.session_state._last_run = None
    if "past_activities" not in st.session_state:
        st.session_state.past_activities = []
    if "past_comparison" not in st.session_state:
        st.session_state.past_comparison = None

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


def _clear_strava_session_tokens() -> None:
    """Drop sticky browser tokens so secrets are used again."""
    st.session_state.strava_access_token = None
    st.session_state.strava_refresh_token = None
    st.session_state.strava_oauth_connected = False
    st.session_state.strava_refreshed_access_token = None
    st.session_state.strava_refreshed_refresh_token = None
    st.session_state.explore_hits = []
    st.session_state.explore_cache = {}


def _sync_strava_secrets_fingerprint() -> None:
    """If Cloud secrets changed, clear sticky session tokens (old refresh is dead)."""
    try:
        from bike_wind.settings import strava_secrets_fingerprint
    except Exception:  # noqa: BLE001
        return
    fp = strava_secrets_fingerprint()
    prev = st.session_state.get("_strava_secrets_fp")
    if prev is None:
        st.session_state._strava_secrets_fp = fp
        return
    if prev != fp:
        _clear_strava_session_tokens()
        st.session_state._strava_secrets_fp = fp
        st.session_state._strava_secrets_rotated = True


def _effective_strava_token(config) -> str | None:
    """OAuth session wins only after Connect; else secrets (+ session refresh cache)."""
    if st.session_state.get("strava_oauth_connected") and st.session_state.strava_access_token:
        return st.session_state.strava_access_token
    # Prefer secrets for anonymous visitors — sticky OAuth must not block them.
    return st.session_state.get("strava_refreshed_access_token") or config.access_token


def _effective_strava_refresh(config) -> str | None:
    """Prefer secrets refresh_token unless visitor explicitly Connected Strava."""
    if st.session_state.get("strava_oauth_connected") and st.session_state.strava_refresh_token:
        return st.session_state.strava_refresh_token
    # Secrets first so a revoked session refresh cannot mask a newly pasted secret.
    return config.refresh_token or st.session_state.get("strava_refreshed_refresh_token")


def _strava_ready(config) -> bool:
    """True if we have an access token, or can mint one via refresh_token grant."""
    if _effective_strava_token(config):
        return True
    return bool(
        config.client_id
        and config.client_secret
        and _effective_strava_refresh(config)
    )


def _persist_refreshed_tokens(tokens) -> None:
    """Keep refreshed tokens in session only — Streamlit secrets are read-only."""
    if st.session_state.get("strava_oauth_connected"):
        st.session_state.strava_access_token = tokens.access_token
        if tokens.refresh_token:
            st.session_state.strava_refresh_token = tokens.refresh_token
    else:
        st.session_state.strava_refreshed_access_token = tokens.access_token
        if tokens.refresh_token:
            st.session_state.strava_refreshed_refresh_token = tokens.refresh_token


def _strava_client(config, *, secrets_only: bool = False) -> StravaClient:
    """Client for explore + segment resolve; auto-refreshes on 401 into session_state."""

    def on_refreshed(tokens) -> None:
        _persist_refreshed_tokens(tokens)

    def on_refresh_failed() -> None:
        # Sticky dead tokens must not outlive a failed refresh.
        if not st.session_state.get("strava_oauth_connected"):
            st.session_state.strava_refreshed_access_token = None
            st.session_state.strava_refreshed_refresh_token = None

    if secrets_only:
        access = config.access_token
        refresh = config.refresh_token
    else:
        access = _effective_strava_token(config)
        refresh = _effective_strava_refresh(config)

    _ = on_refresh_failed  # reserved for future hooks; clear happens in _run_strava_call
    return StravaClient(
        config,
        access_token=access,
        refresh_token=refresh,
        on_tokens_refreshed=on_refreshed,
    )


def _run_strava_call(config, fn):
    """Run ``fn(client)``; on refresh failure clear sticky tokens and retry with secrets."""
    client = _strava_client(config)
    try:
        return fn(client)
    except StravaAuthError as exc:
        if getattr(exc, "kind", None) != "refresh_failed":
            raise
        _clear_strava_session_tokens()
        # Retry once with deployer secrets only (ignore any leftover session).
        client2 = _strava_client(config, secrets_only=True)
        return fn(client2)


def _token_source(config) -> str:
    if st.session_state.get("strava_oauth_connected") and st.session_state.strava_access_token:
        return "session"
    if st.session_state.get("strava_refreshed_access_token"):
        return "secrets-refreshed"
    if config.access_token or (
        config.client_id and config.client_secret and config.refresh_token
    ):
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
        st.session_state.strava_oauth_connected = True
        st.session_state.strava_access_token = tokens.access_token
        st.session_state.strava_refresh_token = tokens.refresh_token
        st.session_state.explore_hits = []
        st.session_state.explore_cache = {}
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
    _clear_strava_session_tokens()
    st.session_state.use_strava_segment = False


def _on_reset_strava_session() -> None:
    _clear_strava_session_tokens()
    st.session_state.use_strava_segment = False
    st.session_state.strava_last_error = None
    st.session_state._strava_reset_flash = True


def _queue_map_center(lat: float, lon: float) -> None:
    """Queue a pin update for the next render (before lat/lon widgets exist)."""
    st.session_state._pending_map_center = (round(float(lat), 4), round(float(lon), 4))


def _apply_geo_query_params() -> None:
    """Read ?geo_lat=&geo_lon= from the geolocation button (before widgets)."""
    params = st.query_params
    raw_lat, raw_lon = params.get("geo_lat"), params.get("geo_lon")
    if raw_lat is None or raw_lon is None:
        return
    try:
        lat, lon = float(raw_lat), float(raw_lon)
    except (TypeError, ValueError):
        return
    if not (-85.0 <= lat <= 85.0 and -180.0 <= lon <= 180.0):
        return
    _queue_map_center(lat, lon)
    st.session_state.location_source = "geolocation"
    st.session_state.explore_cache = {}
    try:
        del st.query_params["geo_lat"]
        del st.query_params["geo_lon"]
    except Exception:  # noqa: BLE001
        pass


def _on_reset_map_location() -> None:
    st.session_state.map_lat = DEFAULT_LAT
    st.session_state.map_lon = DEFAULT_LON
    st.session_state.map_lat_input = DEFAULT_LAT
    st.session_state.map_lon_input = DEFAULT_LON
    st.session_state._pin_auto_centered = False
    st.session_state.location_source = "default"
    st.session_state.explore_cache = {}


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
    _apply_geo_query_params()
    _apply_pending_map_center()

    top_l, top_r = st.columns([4, 1])
    with top_r:
        dark = st.toggle("Dark", value=True, help="Dark / light")
    _apply_theme(dark)

    with top_l:
        st.html(
            """
            <div class="bw-hero">
              <p class="bw-brand">Bike Wind</p>
              <p class="bw-tagline">Predict ride time under wind.</p>
            </div>
            """
        )

    strava_cfg = load_strava_config()
    _sync_strava_secrets_fingerprint()
    _handle_oauth_callback(strava_cfg)
    strava_ready = _strava_ready(strava_cfg)

    default_dt = datetime.now(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    ) + timedelta(hours=3)

    map_lat = float(st.session_state.map_lat_input)
    map_lon = float(st.session_state.map_lon_input)
    st.session_state.map_lat = map_lat
    st.session_state.map_lon = map_lon

    left_col, right_col = st.columns([1.35, 1], gap="large")

    with left_col:
        with st.container(border=True):
            st.html('<p class="bw-section-label">Wind</p>')
            when = st.datetime_input("When (UTC)", value=default_dt)
            pin = fetch_pin_wind(map_lat, map_lon)
            if pin is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("Wind", f"{pin.speed_mph:.1f} mph")
                m2.metric("From", compass_label(pin.direction_deg))
                if pin.temperature_c is not None:
                    m3.metric("Temp", f"{pin.temperature_c * 9 / 5 + 32:.0f} °F")
                else:
                    m3.metric("Temp", "—")
            try:
                nearby_map = build_nearby_segments_map(
                    map_lat,
                    map_lon,
                    [],
                    dark=dark,
                    height_px=NEARBY_MAP_HEIGHT_PX,
                )
                st_folium(
                    nearby_map,
                    width=None,
                    height=NEARBY_MAP_HEIGHT_PX,
                    returned_objects=[],
                    use_container_width=True,
                )
            except Exception as exc:  # noqa: BLE001
                st.caption(f"Map unavailable: {exc}")
            st.link_button(
                "Browse segments on Strava",
                "https://www.strava.com/segments/explore",
                use_container_width=True,
            )
            with st.expander("Map pin", expanded=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.number_input(
                        "Lat",
                        min_value=-85.0,
                        max_value=85.0,
                        step=0.1,
                        format="%.4f",
                        key="map_lat_input",
                    )
                with c2:
                    st.number_input(
                        "Lon",
                        min_value=-180.0,
                        max_value=180.0,
                        step=0.1,
                        format="%.4f",
                        key="map_lon_input",
                    )
                with c3:
                    st.write("")
                    st.button("SE Florida", on_click=_on_reset_map_location)

    with right_col:
        with st.container(border=True):
            st.html('<p class="bw-section-label">Route</p>')
            st.caption("Segment link or ID — load a public Strava segment.")
            strava_ref_text = st.text_input(
                "Segment URL or ID",
                placeholder="https://www.strava.com/segments/229781",
                key="strava_ref_input",
                label_visibility="collapsed",
            )
            do_resolve = st.button(
                "Load segment",
                disabled=not strava_ready or not strava_ref_text.strip(),
                use_container_width=True,
            )
            if do_resolve and strava_ready and strava_ref_text.strip():
                try:
                    def _resolve(client: StravaClient):
                        sid, _ref = client.resolve_ref(strava_ref_text)
                        return sid, client.get_segment_meta(sid)

                    sid, meta = _run_strava_call(strava_cfg, _resolve)
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
                except StravaAuthError as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not load segment: {exc}")

            if st.session_state.strava_meta and st.session_state.strava_segment_id:
                m = st.session_state.strava_meta
                st.caption(f"Selected: **{m['name']}** · {m['segment_id']}")
                st.checkbox("Use this segment", key="use_strava_segment")

            st.caption("GPX — use your own file instead of a segment.")
            uploaded = st.file_uploader("GPX", type=["gpx"], label_visibility="collapsed")
            use_sample = st.checkbox("Demo route", value=uploaded is None and not st.session_state.get("use_strava_segment"))

        with st.container(border=True):
            st.html('<p class="bw-section-label">Rider</p>')
            labels = position_labels()
            current_label = POSITION_PRESETS[st.session_state.position_key].label
            chosen = st.selectbox(
                "Position",
                options=labels,
                index=labels.index(current_label) if current_label in labels else 0,
            )
            position_key = position_key_for_label(chosen)
            if position_key != st.session_state.position_key:
                st.session_state.position_key = position_key
                if not st.session_state.bike_mass_manual:
                    _apply_bike_kg(POSITION_PRESETS[position_key].bike_mass_kg, manual=False)

            power_w = st.slider("Power (W)", 80, 450, int(st.session_state.power_w), 5)
            st.session_state.power_w = power_w

            w1, w2 = st.columns(2)
            with w1:
                st.number_input(
                    "Rider kg",
                    min_value=40.0,
                    max_value=120.0,
                    step=0.5,
                    key="rider_kg_input",
                    on_change=_on_rider_kg,
                )
            with w2:
                st.number_input(
                    "Rider lb",
                    min_value=90.0,
                    max_value=265.0,
                    step=1.0,
                    key="rider_lb_input",
                    on_change=_on_rider_lb,
                )
            st.session_state.rider_kg = float(st.session_state.rider_kg_input)

            full_aero = st.session_state.get("full_aero_kit", False)
            non_aero = st.session_state.get("non_aero", False)
            prefer_hrrr = hrrr_enabled_by_env()
            aero = resolve_aero(
                st.session_state.position_key,
                full_aero_kit=bool(full_aero),
                non_aero=bool(non_aero),
                bike_mass_kg=float(st.session_state.bike_kg),
            )
            st.caption(
                f"{POSITION_PRESETS[aero.position_key].label} · "
                f"Cd·A={aero.cd * aero.area_m2:.3f} · "
                f"{st.session_state.rider_kg + aero.bike_mass_kg:.1f} kg total"
            )

            if uploaded is not None and not use_sample and not st.session_state.get("use_strava_segment"):
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

            run = st.button("Run", type="primary", use_container_width=True)

        with st.expander("Troubleshooting", expanded=False):
            st.button(
                "Reset Strava session",
                on_click=_on_reset_strava_session,
                key="reset_strava_session",
            )
            presence = strava_secrets_presence()
            st.caption(
                f"Token: {'set' if presence.get('ACCESS_TOKEN') else 'missing'} · "
                f"Refresh: {'set' if presence.get('REFRESH_TOKEN') else 'missing'}"
            )
            if st.session_state.get("strava_last_error"):
                st.error(st.session_state.strava_last_error)

    # --- Past ride check ---
    with st.container(border=True):
        st.html('<p class="bw-section-label">Check a past ride</p>')
        st.caption("Compare the model to one of your recent Strava rides (archive wind).")
        if not strava_ready:
            st.caption("Needs the app Strava access token in secrets.")
        else:
            if st.button("Load recent rides", use_container_width=True):
                try:
                    acts = _run_strava_call(
                        strava_cfg,
                        lambda c: c.list_recent_activities(per_page=30),
                    )
                    rides = [
                        a
                        for a in acts
                        if a.sport_type.lower() in {"ride", "virtualride", "gravelride", "mountainbikeride", "e-bike-ride", "ebikeride"}
                        or "ride" in a.sport_type.lower()
                    ]
                    st.session_state.past_activities = rides or acts
                except StravaAuthError as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not list activities: {exc}")

            activities: list[ActivitySummary] = list(
                st.session_state.get("past_activities") or []
            )
            if activities:
                labels = [ride_activity_label(a) for a in activities]
                pick = st.selectbox("Ride", options=labels, key="past_ride_pick")
                chosen = activities[labels.index(pick)]
                if not chosen.has_power:
                    st.caption("No power on this ride — will use your rider power setting.")
                if st.button("Compare to model", type="primary", key="past_compare_btn"):
                    try:
                        with st.spinner("Archive weather + physics…"):
                            comparison = _run_strava_call(
                                strava_cfg,
                                lambda c: compare_activity(
                                    c,
                                    chosen,
                                    rider_mass_kg=float(st.session_state.rider_kg),
                                    bike_mass_kg=float(st.session_state.bike_kg),
                                    position_key=st.session_state.position_key,
                                    fallback_power_w=float(st.session_state.power_w),
                                    height_cm=float(st.session_state.height_cm),
                                    full_aero_kit=bool(st.session_state.get("full_aero_kit", False)),
                                    non_aero=bool(st.session_state.get("non_aero", False)),
                                ),
                            )
                        st.session_state.past_comparison = comparison
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Comparison failed: {exc}")

            comparison = st.session_state.get("past_comparison")
            if comparison is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted", comparison.predicted_label)
                c2.metric("Actual moving", comparison.actual_moving_label)
                c3.metric("Δ (pred − actual)", comparison.delta_moving_label)
                st.caption(
                    f"{comparison.power_w:.0f} W"
                    f"{' from activity' if comparison.power_from_activity else ' from rider setting'}"
                    f" · weather {comparison.weather_source}"
                    f"{'' if comparison.weather_ok else ' (calm fallback)'}"
                )
                st.caption(comparison.note)

    last_run = st.session_state.get("_last_run")
    if not run and not last_run:
        return

    use_strava = bool(st.session_state.get("use_strava_segment")) and bool(
        st.session_state.get("strava_segment_id")
    )

    if run:
        try:
            if use_strava and strava_ready:
                meta, route = _run_strava_call(
                    strava_cfg,
                    lambda c: c.load_segment_route(int(st.session_state.strava_segment_id)),
                )
                st.session_state.strava_meta = {
                    "segment_id": meta.segment_id,
                    "name": meta.name,
                    "distance_m": meta.distance_m,
                    "average_grade": meta.average_grade,
                    "from_cache": meta.from_cache,
                }
                source_label = f"Strava: {meta.name}"
            elif uploaded is not None and not use_sample:
                route = parse_gpx(uploaded.getvalue())
                source_label = uploaded.name
            else:
                route = parse_gpx(FIXTURE)
                source_label = "Demo route"
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
        with st.spinner("Fetching wind and solving…"):
            when_dt = when if isinstance(when, datetime) else datetime.combine(when, datetime.min.time())
            if when_dt.tzinfo is None:
                when_dt = when_dt.replace(tzinfo=timezone.utc)
            result = predict_speeds(
                route,
                params,
                when_dt,
                wind_height_m=wind_z,
                prefer_hrrr=prefer_hrrr,
            )
        mid_lat, mid_lon = route.midpoint()
        st.session_state._last_run = {
            "result": result,
            "source_label": source_label,
            "route_km": route.distances_m[-1] / 1000.0,
            "position_key": aero.position_key,
            "cda": aero.cd * aero.area_m2,
        }
        if _pin_needs_recenter(mid_lat, mid_lon):
            _queue_map_center(mid_lat, mid_lon)
            st.rerun()
        source_label_out = source_label
        route_km = route.distances_m[-1] / 1000.0
        position_key_out = aero.position_key
        cda_out = aero.cd * aero.area_m2
    else:
        result = last_run["result"]
        source_label_out = last_run["source_label"]
        route_km = last_run["route_km"]
        position_key_out = last_run["position_key"]
        cda_out = last_run["cda"]

    with st.container(border=True):
        st.html('<p class="bw-section-label">Results</p>')
        st.caption(
            f"{source_label_out} · {route_km:.1f} km · "
            f"{POSITION_PRESETS[position_key_out].label} · Cd·A={cda_out:.3f}"
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Time (with wind)", format_duration(result.eta_with_wind_s))
        m2.metric("Time (calm)", format_duration(result.eta_no_wind_s))
        avg = (
            sum(result.speeds_with_wind_kmh) / len(result.speeds_with_wind_kmh)
            if result.speeds_with_wind_kmh
            else 0.0
        )
        m3.metric("Avg speed", f"{avg:.1f} km/h")
        if result.weather_ok:
            st.caption(
                f"{result.weather_source} · along-track {result.mean_wind_along_mps:+.1f} m/s"
            )
        elif result.weather_message:
            st.caption(result.weather_message)

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
            st.caption(f"Route map unavailable: {exc}")

        fig, ax = plt.subplots(figsize=(9, 3.4))
        bg = "#0c1210" if dark else "#f7faf8"
        fg = "#e7efea" if dark else "#14201b"
        accent = "#3ecf8e" if dark else "#1a6b52"
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.plot(result.distances_km, result.speeds_no_wind_kmh, label="Calm", color="#8fa398", linestyle="--")
        ax.plot(result.distances_km, result.speeds_with_wind_kmh, label="With wind", color=accent, linewidth=2)
        ax.set_xlabel("Distance (km)", color=fg)
        ax.set_ylabel("km/h", color=fg)
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


if __name__ == "__main__":
    main()
