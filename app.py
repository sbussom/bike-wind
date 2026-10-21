"""Streamlit app: GPX → wind → speed chart with rider / aero controls."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from streamlit_folium import st_folium

from bike_wind.analysis import format_duration
from bike_wind.gpx import parse_gpx
from bike_wind.map_view import MAP_MIN_HEIGHT_PX, build_wind_route_map
from bike_wind.physics import RiderBikeParams
from bike_wind.predict import downsample_route, predict_speeds
from bike_wind.physics.constants import DRIVETRAIN_EFFICIENCY
from bike_wind.rider import (
    DEFAULT_POSITION_KEY,
    DEFAULT_RIDER_KG,
    DEFAULT_SURFACE_KEY,
    POSITION_PRESETS,
    SURFACE_PRESETS,
    kg_to_lb,
    lb_to_kg,
    position_key_for_label,
    position_labels,
    resolve_aero,
    resolve_surface,
    smart_default_height_cm,
    standing_height_to_wind_z_m,
    surface_key_for_label,
    surface_labels,
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
            "REFRESH_TOKEN": False,
        }


from bike_wind.past_ride import ride_activity_label
from bike_wind.strava import (
    ActivitySummary,
    SegmentEffortSummary,
    StravaAuthError,
    StravaClient,
    StravaError,
    load_strava_config,
)
from bike_wind.strava.oauth import (
    DEFAULT_REDIRECT_URI,
    build_authorize_url,
    exchange_code,
    oauth_configured,
)
from bike_wind.weather import hrrr_enabled_by_env


_CYCLING_SPORTS = {
    "ride",
    "virtualride",
    "gravelride",
    "mountainbikeride",
    "ebikeride",
    "e-bike-ride",
}


def _is_cycling_activity(act: ActivitySummary) -> bool:
    sport = (act.sport_type or "").lower().replace(" ", "")
    return sport in _CYCLING_SPORTS or "ride" in sport

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
        st.session_state.rider_kg = DEFAULT_RIDER_KG
    if "bike_kg" not in st.session_state:
        st.session_state.bike_kg = POSITION_PRESETS[DEFAULT_POSITION_KEY].bike_mass_kg
    if "bike_mass_manual" not in st.session_state:
        st.session_state.bike_mass_manual = False
    if "height_cm" not in st.session_state:
        st.session_state.height_cm = smart_default_height_cm(DEFAULT_RIDER_KG)
    if "height_manual" not in st.session_state:
        st.session_state.height_manual = False
    if "position_key" not in st.session_state:
        st.session_state.position_key = DEFAULT_POSITION_KEY
    if "surface_key" not in st.session_state:
        st.session_state.surface_key = DEFAULT_SURFACE_KEY
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
    if "strava_token_expires_at" not in st.session_state:
        st.session_state.strava_token_expires_at = None
    if "strava_refreshed_access_token" not in st.session_state:
        st.session_state.strava_refreshed_access_token = None
    if "strava_refreshed_refresh_token" not in st.session_state:
        st.session_state.strava_refreshed_refresh_token = None
    if "strava_last_error" not in st.session_state:
        st.session_state.strava_last_error = None
    if "strava_oauth_error" not in st.session_state:
        st.session_state.strava_oauth_error = None
    # Canonical route choice — NOT a widget key (never write use_strava_* after
    # a checkbox with that key exists; StreamlitWidgetAlreadyInstantiatedError).
    if "route_source" not in st.session_state:
        st.session_state.route_source = None  # "segment" | "activity" | None
    if "recent_activities" not in st.session_state:
        st.session_state.recent_activities = []
    if "day_activities" not in st.session_state:
        st.session_state.day_activities = []
    if "selected_activity" not in st.session_state:
        st.session_state.selected_activity = None
    if "activity_route" not in st.session_state:
        st.session_state.activity_route = None
    if "activity_efforts" not in st.session_state:
        st.session_state.activity_efforts = []
    if "selected_effort" not in st.session_state:
        st.session_state.selected_effort = None
    if "effort_route" not in st.session_state:
        st.session_state.effort_route = None
    if "activity_stream_error" not in st.session_state:
        st.session_state.activity_stream_error = None
    if "_rides_token_fp" not in st.session_state:
        st.session_state._rides_token_fp = None
    if "_efforts_activity_id" not in st.session_state:
        st.session_state._efforts_activity_id = None
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


def _clear_strava_session_tokens() -> None:
    """Drop sticky browser tokens so secrets are used again."""
    st.session_state.strava_access_token = None
    st.session_state.strava_refresh_token = None
    st.session_state.strava_oauth_connected = False
    st.session_state.strava_token_expires_at = None
    st.session_state.strava_refreshed_access_token = None
    st.session_state.strava_refreshed_refresh_token = None


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
        st.session_state.strava_token_expires_at = tokens.expires_at
    else:
        st.session_state.strava_refreshed_access_token = tokens.access_token
        if tokens.refresh_token:
            st.session_state.strava_refreshed_refresh_token = tokens.refresh_token


def _strava_client(config, *, secrets_only: bool = False) -> StravaClient:
    """Client for segment/activity calls; auto-refreshes on 401 into session_state."""

    def on_refreshed(tokens) -> None:
        _persist_refreshed_tokens(tokens)

    if secrets_only:
        access = config.access_token
        refresh = config.refresh_token
    else:
        access = _effective_strava_token(config)
        refresh = _effective_strava_refresh(config)

    return StravaClient(
        config,
        access_token=access,
        refresh_token=refresh,
        on_tokens_refreshed=on_refreshed,
    )


def _run_strava_call(config, fn):
    """Run ``fn(client)``; secrets path may retry after refresh failure.

    Connected visitors keep their session token only — never silently fall back
    to the owner secrets token for personal rides.
    """
    was_connected = bool(st.session_state.get("strava_oauth_connected"))
    client = _strava_client(config)
    try:
        return fn(client)
    except StravaAuthError as exc:
        if getattr(exc, "kind", None) != "refresh_failed":
            raise
        _clear_strava_session_tokens()
        if was_connected:
            raise StravaAuthError(
                "Strava session expired — reconnect.",
                kind="refresh_failed",
            ) from exc
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


def _clear_oauth_query_params() -> None:
    try:
        for key in ("code", "scope", "state", "error", "error_description"):
            if key in st.query_params:
                del st.query_params[key]
    except Exception:  # noqa: BLE001
        pass


def _handle_oauth_callback(config) -> None:
    """Exchange ?code= from Strava redirect into session_state (never secrets)."""
    params = st.query_params
    if params.get("error"):
        err = str(params.get("error_description") or params.get("error"))
        st.session_state.strava_oauth_error = f"Strava rejected connect: {err}"
        _clear_oauth_query_params()
        return
    code = params.get("code")
    if not code or st.session_state.get("_oauth_handled_code") == code:
        return
    if not oauth_configured(config.client_id, config.client_secret, config.redirect_uri):
        st.session_state.strava_oauth_error = (
            "Connect needs client_id, client_secret, and redirect_uri in secrets."
        )
        _clear_oauth_query_params()
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
        st.session_state.strava_token_expires_at = tokens.expires_at
        st.session_state.strava_oauth_error = None
        _clear_activity_route_state()
        st.session_state._oauth_handled_code = code
        _clear_oauth_query_params()
        st.success("Strava connected for this browser session.")
    except Exception as exc:  # noqa: BLE001 — no response body / tokens in message
        st.session_state.strava_oauth_error = f"Strava OAuth failed: {exc}"
        _clear_oauth_query_params()


def _clear_activity_route_state() -> None:
    if st.session_state.get("route_source") in ("activity", "effort"):
        st.session_state.route_source = None
    st.session_state.recent_activities = []
    st.session_state.day_activities = []
    st.session_state.selected_activity = None
    st.session_state.selected_effort = None
    st.session_state.activity_route = None
    st.session_state.activity_efforts = []
    st.session_state._efforts_activity_id = None
    st.session_state._rides_token_fp = None
    st.session_state.activity_stream_error = None
    st.session_state._recent_label_map = {}
    st.session_state._day_label_map = {}
    st.session_state._effort_label_map = {}


def _on_disconnect_strava() -> None:
    _clear_strava_session_tokens()
    st.session_state.route_source = None
    _clear_activity_route_state()
    st.session_state.strava_oauth_error = None


def _on_reset_strava_session() -> None:
    _clear_strava_session_tokens()
    st.session_state.route_source = None
    st.session_state.strava_last_error = None
    st.session_state.strava_oauth_error = None
    _clear_activity_route_state()
    st.session_state._strava_reset_flash = True


def _select_strava_segment(
    segment_id: int,
    *,
    name: str,
    distance_m: float = 0.0,
    average_grade: float = 0.0,
    from_cache: bool = False,
) -> None:
    """Set active Strava segment as route_source (non-widget key only)."""
    st.session_state.strava_segment_id = segment_id
    st.session_state.strava_meta = {
        "segment_id": segment_id,
        "name": name,
        "distance_m": distance_m,
        "average_grade": average_grade,
        "from_cache": from_cache,
    }
    st.session_state.route_source = "segment"
    st.session_state.selected_activity = None
    st.session_state.selected_effort = None
    st.session_state.activity_route = None
    st.session_state.effort_route = None
    st.session_state.activity_efforts = []
    st.session_state._efforts_activity_id = None
    st.session_state.activity_stream_error = None


def _select_strava_activity(act: ActivitySummary) -> None:
    """Mark a Strava ride as route_source (never touch widget keys)."""
    prev_id = getattr(st.session_state.get("selected_activity"), "activity_id", None)
    st.session_state.selected_activity = act
    st.session_state.selected_effort = None
    st.session_state.effort_route = None
    st.session_state.route_source = "activity"
    st.session_state.activity_stream_error = None
    if prev_id != act.activity_id:
        st.session_state.activity_efforts = []
        st.session_state._efforts_activity_id = None
        st.session_state._effort_label_map = {}
    if act.has_power and act.average_watts is not None:
        # power_w is not a widget key; the slider uses its own default each run.
        st.session_state.power_w = int(round(act.average_watts))


def _filter_cycling(acts: list[ActivitySummary]) -> list[ActivitySummary]:
    rides = [a for a in acts if _is_cycling_activity(a)]
    return rides if rides else list(acts)


def _ensure_recent_rides(config, *, limit: int = 10) -> None:
    """Fetch the latest rides once per token mode (session vs secrets)."""
    if not _strava_ready(config):
        return
    fp = "session" if st.session_state.get("strava_oauth_connected") else "secrets"
    if st.session_state.get("_rides_token_fp") == fp and st.session_state.recent_activities:
        return
    try:
        acts = _run_strava_call(
            config,
            lambda c: c.list_recent_activities(per_page=max(limit * 2, 20)),
        )
        rides = _filter_cycling(acts)[:limit]
        st.session_state.recent_activities = rides
        st.session_state._rides_token_fp = fp
        st.session_state.activity_stream_error = (
            None if rides else "No recent rides found for this athlete."
        )
    except (StravaAuthError, StravaError) as exc:
        st.session_state.recent_activities = []
        st.session_state._rides_token_fp = fp
        st.session_state.activity_stream_error = str(exc)
    except Exception as exc:  # noqa: BLE001
        st.session_state.recent_activities = []
        st.session_state._rides_token_fp = fp
        st.session_state.activity_stream_error = f"Could not list rides: {exc}"


def _load_ride_track(config, act: ActivitySummary) -> bool:
    """Pull GPS/elev/time/speed streams and set as the active route. Returns ok.

    Safe from ``on_change`` callbacks (runs before widgets). Must not assign
    any key that is also a widget ``key=`` later in the script.
    """
    try:
        route = _run_strava_call(
            config,
            lambda c: c.load_activity_route(
                act.activity_id, start_date=act.start_date
            ),
        )
    except (StravaAuthError, StravaError) as exc:
        st.session_state.activity_stream_error = (
            f"Ride streams unavailable ({exc}). Check activity:read scope."
        )
        st.session_state.activity_route = None
        return False
    except Exception as exc:  # noqa: BLE001
        st.session_state.activity_stream_error = f"Could not load ride track: {exc}"
        st.session_state.activity_route = None
        return False
    _select_strava_activity(act)
    st.session_state.activity_route = route
    st.session_state.activity_stream_error = None
    # Refresh segment efforts for this ride (dropdown after the ride pick).
    _fetch_activity_efforts(config, act.activity_id)
    return True


def _fetch_activity_efforts(config, activity_id: int) -> None:
    """Load segment efforts for the chosen ride into session (non-widget keys)."""
    if st.session_state.get("_efforts_activity_id") == activity_id:
        return
    try:
        efforts = _run_strava_call(
            config,
            lambda c: c.list_activity_segment_efforts(activity_id),
        )
        st.session_state.activity_efforts = efforts
        st.session_state._efforts_activity_id = activity_id
    except (StravaAuthError, StravaError) as exc:
        st.session_state.activity_efforts = []
        st.session_state._efforts_activity_id = activity_id
        st.session_state.activity_stream_error = (
            f"Segment efforts unavailable ({exc})."
        )
    except Exception as exc:  # noqa: BLE001
        st.session_state.activity_efforts = []
        st.session_state._efforts_activity_id = activity_id
        st.session_state.activity_stream_error = f"Could not list segment efforts: {exc}"


def _load_effort_segment(config, effort: SegmentEffortSummary) -> bool:
    """Slice the activity streams to the effort window (speed + watts).

    Uses ``start_index``/``end_index`` on the ride's recorded streams — never a
    flat elapsed/distance average. Keeps the full ``activity_route`` intact.
    """
    from bike_wind.strava.client import slice_route_for_effort

    if not effort.has_stream_indices:
        st.session_state.activity_stream_error = (
            "This segment effort has no stream indices from Strava."
        )
        return False

    route = st.session_state.get("activity_route")
    act = st.session_state.get("selected_activity")
    # If activity_route was replaced somehow, reload the full ride first.
    need_reload = (
        route is None
        or len(route) < 2
        or (effort.end_index is not None and len(route) <= effort.end_index)
    )
    if need_reload and act is not None:
        if not _load_ride_track(config, act):
            return False
        route = st.session_state.get("activity_route")
    if route is None or len(route) < 2:
        st.session_state.activity_stream_error = (
            "Load the ride track before picking a segment effort."
        )
        return False

    try:
        sliced = slice_route_for_effort(route, effort)
    except ValueError as exc:
        st.session_state.activity_stream_error = str(exc)
        return False
    except Exception as exc:  # noqa: BLE001
        st.session_state.activity_stream_error = f"Could not slice effort: {exc}"
        return False

    st.session_state.selected_effort = effort
    st.session_state.route_source = "effort"
    st.session_state.effort_route = sliced
    st.session_state.strava_segment_id = effort.segment_id
    st.session_state.strava_meta = {
        "segment_id": effort.segment_id,
        "name": effort.name,
        "distance_m": effort.distance_m,
        "average_grade": 0.0,
        "from_cache": False,
    }
    st.session_state.activity_stream_error = None
    if effort.average_watts is not None and effort.average_watts > 0:
        st.session_state.power_w = int(round(effort.average_watts))
    return True


_RIDE_PLACEHOLDER = "— pick a recent ride —"
_DAY_PLACEHOLDER = "— pick a ride from that day —"
_EFFORT_PLACEHOLDER = "— pick a segment on this ride —"


def _on_effort_pick() -> None:
    """Segment-effort selectbox on_change — before widgets; sets route_source."""
    pick = st.session_state.get("effort_ride_pick")
    if not pick or pick == _EFFORT_PLACEHOLDER:
        return
    effort = (st.session_state.get("_effort_label_map") or {}).get(pick)
    if effort is None:
        return
    cur = st.session_state.get("selected_effort")
    if (
        cur is not None
        and cur.effort_id == effort.effort_id
        and st.session_state.get("route_source") == "effort"
        and st.session_state.get("activity_route") is not None
    ):
        return
    _load_effort_segment(load_strava_config(), effort)



def _on_recent_ride_pick() -> None:
    """Selectbox on_change — runs before widgets; safe to set route_source."""
    pick = st.session_state.get("recent_ride_pick")
    if not pick or pick == _RIDE_PLACEHOLDER:
        return
    act = (st.session_state.get("_recent_label_map") or {}).get(pick)
    if act is None:
        return
    loaded_id = getattr(st.session_state.get("selected_activity"), "activity_id", None)
    if loaded_id == act.activity_id and st.session_state.get("activity_route") is not None:
        return
    _load_ride_track(load_strava_config(), act)


def _on_day_ride_pick() -> None:
    """Day-list selectbox on_change — same callback-order rules as recent rides."""
    pick = st.session_state.get("day_ride_pick")
    if not pick or pick == _DAY_PLACEHOLDER:
        return
    act = (st.session_state.get("_day_label_map") or {}).get(pick)
    if act is None:
        return
    loaded_id = getattr(st.session_state.get("selected_activity"), "activity_id", None)
    if loaded_id == act.activity_id and st.session_state.get("activity_route") is not None:
        return
    _load_ride_track(load_strava_config(), act)


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


def _chart_colors(dark: bool) -> tuple[str, str, str, str]:
    """Background, foreground, accent, spine for grey-canvas charts."""
    if dark:
        return "#0c1210", "#e7efea", "#3ecf8e", "#3a4a42"
    return "#f7faf8", "#14201b", "#1a6b52", "#c5d0ca"


def _style_speed_axes(ax, *, dark: bool, fig) -> tuple[str, str, str]:
    bg, fg, accent, spine = _chart_colors(dark)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlabel("Distance (mi)", color=fg)
    ax.set_ylabel("mph", color=fg)
    ax.tick_params(colors=fg)
    for sp in ax.spines.values():
        sp.set_color(spine)
    ax.grid(True, alpha=0.22, color=fg)
    return bg, fg, accent


def _validation_help_text(
    result,
    *,
    position_label: str,
    cda: float,
    surface_label: str,
    crr_note: str | None,
) -> str:
    """Plain-language parameter blurb for the validation chart popover."""
    p = result.params
    grade = result.mean_grade_pct
    eta_pct = (1.0 - p.drivetrain_efficiency) * 100.0
    power_line = (
        f"Power: recorded meter watts (mean ≈ {p.power_w:.0f} W crank)."
        if result.power_source == "meter"
        else f"Power: entered rider power {p.power_w:.0f} W (no power meter)."
    )
    crr_line = f"Crr={p.crr:.4f} · {surface_label}."
    if crr_note:
        crr_line = f"{crr_line} {crr_note}"
    lines = [
        power_line,
        f"System mass: {p.total_mass_kg:.1f} kg "
        f"(rider {p.rider_mass_kg:.1f} + bike {p.bike_mass_kg:.1f}).",
        f"Position: {position_label} · Cd·A={cda:.3f}.",
        crr_line,
        f"Drivetrain: ~{eta_pct:.0f}% loss (efficiency {p.drivetrain_efficiency:.2f}).",
        f"Air density ρ={result.rho_mean:.3f} kg/m³ from temp/pressure.",
        f"Wind: {result.wind_label(unit='mph')}.",
        f"Grade: elevation rise/run in the solver (mean |grade| ≈ {grade:.1f}%).",
        "Model does not brake; it reacts to every grade change, so descents "
        "spike higher than a real ride.",
        "Units: speed mph, distance miles.",
        "Lines: teal = model with wind+grade; grey = recorded speed "
        "(velocity_smooth or time+distance).",
    ]
    return "\n".join(lines)


def main() -> None:
    _init_state()

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
    oauth_ok = oauth_configured(
        strava_cfg.client_id, strava_cfg.client_secret, strava_cfg.redirect_uri
    )
    connected = bool(st.session_state.get("strava_oauth_connected"))

    # Compact Connect / Disconnect — not a hero.
    conn_l, conn_r = st.columns([3, 2])
    with conn_l:
        if connected:
            st.caption("Strava connected (this browser).")
        elif oauth_ok:
            st.caption("Optional: connect your Strava for personal rides.")
        else:
            st.caption("Strava connect needs client_id / secret in secrets.")
    with conn_r:
        if connected:
            st.button(
                "Disconnect",
                on_click=_on_disconnect_strava,
                key="disconnect_strava",
                use_container_width=True,
            )
        elif oauth_ok:
            auth_url = build_authorize_url(
                strava_cfg.client_id,  # type: ignore[arg-type]
                strava_cfg.redirect_uri or DEFAULT_REDIRECT_URI,
            )
            st.link_button(
                "Connect with Strava",
                auth_url,
                use_container_width=True,
            )
    if st.session_state.get("strava_oauth_error"):
        st.error(st.session_state.strava_oauth_error)

    default_dt = datetime.now(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    ) + timedelta(hours=3)

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
                )
            except StravaAuthError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load segment: {exc}")

        if (
            st.session_state.get("route_source") == "segment"
            and st.session_state.strava_meta
            and st.session_state.strava_segment_id
        ):
            m = st.session_state.strava_meta
            st.caption(f"Using segment: **{m['name']}** · {m['segment_id']}")

        st.caption(
            "Recent ride — your Strava if connected, else the app token. "
            "Choosing a ride loads the track."
        )
        if strava_ready:
            _ensure_recent_rides(strava_cfg, limit=10)

        activities: list[ActivitySummary] = list(
            st.session_state.get("recent_activities") or []
        )
        if activities:
            label_map = {ride_activity_label(a): a for a in activities}
            # Maps must exist before on_change fires on the next run.
            st.session_state._recent_label_map = label_map
            labels = [_RIDE_PLACEHOLDER, *label_map.keys()]
            st.selectbox(
                "Recent rides",
                options=labels,
                key="recent_ride_pick",
                on_change=_on_recent_ride_pick,
            )

        ride_day = st.date_input(
            "Rides on date (UTC)",
            value=datetime.now(timezone.utc).date(),
            key="ride_day_filter",
        )
        if st.button(
            "List rides on date",
            disabled=not strava_ready,
            use_container_width=True,
            key="list_rides_on_date",
        ):
            try:
                day_acts = _run_strava_call(
                    strava_cfg,
                    lambda c: c.list_activities_on_date(ride_day),
                )
                day_rides = _filter_cycling(day_acts)
                st.session_state.day_activities = day_rides
                st.session_state.activity_stream_error = (
                    None
                    if day_rides
                    else f"No rides on {ride_day.isoformat()}."
                )
            except (StravaAuthError, StravaError) as exc:
                st.session_state.day_activities = []
                st.session_state.activity_stream_error = str(exc)
            except Exception as exc:  # noqa: BLE001
                st.session_state.day_activities = []
                st.session_state.activity_stream_error = f"Could not list rides: {exc}"

        day_activities: list[ActivitySummary] = list(
            st.session_state.get("day_activities") or []
        )
        if day_activities:
            day_map = {ride_activity_label(a): a for a in day_activities}
            st.session_state._day_label_map = day_map
            day_labels = [_DAY_PLACEHOLDER, *day_map.keys()]
            st.selectbox(
                "Rides that day",
                options=day_labels,
                key="day_ride_pick",
                on_change=_on_day_ride_pick,
            )

        if st.session_state.get("selected_activity") and st.session_state.get(
            "route_source"
        ) in ("activity", "effort"):
            act = st.session_state.selected_activity
            n_pts = len(st.session_state.activity_route or [])
            if st.session_state.get("route_source") == "effort" and st.session_state.get(
                "selected_effort"
            ):
                eff = st.session_state.selected_effort
                st.caption(
                    f"Segment on ride: **{eff.name}** · {eff.label()}"
                    + (f" · {n_pts} pts" if n_pts else "")
                )
            else:
                st.caption(
                    f"Track loaded: **{act.name}** · {ride_activity_label(act)}"
                    + (f" · {n_pts} pts" if n_pts else "")
                )
            # Segment efforts on this ride → choosing one loads that segment's GPS.
            if st.session_state.get("_efforts_activity_id") != act.activity_id:
                _fetch_activity_efforts(strava_cfg, act.activity_id)
            efforts: list[SegmentEffortSummary] = list(
                st.session_state.get("activity_efforts") or []
            )
            if efforts:
                effort_map = {e.label(): e for e in efforts}
                st.session_state._effort_label_map = effort_map
                st.selectbox(
                    "Segment on this ride",
                    options=[_EFFORT_PLACEHOLDER, *effort_map.keys()],
                    key="effort_ride_pick",
                    on_change=_on_effort_pick,
                )
            elif st.session_state.get("_efforts_activity_id") == act.activity_id:
                st.caption("No segment efforts on this ride.")

        if st.session_state.get("activity_stream_error"):
            st.error(st.session_state.activity_stream_error)

        st.caption("GPX — use your own file instead of a segment or ride.")
        uploaded = st.file_uploader("GPX", type=["gpx"], label_visibility="collapsed")
        use_sample = st.checkbox(
            "Demo route",
            value=(
                uploaded is None
                and st.session_state.get("route_source")
                not in ("segment", "activity", "effort")
            ),
        )

    with st.container(border=True):
        st.html('<p class="bw-section-label">Rider</p>')
        when = st.datetime_input(
            "When (UTC)",
            value=default_dt,
            help="Forecast time for a future attempt. Timed GPX or Strava rides use their start time for wind instead.",
        )
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

        surf_labels = surface_labels()
        current_surf = SURFACE_PRESETS[st.session_state.surface_key].label
        surf_chosen = st.selectbox(
            "Surface / Crr",
            options=surf_labels,
            index=surf_labels.index(current_surf) if current_surf in surf_labels else 0,
            help="Rolling resistance. Packed-sand gravel value is an estimate, not a lab test.",
        )
        surface_key = surface_key_for_label(surf_chosen)
        st.session_state.surface_key = surface_key
        surface = resolve_surface(surface_key)
        st.caption(
            f"Crr = {surface.crr:.4f}"
            + (f" · {surface.estimate_note}" if surface.estimate_note else "")
        )

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
            f"Crr={surface.crr:.4f} · "
            f"{st.session_state.rider_kg:.0f}+{aero.bike_mass_kg:.0f}="
            f"{st.session_state.rider_kg + aero.bike_mass_kg:.0f} kg"
        )

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
            f"Refresh: {'set' if presence.get('REFRESH_TOKEN') else 'missing'} · "
            f"OAuth: {'connected' if connected else 'off'}"
        )
        st.caption(
            "Strava app Authorization Callback Domain: gustypower.streamlit.app "
            f"(redirect {strava_cfg.redirect_uri or DEFAULT_REDIRECT_URI})."
        )
        if st.session_state.get("strava_last_error"):
            st.error(st.session_state.strava_last_error)

    last_run = st.session_state.get("_last_run")
    if not run and not last_run:
        return

    use_effort = st.session_state.get("route_source") == "effort" and bool(
        st.session_state.get("selected_effort")
    )
    use_activity = st.session_state.get("route_source") == "activity" and bool(
        st.session_state.get("selected_activity")
    )
    use_strava = st.session_state.get("route_source") == "segment" and bool(
        st.session_state.get("strava_segment_id")
    )

    if run:
        try:
            if use_effort and strava_ready:
                effort: SegmentEffortSummary = st.session_state.selected_effort
                route = st.session_state.get("effort_route")
                if route is None or len(route) < 2:
                    if not _load_effort_segment(strava_cfg, effort):
                        st.error(
                            st.session_state.activity_stream_error
                            or "Segment streams unavailable."
                        )
                        return
                    route = st.session_state.effort_route
                source_label = f"Strava segment: {effort.name}"
            elif use_activity and strava_ready:
                act: ActivitySummary = st.session_state.selected_activity
                route = st.session_state.get("activity_route")
                if route is None or len(route) < 2:
                    if not _load_ride_track(strava_cfg, act):
                        st.error(
                            st.session_state.activity_stream_error
                            or "Ride streams unavailable."
                        )
                        return
                    route = st.session_state.activity_route
                source_label = f"Strava ride: {act.name}"
            elif use_strava and strava_ready:
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
        surface = resolve_surface(st.session_state.surface_key)
        params = RiderBikeParams(
            rider_mass_kg=float(st.session_state.rider_kg),
            power_w=float(power_w),
            bike_mass_kg=aero.bike_mass_kg,
            cd=aero.cd,
            area_m2=aero.area_m2,
            crr=surface.crr,
            drivetrain_efficiency=DRIVETRAIN_EFFICIENCY,
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
        st.session_state._last_run = {
            "result": result,
            "source_label": source_label,
            "route_km": route.distances_m[-1] / 1000.0,
            "position_key": aero.position_key,
            "cda": aero.cd * aero.area_m2,
            "surface_key": surface.key,
            "surface_label": surface.label,
            "crr_note": surface.estimate_note,
        }
        source_label_out = source_label
        route_km = route.distances_m[-1] / 1000.0
        position_key_out = aero.position_key
        cda_out = aero.cd * aero.area_m2
        surface_label_out = surface.label
        crr_note_out = surface.estimate_note
    else:
        result = last_run["result"]
        source_label_out = last_run["source_label"]
        route_km = last_run["route_km"]
        position_key_out = last_run["position_key"]
        cda_out = last_run["cda"]
        surface_label_out = last_run.get("surface_label") or resolve_surface(
            last_run.get("surface_key", DEFAULT_SURFACE_KEY)
        ).label
        crr_note_out = last_run.get("crr_note")

    with st.container(border=True):
        st.html('<p class="bw-section-label">Results</p>')
        st.caption(
            f"{source_label_out} · {route_km:.1f} km · "
            f"{POSITION_PRESETS[position_key_out].label} · Cd·A={cda_out:.3f} · "
            f"Crr={result.params.crr:.4f}"
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Time (with wind)", format_duration(result.eta_with_wind_s))
        m2.metric("Time (calm)", format_duration(result.eta_no_wind_s))
        avg = (
            sum(result.speeds_with_wind_mph) / len(result.speeds_with_wind_mph)
            if result.speeds_with_wind_mph
            else 0.0
        )
        m3.metric("Avg speed", f"{avg:.1f} mph")
        st.caption(
            "Predicted times: with-wind (solid) vs calm (dashed). "
            + result.wind_label(unit="mph")
        )
        if result.power_source == "entered":
            st.caption(result.power_source_note)
        elif result.weather_message and not result.weather_ok:
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
        bg, fg, accent = _style_speed_axes(ax, dark=dark, fig=fig)
        ax.plot(
            result.distances_mi,
            result.speeds_no_wind_mph,
            label="Calm",
            color="#8fa398",
            linestyle="--",
        )
        ax.plot(
            result.distances_mi,
            result.speeds_with_wind_mph,
            label="With wind",
            color=accent,
            linewidth=2,
        )
        if result.distances_mi:
            ax.set_xlim(result.distances_mi[0], result.distances_mi[-1])
        legend = ax.legend(facecolor=bg, edgecolor="#3a4a42" if dark else "#c5d0ca")
        for text in legend.get_texts():
            text.set_color(fg)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # --- Validation: actual vs model (same grey canvas) ---
        head_l, head_r = st.columns([5, 1])
        with head_l:
            st.caption(
                f"Validation · grade in model (mean |grade| ≈ {result.mean_grade_pct:.1f}%)"
            )
        with head_r:
            with st.popover("Info"):
                st.text(
                    _validation_help_text(
                        result,
                        position_label=POSITION_PRESETS[position_key_out].label,
                        cda=cda_out,
                        surface_label=surface_label_out,
                        crr_note=crr_note_out,
                    )
                )

        fig2, ax2 = plt.subplots(figsize=(9, 3.4))
        bg2, fg2, accent2 = _style_speed_axes(ax2, dark=dark, fig=fig2)
        ax2.plot(
            result.distances_mi,
            result.speeds_with_wind_mph,
            label="Model",
            color=accent2,
            linewidth=2,
        )
        if result.has_actual_speeds and result.actual_speeds_mph is not None:
            ax2.plot(
                result.distances_mi,
                result.actual_speeds_mph,
                label="Actual",
                color="#8fa398",
                linewidth=1.8,
                alpha=0.95,
            )
        else:
            st.caption(
                "Actual speed needs a GPX or Strava velocity_smooth / time+distance — "
                "showing model only."
            )
        st.caption(
            f"{result.power_source_note} · {result.wind_label(unit='mph')}"
        )
        if result.distances_mi:
            ax2.set_xlim(result.distances_mi[0], result.distances_mi[-1])
        leg2 = ax2.legend(facecolor=bg2, edgecolor="#3a4a42" if dark else "#c5d0ca")
        for text in leg2.get_texts():
            text.set_color(fg2)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


if __name__ == "__main__":
    main()
