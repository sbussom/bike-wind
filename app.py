"""Streamlit app: GPX → wind → speed chart with rider / aero controls."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from bike_wind.analysis import (
    format_duration,
    parse_target_time,
    power_for_target_time,
    time_uncertainty_band,
    wind_pace_split,
)
from bike_wind.physics_explain import explain_prediction
from bike_wind.gpx.parse import parse_gpx
from bike_wind.local_wind_map import compass_label, format_wind_speed_dir
from bike_wind.ride_clock import clock_speed_chart, figure_clock_speed
from bike_wind.choice_list import visible_choice
from bike_wind.calibration.metrics import score_segment_time
from bike_wind.forecast_log import (
    ForecastLog,
    PERSISTENCE_NOTE,
    fetch_actual_wind,
    forecast_row_line,
    match_forecasts,
    post_ride_row_line,
    summary_line,
    summarize_forecast,
    summarize_post_rides,
    target_dates,
)
from bike_wind.ride_filter import (
    accuracy_skip_reason,
    collect_outdoor_rides,
    hidden_rides_note,
)
from bike_wind.when_pick import pick_utc_date, pick_utc_when
from bike_wind.wind_chart import figure_wind_along, wind_along_chart
from bike_wind.physics.constants import DRIVETRAIN_EFFICIENCY
from bike_wind.physics.forces import RiderBikeParams
from bike_wind.predict import (
    apply_what_if,
    downsample_route,
    predict_speeds,
    what_if_course,
)
from bike_wind.clothing import (
    CLOTHING_PRESETS,
    DEFAULT_CLOTHING_KEY,
    clothing_key_for_label,
    clothing_labels,
    resolve_clothing,
)
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
from bike_wind.theme import apply_theme, chart_palette
from bike_wind.user_config import clothing_key_from_config, save_rider_config
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
    ACTIVITY_SCOPE_HINT,
    DEFAULT_REDIRECT_URI,
    build_authorize_url,
    exchange_code,
    oauth_configured,
    scope_has_activity_read,
)
from bike_wind.weather import hrrr_enabled_by_env



FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_route.gpx"

st.set_page_config(
    page_title="Bike Wind",
    layout="centered",
    initial_sidebar_state="collapsed",
)



def _init_state() -> None:
    if "bw_dark" not in st.session_state:
        st.session_state.bw_dark = False
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
    if "clothing_key" not in st.session_state:
        st.session_state.clothing_key = clothing_key_from_config()
        if st.session_state.clothing_key not in CLOTHING_PRESETS:
            st.session_state.clothing_key = DEFAULT_CLOTHING_KEY
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
    if "strava_token_scope" not in st.session_state:
        st.session_state.strava_token_scope = None
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
    st.session_state.strava_token_scope = None
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
        if tokens.scope:
            st.session_state.strava_token_scope = tokens.scope
    else:
        st.session_state.strava_refreshed_access_token = tokens.access_token
        if tokens.refresh_token:
            st.session_state.strava_refreshed_refresh_token = tokens.refresh_token


def _session_has_activity_read() -> bool:
    """Connected browser token includes activity:read (not the owner secrets token)."""
    return bool(
        st.session_state.get("strava_oauth_connected")
        and st.session_state.get("strava_access_token")
        and scope_has_activity_read(st.session_state.get("strava_token_scope"))
    )


def _strava_client(config, *, secrets_only: bool = False) -> StravaClient:
    """Client for segment calls; auto-refreshes on 401 into session_state.

    Prefer secrets (or secrets refresh) for public segment-by-id. Personal rides
    must use ``_activity_strava_client`` instead — never this path with scope=read.
    """

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


def _activity_strava_client(config) -> StravaClient | None:
    """Session-only client when scope includes activity:read; else None."""
    if not _session_has_activity_read():
        return None

    def on_refreshed(tokens) -> None:
        _persist_refreshed_tokens(tokens)

    return StravaClient(
        config,
        access_token=st.session_state.strava_access_token,
        refresh_token=st.session_state.strava_refresh_token,
        on_tokens_refreshed=on_refreshed,
    )


def _run_strava_call(config, fn):
    """Run ``fn(client)`` for segments; secrets path may retry after refresh failure.

    Connected visitors keep their session token only — never silently fall back
    to the owner secrets token for personal rides (use ``_run_activity_call``).
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


def _run_activity_call(config, fn):
    """Run ``fn(client)`` for activities/efforts/streams — session + activity:read only.

    Never sends the owner secrets token (scope read) to /athlete/activities.
    """
    client = _activity_strava_client(config)
    if client is None:
        raise StravaAuthError(ACTIVITY_SCOPE_HINT, kind="forbidden_scope")
    try:
        return fn(client)
    except StravaAuthError as exc:
        if getattr(exc, "kind", None) == "refresh_failed":
            _clear_strava_session_tokens()
            raise StravaAuthError(
                "Strava session expired — reconnect.",
                kind="refresh_failed",
            ) from exc
        raise


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
        # Prefer token payload scope; fall back to redirect ?scope=
        scope = tokens.scope or (str(params.get("scope")) if params.get("scope") else None)
        st.session_state.strava_oauth_connected = True
        st.session_state.strava_access_token = tokens.access_token
        st.session_state.strava_refresh_token = tokens.refresh_token
        st.session_state.strava_token_expires_at = tokens.expires_at
        st.session_state.strava_token_scope = scope
        st.session_state.strava_oauth_error = None
        _clear_activity_route_state()
        st.session_state._oauth_handled_code = code
        _clear_oauth_query_params()
        if scope_has_activity_read(scope):
            st.success("Strava connected for this browser session.")
        else:
            st.warning(ACTIVITY_SCOPE_HINT)
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


def _ensure_recent_rides(config, *, limit: int = 10) -> None:
    """Fetch recent outdoor rides with the session activity:read token only.

    Virtual, trainer, e-bike, and no-GPS activities are skipped. If the first
    Strava page is mostly those, later pages are fetched until ``limit``
    outdoor rides are in hand or the history runs out.
    """
    if not _session_has_activity_read():
        st.session_state.recent_activities = []
        st.session_state.recent_hidden_rides = 0
        if st.session_state.get("strava_oauth_connected"):
            st.session_state.activity_stream_error = ACTIVITY_SCOPE_HINT
        return
    # Bumped when the outdoor filter landed so an open browser refetches.
    fp = "session-activity-outdoor"
    if st.session_state.get("_rides_token_fp") == fp and st.session_state.recent_activities:
        return
    per_page = 30
    try:
        def _fetch(client: StravaClient):
            return collect_outdoor_rides(
                lambda page: client.list_recent_activities(per_page=per_page, page=page),
                per_page=per_page,
                want=limit,
            )

        rides, hidden = _run_activity_call(config, _fetch)
        st.session_state.recent_activities = rides
        st.session_state.recent_hidden_rides = hidden
        st.session_state._rides_token_fp = fp
        st.session_state.activity_stream_error = (
            None if rides else "No recent outdoor rides found for this athlete."
        )
    except StravaAuthError as exc:
        st.session_state.recent_activities = []
        st.session_state.recent_hidden_rides = 0
        st.session_state._rides_token_fp = fp
        # Keep the short scope hint; never dump diagnostics / secrets paste text.
        if getattr(exc, "kind", None) == "forbidden_scope":
            st.session_state.activity_stream_error = ACTIVITY_SCOPE_HINT
        else:
            st.session_state.activity_stream_error = ACTIVITY_SCOPE_HINT
    except StravaError:
        st.session_state.recent_activities = []
        st.session_state.recent_hidden_rides = 0
        st.session_state._rides_token_fp = fp
        st.session_state.activity_stream_error = ACTIVITY_SCOPE_HINT
    except Exception:  # noqa: BLE001
        st.session_state.recent_activities = []
        st.session_state.recent_hidden_rides = 0
        st.session_state._rides_token_fp = fp
        st.session_state.activity_stream_error = ACTIVITY_SCOPE_HINT


def _load_ride_track(config, act: ActivitySummary) -> bool:
    """Pull GPS/elev/time/speed streams and set as the active route. Returns ok.

    Safe from ``on_change`` callbacks (runs before widgets). Must not assign
    any key that is also a widget ``key=`` later in the script.
    """
    try:
        route = _run_activity_call(
            config,
            lambda c: c.load_activity_route(
                act.activity_id, start_date=act.start_date
            ),
        )
    except StravaAuthError:
        st.session_state.activity_stream_error = ACTIVITY_SCOPE_HINT
        st.session_state.activity_route = None
        return False
    except StravaError:
        st.session_state.activity_stream_error = ACTIVITY_SCOPE_HINT
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
        efforts = _run_activity_call(
            config,
            lambda c: c.list_activity_segment_efforts(activity_id),
        )
        st.session_state.activity_efforts = efforts
        st.session_state._efforts_activity_id = activity_id
    except (StravaAuthError, StravaError):
        st.session_state.activity_efforts = []
        st.session_state._efforts_activity_id = activity_id
        st.session_state.activity_stream_error = ACTIVITY_SCOPE_HINT
    except Exception:  # noqa: BLE001
        st.session_state.activity_efforts = []
        st.session_state._efforts_activity_id = activity_id
        st.session_state.activity_stream_error = ACTIVITY_SCOPE_HINT


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
    kom_time_s = None
    try:
        meta = _run_strava_call(
            config, lambda c: c.get_segment_meta(effort.segment_id, use_cache=False)
        )
        kom_time_s = meta.kom_time_s
        st.session_state.strava_meta = {
            "segment_id": meta.segment_id,
            "name": meta.name or effort.name,
            "distance_m": meta.distance_m or effort.distance_m,
            "average_grade": meta.average_grade,
            "from_cache": meta.from_cache,
            "kom_time_s": kom_time_s,
        }
    except Exception:  # noqa: BLE001 — keep effort usable without KOM
        st.session_state.strava_meta = {
            "segment_id": effort.segment_id,
            "name": effort.name,
            "distance_m": effort.distance_m,
            "average_grade": 0.0,
            "from_cache": False,
            "kom_time_s": None,
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
    """Background, foreground, accent, spine for the shared theme."""
    pal = chart_palette(dark)
    return pal.chart_bg, pal.text, pal.accent, pal.spine


def _style_speed_axes(ax, *, dark: bool, fig) -> tuple[str, str, str]:
    bg, fg, accent, spine = _chart_colors(dark)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlabel("Distance (mi)", color=fg)
    ax.set_ylabel("mph", color=fg)
    ax.tick_params(colors=fg)
    for sp in ax.spines.values():
        sp.set_color(spine)
    ax.grid(True, alpha=0.45, color=spine)
    return bg, fg, accent


def _style_power_axes(ax, *, dark: bool, fig) -> tuple[str, str, str]:
    """Distance (mi) × watts chart styling (same grey canvas as speed)."""
    bg, fg, accent = _style_speed_axes(ax, dark=dark, fig=fig)
    ax.set_ylabel("watts", color=fg)
    return bg, fg, accent


def _validation_help_text(
    result,
    *,
    position_label: str,
    cda: float,
    surface_label: str,
    crr_note: str | None,
    power_mode_label: str,
    whatif_power_w: float,
    whatif_wind_mph: float,
    whatif_wind_from_deg: float,
    whatif_grade_pct: float,
) -> str:
    """Plain-language parameter blurb for the results charts popover."""
    p = result.params
    crr_line = f"Crr={p.crr:.4f} · {surface_label}."
    if crr_note:
        crr_line = f"{crr_line} {crr_note}"
    wind_s = format_wind_speed_dir(
        whatif_wind_mph / 2.236936,
        whatif_wind_from_deg,
        unit="mph",
    )
    window = result.power_window_s
    if result.has_watts and window:
        power_trace = f"Power chart: recorded watts with {window:.0f}s rolling mean."
    elif result.has_watts:
        power_trace = "Power chart: recorded effort watts (already ~3s average)."
    else:
        power_trace = "No recorded power on this ride — model uses flat watts."
    lines = [
        f"Power mode: {power_mode_label}.",
        f"Flat slider watts (what-if): {whatif_power_w:.0f} W.",
        f"What-if wind: {wind_s}.",
        f"Climb scale: {whatif_grade_pct:.0f}% of this route's grades.",
        f"System mass: {p.total_mass_kg:.1f} kg "
        f"(rider {p.rider_mass_kg:.1f} + bike {p.bike_mass_kg:.1f}).",
        f"Position: {position_label} · Cd·A={cda:.3f}.",
        crr_line,
        power_trace,
        "Predicted speed uses that power series (or flat slider) + grade + wind.",
        "Model does not brake; descents spike higher than a real ride.",
        "Units: mph / miles; watts on the power chart.",
    ]
    return "\n".join(lines)


_POWER_MODE_RECORDED = "Recorded effort"
_POWER_MODE_FLAT = "Flat slider watts"

_MODE_PAST = "Past ride analysis"
_MODE_FUTURE = "Future ride prediction"
# Older session snapshots used shorter labels.
_PAST_MODE_ALIASES = {_MODE_PAST, "Past rides"}
_FUTURE_MODE_ALIASES = {_MODE_FUTURE, "Future prediction"}


def _is_past_mode(mode: str | None) -> bool:
    return mode in _PAST_MODE_ALIASES


def _format_time_error_pct(error_pct: float) -> str:
    """Signed percent for the results strip (+slow / −fast vs actual)."""
    return f"{error_pct:+.1f}%"


def _panel_rider_controls(*, key_prefix: str):
    """Position / surface / clothing / power / mass for one mode panel."""
    labels = position_labels()
    current_label = POSITION_PRESETS[st.session_state.position_key].label
    pos_key = f"{key_prefix}_position"
    if pos_key not in st.session_state:
        st.session_state[pos_key] = (
            current_label if current_label in labels else labels[0]
        )
    chosen = st.radio("Position", options=labels, key=pos_key)
    position_key = position_key_for_label(chosen)
    if position_key != st.session_state.position_key:
        st.session_state.position_key = position_key
        if not st.session_state.bike_mass_manual:
            _apply_bike_kg(POSITION_PRESETS[position_key].bike_mass_kg, manual=False)

    surf_labels = surface_labels()
    current_surf = SURFACE_PRESETS[st.session_state.surface_key].label
    surf_key = f"{key_prefix}_surface"
    if surf_key not in st.session_state:
        st.session_state[surf_key] = (
            current_surf if current_surf in surf_labels else surf_labels[0]
        )
    surf_chosen = st.radio(
        "Surface / Crr",
        options=surf_labels,
        key=surf_key,
        help="Rolling resistance. Packed-sand gravel value is an estimate, not a lab test.",
    )
    surface_key = surface_key_for_label(surf_chosen)
    st.session_state.surface_key = surface_key
    surface = resolve_surface(surface_key)
    st.caption(
        f"Crr = {surface.crr:.4f}"
        + (f" · {surface.estimate_note}" if surface.estimate_note else "")
    )

    cloth_labels = clothing_labels()
    current_cloth = CLOTHING_PRESETS[st.session_state.clothing_key].label
    cloth_key = f"{key_prefix}_clothing"
    if cloth_key not in st.session_state:
        st.session_state[cloth_key] = (
            current_cloth if current_cloth in cloth_labels else cloth_labels[0]
        )
    cloth_chosen = st.radio(
        "Clothing",
        options=cloth_labels,
        key=cloth_key,
        help=(
            "Cd·A multiplier. Skin-tight = 1.00 (default, unchanged predictions). "
            "Loose jersey ≈ +8%. Manual — Strava has no clothing field."
        ),
    )
    clothing_key = clothing_key_for_label(cloth_chosen)
    if clothing_key != st.session_state.clothing_key:
        st.session_state.clothing_key = clothing_key
        save_rider_config({"clothing_key": clothing_key})
    clothing = resolve_clothing(st.session_state.clothing_key)
    st.caption(
        f"Cd·A × {clothing.cda_multiplier:.2f}"
        + (f" · {clothing.note}" if clothing.note else "")
    )

    power_key = f"{key_prefix}_power"
    if power_key not in st.session_state:
        st.session_state[power_key] = int(st.session_state.power_w)
    power_w = st.slider("Power (W)", min_value=80, max_value=450, step=5, key=power_key)
    st.session_state.power_w = int(power_w)

    kg_key = f"{key_prefix}_rider_kg"
    lb_key = f"{key_prefix}_rider_lb"
    if kg_key not in st.session_state:
        st.session_state[kg_key] = float(st.session_state.rider_kg)
    if lb_key not in st.session_state:
        st.session_state[lb_key] = round(kg_to_lb(st.session_state.rider_kg), 1)

    def _sync_kg() -> None:
        st.session_state.rider_kg = float(st.session_state[kg_key])
        st.session_state[lb_key] = round(kg_to_lb(st.session_state.rider_kg), 1)
        st.session_state.rider_kg_input = float(st.session_state.rider_kg)
        st.session_state.rider_lb_input = float(st.session_state[lb_key])
        if not st.session_state.height_manual:
            st.session_state.height_cm = smart_default_height_cm(st.session_state.rider_kg)
            st.session_state.height_cm_input = float(st.session_state.height_cm)

    def _sync_lb() -> None:
        st.session_state.rider_kg = lb_to_kg(float(st.session_state[lb_key]))
        st.session_state[kg_key] = float(st.session_state.rider_kg)
        st.session_state.rider_kg_input = float(st.session_state.rider_kg)
        st.session_state.rider_lb_input = float(st.session_state[lb_key])
        if not st.session_state.height_manual:
            st.session_state.height_cm = smart_default_height_cm(st.session_state.rider_kg)
            st.session_state.height_cm_input = float(st.session_state.height_cm)

    w1, w2 = st.columns(2)
    with w1:
        st.number_input(
            "Rider kg",
            min_value=40.0,
            max_value=120.0,
            step=0.5,
            key=kg_key,
            on_change=_sync_kg,
        )
    with w2:
        st.number_input(
            "Rider lb",
            min_value=90.0,
            max_value=265.0,
            step=1.0,
            key=lb_key,
            on_change=_sync_lb,
        )
    st.session_state.rider_kg = float(st.session_state[kg_key])

    aero = resolve_aero(
        st.session_state.position_key,
        full_aero_kit=bool(st.session_state.get("full_aero_kit", False)),
        non_aero=bool(st.session_state.get("non_aero", False)),
        bike_mass_kg=float(st.session_state.bike_kg),
    )
    clothing = resolve_clothing(st.session_state.clothing_key)
    cda_used = aero.cd * aero.area_m2 * clothing.cda_multiplier
    st.caption(
        f"{POSITION_PRESETS[aero.position_key].label} · "
        f"Cd·A={cda_used:.3f} (×{clothing.cda_multiplier:.2f} clothing"
        f"{'; lower when pushing hard, ~0.48 at 250 W' if aero.power_cda else ''}) · "
        f"Crr={surface.crr:.4f} · "
        f"{st.session_state.rider_kg:.0f}+{aero.bike_mass_kg:.0f}="
        f"{st.session_state.rider_kg + aero.bike_mass_kg:.0f} kg"
    )
    return float(power_w), surface, aero


def _kom_time_s() -> float | None:
    """Strava KOM seconds from the loaded segment, or None. Never invented."""
    meta = st.session_state.get("strava_meta") or {}
    kom = meta.get("kom_time_s")
    if isinstance(kom, (int, float)) and float(kom) > 0.0:
        return float(kom)
    return None


def _on_use_kom() -> None:
    kom = _kom_time_s()
    if kom is None:
        return
    st.session_state.pace_target_time = format_duration(kom)
    st.session_state._pace_run = True


def _pace_fingerprint(result, avg_power_w: float, wind_speed_mps: float, grade_scale: float, when_utc: str | None):
    end = float(result.distances_m[-1]) if result.distances_m else 0.0
    return (
        round(float(avg_power_w), 1),
        round(float(wind_speed_mps), 2),
        round(float(grade_scale), 3),
        len(result.distances_m),
        round(end, 0),
        when_utc or "",
    )


def _when_from_iso(when_utc: str | None) -> datetime | None:
    if not when_utc:
        return None
    try:
        when_dt = datetime.fromisoformat(when_utc)
    except ValueError:
        return None
    if when_dt.tzinfo is None:
        return when_dt.replace(tzinfo=timezone.utc)
    return when_dt.astimezone(timezone.utc)


def _forecast_log_allowed(result, when_dt: datetime) -> bool:
    """True when this run is a plan or a start, not a past-ride replay."""
    if not result.weather_ok:
        return False
    if result.used_ride_start_time:
        return False
    if result.weather_source in {"none", "open-meteo-archive"}:
        return False
    if not result.lats or not result.lons:
        return False
    if when_dt.tzinfo is None:
        when_dt = when_dt.replace(tzinfo=timezone.utc)
    if when_dt < datetime.now(timezone.utc) - timedelta(minutes=30):
        return False
    if st.session_state.get("route_source") in {"activity", "effort"}:
        return False
    return True


def _log_segment(source_label: str) -> tuple[int | None, str]:
    if st.session_state.get("route_source") != "segment":
        return None, source_label
    meta = st.session_state.get("strava_meta") or {}
    name = str(meta.get("name") or source_label)
    try:
        sid = int(meta["segment_id"]) if meta.get("segment_id") else None
    except (TypeError, ValueError, KeyError):
        sid = None
    return sid, name


def _wind_is_forecast(result, wind_speed_mps: float, grade_scale: float) -> bool:
    """Plan logging keeps the forecast the model used, not a hand-scaled wind."""
    wind_mph = float(wind_speed_mps) * 2.236936
    if abs(wind_mph - float(result.mean_wind_speed_mph)) > 0.3:
        return False
    return abs(float(grade_scale) - 1.0) <= 1e-6


def _record_post_ride(result, power_w: float) -> bool:
    """Store a segment-time score for this past ride when it has a power meter."""
    source = st.session_state.get("route_source")
    act = st.session_state.get("selected_activity")
    effort = st.session_state.get("selected_effort") if source == "effort" else None
    activity = act if isinstance(act, ActivitySummary) else None
    if source == "effort":
        if not isinstance(effort, SegmentEffortSummary):
            return False
        if accuracy_skip_reason(activity, effort) is not None or activity is None:
            return False
        actual = float(effort.elapsed_time_s or effort.moving_time_s)
        segment_id = int(effort.segment_id)
        name = effort.name
        activity_id = int(activity.activity_id)
    elif source == "activity":
        if activity is None or accuracy_skip_reason(activity) is not None:
            return False
        actual = float(activity.moving_time_s)
        segment_id = 0
        name = activity.name
        activity_id = int(activity.activity_id)
    else:
        return False
    grade = score_segment_time(float(result.eta_with_wind_s), actual)
    if grade is None:
        return False
    ForecastLog().upsert_post_ride(
        activity_id=activity_id,
        segment_id=segment_id,
        segment_name=name,
        predicted_time_s=grade.predicted_s,
        actual_time_s=grade.actual_s,
        time_error_s=grade.error_s,
        time_error_pct=grade.error_pct,
        passed=grade.passed,
        power_w=float(power_w),
    )
    return True


def _capture_run_snapshot(result, *, when_dt: datetime, power_w: float, source_label: str) -> None:
    """Log a pre-ride forecast, and a post-ride score when this run is a past ride."""
    notes: list[str] = []
    if _forecast_log_allowed(result, when_dt):
        segment_id, name = _log_segment(source_label)
        gap = abs((when_dt - datetime.now(timezone.utc)).total_seconds())
        kind = "ride" if gap <= 45 * 60 else "plan"
        row_id = ForecastLog().log_forecast(
            issued_at=datetime.now(timezone.utc),
            target_time=when_dt,
            lat=float(result.lats[0]),
            lon=float(result.lons[0]),
            wind_speed_mps=float(result.mean_wind_speed_mps),
            wind_from_deg=float(result.mean_wind_from_deg),
            gust_speed_mps=result.mean_gust_speed_mps,
            forecast_source=str(result.weather_source),
            segment_id=segment_id,
            segment_name=name,
            target_power_w=float(power_w),
            predicted_time_s=float(result.eta_with_wind_s),
            kind=kind,
        )
        if row_id is not None:
            notes.append(f"Logged the {kind} forecast for {name}.")
    if _record_post_ride(result, power_w):
        notes.append("Recorded a post-ride segment-time score.")
    if notes:
        st.session_state._forecast_log_note = " ".join(notes)


def _log_planned_forecast(
    result,
    *,
    power_w: float,
    wind_speed_mps: float,
    grade_scale: float,
    when_utc: str | None,
    predicted_s: float | None,
    raw_target: str,
    kom_s: float | None,
    source_label: str,
) -> None:
    """Log Plan / Use KOM when the sliders are still the forecast the model used."""
    when_dt = _when_from_iso(when_utc)
    if when_dt is None or predicted_s is None:
        return
    if not _forecast_log_allowed(result, when_dt):
        return
    if not _wind_is_forecast(result, wind_speed_mps, grade_scale):
        return
    kind = "plan"
    if kom_s is not None and raw_target.strip():
        try:
            if abs(parse_target_time(raw_target) - float(kom_s)) < 0.6:
                kind = "kom"
        except ValueError:
            kind = "plan"
    segment_id, name = _log_segment(source_label)
    row_id = ForecastLog().log_forecast(
        issued_at=datetime.now(timezone.utc),
        target_time=when_dt,
        lat=float(result.lats[0]),
        lon=float(result.lons[0]),
        wind_speed_mps=float(result.mean_wind_speed_mps),
        wind_from_deg=float(result.mean_wind_from_deg),
        gust_speed_mps=result.mean_gust_speed_mps,
        forecast_source=str(result.weather_source),
        segment_id=segment_id,
        segment_name=name,
        target_power_w=float(power_w),
        predicted_time_s=float(predicted_s),
        kind=kind,
    )
    if row_id is not None:
        st.session_state._forecast_log_note = f"Logged the {kind} forecast for {name}."


def _match_open_forecasts(config) -> str:
    """Pair unscored forecast rows with Strava rides and write the scores."""
    store = ForecastLog()
    pending = store.pending_forecasts()
    if not pending:
        return "No forecasts waiting."

    def _fetch(client: StravaClient):
        found: dict[int, ActivitySummary] = {}
        for day in target_dates(pending):
            for act in client.list_activities_on_date(day):
                found[act.activity_id] = act
        activities = list(found.values())
        efforts: dict[int, list[SegmentEffortSummary]] = {}
        if any(row.segment_id is not None for row in pending):
            for act in activities:
                try:
                    efforts[act.activity_id] = client.list_activity_segment_efforts(
                        act.activity_id
                    )
                except (StravaAuthError, StravaError):
                    efforts[act.activity_id] = []
        return activities, efforts

    activities, efforts = _run_activity_call(config, _fetch)
    wind_cache: dict[tuple, tuple[float, float] | None] = {}

    def _wind(lat: float, lon: float, when: datetime):
        key = (round(lat, 3), round(lon, 3), when.strftime("%Y-%m-%dT%H"))
        if key not in wind_cache:
            wind_cache[key] = fetch_actual_wind(lat, lon, when)
        return wind_cache[key]

    updated = match_forecasts(pending, activities, efforts, _wind)
    scored = 0
    for row in updated:
        if row.actual_time_s is not None:
            store.update_forecast(row)
            scored += 1
    noun = "forecast" if scored == 1 else "forecasts"
    return f"Scored {scored} {noun}."


def _render_accuracy(config, *, strava_ready: bool) -> None:
    """Forecast scores and post-ride scores, kept on separate lines."""
    try:
        store = ForecastLog()
        pending = store.pending_forecasts()
    except OSError as exc:
        st.caption(f"Accuracy log unavailable: {exc}")
        return
    with st.expander(
        "Accuracy",
        expanded=bool(st.session_state.get("_forecast_log_note")),
    ):
        note = st.session_state.get("_forecast_log_note")
        if note:
            st.caption(note)
        st.caption(
            "Run a route for now or later, or click Plan / Use KOM, to save the "
            "forecast and the predicted time. After the ride, match it to Strava. "
            "Segment efforts use Strava elapsed time. Whole rides use moving time, "
            "the same clock as the post-ride check. Only outdoor power-meter rides "
            "count (Strava device_watts). Virtual, Zwift, trainer, and estimated "
            "power are left out."
        )
        if pending and strava_ready:
            if st.button("Match forecasts to Strava"):
                try:
                    st.caption(_match_open_forecasts(config))
                except (StravaAuthError, StravaError):
                    st.caption("Connect Strava with activity access to match forecasts.")
                except Exception as exc:  # noqa: BLE001
                    st.caption(f"Could not match forecasts: {exc}")
        elif pending:
            st.caption("Connect Strava to match these forecasts to rides.")
        try:
            forecasts = store.list_forecasts()
            posts = store.list_post_rides()
        except OSError as exc:
            st.caption(f"Accuracy log unavailable: {exc}")
            return
        st.caption(summary_line(summarize_forecast(forecasts)))
        st.caption(summary_line(summarize_post_rides(posts)))
        for row in forecasts[:8]:
            st.caption(forecast_row_line(row))
        for row in posts[:5]:
            st.caption(post_ride_row_line(row))
        st.caption(PERSISTENCE_NOTE)


def _render_pacing(
    result,
    *,
    avg_power_w: float,
    wind_speed_mps: float,
    grade_scale: float,
    when_utc: str | None,
    recorded_power: bool = False,
    predicted_s: float | None = None,
    source_label: str = "",
) -> None:
    """Target-time watts and a headwind/tailwind split on the loaded route."""
    st.html('<p class="bw-section-label">Pacing</p>')
    kom = _kom_time_s()
    if kom is None:
        st.caption("Strava didn't return a KOM. Enter a target time.")
    else:
        st.caption(f"Strava KOM {format_duration(kom)}.")

    if "pace_target_time" not in st.session_state:
        st.session_state.pace_target_time = ""

    if kom is not None:
        entry, kom_col, plan_col = st.columns([2.4, 1, 1])
        with kom_col:
            st.button("Use KOM", on_click=_on_use_kom, use_container_width=True)
    else:
        entry, plan_col = st.columns([3, 1])
    with entry:
        st.text_input(
            "Target time",
            placeholder="26:57",
            key="pace_target_time",
            label_visibility="collapsed",
            help="mm:ss or h:mm:ss. Watts are solved for this time on the route and wind.",
        )
    with plan_col:
        plan_click = st.button("Plan", use_container_width=True)

    fp = _pace_fingerprint(result, avg_power_w, wind_speed_mps, grade_scale, when_utc)
    if plan_click or st.session_state.get("_pace_run"):
        st.session_state._pace_run = False
        raw = (st.session_state.get("pace_target_time") or "").strip()
        try:
            with st.spinner("Planning pace…"):
                grades, winds = what_if_course(
                    result,
                    wind_speed_mps=wind_speed_mps,
                    grade_scale=grade_scale,
                )
                params = result.params.with_updates(power_w=float(avg_power_w))
                target = None
                target_error = None
                if raw:
                    try:
                        seconds = parse_target_time(raw)
                    except ValueError as exc:
                        seconds = None
                        target_error = str(exc)
                    if seconds is not None:
                        target = power_for_target_time(
                            seconds,
                            result.distances_m,
                            grades,
                            winds,
                            params,
                        )
                split = wind_pace_split(
                    float(avg_power_w),
                    result.distances_m,
                    grades,
                    winds,
                    params,
                )
                band = None
                if result.weather_ok and when_utc:
                    when_dt = datetime.fromisoformat(when_utc)
                    if when_dt.tzinfo is None:
                        when_dt = when_dt.replace(tzinfo=timezone.utc)
                    band = time_uncertainty_band(
                        result.distances_m,
                        grades,
                        winds,
                        params,
                        when_dt,
                    )
            st.session_state.pace_result = {
                "fp": fp,
                "target": target,
                "target_error": target_error,
                "target_text": raw,
                "kom_time_s": kom,
                "split": split,
                "band": band,
                "weather_ok": bool(result.weather_ok),
                "distance_m": (
                    float(result.distances_m[-1] - result.distances_m[0])
                    if len(result.distances_m) >= 2
                    else 0.0
                ),
                "when_utc": when_utc,
                "avg_power_w": float(avg_power_w),
            }
            _log_planned_forecast(
                result,
                power_w=float(avg_power_w),
                wind_speed_mps=float(wind_speed_mps),
                grade_scale=float(grade_scale),
                when_utc=when_utc,
                predicted_s=predicted_s,
                raw_target=raw,
                kom_s=kom,
                source_label=source_label,
            )
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Pace plan failed: {exc}")

    saved = st.session_state.get("pace_result")
    if not saved:
        note = (
            "Plan solves watts for the target time, then splits that average "
            "between the headwind and the tailwind."
        )
        if recorded_power:
            note += " The split uses the power slider, not the recorded file."
        st.caption(note)
        return

    if saved.get("fp") != fp:
        st.caption("These numbers are from the last Plan click. Click Plan again after you change power, wind, or the start time.")

    target = saved.get("target")
    if saved.get("target_error"):
        st.caption(saved["target_error"])
    elif not saved.get("target_text"):
        st.caption("Enter a target time to solve watts.")
    elif target is not None:
        dist_m = float(saved.get("distance_m") or 0.0)
        mph = (dist_m / target.target_time_s) * 2.236936 if target.target_time_s > 0 else 0.0
        used_kom = (
            saved.get("kom_time_s") is not None
            and abs(float(saved["kom_time_s"]) - target.target_time_s) < 0.6
        )
        kom_note = "Strava KOM. " if used_kom else ""
        st.html(
            f"""
            <div class="bw-pace-row">
              <div class="bw-time-block">
                <p class="bw-time-label">Watts for {format_duration(target.target_time_s)}</p>
                <p class="bw-pace-value">{target.required_power_w:.0f} W</p>
              </div>
              <div class="bw-time-block">
                <p class="bw-time-label">Speed</p>
                <p class="bw-pace-value">{mph:.1f} mph</p>
              </div>
            </div>
            """
        )
        st.caption(f"{kom_note}{target.message}")

    split = saved.get("split")
    if split is not None and split.split_useful:
        head_mi = split.headwind_distance_m / 1609.344
        tail_mi = split.tailwind_distance_m / 1609.344
        st.html(
            f"""
            <div class="bw-pace-row">
              <div class="bw-time-block">
                <p class="bw-time-label">Headwind · {head_mi:.1f} mi</p>
                <p class="bw-pace-value">{split.headwind_power_w:.0f} W</p>
              </div>
              <div class="bw-time-block">
                <p class="bw-time-label">Tailwind · {tail_mi:.1f} mi</p>
                <p class="bw-pace-value">{split.tailwind_power_w:.0f} W</p>
              </div>
              <div class="bw-time-block">
                <p class="bw-time-label">Saved vs steady</p>
                <p class="bw-pace-value">{split.seconds_saved:.0f} s</p>
              </div>
            </div>
            """
        )
        st.caption(
            f"Steady {format_duration(split.steady_time_s)} at {split.steady_speed_mph:.1f} mph · "
            f"paced {format_duration(split.paced_time_s)} at {split.paced_speed_mph:.1f} mph · "
            f"same {split.avg_power_w:.0f} W average along the route."
        )
        if abs(split.meter_avg_power_w - split.avg_power_w) >= 3.0:
            st.caption(
                f"A power meter will read about {split.meter_avg_power_w:.0f} W, "
                "because the harder leg takes longer."
            )
    elif split is not None:
        st.caption(split.message)

    band = saved.get("band")
    if band is not None and split is not None:
        spread_low = band.eta_s - band.eta_low_s
        spread_high = band.eta_high_s - band.eta_s
        lo = max(0.0, split.steady_time_s - spread_low)
        hi = split.steady_time_s + spread_high
        mph = band.along_uncertainty_mps / 0.44704
        st.caption(
            f"Forecast wind is uncertain (±{mph:.1f} mph along the route, "
            f"{band.horizon_days:.1f} days out). "
            f"Steady {saved['avg_power_w']:.0f} W could be {format_duration(lo)}–{format_duration(hi)}."
        )
    elif not saved.get("weather_ok"):
        st.caption("Wind forecast isn't available, so this plan is calm air.")


def _sync_whatif_slider_defaults(result) -> None:
    """Set what-if widget keys before those widgets are drawn (new Run only)."""
    default_power = int(round(result.default_whatif_power_w))
    default_wind = round(result.mean_wind_speed_mph, 1)
    default_mode = (
        _POWER_MODE_RECORDED if result.has_watts else _POWER_MODE_FLAT
    )
    fp = (
        f"{len(result.distances_m)}:{default_power}:{default_wind:.1f}:"
        f"{result.mean_wind_from_deg:.1f}:{result.weather_source}:"
        f"{int(result.has_watts)}"
    )
    if st.session_state.get("_whatif_fp") == fp:
        return
    st.session_state.whatif_power_w = max(80, min(450, default_power))
    st.session_state.whatif_wind_mph = max(0.0, min(40.0, float(default_wind)))
    st.session_state.whatif_grade_pct = 100
    st.session_state.whatif_power_mode = default_mode
    st.session_state._whatif_fp = fp


def main() -> None:
    _init_state()

    top_l, top_r = st.columns([4, 1])
    with top_r:
        dark = st.toggle("Dark", key="bw_dark", help="Dark / light")
    apply_theme(dark)

    with top_l:
        st.html(
            """
            <div class="bw-hero">
              <p class="bw-brand">Bike Wind</p>
              <p class="bw-tagline">Predicting how you will do on a route or segment</p>
            </div>
            """
        )
        link_map, link_exp = st.columns(2)
        with link_map:
            st.page_link("pages/1_Wind_Map.py", label="Wind map", icon="🌬️")
        with link_exp:
            st.page_link("pages/2_Experimental.py", label="Experimental", icon="🧪")

    strava_cfg = load_strava_config()
    _sync_strava_secrets_fingerprint()
    _handle_oauth_callback(strava_cfg)
    strava_ready = _strava_ready(strava_cfg)
    oauth_ok = oauth_configured(
        strava_cfg.client_id, strava_cfg.client_secret, strava_cfg.redirect_uri
    )
    connected = bool(st.session_state.get("strava_oauth_connected"))
    activity_ok = _session_has_activity_read()
    needs_activity_reconnect = connected and not activity_ok
    prefer_hrrr = hrrr_enabled_by_env()

    # Connect with Strava sits above the two closed mode panels.
    conn_l, conn_r = st.columns([3, 2])
    with conn_l:
        if activity_ok:
            st.caption("Strava connected (this browser).")
        elif needs_activity_reconnect:
            st.caption(ACTIVITY_SCOPE_HINT)
        elif oauth_ok:
            st.caption("Connect Strava for personal rides and past-ride checks.")
        else:
            st.caption("Strava connect needs client_id / secret in secrets.")
    with conn_r:
        if activity_ok:
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
            if needs_activity_reconnect:
                st.button(
                    "Disconnect",
                    on_click=_on_disconnect_strava,
                    key="disconnect_strava_rescope",
                    use_container_width=True,
                )
    if st.session_state.get("strava_oauth_error"):
        st.error(st.session_state.strava_oauth_error)

    def _resolve_segment_from_input(*, key_prefix: str) -> None:
        st.caption("Segment link or ID — load a public Strava segment.")
        strava_ref_text = st.text_input(
            "Segment URL or ID",
            placeholder="https://www.strava.com/segments/229781",
            key=f"{key_prefix}_strava_ref",
            label_visibility="collapsed",
        )
        do_resolve = st.button(
            "Load segment",
            disabled=not strava_ready or not strava_ref_text.strip(),
            use_container_width=True,
            key=f"{key_prefix}_load_segment",
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

    def _past_ride_pickers() -> None:
        st.caption("Load a past Strava ride, then optionally isolate a segment effort.")
        if activity_ok:
            _ensure_recent_rides(strava_cfg, limit=10)
        elif needs_activity_reconnect:
            st.caption(ACTIVITY_SCOPE_HINT)

        activities: list[ActivitySummary] = list(
            st.session_state.get("recent_activities") or []
        )
        if activities:
            label_map = {ride_activity_label(a): a for a in activities}
            st.session_state._recent_label_map = label_map
            visible_choice(
                "Recent rides",
                list(label_map.keys()),
                key="recent_ride_pick",
                placeholder=_RIDE_PLACEHOLDER,
                on_change=_on_recent_ride_pick,
            )
        recent_note = hidden_rides_note(
            int(st.session_state.get("recent_hidden_rides") or 0)
        )
        if recent_note and activity_ok:
            st.caption(recent_note)

        ride_day = pick_utc_date(
            "Rides on date (UTC)",
            key="ride_day",
            default=datetime.now(timezone.utc).date(),
            quick=(("Today", 0), ("Yesterday", -1)),
        )
        if st.button(
            "List rides on date",
            disabled=not activity_ok,
            use_container_width=True,
            key="list_rides_on_date",
        ):
            try:
                def _fetch_day(client: StravaClient):
                    start = datetime(
                        ride_day.year, ride_day.month, ride_day.day, tzinfo=timezone.utc
                    )
                    end = start + timedelta(days=1)
                    per_page = 50
                    return collect_outdoor_rides(
                        lambda page: client.list_recent_activities(
                            per_page=per_page,
                            page=page,
                            after=int(start.timestamp()) - 1,
                            before=int(end.timestamp()),
                        ),
                        per_page=per_page,
                        want=None,
                    )

                day_rides, day_hidden = _run_activity_call(strava_cfg, _fetch_day)
                st.session_state.day_activities = day_rides
                st.session_state.day_hidden_rides = day_hidden
                st.session_state.activity_stream_error = (
                    None
                    if day_rides
                    else f"No outdoor rides on {ride_day.isoformat()}."
                )
            except (StravaAuthError, StravaError):
                st.session_state.day_activities = []
                st.session_state.day_hidden_rides = 0
                st.session_state.activity_stream_error = ACTIVITY_SCOPE_HINT
            except Exception:  # noqa: BLE001
                st.session_state.day_activities = []
                st.session_state.day_hidden_rides = 0
                st.session_state.activity_stream_error = ACTIVITY_SCOPE_HINT

        day_activities: list[ActivitySummary] = list(
            st.session_state.get("day_activities") or []
        )
        if day_activities:
            day_map = {ride_activity_label(a): a for a in day_activities}
            st.session_state._day_label_map = day_map
            visible_choice(
                "Rides that day",
                list(day_map.keys()),
                key="day_ride_pick",
                placeholder=_DAY_PLACEHOLDER,
                on_change=_on_day_ride_pick,
            )
        day_note = hidden_rides_note(int(st.session_state.get("day_hidden_rides") or 0))
        if day_note:
            st.caption(day_note)

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
            if st.session_state.get("_efforts_activity_id") != act.activity_id:
                _fetch_activity_efforts(strava_cfg, act.activity_id)
            efforts: list[SegmentEffortSummary] = list(
                st.session_state.get("activity_efforts") or []
            )
            if efforts:
                effort_map = {e.label(): e for e in efforts}
                st.session_state._effort_label_map = effort_map
                visible_choice(
                    "Segment on this ride",
                    list(effort_map.keys()),
                    key="effort_ride_pick",
                    placeholder=_EFFORT_PLACEHOLDER,
                    on_change=_on_effort_pick,
                )
            elif st.session_state.get("_efforts_activity_id") == act.activity_id:
                st.caption("No segment efforts on this ride.")

        if st.session_state.get("activity_stream_error"):
            st.error(st.session_state.activity_stream_error)

    # Defaults when neither panel runs this turn (results reuse last_run).
    uploaded = None
    use_sample = True
    when = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    power_w = float(st.session_state.power_w)
    surface = resolve_surface(st.session_state.surface_key)
    aero = resolve_aero(
        st.session_state.position_key,
        full_aero_kit=bool(st.session_state.get("full_aero_kit", False)),
        non_aero=bool(st.session_state.get("non_aero", False)),
        bike_mass_kg=float(st.session_state.bike_kg),
    )
    run = False
    pred_mode = st.session_state.get("pred_mode") or _MODE_FUTURE

    with st.expander(_MODE_PAST, expanded=False):
        st.caption(
            "Physics check on a recorded ride or segment effort: archive wind at "
            "the ride start, recorded power when available, then predicted vs actual "
            "time with percent error. No next-week forecast here."
        )
        _past_ride_pickers()
        with st.expander("Or load a segment / GPX", expanded=False):
            _resolve_segment_from_input(key_prefix="past")
            st.caption("GPX — use your own file instead of a segment or ride.")
            past_uploaded = st.file_uploader(
                "GPX", type=["gpx"], label_visibility="collapsed", key="gpx_past"
            )
            past_sample = st.checkbox(
                "Demo route",
                value=(
                    past_uploaded is None
                    and st.session_state.get("route_source")
                    not in ("segment", "activity", "effort")
                ),
                key="demo_past",
            )
        past_power, past_surface, past_aero = _panel_rider_controls(key_prefix="past")
        run_past = st.button(
            "Run past check", type="primary", use_container_width=True, key="run_past"
        )
        if run_past:
            run = True
            pred_mode = _MODE_PAST
            st.session_state.pred_mode = pred_mode
            uploaded = past_uploaded
            use_sample = bool(past_sample)
            power_w = past_power
            surface = past_surface
            aero = past_aero
            # Timed rides use start time for archive wind inside predict_speeds.
            when = datetime.now(timezone.utc).replace(
                minute=0, second=0, microsecond=0
            )

    with st.expander(_MODE_FUTURE, expanded=False):
        st.caption(
            "Predict a segment or GPX under the next week's hourly forecast. "
            "Same physics engine — power, wind, and grade."
        )
        _resolve_segment_from_input(key_prefix="future")
        st.caption("GPX — use your own file instead of a segment.")
        future_uploaded = st.file_uploader(
            "GPX", type=["gpx"], label_visibility="collapsed", key="gpx_future"
        )
        future_sample = st.checkbox(
            "Demo route",
            value=(
                future_uploaded is None
                and st.session_state.get("route_source")
                not in ("segment", "activity", "effort")
            ),
            key="demo_future",
        )
        future_when = pick_utc_when(
            "Start (UTC) — next week, hourly",
            key="when_utc_future",
            default=datetime.now(timezone.utc).replace(
                minute=0, second=0, microsecond=0
            )
            + timedelta(hours=3),
            quick=(
                ("Today", 0),
                ("Tomorrow", 1),
                ("+3d", 3),
                ("+7d", 7),
            ),
        )
        st.caption("Hourly Open-Meteo forecast when available.")
        future_power, future_surface, future_aero = _panel_rider_controls(
            key_prefix="future"
        )
        run_future = st.button(
            "Run prediction",
            type="primary",
            use_container_width=True,
            key="run_future",
        )
        if run_future:
            run = True
            pred_mode = _MODE_FUTURE
            st.session_state.pred_mode = pred_mode
            uploaded = future_uploaded
            use_sample = bool(future_sample)
            power_w = future_power
            surface = future_surface
            aero = future_aero
            when = future_when

    past_mode = _is_past_mode(pred_mode)

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
    if (
        last_run
        and not run
        and last_run.get("pred_mode") not in (None, pred_mode)
        and not (
            _is_past_mode(last_run.get("pred_mode")) and _is_past_mode(pred_mode)
        )
        and not (
            last_run.get("pred_mode") in _FUTURE_MODE_ALIASES
            and pred_mode in _FUTURE_MODE_ALIASES
        )
    ):
        # Keep the other mode's snapshot in session; don't show it here.
        last_run = None
    if not run and last_run:
        pred_mode = last_run.get("pred_mode") or pred_mode
        past_mode = _is_past_mode(pred_mode)
    if not run and not last_run:
        return

    # Past panel owns ride/effort tracks; future sticks to segment / GPX / demo.
    use_effort = (
        past_mode
        and st.session_state.get("route_source") == "effort"
        and bool(st.session_state.get("selected_effort"))
    )
    use_activity = (
        past_mode
        and st.session_state.get("route_source") == "activity"
        and bool(st.session_state.get("selected_activity"))
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
                # Refresh meta for KOM (cache may omit xoms).
                try:
                    meta_live = _run_strava_call(
                        strava_cfg,
                        lambda c: c.get_segment_meta(
                            int(st.session_state.strava_segment_id), use_cache=False
                        ),
                    )
                    kom_time_s = meta_live.kom_time_s
                except Exception:  # noqa: BLE001
                    kom_time_s = getattr(meta, "kom_time_s", None)
                st.session_state.strava_meta = {
                    "segment_id": meta.segment_id,
                    "name": meta.name,
                    "distance_m": meta.distance_m,
                    "average_grade": meta.average_grade,
                    "from_cache": meta.from_cache,
                    "kom_time_s": kom_time_s,
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
        clothing = resolve_clothing(st.session_state.clothing_key)
        from bike_wind.calibration.cda_defaults import YAW_CDA_ENABLED

        params = RiderBikeParams(
            rider_mass_kg=float(st.session_state.rider_kg),
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
        _capture_run_snapshot(
            result,
            when_dt=when_dt,
            power_w=float(power_w),
            source_label=source_label,
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
            "when_utc": when_dt.isoformat(),
            "pred_mode": pred_mode,
        }
        source_label_out = source_label
        route_km = route.distances_m[-1] / 1000.0
        position_key_out = aero.position_key
        cda_out = aero.cd * aero.area_m2
        surface_label_out = surface.label
        crr_note_out = surface.estimate_note
        when_utc_out = when_dt.isoformat()
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
        when_utc_out = last_run.get("when_utc")

    with st.container(border=True):
        st.html('<p class="bw-section-label">Results</p>')
        mode_note = "Past check" if past_mode else "Future forecast"
        st.caption(
            f"{mode_note} · {source_label_out} · {route_km:.1f} km · "
            f"{POSITION_PRESETS[position_key_out].label} · Cd·A={cda_out:.3f} · "
            f"Crr={result.params.crr:.4f}"
        )

        # What-if defaults before widgets exist on this run.
        _sync_whatif_slider_defaults(result)
        wind_dir = compass_label(result.mean_wind_from_deg) if result.weather_ok else "—"

        if result.has_watts:
            power_mode = st.radio(
                "Power source",
                options=[_POWER_MODE_RECORDED, _POWER_MODE_FLAT],
                horizontal=True,
                key="whatif_power_mode",
                help=(
                    "Default: solve each point from the recorded effort power. "
                    "Switch to flat slider watts for what-if."
                ),
            )
        else:
            power_mode = _POWER_MODE_FLAT
            st.caption("No recorded power — model uses flat slider watts.")

        s1, s2, s3 = st.columns(3)
        with s1:
            whatif_power = st.slider(
                "Power (W)",
                min_value=80,
                max_value=450,
                step=5,
                key="whatif_power_w",
                help=(
                    "Flat watts for what-if mode. Ignored while Power source is "
                    "Recorded effort (mean shown as caption only)."
                ),
            )
        with s2:
            whatif_wind_mph = st.slider(
                f"Wind (mph · {wind_dir})",
                min_value=0.0,
                max_value=40.0,
                step=0.5,
                key="whatif_wind_mph",
                help="Scales forecast wind speed; direction stays the same.",
            )
        with s3:
            whatif_grade_pct = st.slider(
                "Climbs (% of route)",
                min_value=0,
                max_value=200,
                step=5,
                key="whatif_grade_pct",
                help="Scales this route's grades (100% = as recorded).",
            )

        use_recorded = power_mode == _POWER_MODE_RECORDED and result.has_watts
        whatif = apply_what_if(
            result,
            power_w=float(whatif_power),
            wind_speed_mps=float(whatif_wind_mph) / 2.236936,
            grade_scale=float(whatif_grade_pct) / 100.0,
            use_recorded_power=use_recorded,
        )

        clock = None
        if result.times_utc and result.actual_speeds_mps and result.has_actual_speeds:
            clock = clock_speed_chart(
                result.times_utc,
                result.distances_m,
                result.actual_speeds_mps,
                whatif.speeds_with_wind_mps,
            )
        actual_s = result.actual_elapsed_s
        time_grade = (
            score_segment_time(float(whatif.eta_with_wind_s), float(actual_s))
            if past_mode and actual_s is not None
            else None
        )
        if past_mode:
            actual_html = (
                f"""
              <div class="bw-time-block">
                <p class="bw-time-label">Actual</p>
                <p class="bw-time-value">{format_duration(actual_s)}</p>
              </div>
            """
                if actual_s is not None
                else """
              <div class="bw-time-block">
                <p class="bw-time-label">Actual</p>
                <p class="bw-time-value">—</p>
              </div>
            """
            )
            error_html = (
                f"""
              <div class="bw-time-block">
                <p class="bw-time-label">Error</p>
                <p class="bw-time-value">{_format_time_error_pct(time_grade.error_pct)}</p>
              </div>
            """
                if time_grade is not None
                else """
              <div class="bw-time-block">
                <p class="bw-time-label">Error</p>
                <p class="bw-time-value">—</p>
              </div>
            """
            )
            st.html(
                f"""
            <div class="bw-time-row">
              <div class="bw-time-block">
                <p class="bw-time-label">Predicted</p>
                <p class="bw-time-value">{format_duration(whatif.eta_with_wind_s)}</p>
              </div>
              {actual_html}
              {error_html}
            </div>
            """
            )
            if time_grade is not None:
                gate = (
                    "within ±5%"
                    if time_grade.passed
                    else "outside ±5% gate"
                )
                st.caption(
                    f"Predicted − actual = {time_grade.error_s:+.0f}s "
                    f"({_format_time_error_pct(time_grade.error_pct)}, {gate}). "
                    "Isolated to the selected ride or segment effort."
                )
            elif st.session_state.get("route_source") in ("activity", "effort"):
                st.caption(
                    "No usable actual time on this selection — pick a segment effort "
                    "or a ride with moving time."
                )
        else:
            st.html(
                f"""
            <div class="bw-time-row">
              <div class="bw-time-block">
                <p class="bw-time-label">Predicted</p>
                <p class="bw-time-value">{format_duration(whatif.eta_with_wind_s)}</p>
              </div>
            </div>
            """
            )
        if whatif.used_recorded_power:
            power_caption = (
                f"Power source: recorded effort (mean {whatif.power_w:.0f} W)"
            )
            if result.power_window_s:
                power_caption += f" · {result.power_window_s:.0f}s rolling mean"
        else:
            power_caption = f"Power source: flat slider ({whatif.power_w:.0f} W)"
        st.caption(
            f"{power_caption} · "
            f"{format_wind_speed_dir(whatif.wind_speed_mps, whatif.wind_from_deg)} · "
            f"climbs {whatif_grade_pct:.0f}% · "
            f"calm {format_duration(whatif.eta_no_wind_s)}"
        )
        if clock is not None and clock.stops:
            st.caption(clock.summary())

        if not past_mode:
            _render_pacing(
                result,
                avg_power_w=float(whatif_power),
                wind_speed_mps=float(whatif_wind_mph) / 2.236936,
                grade_scale=float(whatif_grade_pct) / 100.0,
                when_utc=when_utc_out,
                recorded_power=bool(use_recorded),
                predicted_s=float(whatif.eta_with_wind_s),
                source_label=source_label_out,
            )

        head_l, head_mid, head_r = st.columns([4, 1, 1])
        with head_l:
            st.caption(
                "Power file used · predicted vs actual speed"
                if past_mode
                else "Predicted speed along the route (mph)"
            )
        with head_mid:
            with st.popover("Equations"):
                try:
                    explain = explain_prediction(
                        result,
                        whatif,
                        grade_scale=float(whatif_grade_pct) / 100.0,
                    )
                    st.caption(
                        f"Same solver as the chart · sample at "
                        f"{explain.distance_mi:.2f} mi on this effort"
                    )
                    for line in explain.lines:
                        st.markdown(f"**{line.title}**")
                        st.code(line.latex_or_text, language=None)
                        st.caption(line.plain)
                except Exception as exc:  # noqa: BLE001
                    st.caption(f"Equations unavailable: {exc}")
        with head_r:
            with st.popover("Info"):
                st.text(
                    _validation_help_text(
                        result,
                        position_label=POSITION_PRESETS[position_key_out].label,
                        cda=cda_out,
                        surface_label=surface_label_out,
                        crr_note=crr_note_out,
                        power_mode_label=power_mode,
                        whatif_power_w=float(whatif_power),
                        whatif_wind_mph=float(whatif_wind_mph),
                        whatif_wind_from_deg=whatif.wind_from_deg,
                        whatif_grade_pct=float(whatif_grade_pct),
                    )
                )

        # --- Graph 1: power vs distance (watts used by the model) ---
        pal = chart_palette(dark)
        power_color = pal.power
        fig_p, ax_p = plt.subplots(figsize=(9, 2.8))
        bg_p, fg_p, _accent_p = _style_power_axes(ax_p, dark=dark, fig=fig_p)
        plot_powers = whatif.powers_w if whatif.powers_w is not None else result.watts
        if plot_powers is not None and result.has_watts:
            ax_p.plot(
                result.distances_mi,
                plot_powers,
                label="Power",
                color=power_color,
                linewidth=1.6,
            )
        elif not use_recorded:
            ax_p.axhline(
                whatif.power_w,
                color=power_color,
                linewidth=1.8,
                label=f"Flat {whatif.power_w:.0f} W",
            )
        else:
            ax_p.text(
                0.5,
                0.5,
                "No recorded power",
                transform=ax_p.transAxes,
                ha="center",
                va="center",
                color=fg_p,
            )
        if result.distances_mi:
            ax_p.set_xlim(result.distances_mi[0], result.distances_mi[-1])
        leg_p = ax_p.legend(
            facecolor=bg_p, edgecolor=pal.spine
        )
        for text in leg_p.get_texts():
            text.set_color(fg_p)
        fig_p.tight_layout()
        st.pyplot(fig_p)
        plt.close(fig_p)

        # --- Graph 2: predicted vs actual speed. Time shows pauses; distance does not. ---
        use_time = False
        if clock is not None:
            default_axis = "Time" if clock.stops else "Distance"
            axis = st.radio(
                "Speed chart",
                ["Time", "Distance"],
                index=0 if default_axis == "Time" else 1,
                horizontal=True,
                key="speed_axis",
                help=(
                    "Time is elapsed clock time, so a stop is a flat 0 mph for as "
                    "long as it lasted. Distance skips that pause."
                ),
            )
            use_time = axis == "Time"
        if use_time and clock is not None:
            fig2 = figure_clock_speed(clock, dark=dark)
            st.pyplot(fig2)
            plt.close(fig2)
            st.caption(clock.caption)
        else:
            fig2, ax2 = plt.subplots(figsize=(9, 3.2))
            bg2, fg2, accent2 = _style_speed_axes(ax2, dark=dark, fig=fig2)
            ax2.plot(
                result.distances_mi,
                whatif.speeds_with_wind_mph,
                label="Predicted",
                color=accent2,
                linewidth=2,
            )
            if result.has_actual_speeds and result.actual_speeds_mph is not None:
                ax2.plot(
                    result.distances_mi,
                    result.actual_speeds_mph,
                    label="Actual",
                    color=pal.actual,
                    linewidth=1.8,
                    alpha=0.95,
                )
            else:
                st.caption(
                    "Actual speed needs a GPX or Strava velocity_smooth / time+distance — "
                    "showing predicted only."
                )
            if result.distances_mi:
                ax2.set_xlim(result.distances_mi[0], result.distances_mi[-1])
            leg2 = ax2.legend(
                facecolor=bg2,
                edgecolor=pal.spine,
            )
            for text in leg2.get_texts():
                text.set_color(fg2)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        # Wind the rider feels along the route. Gusts live here, not on the speed chart.
        wind_chart = wind_along_chart(result, wind_speed_mps=whatif.wind_speed_mps)
        if wind_chart is None:
            st.caption("No forecast wind to graph for this run.")
        else:
            fig_w = figure_wind_along(wind_chart, dark=dark)
            st.pyplot(fig_w)
            plt.close(fig_w)
            st.caption(wind_chart.caption)

        _render_accuracy(strava_cfg, strava_ready=strava_ready)

if __name__ == "__main__":
    main()
