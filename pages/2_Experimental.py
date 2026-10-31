"""Experimental setups next to the validated GustyPower prediction.

The prediction page is unchanged. This page re-solves the same route, weather,
and power with optional surface-aware rolling and an unvalidated Cd·A / Crr
override, and shows the difference.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

from bike_wind.analysis import format_duration
from bike_wind.clothing import DEFAULT_CLOTHING_KEY, resolve_clothing
from bike_wind.experimental import (
    AERO_WHEELS_CDA_MULTIPLIER,
    AERO_WHEELS_CRR,
    AERO_WHEELS_PRESET_NAME,
    CDA_MULTIPLIER_DEFAULT,
    CDA_MULTIPLIER_MAX,
    CDA_MULTIPLIER_MIN,
    GRAVEL_SECTION_CRR,
    OSM_TIMEOUT_S,
    PRODUCTION_CRR,
    ROAD_SECTION_CRR,
    average_speed_mph,
    build_production_params,
    compare_with_production,
    crr_for_sections,
    format_surface_shares,
    sections_for_route,
    surface_rule_blurb,
)
from bike_wind.gpx.parse import RouteStreams, parse_gpx
from bike_wind.predict import downsample_route, predict_speeds
from bike_wind.rider import (
    DEFAULT_POSITION_KEY,
    DEFAULT_RIDER_KG,
    DEFAULT_SURFACE_KEY,
    POSITION_PRESETS,
    resolve_surface,
    smart_default_height_cm,
    standing_height_to_wind_z_m,
)
from bike_wind.strava import StravaAuthError, StravaClient, StravaError, load_strava_config
from bike_wind.strava.oauth import scope_has_activity_read
from bike_wind.weather import hrrr_enabled_by_env

st.set_page_config(
    page_title="GustyPower experimental",
    layout="centered",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parents[1]
PUMPHOUSE = ROOT / "fixtures" / "segment_41392335.json"
PUMPHOUSE_ID = 41392335

PRESET_CUSTOM = "Custom (unvalidated)"
ROUTE_PUMPHOUSE = "Round trip to pumphouse"
ROUTE_MAIN = "Route loaded on the prediction page"
ROUTE_SEGMENT = "Strava segment"
ROUTE_GPX = "GPX upload"


def _init_widgets() -> None:
    if "exp_power_w" not in st.session_state:
        st.session_state.exp_power_w = int(st.session_state.get("power_w", 220))
    if "exp_rider_kg" not in st.session_state:
        st.session_state.exp_rider_kg = float(st.session_state.get("rider_kg", DEFAULT_RIDER_KG))
    if "exp_cda_mult" not in st.session_state:
        st.session_state.exp_cda_mult = CDA_MULTIPLIER_DEFAULT
    if "exp_crr" not in st.session_state:
        surface_key = st.session_state.get("surface_key", DEFAULT_SURFACE_KEY)
        st.session_state.exp_crr = float(resolve_surface(surface_key).crr)
    if "exp_preset" not in st.session_state:
        st.session_state.exp_preset = PRESET_CUSTOM
    if "exp_when" not in st.session_state:
        st.session_state.exp_when = datetime.now(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        ) + timedelta(hours=3)
    if "exp_surface_on" not in st.session_state:
        st.session_state.exp_surface_on = False
    if "exp_manual_on" not in st.session_state:
        st.session_state.exp_manual_on = False


def _on_preset() -> None:
    choice = st.session_state.get("exp_preset")
    if choice == AERO_WHEELS_PRESET_NAME:
        st.session_state.exp_cda_mult = AERO_WHEELS_CDA_MULTIPLIER
        st.session_state.exp_crr = AERO_WHEELS_CRR
        return
    surface_key = st.session_state.get("surface_key", DEFAULT_SURFACE_KEY)
    st.session_state.exp_cda_mult = CDA_MULTIPLIER_DEFAULT
    st.session_state.exp_crr = float(resolve_surface(surface_key).crr)


def _strava_client() -> StravaClient:
    """Same token order as the prediction page: connected session, else secrets."""
    cfg = load_strava_config()

    def on_refreshed(tokens) -> None:
        if st.session_state.get("strava_oauth_connected"):
            st.session_state.strava_access_token = tokens.access_token
            if tokens.refresh_token:
                st.session_state.strava_refresh_token = tokens.refresh_token
            return
        st.session_state.strava_refreshed_access_token = tokens.access_token
        if tokens.refresh_token:
            st.session_state.strava_refreshed_refresh_token = tokens.refresh_token

    if st.session_state.get("strava_oauth_connected") and st.session_state.get("strava_access_token"):
        access = st.session_state.strava_access_token
        refresh = st.session_state.get("strava_refresh_token") or cfg.refresh_token
    else:
        access = st.session_state.get("strava_refreshed_access_token") or cfg.access_token
        refresh = cfg.refresh_token or st.session_state.get("strava_refreshed_refresh_token")
    return StravaClient(
        cfg,
        access_token=access,
        refresh_token=refresh,
        on_tokens_refreshed=on_refreshed,
    )


def _activity_client() -> StravaClient | None:
    if not (
        st.session_state.get("strava_oauth_connected")
        and st.session_state.get("strava_access_token")
        and scope_has_activity_read(st.session_state.get("strava_token_scope"))
    ):
        return None
    return _strava_client()


@st.cache_data(ttl=14 * 24 * 3600, show_spinner=False)
def _pumphouse_route() -> RouteStreams:
    seg = json.loads(PUMPHOUSE.read_text())
    return RouteStreams(
        lats=[float(p[0]) for p in seg["latlng"]],
        lons=[float(p[1]) for p in seg["latlng"]],
        elevs_m=[float(x) for x in seg["altitude"]],
        distances_m=[float(x) for x in seg["distance"]],
    )


def _load_segment(segment_id: int) -> tuple[str, RouteStreams]:
    cache: dict = st.session_state.setdefault("exp_segment_cache", {})
    if segment_id in cache:
        return cache[segment_id]
    client = _strava_client()
    meta, route = client.load_segment_route(int(segment_id))
    loaded = (meta.name, route)
    cache[segment_id] = loaded
    return loaded


def _route_from_main_page() -> tuple[str, RouteStreams, int | None]:
    """Reuse the route the prediction page already selected."""
    source = st.session_state.get("route_source")
    if source == "effort" and st.session_state.get("effort_route") is not None:
        effort = st.session_state.get("selected_effort")
        name = getattr(effort, "name", None) or "Segment effort"
        sid = getattr(effort, "segment_id", None)
        return f"Prediction page · {name}", st.session_state.effort_route, sid
    if source == "activity" and st.session_state.get("activity_route") is not None:
        act = st.session_state.get("selected_activity")
        name = getattr(act, "name", None) or "Strava ride"
        return f"Prediction page · {name}", st.session_state.activity_route, None
    sid = st.session_state.get("strava_segment_id")
    if source == "segment" and sid:
        name, route = _load_segment(int(sid))
        return f"Prediction page · {name}", route, int(sid)
    raise RuntimeError(
        "Nothing is loaded on the prediction page yet. Load a segment or ride there, "
        "or pick a Strava segment / GPX here."
    )


def _resolve_route(choice: str, segment_text: str, upload) -> tuple[str, RouteStreams, int | None]:
    if choice == ROUTE_PUMPHOUSE:
        return "Round trip to pumphouse", _pumphouse_route(), PUMPHOUSE_ID
    if choice == ROUTE_MAIN:
        return _route_from_main_page()
    if choice == ROUTE_SEGMENT:
        text = (segment_text or "").strip()
        if not text:
            raise RuntimeError("Paste a Strava segment URL or ID.")
        client = _strava_client()
        sid, _ref = client.resolve_ref(text)
        name, route = _load_segment(int(sid))
        return name, route, int(sid)
    if upload is None:
        raise RuntimeError("Upload a GPX file.")
    route = parse_gpx(upload.getvalue())
    return upload.name or "GPX", route, None
    raise RuntimeError(f"Unknown route choice: {choice}")


def _latest_effort_line(segment_id: int | None) -> str | None:
    if segment_id is None:
        return None
    client = _activity_client()
    if client is None:
        return None
    cache: dict = st.session_state.setdefault("exp_effort_note", {})
    if segment_id in cache:
        return cache[segment_id] or None
    try:
        efforts = client.list_my_segment_efforts(int(segment_id), per_page=1)
    except (StravaAuthError, StravaError):
        return None
    except Exception:  # noqa: BLE001
        return None
    if not efforts:
        cache[segment_id] = ""
        return None
    effort = efforts[0]
    when = (effort.start_date or "")[:10]
    watts = f", avg {effort.average_watts:.0f} W" if effort.average_watts else ""
    note = (
        f"Your latest Strava effort on this segment: "
        f"{format_duration(float(effort.elapsed_time_s or effort.moving_time_s))} "
        f"on {when or 'an unknown date'}{watts}. "
        "That is the recorded clock, not a model run."
    )
    cache[segment_id] = note
    return note


def _production_caption() -> str:
    position_key = st.session_state.get("position_key", DEFAULT_POSITION_KEY)
    surface_key = st.session_state.get("surface_key", DEFAULT_SURFACE_KEY)
    clothing_key = st.session_state.get("clothing_key", DEFAULT_CLOTHING_KEY)
    position = POSITION_PRESETS.get(position_key, POSITION_PRESETS[DEFAULT_POSITION_KEY])
    surface = resolve_surface(surface_key)
    clothing = resolve_clothing(clothing_key)
    return (
        f"Production baseline from the prediction page: {position.label}, "
        f"Crr {surface.crr:.4f}, clothing ×{clothing.cda_multiplier:.2f}. "
        "Change those there; this page only adds experimental setups."
    )


def _fetch_fingerprint(choice, segment_text, upload, power_w, rider_kg, when) -> tuple:
    when_dt = when if isinstance(when, datetime) else datetime.combine(when, datetime.min.time())
    if when_dt.tzinfo is None:
        when_dt = when_dt.replace(tzinfo=timezone.utc)
    return (
        choice,
        (segment_text or "").strip(),
        getattr(upload, "name", None),
        round(float(power_w), 1),
        round(float(rider_kg), 2),
        when_dt.isoformat(),
    )


def _section_flags(sections) -> list[bool]:
    """Road/gravel control for each stretch. Widget keys stick for the session."""
    st.subheader("Sections")
    st.caption(
        f"Road = Crr {ROAD_SECTION_CRR:.4f} and the production Cd·A. "
        f"Gravel = Crr {GRAVEL_SECTION_CRR:.3f}. "
        "OpenStreetMap unpaved starts on gravel. Paved and unmapped start on road. "
        "The pumphouse canal bank is unmapped, so those rows start as road until you flip them. "
        "The By section column updates when you flip a row, without fetching weather again."
    )
    flags: list[bool] = []
    for sec in sections:
        key = f"exp_sec_{sec.section_id}"
        if key not in st.session_state:
            st.session_state[key] = "Gravel" if sec.default_gravel else "Road"
        label_col, toggle_col = st.columns([3, 2])
        with label_col:
            st.markdown(f"**{sec.title}**")
            st.caption(sec.detail)
        with toggle_col:
            choice = st.radio(
                sec.title,
                ["Road", "Gravel"],
                horizontal=True,
                key=key,
                label_visibility="collapsed",
            )
        flags.append(choice == "Gravel")
    return flags


def _render_results(payload: dict) -> None:
    columns = payload["columns"]
    actual_s = payload.get("actual_s")
    n = len(columns) + (1 if actual_s else 0)
    slots = st.columns(n)
    i = 0
    if actual_s:
        with slots[0]:
            st.metric("Actual elapsed", format_duration(float(actual_s)))
            st.caption(f"{payload['actual_mph']:.1f} mph · recorded, not a model")
        i = 1
    for col, slot in zip(columns, slots[i:]):
        with slot:
            if col.label == "Production":
                st.metric("Production", format_duration(col.eta_s))
                st.caption(f"{col.avg_speed_mph:.1f} mph")
            else:
                st.metric(
                    col.label,
                    format_duration(col.eta_s),
                    delta=f"{col.delta_s:+.0f} s vs production",
                    delta_color="inverse",
                )
                st.caption(
                    f"{col.avg_speed_mph:.1f} mph ({col.delta_mph:+.2f} mph vs production)"
                )
            if col.fell_back:
                st.warning("Fell back to the production model.")
    if payload.get("surface_warning"):
        st.warning(payload["surface_warning"])
    if payload.get("surface_shares"):
        st.info(payload["surface_shares"])
    for col in columns:
        if col.note:
            st.caption(f"{col.label}: {col.note}")
    st.caption(payload.get("detail") or "")
    if payload.get("effort_line") and not actual_s:
        st.caption(payload["effort_line"])


def main() -> None:
    _init_widgets()
    st.title("Experimental")
    st.warning(
        "These setups are experimental and are not part of the validated production model. "
        "The prediction page is unchanged."
    )
    st.caption(
        "Same route, weather, and power as a normal prediction. "
        "Production stays in the first column. Each option you turn on adds a column "
        "with time, average speed, and the difference versus production."
    )

    with st.container(border=True):
        st.subheader("Route, date, power")
        choice = st.radio(
            "Route",
            [ROUTE_PUMPHOUSE, ROUTE_MAIN, ROUTE_SEGMENT, ROUTE_GPX],
            key="exp_route_choice",
        )
        segment_text = ""
        upload = None
        if choice == ROUTE_SEGMENT:
            segment_text = st.text_input(
                "Segment URL or ID",
                placeholder="https://www.strava.com/segments/229781",
                key="exp_segment_ref",
            )
        elif choice == ROUTE_GPX:
            upload = st.file_uploader("GPX", type=["gpx"], key="exp_gpx")
        elif choice == ROUTE_MAIN:
            st.caption(
                "Uses the segment, ride, or segment-on-a-ride you last selected "
                "on the prediction page, including recorded power and time when Strava sent them."
            )
        when = st.datetime_input(
            "When (UTC)",
            key="exp_when",
            help="Forecast time. A ride that already has timestamps uses its start time instead, same as the prediction page.",
        )
        power_w = st.slider("Power (W)", 80, 450, step=5, key="exp_power_w")
        rider_kg = st.number_input(
            "Rider kg",
            min_value=40.0,
            max_value=120.0,
            step=0.5,
            key="exp_rider_kg",
        )
        st.caption(_production_caption())

    with st.container(border=True):
        st.subheader("Experimental setups")
        surface_on = st.checkbox(
            "Surface-aware rolling (road vs gravel by OpenStreetMap surface)",
            key="exp_surface_on",
            help=(
                f"Cached lookup, {OSM_TIMEOUT_S:.0f}s timeout. "
                "If OpenStreetMap does not answer, this column falls back to production "
                "and the page says so."
            ),
        )
        if surface_on:
            st.caption(surface_rule_blurb())
            st.caption(
                "Part of St. Lucie County is bundled (including Round trip to pumphouse) "
                "and is used before the network. That segment is mostly unmapped canal bank, "
                f"so it stays on the production setup. Other areas try OpenStreetMap live "
                f"(cached, {OSM_TIMEOUT_S:.0f}s timeout) and fall back to production if it fails."
            )
        manual_on = st.checkbox(
            "Manual bike setup (unvalidated)",
            key="exp_manual_on",
            help="Cd·A multiplier and Crr you pick. Not fitted on the hold-out rides.",
        )
        if manual_on:
            st.selectbox(
                "Preset",
                [PRESET_CUSTOM, AERO_WHEELS_PRESET_NAME],
                key="exp_preset",
                on_change=_on_preset,
                help=(
                    "Aero wheels + road tires is an unvalidated guess "
                    f"(Cd·A ×{AERO_WHEELS_CDA_MULTIPLIER:.2f}, Crr {AERO_WHEELS_CRR:.4f}) "
                    "for 55 mm wheels and 28 mm tires on smooth tarmac. "
                    "It is not a validated setup."
                ),
            )
            c1, c2 = st.columns(2)
            with c1:
                st.slider(
                    "Cd·A multiplier (unvalidated)",
                    min_value=CDA_MULTIPLIER_MIN,
                    max_value=CDA_MULTIPLIER_MAX,
                    step=0.01,
                    key="exp_cda_mult",
                    help=f"1.00 leaves production Cd·A. Default {CDA_MULTIPLIER_DEFAULT:.2f}.",
                )
            with c2:
                st.number_input(
                    "Crr (unvalidated)",
                    min_value=0.0020,
                    max_value=0.0200,
                    step=0.0001,
                    format="%.4f",
                    key="exp_crr",
                    help=f"Production gravel Crr is {PRODUCTION_CRR:.4f}.",
                )
            st.caption(
                "Unvalidated. The preset only fills these two inputs; "
                "the numbers above are what the comparison uses."
            )

    compare = st.button("Compare", type="primary", use_container_width=True)
    fetch_fp = _fetch_fingerprint(choice, segment_text, upload, power_w, rider_kg, when)

    if compare:
        try:
            label, route, segment_id = _resolve_route(choice, segment_text, upload)
            route = downsample_route(route, max_points=300)
            position_key = st.session_state.get("position_key", DEFAULT_POSITION_KEY)
            surface_key = st.session_state.get("surface_key", DEFAULT_SURFACE_KEY)
            clothing_key = st.session_state.get("clothing_key", DEFAULT_CLOTHING_KEY)
            bike_kg = st.session_state.get("bike_kg")
            params = build_production_params(
                rider_kg=float(rider_kg),
                power_w=float(power_w),
                position_key=position_key,
                surface_key=surface_key,
                clothing_key=clothing_key,
                bike_mass_kg=float(bike_kg) if bike_kg is not None else None,
                full_aero_kit=bool(st.session_state.get("full_aero_kit", False)),
                non_aero=bool(st.session_state.get("non_aero", False)),
            )
            height_cm = float(
                st.session_state.get("height_cm", smart_default_height_cm(float(rider_kg)))
            )
            when_dt = when if isinstance(when, datetime) else datetime.combine(when, datetime.min.time())
            if when_dt.tzinfo is None:
                when_dt = when_dt.replace(tzinfo=timezone.utc)
            with st.spinner("Fetching wind and solving the production model…"):
                result = predict_speeds(
                    route,
                    params,
                    when_dt,
                    wind_height_m=standing_height_to_wind_z_m(height_cm),
                    prefer_hrrr=hrrr_enabled_by_env(),
                )
            with st.spinner(
                f"Splitting the route into sections (OpenStreetMap, up to {OSM_TIMEOUT_S:.0f}s, cached)…"
            ):
                sections, surface_report = sections_for_route(
                    list(route.lats),
                    list(route.lons),
                    list(route.distances_m),
                    timeout_s=OSM_TIMEOUT_S,
                )
            effort_line = None if result.actual_elapsed_s else _latest_effort_line(segment_id)
            st.session_state.exp_solved = {
                "fp": fetch_fp,
                "result": result,
                "label": label,
                "sections": sections,
                "surface_report": surface_report,
                "effort_line": effort_line,
            }
        except (StravaAuthError, StravaError) as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not compare: {exc}")

    solved = st.session_state.get("exp_solved")
    live = solved is not None and solved.get("fp") == fetch_fp
    if not solved:
        st.caption(
            "Press Compare to load the route. Section rows show up after that, "
            "and flipping road/gravel recalculates without fetching weather again."
        )
        return
    if not live:
        st.caption(
            "Route, time, or power changed. Press Compare to refresh weather. "
            "Section choices are kept and apply on the next compare."
        )
        last = st.session_state.get("exp_last")
        if last:
            _render_results(last)
        return

    result = solved["result"]
    sections = solved["sections"]
    flags = _section_flags(sections) if sections else []
    section_crr = crr_for_sections(len(result.distances_m), sections, flags) if sections else None
    n_gravel = sum(1 for flag in flags if flag)
    section_note = (
        f"Unvalidated. Road Crr {ROAD_SECTION_CRR:.4f} and production Cd·A. "
        f"Gravel Crr {GRAVEL_SECTION_CRR:.3f}. "
        f"{n_gravel} of {len(sections)} sections set to gravel."
    )
    surface_report = solved["surface_report"] if surface_on else None
    surface_warning = None
    surface_shares = None
    if surface_on and surface_report is not None:
        if surface_report.fell_back:
            surface_warning = (
                "Surface lookup failed"
                + (f": {surface_report.error}" if surface_report.error else "")
                + ". The surface-aware column fell back to the production model. "
                "No share of the route was classified as paved, unpaved, or unmapped."
            )
        else:
            surface_shares = (
                "Route classified by distance: "
                + format_surface_shares(surface_report)
                + "."
            )
    manual_cda = float(st.session_state.exp_cda_mult) if manual_on else None
    manual_crr = float(st.session_state.exp_crr) if manual_on else None
    columns = compare_with_production(
        result,
        surface_report=surface_report,
        manual_cda_multiplier=manual_cda,
        manual_crr=manual_crr,
        section_crr=section_crr,
        section_note=section_note,
    )
    actual_s = result.actual_elapsed_s
    power_note = (
        "Every column uses the recorded power trace, same as the prediction page."
        if result.has_watts
        else f"No power trace on this route. Every column uses flat {float(power_w):.0f} W."
    )
    clock = (
        "Weather uses the ride's start time."
        if result.used_ride_start_time
        else "Weather uses the date you picked."
    )
    detail = (
        f"{solved['label']} · {result.distances_m[-1] / 1000.0:.1f} km · "
        f"{result.wind_label(unit='mph')} · {power_note} {clock}"
    )
    payload = {
        "columns": columns,
        "actual_s": actual_s,
        "actual_mph": average_speed_mph(result.distances_m, actual_s) if actual_s else None,
        "surface_warning": surface_warning,
        "surface_shares": surface_shares,
        "detail": detail,
        "effort_line": solved.get("effort_line"),
    }
    st.session_state.exp_last = payload
    _render_results(payload)


if __name__ == "__main__":
    main()
