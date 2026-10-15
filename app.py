"""Streamlit app: GPX → wind → speed chart with rider / aero controls."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from streamlit_folium import st_folium

from bike_wind.analysis import format_duration, parse_target_time
from bike_wind.gpx import parse_gpx
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
from bike_wind.strava import (
    StravaAuthError,
    StravaClient,
    StravaRateLimitError,
    load_strava_config,
)
from bike_wind.strava.oauth import build_authorize_url, exchange_code

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
    if "strava_access_token" not in st.session_state:
        st.session_state.strava_access_token = None
    if "strava_refresh_token" not in st.session_state:
        st.session_state.strava_refresh_token = None
    if "strava_meta" not in st.session_state:
        st.session_state.strava_meta = None
    if "strava_segment_id" not in st.session_state:
        st.session_state.strava_segment_id = None

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


def _effective_strava_token(config) -> str | None:
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
        st.session_state.strava_refresh_token = tokens.refresh_token
        st.session_state._oauth_handled_code = code
        # Clear code from URL without printing secrets
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


def _sync_rider_widgets(kg: float) -> None:
    st.session_state.rider_kg = round(float(kg), 2)
    st.session_state.rider_kg_input = float(st.session_state.rider_kg)
    st.session_state.rider_lb_input = round(kg_to_lb(st.session_state.rider_kg), 1)


def _sync_bike_widgets(kg: float) -> None:
    st.session_state.bike_kg = round(float(kg), 2)
    st.session_state.bike_kg_input = float(st.session_state.bike_kg)
    st.session_state.bike_lb_input = round(kg_to_lb(st.session_state.bike_kg), 1)


def _maybe_update_height_from_weight() -> None:
    if not st.session_state.height_manual:
        st.session_state.height_cm = smart_default_height_cm(st.session_state.rider_kg)
        st.session_state.height_cm_input = float(st.session_state.height_cm)


def _on_rider_kg() -> None:
    _sync_rider_widgets(st.session_state.rider_kg_input)
    _maybe_update_height_from_weight()


def _on_rider_lb() -> None:
    _sync_rider_widgets(lb_to_kg(st.session_state.rider_lb_input))
    _maybe_update_height_from_weight()


def _on_bike_kg() -> None:
    _sync_bike_widgets(st.session_state.bike_kg_input)
    st.session_state.bike_mass_manual = True


def _on_bike_lb() -> None:
    _sync_bike_widgets(lb_to_kg(st.session_state.bike_lb_input))
    st.session_state.bike_mass_manual = True


def _on_height_cm() -> None:
    st.session_state.height_cm = float(st.session_state.height_cm_input)
    st.session_state.height_manual = True


def main() -> None:
    _init_state()
    dark = st.toggle("Dark theme", value=True)
    _apply_theme(dark)

    st.title("Bike wind speed")
    st.caption(
        "GPX + Open-Meteo forecast → predicted ground speed along the route. "
        "Climbs are noisier when GPX elevation is coarse."
    )

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
                _sync_bike_widgets(POSITION_PRESETS[position_key].bike_mass_kg)

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

        preset_cols = st.columns(len(RIDER_WEIGHT_PRESETS_KG))
        for col, kg in zip(preset_cols, RIDER_WEIGHT_PRESETS_KG):
            with col:
                if st.button(f"{int(kg)}", key=f"wpreset_{int(kg)}", use_container_width=True):
                    _sync_rider_widgets(kg)
                    st.session_state.height_manual = False
                    _maybe_update_height_from_weight()
                    st.rerun()

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
        if st.session_state.bike_mass_manual:
            if st.button("Reset bike weight to position default"):
                st.session_state.bike_mass_manual = False
                _sync_bike_widgets(POSITION_PRESETS[st.session_state.position_key].bike_mass_kg)
                st.rerun()

        p1, p2 = st.columns(2)
        with p1:
            power_w = st.number_input(
                "Sustained power (W)",
                min_value=50,
                max_value=600,
                step=10,
                key="power_w",
            )
        with p2:
            st.caption("Power curve from a prior effort — stubbed for later.")
            st.text_input(
                "Prior effort (stub)",
                value="",
                disabled=True,
                label_visibility="collapsed",
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

    # --- Card 2: Route / time ---
    with st.container(border=True):
        st.subheader("Route & time")
        default_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(
            hours=3
        )
        when = st.datetime_input("Ride datetime (UTC)", value=default_dt)
        target_raw = st.text_input(
            "Optional KOM / target time (seconds or mm:ss)",
            value="",
            placeholder="e.g. 12:30 or 750",
            help="Leave blank to skip required-power vs sustainable comparison.",
        )

        route_source = st.radio(
            "Route source",
            options=["GPX", "Strava segment"],
            horizontal=True,
            key="route_source",
        )

        use_sample = False
        uploaded = None
        strava_ref_text = ""
        selected_starred_id: int | None = None

        if route_source == "GPX":
            use_sample = st.checkbox("Use sample South Florida GPX", value=True)
            uploaded = st.file_uploader("Or upload a GPX file", type=["gpx"])
        else:
            token = _effective_strava_token(strava_cfg)
            rl_client = StravaClient(strava_cfg, access_token=token or "pending")
            rl = rl_client.rate_status()
            st.caption(
                f"Strava API budget: {rl.remaining_15m}/200 per 15 min · "
                f"{rl.remaining_day}/2000 per day"
            )
            if rl.limited:
                st.error(rl.message or "Strava rate limited.")

            # OAuth / token setup
            if token:
                st.success("Strava token available (session or STRAVA_ACCESS_TOKEN).")
            elif strava_cfg.oauth_ready:
                auth_url = build_authorize_url(
                    strava_cfg.client_id,  # type: ignore[arg-type]
                    strava_cfg.redirect_uri,  # type: ignore[arg-type]
                )
                st.markdown(f"[Connect Strava (read scope)]({auth_url})")
            else:
                st.warning(
                    "Strava OAuth not configured. Set `STRAVA_CLIENT_ID`, "
                    "`STRAVA_CLIENT_SECRET`, and `STRAVA_REDIRECT_URI` — or set "
                    "`STRAVA_ACCESS_TOKEN` for segment URL/ID fetches. "
                    "GPX path still works without Strava."
                )

            strava_ref_text = st.text_input(
                "Strava segment URL or ID",
                placeholder="https://www.strava.com/segments/229781",
                help="Also accepts activity effort URLs …/activities/…/segments/…",
            )
            c_resolve, _ = st.columns([1, 3])
            with c_resolve:
                do_resolve = st.button("Resolve segment", disabled=not token or rl.limited)

            if do_resolve and token and strava_ref_text.strip():
                try:
                    client = StravaClient(strava_cfg, access_token=token)
                    sid, _ref = client.resolve_ref(strava_ref_text)
                    meta = client.get_segment_meta(sid)
                    st.session_state.strava_segment_id = sid
                    st.session_state.strava_meta = {
                        "segment_id": meta.segment_id,
                        "name": meta.name,
                        "distance_m": meta.distance_m,
                        "average_grade": meta.average_grade,
                        "from_cache": meta.from_cache,
                    }
                except StravaRateLimitError as exc:
                    st.error(exc.status.message or str(exc))
                except StravaAuthError as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Resolve failed: {exc}")

            if st.session_state.strava_meta:
                m = st.session_state.strava_meta
                st.info(
                    f"**{m['name']}** · id {m['segment_id']} · "
                    f"{m['distance_m']/1000:.2f} km · avg grade {m['average_grade']:.1f}% "
                    f"{'(cache)' if m.get('from_cache') else ''}"
                )

            # Starred segments (OAuth / token)
            if token:
                with st.expander("Starred segments", expanded=False):
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
                                }
                                for s in client.list_starred()
                            ]
                        except StravaRateLimitError as exc:
                            st.error(exc.status.message or str(exc))
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Starred fetch failed: {exc}")
                    starred = st.session_state.get("starred_segments") or []
                    if starred:
                        labels = [s["label"] for s in starred]
                        picked = st.radio(
                            "Select a starred segment",
                            options=["(none)"] + labels,
                            key="starred_pick",
                        )
                        if picked != "(none)":
                            selected_starred_id = next(
                                s["id"] for s in starred if s["label"] == picked
                            )
                            st.session_state.strava_segment_id = selected_starred_id
                    else:
                        st.caption("Click refresh to load starred segments.")

            # Explore (feature-flagged)
            with st.expander("Explore Segments", expanded=False):
                if not strava_cfg.extended_access:
                    st.caption(
                        "Requires Extended Access — set `STRAVA_EXTENDED_ACCESS=true` "
                        "when approved (no code change). Currently disabled."
                    )
                else:
                    st.caption("Top 10 ride segments in a bounding box (Extended Access).")
                    b1, b2, b3, b4 = st.columns(4)
                    sw_lat = b1.number_input("SW lat", value=26.05, format="%.4f")
                    sw_lon = b2.number_input("SW lon", value=-80.20, format="%.4f")
                    ne_lat = b3.number_input("NE lat", value=26.25, format="%.4f")
                    ne_lon = b4.number_input("NE lon", value=-80.05, format="%.4f")
                    if st.button("Explore top 10", disabled=not token or rl.limited):
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
                            selected_starred_id = next(
                                h["id"] for h in explore if h["label"] == epick
                            )
                            st.session_state.strava_segment_id = selected_starred_id

        run = st.button("Run prediction", type="primary")

    if not run:
        st.info("Adjust rider setup and route, then run prediction.")
        return

    target_time_s: float | None = None
    try:
        target_time_s = parse_target_time(target_raw)
    except ValueError as exc:
        st.error(f"Target time: {exc}")
        return

    try:
        if route_source == "Strava segment":
            token = _effective_strava_token(strava_cfg)
            sid = st.session_state.strava_segment_id
            if selected_starred_id is not None:
                sid = selected_starred_id
            if not token:
                st.error("Strava token required to load a segment. Use GPX or configure auth.")
                return
            if not sid and strava_ref_text.strip():
                client = StravaClient(strava_cfg, access_token=token)
                sid, _ = client.resolve_ref(strava_ref_text)
                st.session_state.strava_segment_id = sid
            if not sid:
                st.error("Resolve or select a Strava segment first.")
                return
            client = StravaClient(strava_cfg, access_token=token)
            meta, route = client.load_segment_route(int(sid))
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
    except Exception as exc:  # noqa: BLE001 — show parse/API errors in UI (no payloads)
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
        )

    # --- Card 3: Map (middle) ---
    with st.container(border=True):
        st.subheader("Map")
        st.caption(
            "Route colored teal (tailwind) → red (headwind) from along-track wind. "
            f"{source_label} · {route.distances_m[-1]/1000:.1f} km"
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
            f"{POSITION_PRESETS[aero.position_key].label} · Cd·A={aero.cd * aero.area_m2:.3f} · "
            f"{len(route)} points"
        )
        st.info(
            "Decision aid, not a fortune teller. Forecast accuracy degrades past day 5; "
            "climbs are noisier when GPX elevation/grade is coarse."
        )

        if result.weather_message:
            st.warning(result.weather_message)
        elif result.weather_ok:
            st.success(
                f"Open-Meteo wind applied (ρ ≈ {result.rho_mean:.3f} kg/m³). "
                f"Mean along-track {result.mean_wind_along_mps:+.1f} m/s "
                f"(head {result.mean_headwind_mps:.1f} · tail {result.mean_tailwind_mps:.1f})."
            )

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
                flag = "⚠ implausible" if b.implausible else ""
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
