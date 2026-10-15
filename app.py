"""Streamlit Phase 0 vertical slice: GPX → wind → speed chart."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from bike_wind.gpx import parse_gpx
from bike_wind.physics import BIKE_MASS_ROAD_KG, RiderBikeParams
from bike_wind.predict import downsample_route, predict_speeds

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
            </style>
            """,
            unsafe_allow_html=True,
        )


def _format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def main() -> None:
    dark = st.toggle("Dark theme", value=True)
    _apply_theme(dark)

    st.title("Bike wind speed")
    st.caption(
        "GPX + Open-Meteo forecast → predicted ground speed along the route. "
        "Climbs are noisier when GPX elevation is coarse."
    )

    with st.container():
        st.subheader("Inputs")
        c1, c2, c3 = st.columns(3)
        with c1:
            power_w = st.number_input("Sustained power (W)", min_value=50, max_value=600, value=220, step=10)
        with c2:
            rider_kg = st.number_input("Rider weight (kg)", min_value=40.0, max_value=150.0, value=75.0, step=0.5)
        with c3:
            default_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(hours=3)
            when = st.datetime_input("Ride datetime (UTC)", value=default_dt)

        use_sample = st.checkbox("Use sample South Florida GPX", value=True)
        uploaded = st.file_uploader("Or upload a GPX file", type=["gpx"])

        run = st.button("Run prediction", type="primary")

    if not run:
        st.info("Set power/weight/datetime, then run prediction.")
        return

    try:
        if uploaded is not None and not use_sample:
            route = parse_gpx(uploaded.getvalue())
            source_label = uploaded.name
        elif use_sample or uploaded is None:
            route = parse_gpx(FIXTURE)
            source_label = FIXTURE.name
        else:
            route = parse_gpx(uploaded.getvalue())
            source_label = uploaded.name
    except Exception as exc:  # noqa: BLE001 — show parse errors in UI
        st.error(f"Could not parse GPX: {exc}")
        return

    route = downsample_route(route, max_points=300)
    params = RiderBikeParams(
        rider_mass_kg=float(rider_kg),
        power_w=float(power_w),
        bike_mass_kg=BIKE_MASS_ROAD_KG,
    )

    with st.spinner("Fetching forecast and solving speeds…"):
        when_dt = when if isinstance(when, datetime) else datetime.combine(when, datetime.min.time())
        if when_dt.tzinfo is None:
            when_dt = when_dt.replace(tzinfo=timezone.utc)
        result = predict_speeds(route, params, when_dt)

    st.subheader("Results")
    st.caption(f"Route: {source_label} · {route.distances_m[-1]/1000:.1f} km · {len(route)} points")

    if result.weather_message:
        st.warning(result.weather_message)
    elif result.weather_ok:
        st.success(
            f"Open-Meteo wind applied (ρ ≈ {result.rho_mean:.3f} kg/m³). "
            "Forecast skill declines past ~day 5."
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("ETA with wind", _format_duration(result.eta_with_wind_s))
    m2.metric("ETA calm air", _format_duration(result.eta_no_wind_s))
    avg = (
        sum(result.speeds_with_wind_kmh) / len(result.speeds_with_wind_kmh)
        if result.speeds_with_wind_kmh
        else 0.0
    )
    m3.metric("Avg speed (wind)", f"{avg:.1f} km/h")

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


if __name__ == "__main__":
    main()
