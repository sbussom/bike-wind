"""Date and hour controls that do not open a floating calendar.

Streamlit's date and datetime popovers are clipped by the widgets under them
on a phone, and they close as soon as the page scrolls. These steppers,
quick buttons, and an hour slider stay on the page.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta, timezone


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def clamp_date(year: int, month: int, day: int) -> date:
    """A real calendar day. Day 31 in a short month becomes the last day."""
    year = int(year)
    month = min(12, max(1, int(month)))
    last = calendar.monthrange(year, month)[1]
    day = min(last, max(1, int(day)))
    return date(year, month, day)


def utc_datetime(day: date, hour: int) -> datetime:
    """Timezone-aware UTC time on the hour, same shape the forecast expects."""
    hour = min(23, max(0, int(hour)))
    return datetime(day.year, day.month, day.day, hour, 0, tzinfo=timezone.utc)


def shift_date(day: date, days: int) -> date:
    return day + timedelta(days=int(days))


def format_utc_day(day: date) -> str:
    return f"{day.strftime('%a %b')} {day.day}, {day.year} UTC"


def _remember_day(key: str, day: date) -> None:
    import streamlit as st

    st.session_state[f"{key}_y"] = int(day.year)
    st.session_state[f"{key}_m"] = int(day.month)
    st.session_state[f"{key}_d"] = int(day.day)


def _clamp_stored_day(key: str) -> None:
    import streamlit as st

    fixed = clamp_date(
        st.session_state[f"{key}_y"],
        st.session_state[f"{key}_m"],
        st.session_state[f"{key}_d"],
    )
    _remember_day(key, fixed)


def pick_utc_date(
    label: str,
    *,
    key: str,
    default: date,
    quick: tuple[tuple[str, int], ...] = (),
) -> date:
    """Year / month / day steppers plus optional Today-style buttons.

    ``quick`` entries are ``(button label, days from today UTC)``.
    """
    import streamlit as st

    if f"{key}_y" not in st.session_state:
        _remember_day(key, default)
    st.caption(label)
    if quick:
        cols = st.columns(len(quick))
        for col, (name, delta) in zip(cols, quick):
            with col:
                st.button(
                    name,
                    key=f"{key}_btn_{name}",
                    use_container_width=True,
                    on_click=_remember_day,
                    args=(key, shift_date(utc_today(), delta)),
                )
    year_col, month_col, day_col = st.columns(3)
    with year_col:
        year = st.number_input(
            "Year",
            min_value=2009,
            max_value=2036,
            step=1,
            key=f"{key}_y",
            on_change=_clamp_stored_day,
            args=(key,),
        )
    with month_col:
        month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            step=1,
            key=f"{key}_m",
            on_change=_clamp_stored_day,
            args=(key,),
        )
    with day_col:
        day = st.number_input(
            "Day",
            min_value=1,
            max_value=31,
            step=1,
            key=f"{key}_d",
            on_change=_clamp_stored_day,
            args=(key,),
        )
    chosen = clamp_date(year, month, day)
    st.caption(format_utc_day(chosen))
    return chosen


def pick_utc_when(
    label: str,
    *,
    key: str,
    default: datetime,
    quick: tuple[tuple[str, int], ...] = (("Today", 0), ("Tomorrow", 1)),
) -> datetime:
    """Forecast clock: a UTC day plus an hour slider. No calendar popup."""
    import streamlit as st

    if default.tzinfo is None:
        default = default.replace(tzinfo=timezone.utc)
    else:
        default = default.astimezone(timezone.utc)
    if f"{key}_h" not in st.session_state:
        st.session_state[f"{key}_h"] = int(default.hour)
    day = pick_utc_date(label, key=key, default=default.date(), quick=quick)
    hour = st.slider("Hour (UTC)", min_value=0, max_value=23, key=f"{key}_h")
    when = utc_datetime(day, int(hour))
    st.caption(f"{format_utc_day(day)} · {when.hour:02d}:00")
    return when
