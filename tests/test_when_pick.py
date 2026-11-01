"""UTC date steppers: no calendar popover, and short months clamp."""

from datetime import date, datetime, timezone

from bike_wind.when_pick import clamp_date, format_utc_day, shift_date, utc_datetime


def test_clamp_short_month_and_leap_day():
    assert clamp_date(2026, 2, 31) == date(2026, 2, 28)
    assert clamp_date(2024, 2, 31) == date(2024, 2, 29)
    assert clamp_date(2026, 13, 0) == date(2026, 12, 1)
    assert clamp_date(2026, 0, 40) == date(2026, 1, 31)


def test_utc_datetime_is_on_the_hour():
    when = utc_datetime(date(2026, 11, 1), 15)
    assert when == datetime(2026, 11, 1, 15, 0, tzinfo=timezone.utc)
    assert utc_datetime(date(2026, 11, 1), 99).hour == 23
    assert utc_datetime(date(2026, 11, 1), -4).hour == 0


def test_shift_and_label():
    assert shift_date(date(2026, 11, 1), -1) == date(2026, 10, 31)
    assert shift_date(date(2026, 11, 1), 1) == date(2026, 11, 2)
    assert format_utc_day(date(2026, 11, 1)) == "Sun Nov 1, 2026 UTC"


def test_pages_do_not_use_popover_date_widgets():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("app.py", "pages/2_Experimental.py"):
        text = (root / name).read_text()
        assert "st.date_input" not in text
        assert "st.datetime_input" not in text
        assert "pick_utc_" in text
