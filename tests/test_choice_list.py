"""Ride lists stay on the page instead of opening a dropdown."""

from bike_wind.choice_list import page_window


def test_newest_rides_are_the_first_page():
    rides = [f"ride {i}" for i in range(14)]
    window, index, pages = page_window(rides, page=0, page_size=6)
    assert window == rides[:6]
    assert index == 0
    assert pages == 3
    older, index2, _ = page_window(rides, page=2, page_size=6)
    assert older == rides[12:]
    assert index2 == 2


def test_page_index_is_clamped():
    window, index, pages = page_window(["only"], page=9, page_size=6)
    assert window == ["only"]
    assert index == 0
    assert pages == 1
    assert page_window([], page=0, page_size=6) == ([], 0, 1)


def test_ride_menus_are_not_selectboxes():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    app = (root / "app.py").read_text()
    experimental = (root / "pages" / "2_Experimental.py").read_text()
    assert "st.selectbox" not in app
    assert "st.selectbox" not in experimental
    assert "st.multiselect" not in app
    assert "visible_choice" in app
    assert "Recent rides" in app
