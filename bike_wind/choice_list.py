"""Always-visible choices instead of a dropdown menu.

A Streamlit select menu is a floating list. On a phone its lower options sit
under the next text field, and scrolling the page closes the menu. A radio
list is part of the page, so every option stays put.
"""

from __future__ import annotations


def page_window(items: list[str], page: int, page_size: int) -> tuple[list[str], int, int]:
    """Slice ``items`` (newest first) into a 1-based page window.

    Returns ``(rows, page_index, page_count)``. ``page_index`` is clamped.
    """
    if page_size < 1:
        raise ValueError("page_size must be at least 1")
    count = len(items)
    pages = max(1, (count + page_size - 1) // page_size) if count else 1
    index = min(max(0, int(page)), pages - 1)
    start = index * page_size
    return items[start : start + page_size], index, pages


def visible_choice(
    label: str,
    options: list[str],
    *,
    key: str,
    placeholder: str,
    on_change=None,
    page_size: int = 6,
) -> str:
    """Radio list of ``options``, newest first, paged when the list is long.

    ``options`` does not include ``placeholder``. The placeholder is the
    default so opening the page does not load a ride by itself.
    """
    import streamlit as st

    page_key = f"{key}_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    window, index, pages = page_window(list(options), int(st.session_state[page_key]), page_size)
    st.session_state[page_key] = index
    shown = [placeholder, *window]
    if st.session_state.get(key) not in shown:
        st.session_state[key] = placeholder
    st.radio(label, shown, key=key, on_change=on_change)
    if pages > 1:
        newer, mid, older = st.columns([1, 2, 1])
        newer.button(
            "Newer",
            key=f"{key}_newer",
            disabled=index == 0,
            use_container_width=True,
            on_click=_shift_page,
            args=(page_key, -1, pages),
        )
        first = index * page_size + 1
        last = first + len(window) - 1
        mid.caption(f"{first}–{last} of {len(options)}, newest first")
        older.button(
            "Older",
            key=f"{key}_older",
            disabled=index >= pages - 1,
            use_container_width=True,
            on_click=_shift_page,
            args=(page_key, 1, pages),
        )
    return str(st.session_state[key])


def _shift_page(page_key: str, delta: int, pages: int) -> None:
    import streamlit as st

    current = int(st.session_state.get(page_key, 0))
    st.session_state[page_key] = min(pages - 1, max(0, current + delta))
