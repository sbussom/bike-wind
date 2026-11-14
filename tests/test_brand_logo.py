"""GustyPower logo asset and the shared header/footer markup."""

from __future__ import annotations

from PIL import Image

from bike_wind.theme import ICON_PATH, LOGO_PATH, footer_html, header_html


def test_logo_is_trimmed_and_transparent() -> None:
    logo = Image.open(LOGO_PATH)
    assert logo.mode == "RGBA"
    assert logo.size[0] <= 1000
    width, height = logo.size
    for point in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        assert logo.getpixel(point)[3] == 0
    icon = Image.open(ICON_PATH)
    assert icon.mode == "RGBA"
    assert icon.size == (256, 256)
    assert icon.getpixel((0, 0))[3] == 0


def test_header_and_footer_use_the_logo_once_each() -> None:
    header = header_html()
    footer = footer_html()
    assert 'class="bw-logo-header"' in header
    assert 'class="bw-logo-footer"' in footer
    assert header.count('alt="GustyPower"') == 1
    assert footer.count('alt="GustyPower"') == 1
    assert "Bike Wind" not in header
    assert "Bike Wind" not in footer
    assert "data:image/png;base64," in header
