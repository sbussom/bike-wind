"""Keep the OSM surface lookup off the network in tests."""

import pytest


@pytest.fixture(autouse=True)
def _osm_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("GUSTYPOWER_OSM_OFFLINE", "1")
    monkeypatch.setenv("GUSTYPOWER_OSM_CACHE", str(tmp_path / "osm_cache"))
    # Hide shipped St. Lucie tiles so offline tests stay network-free and
    # independent of which tiles the package happens to include.
    monkeypatch.setenv("GUSTYPOWER_OSM_BUNDLED", str(tmp_path / "osm_bundled"))
    from bike_wind.surface import osm_surface

    osm_surface._MEMO.clear()
    yield
