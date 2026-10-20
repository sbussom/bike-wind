"""Local Strava explore map: bounds around a pin + Folium markers."""

from __future__ import annotations

from typing import Sequence

import folium

from bike_wind.map_view import DARK_ATTR, DARK_TILES, LIGHT_ATTR, LIGHT_TILES
from bike_wind.strava.client import ExploreHit

# ~8–10 km box at mid-latitudes (degrees)
DEFAULT_HALF_SPAN_DEG = 0.08
NEARBY_MAP_HEIGHT_PX = 420


def bounds_around(
    lat: float,
    lon: float,
    *,
    half_span_deg: float = DEFAULT_HALF_SPAN_DEG,
) -> list[float]:
    """Return [sw_lat, sw_lon, ne_lat, ne_lon] for Strava explore."""
    return [
        lat - half_span_deg,
        lon - half_span_deg,
        lat + half_span_deg,
        lon + half_span_deg,
    ]


def strava_public_map_url(lat: float, lon: float, *, zoom: int = 12) -> str:
    """Public Strava map centered on a pin (fallback when API explore is blocked)."""
    return f"https://www.strava.com/maps#{zoom}/{lat:.5f}/{lon:.5f}"


def build_nearby_segments_map(
    center_lat: float,
    center_lon: float,
    hits: Sequence[ExploreHit],
    *,
    dark: bool = True,
    height_px: int = NEARBY_MAP_HEIGHT_PX,
    zoom_start: int = 12,
) -> folium.Map:
    """Folium map with a location pin and nearby Strava segment markers."""
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles=DARK_TILES if dark else LIGHT_TILES,
        attr=DARK_ATTR if dark else LIGHT_ATTR,
        name="basemap",
        control=False,
    ).add_to(fmap)

    folium.CircleMarker(
        location=[center_lat, center_lon],
        radius=8,
        color="#3ecf8e",
        fill=True,
        fill_color="#3ecf8e",
        fill_opacity=0.9,
        tooltip="Your location / search center",
    ).add_to(fmap)

    for hit in hits:
        if hit.start_lat is None or hit.start_lon is None:
            continue
        strava_url = f"https://www.strava.com/segments/{hit.segment_id}"
        popup = (
            f"<b>{_esc(hit.name)}</b><br>"
            f"{hit.distance_m/1000:.1f} km · {hit.average_grade:.1f}%<br>"
            f"id {hit.segment_id}<br>"
            f'<a href="{strava_url}" target="_blank" rel="noopener">Open on Strava</a>'
        )
        folium.Marker(
            location=[hit.start_lat, hit.start_lon],
            tooltip=f"{hit.name} · id {hit.segment_id}",
            popup=folium.Popup(popup, max_width=260),
            icon=folium.Icon(color="orange", icon="flag", prefix="fa"),
        ).add_to(fmap)
        if hit.end_lat is not None and hit.end_lon is not None:
            folium.PolyLine(
                locations=[
                    [hit.start_lat, hit.start_lon],
                    [hit.end_lat, hit.end_lon],
                ],
                color="#f59e0b",
                weight=3,
                opacity=0.75,
                tooltip=hit.name,
            ).add_to(fmap)

    if hits:
        pts = [[center_lat, center_lon]]
        for hit in hits:
            if hit.start_lat is not None and hit.start_lon is not None:
                pts.append([hit.start_lat, hit.start_lon])
            if hit.end_lat is not None and hit.end_lon is not None:
                pts.append([hit.end_lat, hit.end_lon])
        if len(pts) >= 2:
            fmap.fit_bounds(pts, padding=(28, 28))

    fmap.get_root().html.add_child(
        folium.Element(
            f"<style>.folium-map {{ min-height: {height_px}px !important; }}</style>"
        )
    )
    return fmap


def geolocation_redirect_html() -> str:
    """Button that writes geo_lat/geo_lon into the top window query string."""
    return """
    <div style="font-family:Figtree,sans-serif;margin:0.25rem 0 0.5rem;">
      <button id="bw-geo-btn" type="button"
        style="min-height:44px;padding:0.55rem 1rem;border-radius:10px;
               border:1px solid rgba(62,207,142,0.35);background:#1f6b4a;
               color:#e7efea;font-weight:600;cursor:pointer;width:100%;">
        Use my location
      </button>
      <p id="bw-geo-msg" style="color:#8fa398;font-size:0.85rem;margin:0.35rem 0 0;"></p>
    </div>
    <script>
    (function () {
      const btn = document.getElementById("bw-geo-btn");
      const msg = document.getElementById("bw-geo-msg");
      if (!btn) return;
      btn.addEventListener("click", function () {
        if (!navigator.geolocation) {
          msg.textContent = "Geolocation is not supported in this browser.";
          return;
        }
        msg.textContent = "Locating…";
        navigator.geolocation.getCurrentPosition(
          function (pos) {
            try {
              const u = new URL(window.top.location.href);
              u.searchParams.set("geo_lat", pos.coords.latitude.toFixed(5));
              u.searchParams.set("geo_lon", pos.coords.longitude.toFixed(5));
              window.top.location.href = u.toString();
            } catch (e) {
              msg.textContent = "Could not update the page URL with your location.";
            }
          },
          function () {
            msg.textContent = "Location permission denied — set the map pin manually.";
          },
          { enableHighAccuracy: true, timeout: 12000, maximumAge: 60000 }
        );
      });
    })();
    </script>
    """


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
