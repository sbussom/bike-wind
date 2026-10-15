# Bike wind speed predictions

Upload a GPX (or use the sample), set position / aero / weight / power, pick a datetime, pull Open-Meteo forecast wind along the route, and plot predicted ground speed vs distance.

**Phase 1 rider UI:** position radios (TT / drops / hoods / upright), kg↔lbs weight fields with presets, bike weight, aero kit toggles, height under Advanced.

**Phase 2 results:** optional KOM/target time → required vs sustainable power + implausible flags; ETA uncertainty band by forecast horizon; head/tail wind summary; distance-bin table; reliability copy.

**Phase 3 map:** Folium middle card — dark/light Esri basemap, route colored teal→red by along-track wind, auto-fit bounds.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or: pip install -e ".[dev]"
```

## Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Toggle dark/light theme at the top. Check **Use sample South Florida GPX** to demo without uploading.

If the environment blocks outbound HTTPS to `api.open-meteo.com`, the app still charts a calm-air prediction and shows a clear offline fallback message.

## Tests

```bash
pytest
```

Physics unit tests cover aero/rolling/gravity forces, tuck area, crosswind drag, iterative speed solve, and the log wind-profile correction.

## Package layout

```
bike_wind/
  physics/     # forces + iterative velocity solver
  gpx/         # GPX → lat/lon/elev/distance streams
  weather/     # Open-Meteo forecast + log-profile wind
  rider.py     # position / aero presets
  analysis.py  # KOM power, ETA bands, bins
  map_view.py  # Folium teal→red wind map
  predict.py   # end-to-end orchestration
app.py         # Streamlit UI
fixtures/      # sample_route.gpx
tests/
```

## Out of scope (later phases)

Strava, HRRR, Explore Segments, full map UI, OAuth.
