# Bike wind speed predictions

Predict ground speed on a road segment under forecast wind: GPX or Strava segment → Open-Meteo → physics → map + chart.

**Phase 1:** position / aero / dual-unit weights.  
**Phase 2:** KOM power, ETA uncertainty band, bins.  
**Phase 3:** Folium map teal→red by along-track wind.  
**Phase 4:** Strava segment URL/ID, streams cache, OAuth starred, Explore (feature-flagged).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). GPX works with no Strava credentials.

## Strava (optional)

| Env var | Purpose |
| --- | --- |
| `STRAVA_ACCESS_TOKEN` | Personal access token for segment URL/ID + streams (no OAuth UI) |
| `STRAVA_CLIENT_ID` | OAuth app client id |
| `STRAVA_CLIENT_SECRET` | OAuth app client secret |
| `STRAVA_REDIRECT_URI` | Must match Strava app settings (e.g. `http://localhost:8501`) |
| `STRAVA_EXTENDED_ACCESS` | Default `false`. Set `true` to enable Explore Segments (top 10 in bbox) when approved — no code change |
| `BIKE_WIND_CACHE` | Optional override for SQLite cache dir (default `.cache/strava`) |

OAuth uses **read** scope. Tokens live in Streamlit session state after redirect; rate limits (200/15 min, 2000/day) are tracked locally and shown in the UI.

**Privacy:** Strava payloads (streams, athlete data, tokens) are never sent to LLMs and are not written to application logs. Segment details + streams are stored only in local SQLite under `.cache/strava/`.

## Tests

```bash
pytest
```

Includes physics, analysis, map color helpers, and mocked Strava URL/client/cache/rate-limit tests (no live Strava calls).

## Package layout

```
bike_wind/
  physics/     # forces + iterative velocity solver
  gpx/         # GPX → lat/lon/elev/distance streams
  weather/     # Open-Meteo forecast + log-profile wind
  strava/      # URL parse, API client, OAuth, SQLite cache, rate limits
  rider.py     # position / aero presets
  analysis.py  # KOM power, ETA bands, bins
  map_view.py  # Folium teal→red wind map
  predict.py   # end-to-end orchestration
app.py         # Streamlit UI
fixtures/      # sample_route.gpx
tests/
```

## Out of scope (Phase 5)

NOAA HRRR, wind forecast SQLite cache polish, back-test notes.
