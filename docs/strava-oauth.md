# Strava OAuth on Streamlit Cloud

Live app: https://bike-wind-prm8ndkr5xgfmy6rxdd4tj.streamlit.app

Settings resolve **environment first**, then `st.secrets` (same key names). Cloud secrets are usually only in `st.secrets`, not `os.environ`.

## Strava API application

In https://www.strava.com/settings/api :

| Field | Value |
| --- | --- |
| Authorization Callback Domain | `bike-wind-prm8ndkr5xgfmy6rxdd4tj.streamlit.app` |
| (redirect used by app) | `https://bike-wind-prm8ndkr5xgfmy6rxdd4tj.streamlit.app` |

No path on the redirect URI — the app reads `?code=` from the root URL after OAuth.

## Streamlit Cloud secrets

App settings → **Secrets** → paste:

```toml
STRAVA_CLIENT_ID = "your_client_id"
STRAVA_CLIENT_SECRET = "your_client_secret"
STRAVA_REDIRECT_URI = "https://bike-wind-prm8ndkr5xgfmy6rxdd4tj.streamlit.app"

# Optional: skip OAuth UI and call segment APIs directly
# STRAVA_ACCESS_TOKEN = "your_access_token"

# Optional flags
# STRAVA_EXTENDED_ACCESS = "false"
# BIKE_WIND_HRRR = "false"
```

After saving secrets, reboot the app. Under **Route source → Strava segment** you should see **Connect Strava (read scope)** when client id/secret/redirect are present.

## Privacy

Never commit secrets. The app does not log Strava tokens or stream payloads.
