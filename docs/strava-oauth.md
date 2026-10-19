# Strava OAuth on Streamlit Cloud

Live app: https://bike-wind-prm8ndkr5xgfmy6rxdd4tj.streamlit.app

Settings resolve **environment first**, then `st.secrets` (same key names).

## Multi-user / Connect with Strava

Visitors click **Connect with Strava** → authorize → token stored in **that browser’s session only** (not shared secrets). Their starred list is their own.

- Session OAuth token wins for API calls.
- Deployer `access_token` in secrets is a fallback when nobody has connected in this session.
- **Disconnect Strava** clears the session token.

## Strava API application

In https://www.strava.com/settings/api :

| Field | Value |
| --- | --- |
| Authorization Callback Domain | `bike-wind-prm8ndkr5xgfmy6rxdd4tj.streamlit.app` |
| Redirect URI (exact, no path) | `https://bike-wind-prm8ndkr5xgfmy6rxdd4tj.streamlit.app` |

Same string must be `STRAVA_REDIRECT_URI` / `[strava].redirect_uri` in secrets. The app reads `?code=` from the root URL.

## Secrets example

```toml
[strava]
client_id = "your_client_id"
client_secret = "your_client_secret"
redirect_uri = "https://bike-wind-prm8ndkr5xgfmy6rxdd4tj.streamlit.app"
# access_token = "optional owner fallback"
```

Flat `STRAVA_*` keys and lowercase nested keys are accepted. Reboot after saving.

## Starred segments

Starred lists are fully paginated (all pages). The UI shows **N starred**.

## Privacy

Never commit secrets. Tokens and stream payloads are not application-logged.
