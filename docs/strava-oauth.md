# Strava OAuth on Streamlit Cloud

Live app: https://bike-wind-prm8ndkr5xgfmy6rxdd4tj.streamlit.app

Settings resolve **environment first**, then `st.secrets` (same key names).

## Multi-user / Connect with Strava

Visitors click **Connect with Strava** → authorize → token stored in **that browser’s session only**. Used for segment resolve and the **nearby explore map**.

- Session OAuth token wins for API calls.
- Deployer `access_token` in secrets is a fallback when nobody has connected in this session.
- **Disconnect Strava** clears the session token.

## Nearby segments map

Uses the same Strava access token as segment loads (session OAuth or secrets `access_token`). No extra secret flag.

1. **Use my location** (browser geolocation) or set the map pin.
2. **Find nearby segments** → `GET /segments/explore` in a ~10 km box (results cached ~5 min in session).
3. Select a segment from the list (or open its Strava link) to load it for prediction.

If Strava returns HTTP 403, the token is missing the `read` scope — reconnect via **Connect with Strava** or mint a token that includes `read`.

Starred-segment lists were removed from the UI.

## Access token expiry / refresh

Strava access tokens expire. On HTTP 401 the app calls the OAuth **refresh_token** grant with secrets (or session) `client_id` / `client_secret` / `refresh_token`, then retries explore and segment loads.

- A successful refresh updates **`st.session_state` only** for this browser session.
- **Streamlit Cloud secrets are read-only at runtime** — the app cannot write the new access token back into secrets. Paste updated tokens into secrets when you want them to survive reboots / cold starts.
- If refresh also fails (dead or revoked refresh token): open https://www.strava.com/settings/api → copy a fresh **Access Token** (and **Refresh Token** if shown) → paste under `[strava]` as `access_token` / `refresh_token` → reboot the app. Or use **Connect with Strava**.

## Strava API application

In https://www.strava.com/settings/api :

| Field | Value |
| --- | --- |
| Authorization Callback Domain | `bike-wind-prm8ndkr5xgfmy6rxdd4tj.streamlit.app` |
| Redirect URI (exact, no path) | `https://bike-wind-prm8ndkr5xgfmy6rxdd4tj.streamlit.app` |

Same string must be `STRAVA_REDIRECT_URI` / `[strava].redirect_uri` in secrets.

## Secrets example

```toml
[strava]
client_id = "your_client_id"
client_secret = "your_client_secret"
redirect_uri = "https://bike-wind-prm8ndkr5xgfmy6rxdd4tj.streamlit.app"
# access_token = "optional owner fallback (expires — pair with refresh_token)"
# refresh_token = "optional; used to refresh expired access_token"
```

## Privacy

Never commit secrets. Tokens and stream payloads are not application-logged.
