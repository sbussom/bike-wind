"""User-visible Strava error kinds (safe strings — never include tokens/bodies)."""

from __future__ import annotations

MISSING_SECRETS = (
    "[missing secrets] No Strava access_token or refresh_token available. "
    "In Streamlit Cloud secrets add under `[strava]`: `client_id`, `client_secret`, "
    "`access_token`, and `refresh_token` (from https://www.strava.com/settings/api), "
    "then reboot. Or click Connect with Strava."
)

REFRESH_FAILED = (
    "[refresh failed] Could not refresh the Strava access token. "
    "Common causes: dead/revoked refresh_token, or client_id/client_secret mismatch "
    "after rotating the secret. Click **Reset Strava session**, then verify "
    "`client_id` / `client_secret` / `refresh_token` in `[strava]` secrets match "
    "https://www.strava.com/settings/api ."
)

REFRESH_FAILED_DETAIL = (
    "[refresh failed] Strava said: {detail}. "
    "Typical: invalid refresh_token (revoked when a newer token was issued) or "
    "wrong client_secret after rotation. Click **Reset Strava session** and confirm "
    "secrets match the API settings page."
)

FORBIDDEN_SCOPE = (
    "[Strava 403 scope] Token lacks required OAuth scope `read` for {path}. "
    "Diagnostics: {diag}. Strava “Get Access Token” must include scope `read`. "
    "`activity:read` is not required for public segment lookup."
)

FORBIDDEN_EXPLORE_TIER = (
    "[Strava explore blocked] GET /segments/explore is restricted to apps on "
    "Strava’s Extended Access Tier (changelog Sept 1, 2026). Your token may be "
    "valid (refresh OK) but this API app is not approved for explore. "
    "Go to https://www.strava.com/settings/api → API Settings Dashboard → request "
    "Extended / Expanded Access. This is not a Streamlit secret. "
    "Until then: paste a segment ID/URL, upload GPX, or open Strava’s public map "
    "near your pin (link shown below). Diagnostics: {diag}"
)

RATE_LIMIT = (
    "[rate limit] Strava HTTP 429 — wait and retry. "
    "15-min and daily quotas apply to explore and segment calls."
)

ZERO_SEGMENTS = (
    "[zero segments] Explore returned 0 segments near lat={lat:.4f}, lon={lon:.4f} "
    "(~10 km box). Try Use my location, move the map pin, or widen the search by "
    "picking a denser riding area. Auth succeeded — this is not a token error."
)

UNAUTHORIZED_AFTER_REFRESH = (
    "[auth after refresh] Refresh returned a new access_token (expires_at={expires_at}, "
    "scope={scope}), bearer was applied to the retry, but {path} still returned "
    "HTTP {status}. Strava said: {detail}. "
    "Diagnostics: {diag}. Click **Reset Strava session** if a sticky browser token "
    "might be involved; do not keep re-pasting the same secrets unless Strava says "
    "the refresh_token is invalid."
)

EXPLORE_UNAUTHORIZED_TOKEN_OK = (
    "[Strava explore blocked] Refresh succeeded and GET /athlete works with the new "
    "token (scope={scope}), but {path} still returned HTTP {status}. "
    "That usually means this API app lacks Extended Access for explore "
    "(required since Sept 1, 2026) — not that your secrets are wrong. "
    "Request Extended Access at https://www.strava.com/settings/api . "
    "Meanwhile use segment ID/URL, GPX, or the public Strava map link below. "
    "Strava said: {detail}. Diagnostics: {diag}"
)
