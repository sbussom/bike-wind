"""User-visible Strava error kinds (safe strings — never include tokens/bodies)."""

from __future__ import annotations

MISSING_SECRETS = (
    "[missing secrets] No Strava access_token or refresh_token available. "
    "In Streamlit Cloud secrets add under `[strava]`: `client_id`, `client_secret`, "
    "`access_token`, and `refresh_token` (from https://www.strava.com/settings/api), "
    "then reboot. Or click Connect with Strava."
)

EXPIRED_ACCESS = (
    "[expired access] Strava returned HTTP 401 (access token missing/expired). "
    "Attempting refresh…"
)

REFRESH_FAILED = (
    "[refresh failed] Could not refresh the Strava access token. "
    "Common causes: dead/revoked refresh_token, or client_id/client_secret mismatch "
    "after rotating the secret. Fix: click **Reset Strava session** (clears sticky "
    "browser tokens), then https://www.strava.com/settings/api → copy a fresh "
    "Access Token + Refresh Token → paste into Streamlit `[strava]` as "
    "`access_token` / `refresh_token` → reboot. "
    "Secrets are read-only at runtime."
)

REFRESH_FAILED_DETAIL = (
    "[refresh failed] Strava said: {detail}. "
    "Typical: `invalid` / `unauthorized` refresh_token (revoked when a newer token "
    "was issued) or wrong client_secret after rotation. "
    "Click **Reset Strava session**, paste matching Access+Refresh from "
    "https://www.strava.com/settings/api into `[strava]` secrets, reboot."
)

FORBIDDEN_SCOPE = (
    "[Strava 403 scope] Token lacks required OAuth scope `read` for {path}. "
    "Strava “Get Access Token” / Connect with Strava must include scope `read` "
    "(public segments). activity:read is not required for explore or public segment "
    "lookup. Re-authorize with Connect Strava (scope=read) or mint a new token "
    "with `read` checked."
)

FORBIDDEN_EXPLORE_TIER = (
    "[Strava 403 explore] GET /segments/explore is restricted to apps on Strava’s "
    "Extended Access Tier (changelog: Sept 1, 2026). Scope `read` alone is not enough. "
    "Go to https://www.strava.com/settings/api → API Settings Dashboard → request "
    "Expanded / Extended Access (self-service for single-player apps when offered). "
    "Until approved, use segment ID/URL or GPX — nearby explore will keep failing. "
    "This is not a Streamlit secret flag."
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
    "[expired access] Still unauthorized after a successful-looking refresh. "
    "Paste fresh tokens from https://www.strava.com/settings/api into `[strava]` "
    "secrets and reboot."
)
