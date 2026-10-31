"""CdA / yaw / wind defaults from the held-out calibration gate.

``FITTED_CDA`` is the personal drag area from the earlier rides of the
3-year measured-power set (Oct 2023 through Dec 2025). ``USE_FITTED_CDA``
is True only because that fit, with the wind handling below, lowered
held-out moving MAE (actual speed > 3 mph) versus the previous live model
(CdA 0.380, Crr 0.0075, yaw off, each sample's own wind × 0.75).

The latest 25% of rides, from 2025-12-26, were not used to train.
Hold-out moving MAE: 2.298 mph before, 2.292 mph after.

A later time-error refit (CdA 0.48, Crr 0.0075, same 78 kg) cut held-out
time error but raised held-out speed MAE, so these coefficients stayed.
0 W is coasting (momentum), not a 0.1 m/s crawl. Anchor wind is blended,
then smoothed over ``WIND_SMOOTH_M``.

Power-dependent Cd·A (``POWER_CDA_ENABLED``): the single 0.505 ran slow on
hard efforts and fast on easy ones. On the stock fitted path, Cd·A now
follows the 3 s power at each sample::

    CdA(P) = 0.5125 - 0.05 * (clamp(P, 100, 350) - 180) / 100

so 0.553 coasting or under 100 W, 0.505 at 195 W, 0.478 at 250 W and 0.428
at 350 W. The intercept and slope were picked by lowest pooled training
speed MAE (434 rides before 2025-12-26) over global, power-step,
ride-mean-power and per-sample families. Same 144-ride holdout:
time error -0.290% -> +0.265%, speed MAE 2.276 -> 2.251 mph.
"""

from __future__ import annotations

# Preset gravel-hoods Cd·A (cd=1.0 × area) before the personal fit.
DEFAULT_CDA = 0.408

# Train optimum under time-resolved wind ×0.75. Applied: hold-out MAE dropped.
FITTED_CDA = 0.505

USE_FITTED_CDA = True
YAW_CDA_ENABLED = False

YAW_K = 1.5

# Each sample uses the weather clock the app can actually resolve
# (15-min when the ride is under 90 days old, otherwise the archive hour).
# The sustained vector is then scaled. Gusts use the same scale and stay a band.
WIND_TIME_RESOLVED = True
WIND_SCALE = 0.75
# Centered distance smooth on along- and cross-track wind after the anchors
# are blended. Kills single-sample bearing flicker without flattening a
# real headwind on an out-and-back leg. 0 disables it.
WIND_SMOOTH_M = 150.0


# Cd·A that falls with effort power (fitted path only; see module docstring).
POWER_CDA_ENABLED = True
POWER_CDA_AT_REF = 0.5125
POWER_CDA_REF_W = 180.0
POWER_CDA_SLOPE_PER_100W = -0.05
POWER_CDA_MIN_W = 100.0
POWER_CDA_MAX_W = 350.0


def power_cda(power_w: float) -> float:
    """Personal Cd·A at this pedaling power (3 s average), clamped range."""
    p = float(power_w)
    if p != p:  # NaN
        p = POWER_CDA_REF_W
    p = min(max(p, POWER_CDA_MIN_W), POWER_CDA_MAX_W)
    return POWER_CDA_AT_REF + POWER_CDA_SLOPE_PER_100W * (p - POWER_CDA_REF_W) / 100.0


def power_cda_factor(power_w: float) -> float:
    """Multiplier on the fitted Cd·A so the product equals ``power_cda``."""
    return power_cda(power_w) / FITTED_CDA


def cda_in_use() -> float:
    """Cd·A the solver uses for the default gravel / packed-sand path."""
    return FITTED_CDA if USE_FITTED_CDA else DEFAULT_CDA


def cda_fit_label() -> str:
    """Short label for Equations: fitted value vs what prediction uses."""
    fitted = FITTED_CDA
    in_use = cda_in_use()
    status = "in use" if USE_FITTED_CDA else "not applied (hold-out MAE did not drop)"
    return f"fitted Cd·A = {fitted:.3f} m² ({status}); in use = {in_use:.3f} m²"


# Surface-aware rolling setup. Each route point can be classed from
# OpenStreetMap (bike_wind.surface): paved, unpaved, or off-map (no mapped
# road: canal banks, levees, fields). Paved and OSM-unpaved points get their
# own Crr and a scale on the power-dependent Cd·A; off-map points keep the
# single setup above (Crr 0.0052, scale 1.0). A failed OSM lookup keeps that
# single setup at every point.
#
# Fit on the same 434 training rides (before 2025-12-26): lowest pooled speed
# MAE with training time error no worse than before, off-map Crr held at
# 0.0052 so canal-bank segments (Round trip to pumphouse) do not move.
# Same 144-ride holdout: time error +0.265% -> -0.157%, speed MAE
# 2.251 -> 2.194 mph. Of the ridden distance ~63% is paved, ~11% OSM-unpaved
# and ~26% off-map.
#
# ``SURFACE_AWARE_ENABLED`` stays False. The prediction page never turns this
# on, so production times stay on the validated single setup. The Experimental
# page opts in per run by setting ``RiderBikeParams.surface_aware`` itself.
SURFACE_AWARE_ENABLED = False
SURFACE_PAVED_CRR = 0.0052
SURFACE_PAVED_CDA_SCALE = 0.95
SURFACE_UNPAVED_CRR = 0.010
SURFACE_UNPAVED_CDA_SCALE = 1.0
SURFACE_OFFMAP_CRR = 0.0052
SURFACE_OFFMAP_CDA_SCALE = 1.0


def surface_setup(surface_class: int) -> tuple[float, float]:
    """(Crr, Cd·A scale) for an OSM surface class (1 paved, 0 unpaved, -1 off-map)."""
    if surface_class == 1:
        return SURFACE_PAVED_CRR, SURFACE_PAVED_CDA_SCALE
    if surface_class == 0:
        return SURFACE_UNPAVED_CRR, SURFACE_UNPAVED_CDA_SCALE
    return SURFACE_OFFMAP_CRR, SURFACE_OFFMAP_CDA_SCALE
