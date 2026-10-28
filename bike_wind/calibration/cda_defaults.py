"""CdA / yaw / wind defaults from the held-out calibration gate.

``FITTED_CDA`` is the personal drag area from the earlier of the 57
measured-power rides (Jul–Oct 2026). ``USE_FITTED_CDA`` is True only because
that fit, with the wind handling below, lowered later-ride hold-out moving
MAE (actual speed > 3 mph) versus CdA 0.408, Crr 0.008, yaw off, and a
single ride-start wind snapshot.

Hold-out moving MAE: 2.314 mph before, 2.244 mph after.
"""

from __future__ import annotations

# Preset gravel-hoods Cd·A (cd=1.0 × area) before the personal fit.
DEFAULT_CDA = 0.408

# Train optimum under time-resolved wind ×0.75. Applied: hold-out MAE dropped.
FITTED_CDA = 0.38

USE_FITTED_CDA = True
YAW_CDA_ENABLED = False

YAW_K = 1.5

# Each sample uses the weather clock the app can actually resolve
# (15-min when the ride is under 90 days old, otherwise the archive hour).
# The sustained vector is then scaled. Gusts use the same scale and stay a band.
WIND_TIME_RESOLVED = True
WIND_SCALE = 0.75


def cda_in_use() -> float:
    """Cd·A the solver uses for the default gravel / packed-sand path."""
    return FITTED_CDA if USE_FITTED_CDA else DEFAULT_CDA


def cda_fit_label() -> str:
    """Short label for Equations: fitted value vs what prediction uses."""
    fitted = FITTED_CDA
    in_use = cda_in_use()
    status = "in use" if USE_FITTED_CDA else "not applied (hold-out MAE did not drop)"
    return f"fitted Cd·A = {fitted:.3f} m² ({status}); in use = {in_use:.3f} m²"
