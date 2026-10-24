"""CdA / yaw defaults from the held-out calibration gate.

``FITTED_CDA`` is always the least-squares train estimate (Crr fixed at
packed-sand 0.008). ``USE_FITTED_CDA`` and ``YAW_CDA_ENABLED`` are True only
when that change dropped 2024+ hold-out moving MAE vs current main
(100 m elev smooth, CdA 0.408, no yaw).
"""

from __future__ import annotations

# Preset gravel-hoods Cd·A (cd=1.0 × area).
DEFAULT_CDA = 0.408

# Least-squares train Cd·A (Crr=0.008, elev=100 m, calm). Not applied:
# hold-out MAE 2.484 > baseline 2.444.
FITTED_CDA = 0.32

# Gate results — both False: hold-out MAE did not drop.
USE_FITTED_CDA = False
YAW_CDA_ENABLED = False

YAW_K = 1.5


def cda_in_use() -> float:
    """Cd·A the solver uses for the default gravel / packed-sand path."""
    return FITTED_CDA if USE_FITTED_CDA else DEFAULT_CDA


def cda_fit_label() -> str:
    """Short label for Equations: fitted value vs what prediction uses."""
    fitted = FITTED_CDA
    in_use = cda_in_use()
    status = "in use" if USE_FITTED_CDA else "not applied (hold-out MAE did not drop)"
    return f"fitted Cd·A = {fitted:.3f} m² ({status}); in use = {in_use:.3f} m²"
