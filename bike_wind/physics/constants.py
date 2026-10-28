"""Shared physics defaults and constants."""

G: float = 9.80665  # m/s²
RHO_DEFAULT: float = 1.2  # kg/m³
R_DRY_AIR: float = 287.05  # J/(kg·K)

CD_DEFAULT: float = 0.9
AREA_DEFAULT_M2: float = 0.4
TUCK_AREA_FACTOR: float = 0.85  # −15% frontal area on descents
# Road-tire pavement baseline; packed-sand gravel rides use CRR_PACKED_SAND.
CRR_DEFAULT: float = 0.004
# 45 mm gravel tire on packed sand. 0.0052 is the personal fit on the 3-year
# measured-power rides (held-out moving MAE dropped). Not a road tire.
CRR_PACKED_SAND: float = 0.0052

BIKE_MASS_ROAD_KG: float = 8.5
BIKE_MASS_TT_KG: float = 7.0
BIKE_MASS_GRAVEL_KG: float = 9.0

# Crank → wheel; ~3% loss so meter watts are not 100% propulsive.
DRIVETRAIN_EFFICIENCY: float = 0.97

RIDER_HEIGHT_M: float = 1.2
WIND_REF_HEIGHT_M: float = 10.0
Z0_OPEN_GRASS: float = 0.03
