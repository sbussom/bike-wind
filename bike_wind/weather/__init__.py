"""Weather fetch and wind-height correction."""

from bike_wind.weather.cache import ForecastCache
from bike_wind.weather.fetch import WeatherFetchResult, fetch_weather_along_route
from bike_wind.weather.hrrr import hrrr_enabled_by_env, in_conus, route_in_conus
from bike_wind.weather.open_meteo import (
    ForecastPoint,
    WeatherFetchError,
    fetch_forecast_along_route,
)
from bike_wind.weather.wind_profile import correct_wind_speed, wind_uv_at_rider_height

__all__ = [
    "ForecastCache",
    "ForecastPoint",
    "WeatherFetchError",
    "WeatherFetchResult",
    "correct_wind_speed",
    "fetch_forecast_along_route",
    "fetch_weather_along_route",
    "hrrr_enabled_by_env",
    "in_conus",
    "route_in_conus",
    "wind_uv_at_rider_height",
]
