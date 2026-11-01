"""Headwind/tailwind chart series. Display only; speeds are not involved."""

import math

import pytest

from bike_wind.physics.forces import RiderBikeParams, WindComponents
from bike_wind.predict import PredictionResult
from bike_wind.wind_chart import figure_wind_along, wind_along_chart, wind_scale_for_slider

MPS = 2.2369362920544


def _result(**kwargs) -> PredictionResult:
    base = dict(
        distances_m=[0.0, 1609.344, 3218.688],
        speeds_with_wind_mps=[8.0, 8.0, 8.0],
        speeds_no_wind_mps=[8.0, 8.0, 8.0],
        grades=[0.0, 0.0, 0.0],
        wind_along_mps=[2.0, -1.0, 0.0],  # model sign: + tailwind
        wind_cross_mps=[0.5, 0.5, 0.0],
        winds=[
            WindComponents(along_track=2.0, cross_track=0.5),
            WindComponents(along_track=-1.0, cross_track=0.5),
            WindComponents(),
        ],
        params=RiderBikeParams(rider_mass_kg=69.0, power_w=200.0),
        rho_mean=1.2,
        weather_ok=True,
        mean_wind_speed_mps=2.0,
        mean_wind_from_deg=90.0,
        gust_along_mps=[3.0, -1.5, float("nan")],
        wind_from_deg_samples=[90.0, 90.0, 180.0],
    )
    base.update(kwargs)
    return PredictionResult(**base)


def test_headwind_is_positive_and_gust_is_separate():
    chart = wind_along_chart(_result())
    assert chart is not None
    assert chart.headwind_mph[0] == pytest.approx(-2.0 * MPS)
    assert chart.headwind_mph[1] == pytest.approx(1.0 * MPS)
    assert chart.headwind_mph[2] == pytest.approx(0.0)
    assert chart.has_gust
    assert chart.gust_headwind_mph is not None
    assert chart.gust_headwind_mph[0] == pytest.approx(-3.0 * MPS)
    assert chart.gust_headwind_mph[1] == pytest.approx(1.5 * MPS)
    assert math.isnan(chart.gust_headwind_mph[2])
    assert chart.from_labels
    assert any(label == "E" for _x, label in chart.from_labels)
    assert "Headwind" in chart.caption
    assert "mph" in chart.caption


def test_no_gust_line_without_gust_product():
    chart = wind_along_chart(_result(gust_along_mps=None))
    assert chart is not None
    assert chart.gust_headwind_mph is None
    assert chart.has_gust is False


def test_no_chart_when_weather_failed():
    assert wind_along_chart(_result(weather_ok=False)) is None


def test_slider_scale_matches_a_stronger_wind():
    result = _result()
    assert wind_scale_for_slider(result, None) == 1.0
    assert wind_scale_for_slider(result, 4.0) == pytest.approx(2.0)
    chart = wind_along_chart(result, wind_speed_mps=4.0)
    assert chart is not None
    assert chart.headwind_mph[1] == pytest.approx(2.0 * MPS)


def test_figure_labels_axes_in_mph():
    chart = wind_along_chart(_result())
    assert chart is not None
    fig = figure_wind_along(chart, dark=True)
    ax = fig.axes[0]
    assert ax.get_ylabel() == "mph"
    assert "Distance" in ax.get_xlabel()
    assert "Wind along the route" in ax.get_title(loc="left")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Sustained" in text for text in labels)
    assert any("Gust" in text for text in labels)
    fig.clf()
