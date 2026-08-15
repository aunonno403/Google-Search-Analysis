"""Tests for visualizer.charts — figure construction and the choropleth fix."""

import numpy as np
import pandas as pd
import pytest

import config
from visualizer.charts import ChartBuilder, PALETTE


@pytest.fixture
def iot_df():
    idx = pd.date_range("2024-01-07", periods=30, freq="W")
    return pd.DataFrame(
        {"alpha": np.linspace(10, 90, 30), "beta": np.linspace(90, 10, 30)},
        index=idx,
    )


@pytest.fixture
def region_df():
    """Mirrors pytrends' interest_by_region(inc_geo_code=True) shape."""
    return pd.DataFrame(
        {"alpha": [100, 80, 60], "geoCode": ["US", "IN", "GB"]},
        index=pd.Index(["United States", "India", "United Kingdom"], name="geoName"),
    )


class TestChoropleth:
    def test_converts_alpha2_codes_to_iso3(self, region_df):
        """
        Regression: pytrends' geoCode is ISO 3166-1 alpha-2 ("US"), but
        Plotly's default locationmode is ISO-3 ("USA"). Passing alpha-2 codes
        with locationmode unset matched nothing, and Plotly drops unmatched
        rows silently — the map rendered blank with no error.
        """
        fig = ChartBuilder.choropleth_map(region_df, "alpha")
        trace = fig.data[0]
        assert trace.locationmode == "ISO-3"
        assert set(trace.locations) == {"USA", "IND", "GBR"}

    def test_all_locations_are_three_letter_codes(self, region_df):
        fig = ChartBuilder.choropleth_map(region_df, "alpha")
        assert all(len(loc) == 3 for loc in fig.data[0].locations)

    def test_values_stay_attached_to_the_right_country(self, region_df):
        fig = ChartBuilder.choropleth_map(region_df, "alpha")
        trace = fig.data[0]
        assert dict(zip(trace.locations, trace.z)) == {"USA": 100, "IND": 80, "GBR": 60}

    def test_unmappable_codes_are_dropped_not_crashed(self):
        df = pd.DataFrame(
            {"alpha": [100, 50], "geoCode": ["US", "ZZ"]},
            index=pd.Index(["United States", "Nowhere"], name="geoName"),
        )
        fig = ChartBuilder.choropleth_map(df, "alpha")
        assert list(fig.data[0].locations) == ["USA"]

    def test_falls_back_to_names_without_geocode(self):
        df = pd.DataFrame(
            {"alpha": [100, 80]},
            index=pd.Index(["United States", "India"], name="geoName"),
        )
        fig = ChartBuilder.choropleth_map(df, "alpha")
        assert fig.data[0].locationmode == "country names"

    def test_missing_keyword_returns_empty_figure(self, region_df):
        assert ChartBuilder.choropleth_map(region_df, "absent").data == ()


class TestTopRegionsBar:
    def test_shows_all_requested_regions(self):
        """
        Regression: the default n was 15 while the caller sliced to 20 and
        the heading said "Top 20 Regions", so five rows silently vanished.
        """
        df = pd.DataFrame(
            {"alpha": list(range(20))},
            index=[f"C{i}" for i in range(20)],
        )
        fig = ChartBuilder.top_regions_bar(df, "alpha", n=config.TOP_REGIONS_N)
        assert len(fig.data[0].y) == 20

    def test_default_n_matches_config(self):
        import inspect
        default = inspect.signature(ChartBuilder.top_regions_bar).parameters["n"].default
        assert default == config.TOP_REGIONS_N

    def test_empty_frame_returns_empty_figure(self):
        assert ChartBuilder.top_regions_bar(pd.DataFrame(), "alpha").data == ()


class TestForecastChart:
    def test_forecast_line_connects_to_history(self, iot_df):
        """The dashed forecast should start at the last historical point."""
        series = iot_df["alpha"]
        fdates = pd.date_range("2024-08-04", periods=5, freq="W")
        fvals = np.array([90.0] * 5)
        fig = ChartBuilder.forecast_chart(
            historical_dates=series.index,
            historical_values=series.values,
            forecast_dates=fdates,
            forecast_values=fvals,
            lower_ci=fvals - 5,
            upper_ci=fvals + 5,
            keyword="alpha",
            method="Test",
        )
        forecast_trace = next(t for t in fig.data if t.name == "Forecast")
        assert forecast_trace.x[0] == series.index[-1]
        assert forecast_trace.y[0] == pytest.approx(series.values[-1])
        assert len(forecast_trace.x) == len(fdates) + 1

    def test_band_is_labelled_a_prediction_interval(self, iot_df):
        series = iot_df["alpha"]
        fdates = pd.date_range("2024-08-04", periods=3, freq="W")
        fvals = np.array([50.0] * 3)
        fig = ChartBuilder.forecast_chart(
            historical_dates=series.index, historical_values=series.values,
            forecast_dates=fdates, forecast_values=fvals,
            lower_ci=fvals - 5, upper_ci=fvals + 5,
            keyword="alpha", method="Test",
        )
        assert any("prediction" in (t.name or "") for t in fig.data)


class TestInterestOverTime:
    def test_builds_two_traces_per_keyword(self, iot_df):
        fig = ChartBuilder.interest_over_time(iot_df, ["alpha", "beta"])
        assert len(fig.data) == 4

    def test_ignores_unknown_keywords(self, iot_df):
        fig = ChartBuilder.interest_over_time(iot_df, ["alpha", "missing"])
        assert len(fig.data) == 2

    def test_smoothing_toggle_changes_values(self, iot_df):
        noisy = iot_df.copy()
        noisy["alpha"] = [10, 90] * 15
        smooth = ChartBuilder.interest_over_time(noisy, ["alpha"], smooth=True)
        raw = ChartBuilder.interest_over_time(noisy, ["alpha"], smooth=False)
        assert list(smooth.data[1].y) != list(raw.data[1].y)


class TestRadar:
    def test_stability_axis_replaces_volatility(self, iot_df):
        """
        A radar chart reads as "further out is better". Volatility inverted
        that on one axis, so an erratic keyword drew the biggest shape.
        """
        fig = ChartBuilder.comparison_radar(iot_df, ["alpha", "beta"])
        assert "Stability" in list(fig.data[0].theta)
        assert "Volatility" not in list(fig.data[0].theta)

    def test_steady_series_scores_higher_stability_than_erratic(self):
        idx = pd.date_range("2024-01-07", periods=20, freq="W")
        df = pd.DataFrame(
            {"steady": [50.0] * 20, "erratic": [0.0, 100.0] * 10},
            index=idx,
        )
        fig = ChartBuilder.comparison_radar(df, ["steady", "erratic"])
        axis = list(fig.data[0].theta).index("Stability")
        assert fig.data[0].r[axis] > fig.data[1].r[axis]

    def test_all_radial_values_within_scale(self, iot_df):
        fig = ChartBuilder.comparison_radar(iot_df, ["alpha", "beta"])
        for trace in fig.data:
            assert all(0 <= v <= 100 for v in trace.r)


class TestPalette:
    def test_derived_from_config(self):
        assert PALETTE[0] == config.PRIMARY_COLOR
        assert PALETTE[1] == config.SECONDARY_COLOR

    def test_all_entries_are_hex(self):
        assert all(c.startswith("#") and len(c) == 7 for c in PALETTE)


class TestStackedArea:
    def test_shares_sum_to_100(self, iot_df):
        fig = ChartBuilder.stacked_area(iot_df, ["alpha", "beta"])
        totals = np.array(fig.data[0].y) + np.array(fig.data[1].y)
        assert np.allclose(totals, 100.0)

    def test_all_zero_rows_do_not_produce_nan(self):
        idx = pd.date_range("2024-01-07", periods=5, freq="W")
        df = pd.DataFrame({"a": [0, 0, 10, 20, 0], "b": [0, 0, 10, 20, 0]}, index=idx)
        fig = ChartBuilder.stacked_area(df, ["a", "b"])
        assert not np.isnan(np.array(fig.data[0].y, dtype=float)).any()
