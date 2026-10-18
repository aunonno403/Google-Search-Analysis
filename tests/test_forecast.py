"""Tests for forecaster.forecast — frequency inference and date generation."""

import numpy as np
import pandas as pd
import pytest

from forecaster.forecast import TrendForecaster


def make_series(n=60, freq="W", start="2024-01-07"):
    idx = pd.date_range(start, periods=n, freq=freq)
    values = 50 + 10 * np.sin(np.arange(n) / 5.0)
    return pd.Series(values, index=idx)


class TestInferFreq:
    @pytest.mark.parametrize("freq,expected", [
        ("min", "min"),
        ("8min", "8min"),
        ("h",   "h"),
        ("D",   "D"),
        ("W",   "W"),
        ("MS",  "MS"),
    ])
    def test_roundtrips_common_frequencies(self, freq, expected):
        """
        Regression: infer_freq used `delta.days`, which is 0 for any
        sub-daily spacing, so hourly and minute data both came back as "D".
        The forecast then generated daily dates for an hourly model.
        """
        assert TrendForecaster.infer_freq(make_series(40, freq=freq)) == expected

    def test_non_datetime_index_defaults_to_weekly(self):
        assert TrendForecaster.infer_freq(pd.Series([1, 2, 3])) == "W"

    def test_single_point_defaults_to_weekly(self):
        s = pd.Series([1.0], index=pd.date_range("2024-01-07", periods=1))
        assert TrendForecaster.infer_freq(s) == "W"


class TestGenerateForecastDates:
    def test_weekly_continues_from_last_point(self):
        fc = TrendForecaster(forecast_periods=4)
        dates = fc.generate_forecast_dates(pd.Timestamp("2024-03-17"), freq="W")
        assert len(dates) == 4
        assert dates[0] == pd.Timestamp("2024-03-24")

    def test_month_start_does_not_skip_a_period(self):
        """
        Regression: date_range(start=last, periods=n+1)[1:] assumed `last`
        would be generated. With freq="MS" and a mid-month date it is not, so
        the slice dropped a real forecast period — April vanished and the
        series began in May, shifting every point a month late.
        """
        fc = TrendForecaster(forecast_periods=4)
        dates = fc.generate_forecast_dates(pd.Timestamp("2024-03-15"), freq="MS")
        assert dates[0] == pd.Timestamp("2024-04-01")
        assert len(dates) == 4

    def test_length_always_matches_forecast_periods(self):
        for freq in ["min", "h", "D", "W", "MS", "QS"]:
            fc = TrendForecaster(forecast_periods=7)
            dates = fc.generate_forecast_dates(pd.Timestamp("2024-03-15"), freq=freq)
            assert len(dates) == 7, freq

    def test_all_dates_are_after_last_observation(self):
        last = pd.Timestamp("2024-03-15 10:30")
        for freq in ["min", "h", "D", "W", "MS"]:
            fc = TrendForecaster(forecast_periods=5)
            dates = fc.generate_forecast_dates(last, freq=freq)
            assert (dates > last).all(), freq


class TestSeasonalPeriod:
    @pytest.mark.parametrize("freq,expected", [("h", 24), ("D", 7), ("W", 52), ("MS", 12)])
    def test_matches_granularity(self, freq, expected):
        assert TrendForecaster.seasonal_period(freq) == expected

    def test_unknown_freq_falls_back(self):
        assert TrendForecaster.seasonal_period("bogus") == 52


class TestForecasting:
    def test_polynomial_shape_and_bounds(self):
        fc = TrendForecaster(forecast_periods=12)
        out = fc.fit_polynomial_trend(make_series(50))
        assert len(out["forecast"]) == 12
        assert out["forecast"].min() >= 0 and out["forecast"].max() <= 100
        assert (out["lower_ci"] <= out["upper_ci"]).all()

    def test_get_best_forecast_always_has_required_keys(self):
        fc = TrendForecaster(forecast_periods=8)
        out = fc.get_best_forecast(make_series(60))
        for key in ("forecast", "lower_ci", "upper_ci", "method", "trend_direction"):
            assert key in out
        assert len(out["forecast"]) == 8

    def test_output_clipped_to_trends_scale(self):
        """Google Trends is a 0–100 index; forecasts must not leave it."""
        idx = pd.date_range("2024-01-07", periods=40, freq="W")
        exploding = pd.Series(np.linspace(10, 100, 40), index=idx)
        out = TrendForecaster(forecast_periods=20).get_best_forecast(exploding)
        assert out["forecast"].max() <= 100
        assert out["lower_ci"].min() >= 0

    def test_short_series_still_returns_a_forecast(self):
        fc = TrendForecaster(forecast_periods=4)
        out = fc.get_best_forecast(make_series(12))
        assert len(out["forecast"]) == 4

    def test_forecast_dates_align_with_forecast_values(self):
        """The two must be the same length or the chart silently misaligns."""
        fc = TrendForecaster(forecast_periods=10)
        s = make_series(60, freq="h")
        out = fc.get_best_forecast(s)
        dates = fc.generate_forecast_dates(s.index[-1], freq=fc.infer_freq(s))
        assert len(dates) == len(out["forecast"])


class TestTrendDirection:
    def test_rising(self):
        s = pd.Series(np.linspace(10, 90, 30),
                      index=pd.date_range("2024-01-07", periods=30, freq="W"))
        assert "Rising" in TrendForecaster.detect_trend_direction(s)

    def test_falling(self):
        s = pd.Series(np.linspace(90, 10, 30),
                      index=pd.date_range("2024-01-07", periods=30, freq="W"))
        assert "Falling" in TrendForecaster.detect_trend_direction(s)

    def test_stable(self):
        s = pd.Series([50.0] * 30,
                      index=pd.date_range("2024-01-07", periods=30, freq="W"))
        assert "Stable" in TrendForecaster.detect_trend_direction(s)

    def test_too_short(self):
        assert TrendForecaster.detect_trend_direction(pd.Series([1, 2])) == "Insufficient data"

    def test_all_zero_series_does_not_divide_by_zero(self):
        s = pd.Series([0.0] * 20,
                      index=pd.date_range("2024-01-07", periods=20, freq="W"))
        assert "Stable" in TrendForecaster.detect_trend_direction(s)
