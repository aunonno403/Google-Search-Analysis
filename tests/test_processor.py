"""Tests for analyzer.processor — summary statistics and cleaning."""

import numpy as np
import pandas as pd
import pytest

from analyzer.processor import DataProcessor


def series(values, start="2024-01-07", freq="W"):
    idx = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, name="kw")


class TestComputeGrowthRate:
    def test_zero_values_do_not_produce_inf(self):
        """
        Regression: the old pct_change().mean() implementation divided by the
        previous value, so a single 0 — common in low-volume Trends data —
        made the metric `inf` and rendered as "inf% avg weekly change".
        """
        s = series([10, 0, 5, 0, 20, 30, 25, 40])
        rate = DataProcessor.compute_growth_rate(s)
        assert np.isfinite(rate)

    def test_all_zeros_returns_zero(self):
        assert DataProcessor.compute_growth_rate(series([0, 0, 0, 0])) == 0.0

    def test_constant_series_has_no_growth(self):
        assert DataProcessor.compute_growth_rate(series([50] * 10)) == 0.0

    def test_zero_gap_keeps_real_elapsed_time(self):
        """Regression: dropped zeros used to renumber x, so 10→20 over 9 weeks read as +100%/week."""
        s = series([10] + [0] * 8 + [20])
        assert DataProcessor.compute_growth_rate(s) == pytest.approx(8.0, abs=0.1)

    def test_steady_doubling_is_about_100_percent(self):
        s = series([1, 2, 4, 8, 16, 32, 64])
        assert DataProcessor.compute_growth_rate(s) == pytest.approx(100.0, abs=0.5)

    def test_steady_halving_is_about_minus_50_percent(self):
        s = series([64, 32, 16, 8, 4, 2, 1])
        assert DataProcessor.compute_growth_rate(s) == pytest.approx(-50.0, abs=0.5)

    def test_up_then_down_is_not_positively_biased(self):
        """
        +50% then -50% is a net loss. The arithmetic mean of percentage
        changes reports 0%, which overstates performance; a geometric rate
        must come out negative.
        """
        s = series([100.0, 150.0, 75.0, 112.5, 56.25])
        assert DataProcessor.compute_growth_rate(s) < 0

    def test_too_short_returns_zero(self):
        assert DataProcessor.compute_growth_rate(series([5])) == 0.0
        assert DataProcessor.compute_growth_rate(pd.Series([], dtype=float)) == 0.0

    def test_nan_values_are_ignored(self):
        s = series([10, np.nan, 20, 40, np.nan, 80])
        assert np.isfinite(DataProcessor.compute_growth_rate(s))


class TestComputeSummaryStats:
    def test_basic_fields(self):
        df = series([10, 20, 100, 40, 30]).to_frame()
        stats = DataProcessor.compute_summary_stats(df, "kw")
        assert stats["current"] == 30
        assert stats["peak"] == 100
        assert stats["avg"] == pytest.approx(40.0)
        assert "2024" in stats["peak_date"]

    def test_missing_keyword_returns_empty(self):
        df = series([1, 2, 3]).to_frame()
        assert DataProcessor.compute_summary_stats(df, "absent") == {}

    def test_empty_frame_returns_empty(self):
        assert DataProcessor.compute_summary_stats(pd.DataFrame(), "kw") == {}

    def test_trailing_nan_does_not_raise(self):
        """int(series.iloc[-1]) used to raise ValueError on a NaN tail."""
        df = series([10, 20, 30, np.nan]).to_frame()
        stats = DataProcessor.compute_summary_stats(df, "kw")
        assert stats["current"] == 30

    def test_all_nan_column_returns_empty(self):
        df = series([np.nan, np.nan, np.nan]).to_frame()
        assert DataProcessor.compute_summary_stats(df, "kw") == {}

    def test_trend_is_finite_with_zeros(self):
        df = series([0, 0, 50, 0, 100]).to_frame()
        stats = DataProcessor.compute_summary_stats(df, "kw")
        assert np.isfinite(stats["trend"])


class TestCleanInterestOverTime:
    def test_none_and_empty(self):
        assert DataProcessor.clean_interest_over_time(None).empty
        assert DataProcessor.clean_interest_over_time(pd.DataFrame()).empty

    def test_fills_na_and_sets_datetime_index(self):
        df = pd.DataFrame({"kw": [1.0, np.nan, 3.0]},
                          index=["2024-01-07", "2024-01-14", "2024-01-21"])
        out = DataProcessor.clean_interest_over_time(df)
        assert isinstance(out.index, pd.DatetimeIndex)
        assert out["kw"].tolist() == [1.0, 0.0, 3.0]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"kw": [1.0, np.nan]},
                          index=pd.date_range("2024-01-07", periods=2, freq="W"))
        DataProcessor.clean_interest_over_time(df)
        assert df["kw"].isna().any()


class TestGetTopRegions:
    def test_returns_n_sorted_desc(self):
        df = pd.DataFrame({"kw": [5, 90, 40, 70]},
                          index=["A", "B", "C", "D"])
        top = DataProcessor.get_top_regions(df, "kw", n=2)
        assert list(top.index) == ["B", "D"]

    def test_missing_keyword_returns_empty(self):
        df = pd.DataFrame({"kw": [1, 2]}, index=["A", "B"])
        assert DataProcessor.get_top_regions(df, "absent").empty
