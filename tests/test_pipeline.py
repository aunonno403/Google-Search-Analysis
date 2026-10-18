"""
End-to-end pipeline tests.

These walk the same sequence main.py does after a fetch — clean, summarise,
chart, forecast — using synthetic data shaped like real pytrends output, so
the modules are exercised together without touching the network.
"""

import numpy as np
import pandas as pd
import pytest

import config
from analyzer.processor import DataProcessor
from analyzer.trends import TrendsAnalyzer
from forecaster.forecast import TrendForecaster
from visualizer.charts import ChartBuilder

KEYWORDS = ["alpha", "beta", "gamma"]


def fake_iot(freq="W", n=60, partial=True):
    """Mimic pytrends.interest_over_time() output, isPartial column included."""
    idx = pd.date_range("2024-01-07", periods=n, freq=freq)
    rng = np.random.default_rng(0)
    data = {kw: rng.integers(0, 101, n) for kw in KEYWORDS}
    if partial:
        data["isPartial"] = [False] * (n - 1) + [True]
    return pd.DataFrame(data, index=idx)


def fake_region():
    """Mimic pytrends.interest_by_region(inc_geo_code=True) output."""
    names = ["United States", "India", "United Kingdom", "Germany", "Brazil"]
    codes = ["US", "IN", "GB", "DE", "BR"]
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {**{kw: rng.integers(0, 101, len(names)) for kw in KEYWORDS},
         "geoCode": codes},
        index=pd.Index(names, name="geoName"),
    )


@pytest.mark.parametrize("freq", ["min", "8min", "h", "D", "W", "MS"])
def test_full_pipeline_across_every_granularity(freq):
    """
    Every timeframe preset maps to one of these frequencies. The forecast
    horizon, the generated dates, and the chart must line up for all of them —
    not just the weekly case the code originally assumed.
    """
    raw = fake_iot(freq=freq, n=60).drop(columns=["isPartial"])
    iot = DataProcessor.clean_interest_over_time(raw)
    assert not iot.empty

    for kw in KEYWORDS:
        stats = DataProcessor.compute_summary_stats(iot, kw)
        assert np.isfinite(stats["trend"]), f"{kw} @ {freq} produced a non-finite trend"

    series = iot["alpha"]
    forecaster = TrendForecaster(forecast_periods=8)
    result = forecaster.get_best_forecast(series)
    inferred = forecaster.infer_freq(series)
    dates = forecaster.generate_forecast_dates(series.index[-1], freq=inferred)

    assert inferred == freq
    assert len(dates) == len(result["forecast"]) == 8
    assert dates[0] > series.index[-1]

    fig = ChartBuilder.forecast_chart(
        historical_dates=series.index,
        historical_values=series.values,
        forecast_dates=dates,
        forecast_values=result["forecast"],
        lower_ci=result["lower_ci"],
        upper_ci=result["upper_ci"],
        keyword="alpha",
        method=result["method"],
    )
    assert len(fig.data) == 3


def test_charts_build_from_pipeline_output():
    iot = DataProcessor.clean_interest_over_time(fake_iot().drop(columns=["isPartial"]))
    region = fake_region()

    assert ChartBuilder.interest_over_time(iot, KEYWORDS).data
    assert ChartBuilder.comparison_radar(iot, KEYWORDS).data
    assert ChartBuilder.stacked_area(iot, KEYWORDS).data
    assert ChartBuilder.choropleth_map(region, "alpha").data

    top = DataProcessor.get_top_regions(region, "alpha", n=config.TOP_REGIONS_N)
    assert ChartBuilder.top_regions_bar(top, "alpha", n=config.TOP_REGIONS_N).data


def test_partial_column_is_dropped_before_use():
    """isPartial is a bool flag, not a keyword — it must not reach the charts."""
    raw = fake_iot(partial=True)
    assert "isPartial" in raw.columns
    cleaned = DataProcessor.clean_interest_over_time(raw.drop(columns=["isPartial"]))
    assert "isPartial" not in cleaned.columns


def test_dropped_keyword_does_not_break_the_results_path():
    """
    Google silently omits low-volume keywords. main.py filters to the columns
    that came back; everything downstream must cope with the shorter list.
    """
    iot = DataProcessor.clean_interest_over_time(
        fake_iot().drop(columns=["isPartial", "gamma"])
    )
    requested = KEYWORDS
    returned = [k for k in requested if k in iot.columns]
    dropped = [k for k in requested if k not in iot.columns]

    assert returned == ["alpha", "beta"]
    assert dropped == ["gamma"]

    assert ChartBuilder.interest_over_time(iot, returned).data
    assert ChartBuilder.comparison_radar(iot, returned).data
    assert DataProcessor.compute_summary_stats(iot, "gamma") == {}


def test_single_keyword_comparison_paths_are_safe():
    iot = DataProcessor.clean_interest_over_time(
        fake_iot().drop(columns=["isPartial", "beta", "gamma"])
    )
    assert ChartBuilder.stacked_area(iot, ["alpha"]).data
    assert ChartBuilder.comparison_radar(iot, ["alpha"]).data


def test_empty_results_degrade_quietly():
    empty = DataProcessor.clean_interest_over_time(pd.DataFrame())
    assert empty.empty
    assert ChartBuilder.interest_over_time(empty, KEYWORDS).data == ()
    assert ChartBuilder.top_regions_bar(pd.DataFrame(), "alpha").data == ()
    assert ChartBuilder.choropleth_map(pd.DataFrame(), "alpha").data == ()


def test_fetch_all_is_offline_safe_when_every_call_fails(monkeypatch):
    """
    Only interest-over-time is mandatory. The optional sections must degrade
    to empty values with recorded warnings rather than failing the whole run.
    """
    analyzer = TrendsAnalyzer.__new__(TrendsAnalyzer)  # skip network in __init__
    analyzer._last_request_time = 0.0

    monkeypatch.setattr(analyzer, "build_payload", lambda *a, **k: None)
    monkeypatch.setattr(
        analyzer, "get_interest_over_time",
        lambda: fake_iot().drop(columns=["isPartial"]),
    )

    def boom(*a, **k):
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(analyzer, "get_interest_by_region", boom)
    monkeypatch.setattr(analyzer, "get_related_queries", boom)
    monkeypatch.setattr(analyzer, "get_suggestions", boom)

    result = analyzer.fetch_all(KEYWORDS, timeframe="today 12-m", geo="")

    assert not result["iot_df"].empty
    assert result["region_df"].empty
    assert result["related"] == {}
    assert result["suggestions"] == []
    assert len(result["warnings"]) == 2


def test_fetch_all_returns_plain_cacheable_data(monkeypatch):
    """st.cache_data pickles the return value, so it must be plain data."""
    import pickle

    analyzer = TrendsAnalyzer.__new__(TrendsAnalyzer)
    analyzer._last_request_time = 0.0
    monkeypatch.setattr(analyzer, "build_payload", lambda *a, **k: None)
    monkeypatch.setattr(
        analyzer, "get_interest_over_time",
        lambda: fake_iot().drop(columns=["isPartial"]),
    )
    monkeypatch.setattr(analyzer, "get_interest_by_region", fake_region)
    monkeypatch.setattr(analyzer, "get_related_queries", dict)
    monkeypatch.setattr(analyzer, "get_suggestions", list)

    result = analyzer.fetch_all(KEYWORDS[:1], timeframe="today 12-m", geo="")
    assert pickle.loads(pickle.dumps(result))["iot_df"].shape[0] == 60


def _offline_analyzer(monkeypatch, pytrends):
    monkeypatch.setattr("config.RETRY_DELAY", 0)
    analyzer = TrendsAnalyzer.__new__(TrendsAnalyzer)
    analyzer._last_request_time = 0.0
    analyzer.pytrends = pytrends
    monkeypatch.setattr(analyzer, "_rate_limit", lambda *a: None)
    return analyzer


def test_interest_over_time_retries_and_drops_partial_row(monkeypatch):
    """A 429 on the data call is retried, and the unfinished period is removed."""
    calls = []

    class FakeTrends:
        def interest_over_time(self):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("429")
            return fake_iot(n=10, partial=True)

    df = _offline_analyzer(monkeypatch, FakeTrends()).get_interest_over_time()
    assert len(calls) == 2
    assert len(df) == 9 and "isPartial" not in df.columns


def test_optional_fetch_errors_become_warnings(monkeypatch):
    """The real getters must raise, so fetch_all can tell the user."""
    class FakeTrends:
        def build_payload(self, *a, **k): pass
        def interest_over_time(self): return fake_iot(n=10)
        def interest_by_region(self, **k): raise RuntimeError("429")
        def related_queries(self): raise RuntimeError("429")
        def suggestions(self, keyword): raise RuntimeError("429")

    result = _offline_analyzer(monkeypatch, FakeTrends()).fetch_all(KEYWORDS)
    assert len(result["warnings"]) == 2
