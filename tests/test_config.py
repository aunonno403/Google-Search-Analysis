"""Tests for config — timeframe span parsing and granularity mapping."""

import pytest

import config


class TestTimeframeSpanDays:
    @pytest.mark.parametrize("timeframe,expected", [
        ("now 1-H",    1 / 24),
        ("now 7-d",    7.0),
        ("today 1-m",  30.0),
        ("today 12-m", 365.0),
        ("today 5-y",  1825.0),
    ])
    def test_presets(self, timeframe, expected):
        assert config.timeframe_span_days(timeframe) == pytest.approx(expected)

    def test_custom_range(self):
        assert config.timeframe_span_days("2023-01-01 2023-01-31") == 30

    def test_unknown_falls_back_to_a_year(self):
        assert config.timeframe_span_days("nonsense") == 365.0

    def test_every_preset_is_covered(self):
        """A new entry in TIMEFRAME_OPTIONS must not silently hit the fallback."""
        for timeframe in config.TIMEFRAME_OPTIONS.values():
            assert timeframe in config._PRESET_SPAN_DAYS, timeframe


class TestGranularityFor:
    @pytest.mark.parametrize("timeframe,freq", [
        ("now 1-H",     "min"),
        ("now 4-H",     "min"),
        ("now 7-d",     "h"),
        ("today 1-m",   "D"),
        ("today 3-m",   "D"),
        ("today 12-m",  "W"),
        ("today 5-y",   "W"),
        ("all",         "MS"),
    ])
    def test_matches_google_resolution(self, timeframe, freq):
        """
        Google changes the resolution of the returned series with the width
        of the window. The forecast horizon label depends on getting this
        right — it used to always claim "weeks".
        """
        assert config.granularity_for(timeframe)[0] == freq

    def test_returns_singular_and_plural(self):
        freq, one, many = config.granularity_for("today 12-m")
        assert (freq, one, many) == ("W", "week", "weeks")

    def test_custom_short_range_is_daily(self):
        assert config.granularity_for("2023-01-01 2023-03-01")[0] == "D"

    def test_custom_long_range_is_monthly(self):
        assert config.granularity_for("2010-01-01 2023-01-01")[0] == "MS"


class TestUnitsForFreq:
    def test_known(self):
        assert config.units_for_freq("W") == ("week", "weeks")

    def test_unknown_has_a_generic_fallback(self):
        assert config.units_for_freq("???") == ("period", "periods")

    def test_every_granularity_freq_has_units(self):
        for _, freq in config._GRANULARITY_RULES:
            assert freq in config.FREQ_UNITS, freq
        assert config._GRANULARITY_FALLBACK in config.FREQ_UNITS

    def test_every_freq_has_a_seasonal_period(self):
        for freq in config.FREQ_UNITS:
            assert freq in config.SEASONAL_PERIODS, freq
