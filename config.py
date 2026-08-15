"""Central configuration for Google Search Analysis Dashboard."""

from __future__ import annotations

import re
from datetime import date

# ─── Default Settings ──────────────────────────────────────────────────────────
DEFAULT_KEYWORDS = ["Artificial Intelligence", "Machine Learning", "Deep Learning"]
DEFAULT_TIMEFRAME = "today 12-m"
DEFAULT_GEO = ""  # empty = worldwide

# ─── Timeframe Options ─────────────────────────────────────────────────────────
TIMEFRAME_OPTIONS = {
    "Past hour":      "now 1-H",
    "Past 4 hours":   "now 4-H",
    "Past day":       "now 1-d",
    "Past 7 days":    "now 7-d",
    "Past 30 days":   "today 1-m",
    "Past 90 days":   "today 3-m",
    "Past 12 months": "today 12-m",
    "Past 5 years":   "today 5-y",
    "2004 – present": "all",
}

# ─── Country / Region Options ──────────────────────────────────────────────────
GEO_OPTIONS = {
    "Worldwide":    "",
    "United States":"US",
    "United Kingdom":"GB",
    "India":        "IN",
    "Canada":       "CA",
    "Australia":    "AU",
    "Germany":      "DE",
    "France":       "FR",
    "Brazil":       "BR",
    "Japan":        "JP",
    "China":        "CN",
    "South Korea":  "KR",
    "Russia":       "RU",
    "Mexico":       "MX",
    "Indonesia":    "ID",
    "Italy":        "IT",
    "Spain":        "ES",
    "Netherlands":  "NL",
    "Sweden":       "SE",
    "Singapore":    "SG",
    "South Africa": "ZA",
    "Nigeria":      "NG",
    "Pakistan":     "PK",
    "Bangladesh":   "BD",
    "Argentina":    "AR",
}

# ─── Analysis Settings ─────────────────────────────────────────────────────────
MAX_KEYWORDS = 5
RETRY_ATTEMPTS = 3
RETRY_DELAY = 15          # seconds between retries on rate limit (grows per attempt)
TOP_REGIONS_N = 20        # how many regions the "top regions" bar chart shows
CACHE_TTL = 3600          # seconds a fetched result stays cached

# ─── Forecast Settings ─────────────────────────────────────────────────────────
FORECAST_PERIODS = 12     # default number of periods to forecast forward
FORECAST_MIN_PERIODS = 4
FORECAST_MAX_PERIODS = 52
FORECAST_CONFIDENCE = 0.95

# ─── Plotly / Chart Theme ──────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
PRIMARY_COLOR   = "#6c63ff"
SECONDARY_COLOR = "#00d4aa"
WARNING_COLOR   = "#ffa726"
DANGER_COLOR    = "#ff6b6b"

# Extra series colours, appended after the four themed ones above.
EXTRA_SERIES_COLORS = ["#42a5f5", "#ab47bc", "#66bb6a", "#ef5350"]


# ─── Timeframe → data granularity ──────────────────────────────────────────────
#
# Google Trends silently changes the resolution of the series it returns based
# on how wide the requested window is. Everything downstream (forecast horizon
# labels, forecast date generation, seasonal periods) depends on knowing which
# resolution to expect, so the rules live here in one place.
#
# Human-readable units for each pandas frequency alias, in one place so the
# slider label, the metric cards, and the forecast heading always agree.
FREQ_UNITS = {
    "min": ("minute",  "minutes"),
    "h":   ("hour",    "hours"),
    "D":   ("day",     "days"),
    "W":   ("week",    "weeks"),
    "MS":  ("month",   "months"),
    "QS":  ("quarter", "quarters"),
}


def units_for_freq(freq: str) -> tuple[str, str]:
    """Return ``(singular, plural)`` unit names for a pandas frequency alias."""
    return FREQ_UNITS.get(freq, ("period", "periods"))


# Each entry is (max_span_in_days, pandas_freq_alias).
_GRANULARITY_RULES = [
    (0.21,    "min"),   # < ~5 hours  → 1-minute buckets
    (1.5,     "min"),   # < ~36 hours → 8-minute buckets
    (8.0,     "h"),     # < 8 days    → hourly
    (270.0,   "D"),     # < ~9 months → daily
    (1826.0,  "W"),     # < ~5 years  → weekly
]
_GRANULARITY_FALLBACK = "MS"  # ≥ 5 years → monthly

_PRESET_SPAN_DAYS = {
    "now 1-H":   1 / 24,
    "now 4-H":   4 / 24,
    "now 1-d":   1.0,
    "now 7-d":   7.0,
    "today 1-m": 30.0,
    "today 3-m": 90.0,
    "today 12-m": 365.0,
    "today 5-y": 1825.0,
    "all":       99999.0,
}

_CUSTOM_RANGE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+(\d{4}-\d{2}-\d{2})$")


def timeframe_span_days(timeframe: str) -> float:
    """
    Return the width of a Google Trends *timeframe* string in days.

    Handles both the presets in TIMEFRAME_OPTIONS and custom
    "YYYY-MM-DD YYYY-MM-DD" ranges. Unknown strings fall back to 365 days.
    """
    if timeframe in _PRESET_SPAN_DAYS:
        return _PRESET_SPAN_DAYS[timeframe]

    match = _CUSTOM_RANGE_RE.match(timeframe.strip())
    if match:
        start = date.fromisoformat(match.group(1))
        end = date.fromisoformat(match.group(2))
        return max((end - start).days, 1)

    return 365.0


def granularity_for(timeframe: str) -> tuple[str, str, str]:
    """
    Return ``(pandas_freq, unit_singular, unit_plural)`` for a *timeframe*.

    This lets the UI label the forecast horizon in the units Google will
    actually return, instead of assuming weeks.
    """
    span = timeframe_span_days(timeframe)
    freq = next(
        (f for max_span, f in _GRANULARITY_RULES if span <= max_span),
        _GRANULARITY_FALLBACK,
    )
    singular, plural = units_for_freq(freq)
    return freq, singular, plural


# Seasonal cycle length to try for each granularity, used by the forecaster.
SEASONAL_PERIODS = {
    "min": 60,    # an hour of minute buckets
    "h":   24,    # daily cycle
    "D":   7,     # weekly cycle
    "W":   52,    # yearly cycle
    "MS":  12,    # yearly cycle
    "QS":  4,
}
