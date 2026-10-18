"""
Data cleaning, transformation, and summary-statistics utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DataProcessor:
    """Stateless helpers for cleaning and summarising trend data."""

    # ── Interest over time ────────────────────────────────────────────────────

    @staticmethod
    def clean_interest_over_time(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the index is a DatetimeIndex and fill missing values with 0."""
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        return df.fillna(0)

    @staticmethod
    def compute_growth_rate(series: pd.Series) -> float:
        """
        Return the average per-period growth rate of *series* as a percentage.

        Fitting a line to log(values) and exponentiating the slope gives a
        geometric growth rate. Two things make this preferable to
        ``series.pct_change().mean()``:

        * Google Trends returns 0 for low-volume periods, and ``pct_change``
          divides by the previous value — a single 0 turns the mean into
          ``inf`` (which used to render as "inf% avg weekly change").
        * The arithmetic mean of percentage changes is upward biased: +50%
          followed by -50% averages to 0% but is really a 25% loss.

        Zero and missing periods are dropped before fitting. Returns 0.0 when
        fewer than two usable points remain.
        """
        if series is None or len(series) < 2:
            return 0.0

        values = pd.to_numeric(series, errors="coerce")
        # log() needs strictly positive input; 0-interest periods carry no
        # usable ratio information, so they are excluded from the fit.
        mask = (values > 0).to_numpy()
        if mask.sum() < 2:
            return 0.0

        # Fit against real elapsed position so gaps don't distort the slope.
        x = np.flatnonzero(mask).astype(float)
        slope = np.polyfit(x, np.log(values[mask].to_numpy(dtype=float)), 1)[0]

        rate = (np.exp(slope) - 1.0) * 100.0
        if not np.isfinite(rate):
            return 0.0
        return float(round(rate, 2))

    @staticmethod
    def compute_summary_stats(df: pd.DataFrame, keyword: str) -> dict:
        """
        Return a dict with current, peak, average, trend (avg % change per
        period), and the date of the peak for *keyword*.

        Returns ``{}`` when the keyword is absent or has no usable values.
        """
        if df is None or df.empty or keyword not in df.columns:
            return {}

        series = pd.to_numeric(df[keyword], errors="coerce").dropna()
        if series.empty:
            return {}

        peak_date = series.idxmax()
        return {
            "current":   int(series.iloc[-1]),
            "peak":      int(series.max()),
            "avg":       float(round(series.mean(), 1)),
            "trend":     DataProcessor.compute_growth_rate(series),
            "peak_date": peak_date.strftime("%b %d, %Y")
                         if hasattr(peak_date, "strftime") else str(peak_date),
        }

    @staticmethod
    def compute_rolling_average(series: pd.Series, window: int = 4) -> pd.Series:
        """Return a centred rolling mean of *series*."""
        return series.rolling(window=window, center=True, min_periods=1).mean()

    # ── Regional data ─────────────────────────────────────────────────────────

    @staticmethod
    def get_top_regions(df: pd.DataFrame, keyword: str, n: int = 20) -> pd.DataFrame:
        """Return the top *n* regions sorted by *keyword* interest (descending)."""
        if df is None or df.empty or keyword not in df.columns:
            return pd.DataFrame()
        return df.nlargest(n, keyword)
