"""
Data cleaning, transformation, and summary-statistics utilities.
"""

import pandas as pd
import numpy as np
from typing import Optional


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
    def compute_summary_stats(df: pd.DataFrame, keyword: str) -> dict:
        """
        Return a dict with current, peak, average, trend (avg % weekly change),
        and the date of the peak for *keyword*.
        """
        if df is None or df.empty or keyword not in df.columns:
            return {}
        series = df[keyword]
        peak_date = series.idxmax()
        return {
            "current":   int(series.iloc[-1]),
            "peak":      int(series.max()),
            "avg":       float(round(series.mean(), 1)),
            "trend":     float(round(series.pct_change().mean() * 100, 2)),
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

    # ── Normalisation ─────────────────────────────────────────────────────────

    @staticmethod
    def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Rescale every column to [0, 100] independently."""
        if df is None or df.empty:
            return df
        result = df.copy()
        for col in result.columns:
            col_max = result[col].max()
            if col_max > 0:
                result[col] = (result[col] / col_max * 100).round(1)
        return result
