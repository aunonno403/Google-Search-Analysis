"""
Trend forecasting module.

Primary method  : Holt-Winters Exponential Smoothing (statsmodels)
Fallback method : Polynomial Regression (numpy)

Both produce point forecasts and 95 % prediction intervals, clipped to the
Google Trends [0, 100] scale.

Note on interval semantics: these are *prediction* intervals (the range a
future observation is expected to fall in), not confidence intervals on the
mean. They also inherit the limits of the underlying data — Google Trends
values are a relative index rescaled so the window's peak is 100, not an
absolute search volume.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


class TrendForecaster:
    """
    Provides time-series forecasting for Google Trends interest data.

    Usage
    -----
    forecaster = TrendForecaster(forecast_periods=12)
    result = forecaster.get_best_forecast(series)   # pandas Series with DatetimeIndex
    """

    def __init__(self, forecast_periods: int = config.FORECAST_PERIODS):
        self.forecast_periods = forecast_periods

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _clip(arr: np.ndarray) -> np.ndarray:
        return np.clip(arr, 0.0, 100.0)

    # ── Frequency inference ───────────────────────────────────────────────────

    @staticmethod
    def infer_freq(series: pd.Series) -> str:
        """
        Infer a pandas frequency alias from a DatetimeIndex series.

        Uses the mean spacing in *seconds* rather than whole days — sub-daily
        Trends data (the "past hour" through "past 7 days" windows) has a
        spacing of 0 days, which previously collapsed every intraday series
        to a daily frequency.
        """
        if len(series) < 2 or not isinstance(series.index, pd.DatetimeIndex):
            return "W"

        span = series.index[-1] - series.index[0]
        seconds = span.total_seconds() / max(len(series) - 1, 1)

        if seconds <= 60 * 2:             # up to 2 min apart
            return "min"
        if seconds <= 60 * 15:            # "Past day" comes in 8-minute buckets
            return "8min"
        if seconds <= 3600 * 3:           # up to 3 h apart
            return "h"
        if seconds <= 86400 * 3:          # up to 3 days apart
            return "D"
        if seconds <= 86400 * 10:         # up to 10 days apart
            return "W"
        if seconds <= 86400 * 45:         # up to ~6 weeks apart
            return "MS"
        return "QS"

    @staticmethod
    def seasonal_period(freq: str) -> int:
        """Return the seasonal cycle length to try for a given frequency."""
        return config.SEASONAL_PERIODS.get(freq, 52)

    # ── Forecasting methods ───────────────────────────────────────────────────

    def fit_polynomial_trend(self, series: pd.Series, degree: int = 3) -> dict:
        """
        Fit a polynomial regression of *degree* to *series* and forecast
        *self.forecast_periods* periods ahead.
        """
        y = series.to_numpy(dtype=float)
        x = np.arange(len(y))
        coef = np.polyfit(x, y, degree)

        x_future   = np.arange(len(y), len(y) + self.forecast_periods)
        y_fitted   = np.polyval(coef, x)
        y_forecast = self._clip(np.polyval(coef, x_future))

        residuals    = y - y_fitted
        std_res      = np.std(residuals)
        margin       = 1.96 * std_res          # ~95 % prediction interval

        return {
            "fitted":    y_fitted,
            "forecast":  y_forecast,
            "lower_ci":  self._clip(y_forecast - margin),
            "upper_ci":  self._clip(y_forecast + margin),
            "method":    f"Polynomial Regression (degree {degree})",
        }

    def fit_exponential_smoothing(self, series: pd.Series) -> dict:
        """
        Fit Holt-Winters Exponential Smoothing.
        Falls back to polynomial regression if statsmodels is unavailable
        or the series is too short.
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore

            n = len(series)

            # Match the seasonal cycle to the actual spacing of the data. The
            # old code assumed 52 (weekly-yearly) for every series, which is
            # meaningless for hourly or daily windows.
            freq = self.infer_freq(series)
            sp = self.seasonal_period(freq)

            # Need at least two full cycles to estimate a seasonal component.
            use_seasonal = n >= sp * 2

            # statsmodels emits convergence and frequency warnings freely.
            # Suppress them only around the fit rather than globally at
            # import time, so the rest of the app keeps its warnings.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = ExponentialSmoothing(
                    series,
                    trend="add",
                    seasonal="add" if use_seasonal else None,
                    seasonal_periods=sp if use_seasonal else None,
                    initialization_method="estimated",
                )
                fit = model.fit(optimized=True, remove_bias=False)

                y_forecast = self._clip(fit.forecast(self.forecast_periods).values)
                y_fitted   = fit.fittedvalues.values

                # Simulation-based 95 % prediction interval
                try:
                    sims = fit.simulate(self.forecast_periods, repetitions=200, error="add")
                    lower_ci = self._clip(np.percentile(sims, 2.5,  axis=1))
                    upper_ci = self._clip(np.percentile(sims, 97.5, axis=1))
                except Exception:
                    std_res  = np.std(series.values - y_fitted)
                    margin   = 1.96 * std_res
                    lower_ci = self._clip(y_forecast - margin)
                    upper_ci = self._clip(y_forecast + margin)

            return {
                "fitted":    y_fitted,
                "forecast":  y_forecast,
                "lower_ci":  lower_ci,
                "upper_ci":  upper_ci,
                "seasonal":  sp if use_seasonal else None,
                "method":    "Holt-Winters Exponential Smoothing",
            }

        except Exception as exc:
            logger.info("Holt-Winters failed (%s); falling back to polynomial.", exc)
            return self.fit_polynomial_trend(series, degree=2)

    # ── Date generation ───────────────────────────────────────────────────────

    def generate_forecast_dates(
        self,
        last_date: pd.Timestamp,
        freq: str = "W",
    ) -> pd.DatetimeIndex:
        """
        Return *forecast_periods* future dates starting one period after
        *last_date*.

        Stepping forward by one offset before building the range avoids an
        off-by-one that used to appear with anchored frequencies. The previous
        approach — ``date_range(start=last_date, periods=n+1)[1:]`` — assumed
        ``last_date`` itself would be generated, but pandas only includes it
        when it happens to align to the anchor. For a mid-month date with
        freq="MS" the range began at the *next* month start, so slicing off
        the first element silently dropped a whole period.
        """
        offset = pd.tseries.frequencies.to_offset(freq)
        return pd.date_range(
            start=last_date + offset,
            periods=self.forecast_periods,
            freq=freq,
        )

    # ── Trend direction ───────────────────────────────────────────────────────

    @staticmethod
    def detect_trend_direction(series: pd.Series) -> str:
        """Return a human-readable label for the overall trend direction."""
        if len(series) < 4:
            return "Insufficient data"

        n       = len(series)
        window  = max(4, n // 5)
        recent  = series.iloc[-window:].mean()
        earlier = series.iloc[:window].mean()

        if earlier == 0:
            return "➡️ Stable"

        pct = (recent - earlier) / earlier * 100
        if pct > 10:
            return f"📈 Rising (+{pct:.1f}%)"
        elif pct < -10:
            return f"📉 Falling ({pct:.1f}%)"
        else:
            return f"➡️ Stable ({pct:+.1f}%)"

    # ── Best-effort entry point ───────────────────────────────────────────────

    def get_best_forecast(self, series: pd.Series) -> dict:
        """
        Try Holt-Winters first; fall back to polynomial regression.
        Always attaches a 'trend_direction' key to the result dict.
        """
        result = self.fit_exponential_smoothing(series)
        result["trend_direction"] = self.detect_trend_direction(series)
        return result
