"""
Trend forecasting module.

Primary method  : Holt-Winters Exponential Smoothing (statsmodels)
Fallback method : Polynomial Regression (scikit-learn / numpy)

Both produce point forecasts and 95 % confidence / prediction intervals,
clipped to the Google Trends [0, 100] scale.
"""

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")


class TrendForecaster:
    """
    Provides time-series forecasting for Google Trends interest data.

    Usage
    -----
    forecaster = TrendForecaster(forecast_periods=12)
    result = forecaster.get_best_forecast(series)   # pandas Series with DatetimeIndex
    """

    def __init__(self, forecast_periods: int = 12):
        self.forecast_periods = forecast_periods

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_array(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Return integer-indexed X and the raw values y."""
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values.astype(float)
        return X, y

    @staticmethod
    def _clip(arr: np.ndarray) -> np.ndarray:
        return np.clip(arr, 0.0, 100.0)

    # ── Forecasting methods ───────────────────────────────────────────────────

    def fit_polynomial_trend(self, series: pd.Series, degree: int = 3) -> dict:
        """
        Fit a polynomial regression of *degree* to *series* and forecast
        *self.forecast_periods* periods ahead.
        """
        X, y = self._to_array(series)

        pipe = Pipeline([
            ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
            ("linear", LinearRegression()),
        ])
        pipe.fit(X, y)

        X_future = np.arange(len(series), len(series) + self.forecast_periods).reshape(-1, 1)
        y_fitted   = pipe.predict(X)
        y_forecast = self._clip(pipe.predict(X_future))

        residuals    = y - y_fitted
        std_res      = np.std(residuals)
        margin       = 1.96 * std_res          # ~95 % prediction interval

        return {
            "fitted":    y_fitted,
            "forecast":  y_forecast,
            "lower_ci":  self._clip(y_forecast - margin),
            "upper_ci":  self._clip(y_forecast + margin),
            "r2":        float(r2_score(y, y_fitted)),
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

            # Infer a seasonal period from the DatetimeIndex frequency
            freq = pd.infer_freq(series.index) if isinstance(series.index, pd.DatetimeIndex) else None
            if freq and freq.startswith("W"):
                sp = 52
            elif freq and freq.startswith(("M", "Q")):
                sp = 12
            else:
                sp = 52  # default: weekly

            use_seasonal = (n >= sp * 2)

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
                "aic":       float(fit.aic),
                "method":    "Holt-Winters Exponential Smoothing",
            }

        except Exception:
            # Graceful fallback
            return self.fit_polynomial_trend(series, degree=2)

    # ── Date generation ───────────────────────────────────────────────────────

    def generate_forecast_dates(
        self,
        last_date: pd.Timestamp,
        freq: str = "W",
    ) -> pd.DatetimeIndex:
        """Return *forecast_periods* future dates starting after *last_date*."""
        return pd.date_range(
            start=last_date,
            periods=self.forecast_periods + 1,
            freq=freq,
        )[1:]

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
        try:
            result = self.fit_exponential_smoothing(series)
        except Exception:
            result = self.fit_polynomial_trend(series, degree=2)

        result["trend_direction"] = self.detect_trend_direction(series)
        return result

    # ── Frequency inference ───────────────────────────────────────────────────

    @staticmethod
    def infer_freq(series: pd.Series) -> str:
        """Infer pandas frequency string from a DatetimeIndex series."""
        if len(series) < 2 or not isinstance(series.index, pd.DatetimeIndex):
            return "W"
        delta = (series.index[-1] - series.index[0]) / max(len(series) - 1, 1)
        days  = delta.days
        if days <= 1:
            return "D"
        elif days <= 7:
            return "W"
        elif days <= 31:
            return "MS"
        else:
            return "QS"
