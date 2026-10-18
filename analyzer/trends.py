"""
Pytrends data fetching wrapper with retry logic and rate limiting.
"""

from __future__ import annotations

import time
import logging

import pandas as pd
from pytrends.request import TrendReq

import config

logger = logging.getLogger(__name__)


class TrendsAnalyzer:
    """
    Robust wrapper around pytrends that adds:
      - Configurable retry logic with exponential back-off
      - Per-request rate limiting to avoid 429 errors
      - Consistent error logging
    """

    def __init__(self, hl: str = "en-US", tz: int = 360):
        # NOTE: Do NOT pass `retries` or `backoff_factor` to TrendReq.
        # pytrends forwards them to urllib3.Retry using the deprecated
        # `method_whitelist` kwarg, which was removed in urllib3 v2.0
        # (replaced by `allowed_methods`). Our own retry loop below
        # already handles all retries, so we don't need pytrends to do it.
        self.pytrends = TrendReq(
            hl=hl,
            tz=tz,
            timeout=(10, 25),
        )
        self._last_request_time: float = 0.0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _rate_limit(self, min_interval: float = 1.5) -> None:
        """Enforce a minimum interval between consecutive API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def _stamp(self) -> None:
        """Record the time of the last request."""
        self._last_request_time = time.time()

    def _retry(self, name: str, fn, *args, **kwargs):
        """Call *fn* with rate limiting, retrying with a growing delay."""
        for attempt in range(1, config.RETRY_ATTEMPTS + 1):
            try:
                self._rate_limit()
                return fn(*args, **kwargs)
            except Exception as exc:
                logger.warning("%s attempt %d/%d failed: %s", name, attempt, config.RETRY_ATTEMPTS, exc)
            finally:
                # Stamp failures too, so the rate limiter still spaces out
                # the retry attempts instead of measuring from the last
                # *successful* call.
                self._stamp()
            if attempt < config.RETRY_ATTEMPTS:
                wait = config.RETRY_DELAY * attempt
                logger.info("Waiting %ds before retry …", wait)
                time.sleep(wait)
        raise RuntimeError(
            f"{name} failed after {config.RETRY_ATTEMPTS} attempts. "
            "Google may be rate-limiting your IP. Please wait a minute and try again."
        )

    # ── Payload builder ───────────────────────────────────────────────────────

    def build_payload(
        self,
        kw_list: list[str],
        timeframe: str = config.DEFAULT_TIMEFRAME,
        geo: str = config.DEFAULT_GEO,
        cat: int = 0,
    ) -> None:
        """Build pytrends payload with retry logic."""
        self._retry(
            "build_payload", self.pytrends.build_payload,
            kw_list, cat=cat, timeframe=timeframe, geo=geo,
        )

    # ── Data fetchers ─────────────────────────────────────────────────────────
    #
    # The optional fetchers below let errors propagate: fetch_all() turns
    # them into user-visible warnings.

    def get_interest_over_time(self) -> pd.DataFrame:
        """Fetch interest-over-time data without the unfinished last period."""
        df = self._retry("interest_over_time", self.pytrends.interest_over_time)
        if df is None:
            return pd.DataFrame()
        if "isPartial" in df.columns:
            # The partial period (e.g. the current week) is incomplete and
            # reads low, which skews "current", the trend and the forecast.
            df = df[~df["isPartial"].astype(bool)].drop(columns=["isPartial"])
        return df

    def get_interest_by_region(self, resolution: str = "COUNTRY") -> pd.DataFrame:
        """Fetch interest-by-region data at the specified resolution."""
        self._rate_limit(2.0)
        try:
            df = self.pytrends.interest_by_region(
                resolution=resolution, inc_low_vol=True, inc_geo_code=True
            )
        finally:
            self._stamp()
        return df if df is not None else pd.DataFrame()

    def get_related_queries(self) -> dict:
        """Fetch related queries dict keyed by keyword."""
        self._rate_limit(2.0)
        try:
            return self.pytrends.related_queries() or {}
        finally:
            self._stamp()

    def get_suggestions(self, keyword: str) -> list:
        """Fetch autocomplete keyword suggestions for *keyword*."""
        self._rate_limit()
        try:
            return self.pytrends.suggestions(keyword=keyword) or []
        finally:
            self._stamp()

    # ── Orchestration ─────────────────────────────────────────────────────────

    def fetch_all(
        self,
        keywords: list[str],
        timeframe: str = config.DEFAULT_TIMEFRAME,
        geo: str = config.DEFAULT_GEO,
        suggestion_limit: int = 2,
    ) -> dict:
        """
        Run the full fetch sequence and return a plain dict of results.

        Only interest-over-time is required; the optional sections degrade to
        empty values with a recorded warning rather than failing the run.

        Pacing between calls is handled by ``_rate_limit`` alone — callers
        should not add their own sleeps on top.

        The returned dict is plain data (DataFrames, dicts, lists), which
        makes it safe to memoise with ``st.cache_data``.
        """
        self.build_payload(keywords, timeframe=timeframe, geo=geo)

        result: dict = {
            "iot_df":      self.get_interest_over_time(),
            "region_df":   pd.DataFrame(),
            "related":     {},
            "suggestions": [],
            "warnings":    [],
        }

        try:
            result["region_df"] = self.get_interest_by_region()
        except Exception as exc:
            logger.warning("interest_by_region failed: %s", exc)
            result["warnings"].append(f"Regional data unavailable: {exc}")

        try:
            result["related"] = self.get_related_queries()
        except Exception as exc:
            logger.warning("related_queries failed: %s", exc)
            result["warnings"].append(f"Related queries unavailable: {exc}")

        # Suggestions are the least important section and the most requests,
        # so only the first few keywords are looked up.
        for kw in keywords[:suggestion_limit]:
            try:
                suggestions = self.get_suggestions(kw)
                if suggestions:
                    result["suggestions"].append(
                        {"keyword": kw, "suggestions": suggestions}
                    )
            except Exception as exc:
                logger.warning("suggestions failed for '%s': %s", kw, exc)

        return result
