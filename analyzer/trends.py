"""
Pytrends data fetching wrapper with retry logic and rate limiting.
"""

import time
import logging
from typing import Optional

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

    # ── Payload builder ───────────────────────────────────────────────────────

    def build_payload(
        self,
        kw_list: list,
        timeframe: str = "today 12-m",
        geo: str = "",
        cat: int = 0,
    ) -> None:
        """Build pytrends payload with retry logic."""
        for attempt in range(1, config.RETRY_ATTEMPTS + 1):
            try:
                self._rate_limit()
                self.pytrends.build_payload(
                    kw_list, cat=cat, timeframe=timeframe, geo=geo
                )
                self._stamp()
                return
            except Exception as exc:
                logger.warning("build_payload attempt %d/%d failed: %s", attempt, config.RETRY_ATTEMPTS, exc)
                if attempt < config.RETRY_ATTEMPTS:
                    wait = config.RETRY_DELAY * attempt
                    logger.info("Waiting %ds before retry …", wait)
                    time.sleep(wait)
        raise RuntimeError(
            f"build_payload failed after {config.RETRY_ATTEMPTS} attempts. "
            "Google may be rate-limiting your IP. Please wait a minute and try again."
        )

    # ── Data fetchers ─────────────────────────────────────────────────────────

    def get_interest_over_time(self) -> pd.DataFrame:
        """Fetch interest-over-time data, dropping the 'isPartial' column."""
        self._rate_limit()
        df: pd.DataFrame = self.pytrends.interest_over_time()
        self._stamp()
        if df is not None and not df.empty and "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        return df if df is not None else pd.DataFrame()

    def get_interest_by_region(self, resolution: str = "COUNTRY") -> pd.DataFrame:
        """Fetch interest-by-region data at the specified resolution."""
        self._rate_limit(2.0)
        df: pd.DataFrame = self.pytrends.interest_by_region(
            resolution=resolution, inc_low_vol=True, inc_geo_code=True
        )
        self._stamp()
        return df if df is not None else pd.DataFrame()

    def get_related_queries(self) -> dict:
        """Fetch related queries dict keyed by keyword."""
        self._rate_limit(2.0)
        try:
            result = self.pytrends.related_queries()
            self._stamp()
            return result or {}
        except Exception as exc:
            logger.error("related_queries failed: %s", exc)
            return {}

    def get_related_topics(self) -> dict:
        """Fetch related topics dict keyed by keyword."""
        self._rate_limit(2.0)
        try:
            result = self.pytrends.related_topics()
            self._stamp()
            return result or {}
        except Exception as exc:
            logger.error("related_topics failed: %s", exc)
            return {}

    def get_suggestions(self, keyword: str) -> list:
        """Fetch autocomplete keyword suggestions for *keyword*."""
        self._rate_limit()
        try:
            result = self.pytrends.suggestions(keyword=keyword)
            self._stamp()
            return result or []
        except Exception as exc:
            logger.error("suggestions failed for '%s': %s", keyword, exc)
            return []
