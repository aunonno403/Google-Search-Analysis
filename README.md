# Google Search Analysis

An interactive Streamlit dashboard for exploring Google Trends data: track
search interest over time, compare keywords, map regional interest, and project
near-term trends.

## Quick start

```bash
pip install -r requirements.txt
```

```bash
streamlit run main.py
```

Then open <http://localhost:8501>, enter up to five keywords in the sidebar,
pick a timeframe and region, and click **Analyse Trends**.

## Project structure

```
main.py                 Streamlit dashboard (UI and layout only)
config.py               Settings, timeframe presets, granularity rules
requirements.txt        Runtime dependencies
requirements-dev.txt    Runtime + test dependencies

analyzer/
  trends.py             pytrends wrapper: retries, rate limiting, fetch_all()
  processor.py          Cleaning and summary statistics
  geo.py                ISO 3166-1 alpha-2 to alpha-3 country codes

forecaster/
  forecast.py           Holt-Winters with polynomial-regression fallback

visualizer/
  charts.py             Plotly figure builders

tests/                  pytest suite
```

## The tabs

| Tab | Shows |
|-----|-------|
| Interest Over Time | Multi-keyword line chart with smoothing, range slider, CSV export |
| Regional | World choropleth plus a top-regions bar chart |
| Comparison | Radar chart, share-of-interest area chart, summary table |
| Forecast | Projection with a 95% prediction band and downloadable data |
| Related | Top and rising related queries, plus keyword suggestions |

## How the forecast works

Two methods are tried in order:

1. **Holt-Winters exponential smoothing** (primary) — captures trend and, when
   there are at least two full cycles of data, seasonality.
2. **Polynomial regression** (fallback) — works at any data size.

The seasonal cycle length is chosen from the data's actual frequency (24 for
hourly, 7 for daily, 52 for weekly, 12 for monthly), and the forecast horizon
is expressed in whatever unit the selected timeframe returns.

### Reading the output honestly

Google Trends does not report search volume. It reports a **relative index
rescaled so that the peak of the requested window is 100**. Two consequences:

- The same keyword produces different numbers depending on the timeframe,
  region, and which other keywords it was queried alongside.
- The shaded band is a **prediction interval** — where a future observation is
  expected to land — not a confidence interval on a mean, and it inherits all
  the limits above.

Treat the forecast as a direction, not a traffic estimate.

## Timeframes and data granularity

Google changes the resolution of the returned series based on how wide the
window is. `config.granularity_for()` encodes the rules, and the UI labels the
forecast horizon accordingly:

| Timeframe | Resolution |
|-----------|-----------|
| Past hour / 4 hours | minutes |
| Past day / 7 days | hours |
| Past 30 / 90 days | days |
| Past 12 months / 5 years | weeks |
| 2004 – present | months |

## Caching and rate limits

Google Trends rate-limits aggressively and pytrends talks to undocumented
endpoints, so HTTP 429 responses are common. The app mitigates this by:

- Memoising each `(keywords, timeframe, geo)` query for one hour
  (`config.CACHE_TTL`), so reruns and tab switches cost no requests.
- Spacing requests inside `TrendsAnalyzer` with a minimum interval.
- Retrying `build_payload` with a growing delay.

If you still get rate-limited, wait a minute and try again. Cached results stay
available in the meantime.

## Running the tests

```bash
pip install -r requirements-dev.txt && python -m pytest tests/ -q
```

The suite runs entirely offline — network calls are stubbed — and covers the
statistics, frequency inference, forecast date generation, chart construction,
and the end-to-end pipeline at every granularity.

## Known limitations

- **pytrends depends on undocumented Google endpoints.** They change without
  notice, and rate limiting has grown stricter over time. If fetching starts
  failing consistently, the data source rather than this code is the likely
  cause; `TrendsAnalyzer.fetch_all()` is the single seam to swap.
- The radar chart plots non-commensurate quantities (an interest index and a
  dispersion measure) on one 0–100 axis. Read it as a shape comparison, not a
  score.
- Keyword suggestions are fetched for the first two keywords only, to limit
  request volume.
