"""Central configuration for Google Search Analysis Dashboard."""

# ─── Default Settings ──────────────────────────────────────────────────────────
DEFAULT_KEYWORDS = ["Python", "Machine Learning", "Data Science"]
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
RETRY_DELAY = 30          # seconds between retries on rate limit

# ─── Forecast Settings ─────────────────────────────────────────────────────────
FORECAST_PERIODS = 12     # default number of periods to forecast forward
FORECAST_CONFIDENCE = 0.95

# ─── Plotly / Chart Theme ──────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
PRIMARY_COLOR   = "#6c63ff"
SECONDARY_COLOR = "#00d4aa"
WARNING_COLOR   = "#ffa726"
DANGER_COLOR    = "#ff6b6b"
