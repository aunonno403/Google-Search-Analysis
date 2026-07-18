"""
Google Search Analysis — Streamlit Dashboard
=============================================
A fully interactive web dashboard for analysing Google Trends data.

Features
--------
• Generic keyword search (up to 5 keywords)
• Interest-over-time with range slider & smoothing
• Global choropleth heat-map + top-regions bar chart
• Multi-keyword radar & stacked-area comparison
• Statistical trend forecasting with 95 % confidence intervals
• Rising / top related queries
• CSV / data export

Run with:
    streamlit run main.py
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

import config
from analyzer.processor import DataProcessor
from analyzer.trends import TrendsAnalyzer
from forecaster.forecast import TrendForecaster
from visualizer.charts import ChartBuilder

# ─── Page setup (MUST be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Google Search Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Google Search Analysis · powered by pytrends & Streamlit"},
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Hide default chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }

/* App background */
.stApp {
    background: linear-gradient(140deg, #080b18 0%, #0e1228 55%, #0b1520 100%);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0f1e 0%, #10132a 100%);
    border-right: 1px solid rgba(108,99,255,0.18);
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c0b8ff;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.4rem;
}

/* ── Metric cards ── */
.gsa-card {
    background: linear-gradient(135deg,
        rgba(108,99,255,0.10) 0%,
        rgba(0,212,170,0.05) 100%);
    border: 1px solid rgba(108,99,255,0.22);
    border-radius: 18px;
    padding: 1.25rem 1.5rem 1rem;
    backdrop-filter: blur(12px);
    transition: transform 0.22s ease, box-shadow 0.22s ease;
    height: 100%;
}
.gsa-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 36px rgba(108,99,255,0.22);
}
.gsa-label {
    font-size: 0.73rem;
    color: #777;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 600;
}
.gsa-value {
    font-size: 2.4rem;
    font-weight: 800;
    color: #6c63ff;
    line-height: 1.05;
    margin: 0.25rem 0 0.15rem;
}
.gsa-sub {
    font-size: 0.8rem;
    color: #666;
    margin-top: 0.1rem;
}
.gsa-delta        { font-size: 0.88rem; font-weight: 600; margin-top: 0.35rem; }
.gsa-delta.up     { color: #00d4aa; }
.gsa-delta.down   { color: #ff6b6b; }
.gsa-delta.neutral{ color: #aaa; }

/* ── Hero ── */
.gsa-hero { text-align: center; padding: 2.2rem 1rem 1.4rem; }
.gsa-hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(130deg, #6c63ff 0%, #00d4aa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
    line-height: 1.1;
}
.gsa-hero p { color: #777; font-size: 1rem; margin: 0; }

/* ── Section headers ── */
.gsa-section {
    font-size: 1.05rem;
    font-weight: 600;
    color: #d4d0ff;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 14px;
    padding: 5px;
    border: 1px solid rgba(255,255,255,0.07);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #777;
    font-weight: 500;
    padding: 9px 22px;
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,
        rgba(108,99,255,0.35),
        rgba(0,212,170,0.18)) !important;
    color: #fff !important;
    border: 1px solid rgba(108,99,255,0.35) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff 0%, #5148d4 100%);
    color: #fff;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 0.65rem 1.6rem;
    transition: all 0.22s ease;
    width: 100%;
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #7c73ff 0%, #6c63ff 100%);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(108,99,255,0.45);
}

/* ── Inputs ── */
.stTextArea textarea,
.stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(108,99,255,0.22) !important;
    border-radius: 10px !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: rgba(108,99,255,0.55) !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.12) !important;
}

/* ── Keyword badges ── */
.gsa-badge {
    display: inline-block;
    background: linear-gradient(135deg,
        rgba(108,99,255,0.22),
        rgba(0,212,170,0.12));
    border: 1px solid rgba(108,99,255,0.32);
    border-radius: 20px;
    padding: 3px 13px;
    font-size: 0.82rem;
    font-weight: 500;
    color: #b0a8ff;
    margin: 2px 2px 0 0;
}

/* ── Feature card (welcome screen) ── */
.gsa-feat {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1.5rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.gsa-feat:hover { border-color: rgba(108,99,255,0.35); }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.8rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(108,99,255,0.35); border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Session state initialisation ─────────────────────────────────────────────
for key, default in {
    "analysis_done": False,
    "data": {},
    "keywords": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Hero header ──────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="gsa-hero">
  <h1>🔍 Google Search Analysis</h1>
  <p>Discover trends · Compare keywords · Forecast future interest — powered by Google Trends</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("---")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # --- Keywords ---
    st.markdown("### 🔤 Keywords")
    keywords_raw = st.text_area(
        "keywords",
        value="Artificial Intelligence\nMachine Learning\nDeep Learning",
        height=130,
        label_visibility="collapsed",
        placeholder="Enter up to 5 keywords, one per line",
    )
    keywords: list[str] = [
        k.strip() for k in keywords_raw.strip().splitlines() if k.strip()
    ][: config.MAX_KEYWORDS]

    if len([k for k in keywords_raw.strip().splitlines() if k.strip()]) > config.MAX_KEYWORDS:
        st.warning(f"Only the first {config.MAX_KEYWORDS} keywords will be used.")

    if keywords:
        badges = " ".join(f'<span class="gsa-badge">{k}</span>' for k in keywords)
        st.markdown(badges, unsafe_allow_html=True)

    st.markdown("---")

    # --- Timeframe ---
    st.markdown("### 📅 Timeframe")
    tf_label: str = st.selectbox(
        "Timeframe",
        options=list(config.TIMEFRAME_OPTIONS),
        index=6,
        label_visibility="collapsed",
    )
    timeframe: str = config.TIMEFRAME_OPTIONS[tf_label]

    use_custom = st.checkbox("Custom date range", value=False)
    if use_custom:
        c1, c2 = st.columns(2)
        start_d = c1.date_input("Start", value=date.today() - timedelta(days=365))
        end_d   = c2.date_input("End",   value=date.today())
        if start_d < end_d:
            timeframe = f"{start_d} {end_d}"
            tf_label  = f"{start_d} → {end_d}"
        else:
            st.error("Start must be before End.")

    st.markdown("---")

    # --- Geography ---
    st.markdown("### 🌍 Geography")
    geo_label: str = st.selectbox(
        "Region",
        options=list(config.GEO_OPTIONS),
        index=0,
        label_visibility="collapsed",
    )
    geo: str = config.GEO_OPTIONS[geo_label]

    st.markdown("---")

    # --- Options ---
    st.markdown("### 🔧 Options")
    smooth       = st.checkbox("Smooth trend lines", value=True)
    forecast_n   = st.slider("Forecast periods (weeks)", 4, 52, 12, 4)

    st.markdown("---")

    run_analysis = st.button("🚀 Analyse Trends", use_container_width=True)

    st.markdown(
        '<div style="color:#444;font-size:0.72rem;text-align:center;margin-top:0.5rem;">'
        "Powered by Google Trends · pytrends · Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )

# ─── Analysis execution ───────────────────────────────────────────────────────
if run_analysis:
    if not keywords:
        st.error("Please enter at least one keyword.")
    else:
        with st.spinner("🔄 Fetching data from Google Trends …"):
            try:
                analyzer  = TrendsAnalyzer()
                processor = DataProcessor()

                # Interest over time
                analyzer.build_payload(keywords, timeframe=timeframe, geo=geo)
                time.sleep(1)
                iot_df = analyzer.get_interest_over_time()

                store: dict = {
                    "keywords":  keywords,
                    "tf_label":  tf_label,
                    "geo_label": geo_label,
                    "iot_df":    iot_df,
                    "region_df": pd.DataFrame(),
                    "related":   {},
                    "suggestions": [],
                }

                # Interest by region
                try:
                    time.sleep(2)
                    store["region_df"] = analyzer.get_interest_by_region()
                except Exception as exc:
                    st.warning(f"Regional data unavailable: {exc}")

                # Related queries
                try:
                    time.sleep(2)
                    store["related"] = analyzer.get_related_queries()
                except Exception as exc:
                    logger.warning("Related queries failed: %s", exc)

                # Keyword suggestions (first 2 keywords only to avoid rate limits)
                for kw in keywords[:2]:
                    try:
                        time.sleep(1)
                        suggs = analyzer.get_suggestions(kw)
                        if suggs:
                            store["suggestions"].append({"keyword": kw, "suggestions": suggs})
                    except Exception:
                        pass

                st.session_state.data          = store
                st.session_state.keywords      = keywords
                st.session_state.analysis_done = True
                st.success("✅ Data loaded successfully!")

            except Exception as exc:
                st.error(f"❌ Analysis failed: {exc}")
                st.info(
                    "💡 Google Trends may be rate-limiting your IP. "
                    "Wait 60 seconds and try again."
                )
                st.session_state.analysis_done = False

# ─── Results display ──────────────────────────────────────────────────────────
if st.session_state.analysis_done and st.session_state.data:
    store     = st.session_state.data
    keywords  = st.session_state.keywords
    iot_df    = store.get("iot_df", pd.DataFrame())
    region_df = store.get("region_df", pd.DataFrame())
    processor = DataProcessor()

    # ── Summary metric cards ───────────────────────────────────────────────────
    if not iot_df.empty:
        st.markdown("### 📊 Snapshot")
        cols = st.columns(len(keywords))
        for i, kw in enumerate(keywords):
            if kw not in iot_df.columns:
                continue
            stats = processor.compute_summary_stats(iot_df, kw)
            trend = stats.get("trend", 0.0)
            dc    = "up" if trend > 0 else ("down" if trend < 0 else "neutral")
            sym   = "▲" if trend > 0 else ("▼" if trend < 0 else "→")
            with cols[i]:
                st.markdown(
                    f"""
<div class="gsa-card">
  <div class="gsa-label">{kw}</div>
  <div class="gsa-value">{stats.get('current','–')}</div>
  <div class="gsa-sub">Peak: <b>{stats.get('peak','–')}</b> on {stats.get('peak_date','–')}</div>
  <div class="gsa-sub">Avg: {stats.get('avg','–')}</div>
  <div class="gsa-delta {dc}">{sym} {abs(trend):.2f}% avg weekly change</div>
</div>""",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs(
        ["📈 Interest Over Time", "🌍 Regional", "⚖️ Comparison", "🔮 Forecast", "🔍 Related"]
    )

    # ── Tab 1 · Interest Over Time ────────────────────────────────────────────
    with t1:
        if not iot_df.empty:
            st.markdown('<div class="gsa-section">📈 Search Interest Over Time</div>', unsafe_allow_html=True)
            st.plotly_chart(
                ChartBuilder.interest_over_time(iot_df, keywords, smooth=smooth),
                use_container_width=True,
            )
            with st.expander("📋 Raw Data & Download"):
                st.dataframe(iot_df.style.background_gradient(cmap="Purples"), use_container_width=True)
                st.download_button(
                    "⬇️ Download CSV",
                    data=iot_df.to_csv(),
                    file_name=f"gsa_iot_{'_'.join(k.replace(' ','_') for k in keywords[:2])}.csv",
                    mime="text/csv",
                )
        else:
            st.warning("No interest-over-time data. Try different keywords or a longer timeframe.")

    # ── Tab 2 · Regional ─────────────────────────────────────────────────────
    with t2:
        if not region_df.empty:
            valid_kws = [k for k in keywords if k in region_df.columns]
            sel_kw = st.selectbox("Keyword for regional view", valid_kws, key="reg_kw")

            st.markdown('<div class="gsa-section">🌍 Global Interest Map</div>', unsafe_allow_html=True)
            st.plotly_chart(ChartBuilder.choropleth_map(region_df, sel_kw), use_container_width=True)

            top20 = processor.get_top_regions(region_df, sel_kw, n=20)
            st.markdown('<div class="gsa-section">🏆 Top 20 Regions</div>', unsafe_allow_html=True)
            st.plotly_chart(ChartBuilder.top_regions_bar(top20, sel_kw), use_container_width=True)

            with st.expander("📋 Full Regional Data"):
                st.dataframe(
                    region_df.sort_values(by=sel_kw, ascending=False).style.background_gradient(cmap="Purples"),
                    use_container_width=True,
                )
        else:
            st.warning("Regional data not available for this query.")

    # ── Tab 3 · Comparison ────────────────────────────────────────────────────
    with t3:
        if not iot_df.empty:
            avail = [k for k in keywords if k in iot_df.columns]
            if len(avail) > 1:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="gsa-section">📡 Radar Comparison</div>', unsafe_allow_html=True)
                    st.plotly_chart(ChartBuilder.comparison_radar(iot_df, avail), use_container_width=True)
                with c2:
                    st.markdown('<div class="gsa-section">📊 Share of Interest Over Time</div>', unsafe_allow_html=True)
                    st.plotly_chart(ChartBuilder.stacked_area(iot_df, avail), use_container_width=True)

                # Stats table
                st.markdown('<div class="gsa-section">📋 Comparison Summary</div>', unsafe_allow_html=True)
                rows = []
                for kw in avail:
                    s = processor.compute_summary_stats(iot_df, kw)
                    rows.append({
                        "Keyword":   kw,
                        "Current":   s.get("current", 0),
                        "Peak":      s.get("peak", 0),
                        "Average":   s.get("avg", 0),
                        "Peak Date": s.get("peak_date", "–"),
                        "Weekly Δ":  f"{s.get('trend', 0):+.2f}%",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Enter 2–5 keywords to enable comparison charts.")

    # ── Tab 4 · Forecast ──────────────────────────────────────────────────────
    with t4:
        if not iot_df.empty:
            avail_fc = [k for k in keywords if k in iot_df.columns]
            fc_kw    = st.selectbox("Keyword to forecast", avail_fc, key="fc_kw")

            if fc_kw:
                series = iot_df[fc_kw].dropna()
                if len(series) >= 10:
                    with st.spinner(f"🧮 Computing forecast for '{fc_kw}' …"):
                        forecaster   = TrendForecaster(forecast_periods=forecast_n)
                        result       = forecaster.get_best_forecast(series)
                        freq         = forecaster.infer_freq(series)
                        future_dates = forecaster.generate_forecast_dates(series.index[-1], freq=freq)

                    st.markdown('<div class="gsa-section">🔮 Trend Forecast with Confidence Interval</div>', unsafe_allow_html=True)
                    st.plotly_chart(
                        ChartBuilder.forecast_chart(
                            historical_dates=series.index,
                            historical_values=series.values,
                            forecast_dates=future_dates,
                            forecast_values=result["forecast"],
                            lower_ci=result["lower_ci"],
                            upper_ci=result["upper_ci"],
                            keyword=fc_kw,
                            method=result.get("method", "Statistical Model"),
                        ),
                        use_container_width=True,
                    )

                    # Insight cards
                    ic1, ic2, ic3 = st.columns(3)
                    avg_fc = float(np.mean(result["forecast"]))
                    with ic1:
                        st.markdown(
                            f'<div class="gsa-card"><div class="gsa-label">Trend Direction</div>'
                            f'<div style="font-size:1.25rem;font-weight:700;color:#e0e0e0;margin:.5rem 0;">'
                            f'{result.get("trend_direction","–")}</div></div>',
                            unsafe_allow_html=True,
                        )
                    with ic2:
                        st.markdown(
                            f'<div class="gsa-card"><div class="gsa-label">Avg Forecasted Interest</div>'
                            f'<div class="gsa-value">{avg_fc:.1f}</div></div>',
                            unsafe_allow_html=True,
                        )
                    with ic3:
                        st.markdown(
                            f'<div class="gsa-card"><div class="gsa-label">Method</div>'
                            f'<div style="font-size:0.88rem;font-weight:600;color:#00d4aa;margin:.5rem 0;">'
                            f'{result.get("method","–")}</div></div>',
                            unsafe_allow_html=True,
                        )

                    with st.expander("📋 Forecast Data & Download"):
                        fc_df = pd.DataFrame({
                            "Date":         future_dates,
                            "Forecast":     np.round(result["forecast"], 1),
                            "Lower CI (95%)": np.round(result["lower_ci"], 1),
                            "Upper CI (95%)": np.round(result["upper_ci"], 1),
                        })
                        st.dataframe(fc_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "⬇️ Download Forecast CSV",
                            data=fc_df.to_csv(index=False),
                            file_name=f"forecast_{fc_kw.replace(' ','_')}.csv",
                            mime="text/csv",
                        )
                else:
                    st.warning(
                        f"Not enough data to forecast '{fc_kw}'. "
                        "Try a longer timeframe (e.g. Past 5 years)."
                    )
        else:
            st.warning("No data available for forecasting.")

    # ── Tab 5 · Related Queries ───────────────────────────────────────────────
    with t5:
        related = store.get("related", {})
        if related:
            valid_rel = [k for k in keywords if k in related and related[k]]
            if valid_rel:
                rel_kw = st.selectbox("Keyword", valid_rel, key="rel_kw")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="gsa-section">🔥 Top Queries</div>', unsafe_allow_html=True)
                    fig_top = ChartBuilder.related_queries_chart(related, rel_kw, "top")
                    if fig_top.data:
                        st.plotly_chart(fig_top, use_container_width=True)
                    else:
                        st.info("No top queries found.")
                with c2:
                    st.markdown('<div class="gsa-section">🚀 Rising Queries</div>', unsafe_allow_html=True)
                    fig_rise = ChartBuilder.related_queries_chart(related, rel_kw, "rising")
                    if fig_rise.data:
                        st.plotly_chart(fig_rise, use_container_width=True)
                    else:
                        st.info("No rising queries found.")
            else:
                st.info("No related queries available for the selected keywords.")
        else:
            st.info("Related queries data was not fetched.")

        # Keyword suggestions
        suggestions = store.get("suggestions", [])
        if suggestions:
            st.markdown("---")
            st.markdown('<div class="gsa-section">💡 Keyword Suggestions</div>', unsafe_allow_html=True)
            for item in suggestions:
                with st.expander(f"Suggestions for **{item['keyword']}**"):
                    df_s = pd.DataFrame(item["suggestions"]).drop(columns=["mid"], errors="ignore")
                    st.dataframe(df_s, use_container_width=True, hide_index=True)

# ─── Welcome / empty state ─────────────────────────────────────────────────────
else:
    st.markdown(
        """
<div style="text-align:center;padding:3.5rem 2rem 2rem;">
  <div style="font-size:5.5rem;margin-bottom:1.2rem;">📊</div>
  <h2 style="color:#e0e0e0;font-size:1.9rem;font-weight:700;margin-bottom:.7rem;">
    Ready to Analyse
  </h2>
  <p style="color:#555;font-size:1rem;max-width:480px;margin:0 auto;line-height:1.7;">
    Type your keywords in the sidebar, pick a timeframe &amp; region,
    then hit <strong style="color:#6c63ff;">Analyse Trends</strong> to explore the data.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    feats = [
        ("📈", "Trend Analysis",   "Track how search interest evolves over any timeframe for any topic"),
        ("🌍", "Global Heatmap",   "See which countries drive search interest with an interactive world map"),
        ("🔮", "AI Forecasting",   "Predict future search interest using Holt-Winters & polynomial models"),
        ("⚖️",  "Multi-Keyword",   "Compare up to 5 keywords with radar charts and share-of-interest view"),
        ("🔍", "Related Queries",  "Uncover rising and top queries associated with your keywords"),
        ("⬇️", "Data Export",     "Download raw and forecast data as CSV with one click"),
    ]
    cols = st.columns(3)
    for idx, (icon, title, desc) in enumerate(feats):
        with cols[idx % 3]:
            st.markdown(
                f"""
<div class="gsa-feat">
  <div style="font-size:2.4rem;margin-bottom:.7rem;">{icon}</div>
  <div style="font-size:1rem;font-weight:600;color:#e0e0e0;margin-bottom:.45rem;">{title}</div>
  <div style="font-size:0.82rem;color:#555;line-height:1.55;">{desc}</div>
</div>""",
                unsafe_allow_html=True,
            )
        if idx == 2:
            st.markdown("")  # small spacer between rows