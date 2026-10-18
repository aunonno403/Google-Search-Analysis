"""
Interactive Plotly chart builders for Google Search Analysis.

All charts use the dark plotly_dark template and a consistent
purple → teal color palette to match the Streamlit dashboard theme.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config
from analyzer.geo import to_alpha3
from analyzer.processor import DataProcessor

# ─── Palette ──────────────────────────────────────────────────────────────────
# Built from the themed colours in config so the charts and the dashboard CSS
# can't drift apart.
PALETTE = [
    config.PRIMARY_COLOR,
    config.SECONDARY_COLOR,
    config.WARNING_COLOR,
    config.DANGER_COLOR,
    *config.EXTRA_SERIES_COLORS,
]

_GRID   = "rgba(255,255,255,0.06)"
_PAPER  = "rgba(0,0,0,0)"
_FONT   = dict(family="Inter, sans-serif", color="#e0e0e0")
_LEGEND = dict(
    orientation="h",
    yanchor="bottom", y=1.02,
    xanchor="right",  x=1,
    bgcolor="rgba(255,255,255,0.05)",
    bordercolor="rgba(255,255,255,0.1)",
    borderwidth=1,
)


def _rgba(hex_color: str, alpha: float) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─── Chart builders ────────────────────────────────────────────────────────────

class ChartBuilder:
    """Collection of static factory methods that return go.Figure objects."""

    # ── Interest over time ────────────────────────────────────────────────────

    @staticmethod
    def interest_over_time(
        df: pd.DataFrame,
        keywords: List[str],
        smooth: bool = True,
    ) -> go.Figure:
        """
        Multi-keyword line chart of search interest over time.
        Renders a semi-transparent raw line and a smoothed overlay.
        Single-keyword view gets a filled area.
        """
        fig = go.Figure()
        single = len([k for k in keywords if k in df.columns]) == 1

        for i, kw in enumerate(keywords):
            if kw not in df.columns:
                continue
            color   = PALETTE[i % len(PALETTE)]
            series  = df[kw]
            fill    = "tozeroy" if single else None
            fill_c  = _rgba(color, 0.08) if single else None

            # Raw (transparent) trace
            fig.add_trace(go.Scatter(
                x=df.index, y=series,
                name=kw,
                mode="lines",
                line=dict(color=color, width=1.2),
                opacity=0.3,
                showlegend=False,
                hoverinfo="skip",
            ))

            # Smoothed / main trace
            y_display = (
                DataProcessor.compute_rolling_average(series, window=4)
                if smooth and len(series) > 4
                else series
            )
            fig.add_trace(go.Scatter(
                x=df.index, y=y_display,
                name=kw,
                mode="lines",
                line=dict(color=color, width=2.5),
                fill=fill,
                fillcolor=fill_c,
                hovertemplate=(
                    f"<b>{kw}</b><br>%{{x}}<br>"
                    "Interest: %{y:.1f}<extra></extra>"
                ),
            ))

        fig.update_layout(
            template=config.PLOTLY_TEMPLATE,
            paper_bgcolor=_PAPER,
            plot_bgcolor=_PAPER,
            font=_FONT,
            legend=_LEGEND,
            hovermode="x unified",
            xaxis=dict(
                showgrid=True, gridwidth=0.5, gridcolor=_GRID,
                showline=True, linecolor="rgba(255,255,255,0.1)",
                rangeslider=dict(visible=True, bgcolor="rgba(255,255,255,0.03)"),
            ),
            yaxis=dict(
                title="Search interest (% of peak)",
                showgrid=True, gridwidth=0.5, gridcolor=_GRID,
                range=[0, 108],
            ),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        return fig

    # ── Choropleth world map ──────────────────────────────────────────────────

    @staticmethod
    def choropleth_map(df: pd.DataFrame, keyword: str) -> go.Figure:
        """Interactive world choropleth of regional search interest."""
        if keyword not in df.columns:
            return go.Figure()

        df_r = df.reset_index()
        if "geoName" not in df_r.columns:
            return go.Figure()

        # pytrends' `geoCode` column holds ISO 3166-1 *alpha-2* codes ("US",
        # "IN"), but Plotly's default locationmode is ISO-3 ("USA", "IND").
        # Passing alpha-2 codes with locationmode left at None meant nothing
        # matched, and Plotly drops unmatched rows silently instead of
        # erroring — so the map rendered empty with no indication why.
        #
        # Convert to alpha-3 and match on ISO-3 explicitly. The alternative,
        # locationmode="country names", is deprecated in Plotly and matches on
        # fragile English names rather than codes.
        if "geoCode" in df_r.columns:
            df_r = df_r.assign(_iso3=df_r["geoCode"].map(to_alpha3))
            df_r = df_r[df_r["_iso3"].notna()]
            if df_r.empty:
                return go.Figure()
            locations, locationmode = "_iso3", "ISO-3"
        else:
            # No codes available — fall back to name matching.
            locations, locationmode = "geoName", "country names"

        fig = px.choropleth(
            df_r,
            locations=locations,
            locationmode=locationmode,
            color=keyword,
            hover_name="geoName",
            color_continuous_scale=[
                [0.0,  "#0d0d1a"],
                [0.2,  "#1a1040"],
                [0.45, "#2e1a6e"],
                [0.70, "#6c63ff"],
                [0.87, "#a594ff"],
                [1.0,  "#e8d5ff"],
            ],
            template=config.PLOTLY_TEMPLATE,
        )
        fig.update_layout(
            paper_bgcolor=_PAPER,
            geo=dict(
                bgcolor=_PAPER,
                showcoastlines=True,  coastlinecolor="rgba(255,255,255,0.15)",
                showland=True,        landcolor="#1a1d2e",
                showocean=True,       oceancolor="#0d0f1a",
                showframe=False,
                projection_type="natural earth",
            ),
            coloraxis_colorbar=dict(
                title=dict(text="Interest", font=dict(color="#e0e0e0")),
                tickfont=dict(color="#e0e0e0"),
                bgcolor="rgba(255,255,255,0.05)",
                bordercolor="rgba(255,255,255,0.1)",
            ),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        return fig

    # ── Top regions horizontal bar chart ─────────────────────────────────────

    @staticmethod
    def top_regions_bar(
        df: pd.DataFrame,
        keyword: str,
        n: int = config.TOP_REGIONS_N,
    ) -> go.Figure:
        """
        Horizontal gradient bar chart ranking regions by interest.

        *n* defaults to the same config value the caller uses to slice the
        data and to build the heading, so the chart can no longer show 15
        bars under a "Top 20 Regions" title.
        """
        if df is None or df.empty or keyword not in df.columns:
            return go.Figure()

        top = df[[keyword]].sort_values(by=keyword, ascending=True).tail(n)

        fig = go.Figure(go.Bar(
            x=top[keyword],
            y=top.index,
            orientation="h",
            marker=dict(
                color=top[keyword],
                colorscale=[[0, "#2e1a6e"], [0.5, "#6c63ff"], [1, "#a594ff"]],
                showscale=False,
            ),
            text=top[keyword],
            textposition="outside",
            textfont=dict(color="#e0e0e0", size=11),
            hovertemplate="<b>%{y}</b><br>Interest: %{x}<extra></extra>",
        ))
        fig.update_layout(
            template=config.PLOTLY_TEMPLATE,
            paper_bgcolor=_PAPER,
            plot_bgcolor=_PAPER,
            font=_FONT,
            xaxis=dict(
                title="Search interest (% of top region)",
                showgrid=True, gridwidth=0.5, gridcolor=_GRID,
                range=[0, 120],
            ),
            yaxis=dict(showgrid=False, tickfont=dict(size=12)),
            margin=dict(l=10, r=50, t=10, b=10),
        )
        return fig

    # ── Forecast chart ────────────────────────────────────────────────────────

    @staticmethod
    def forecast_chart(
        historical_dates,
        historical_values: np.ndarray,
        forecast_dates,
        forecast_values: np.ndarray,
        lower_ci: np.ndarray,
        upper_ci: np.ndarray,
        keyword: str,
        method: str,
    ) -> go.Figure:
        """
        Combined historical + forecast chart with a shaded 95 % prediction
        band and a vertical dashed divider marking where the forecast begins.
        """
        fig = go.Figure()

        hist_dates  = list(historical_dates)
        hist_values = list(historical_values)

        # Historical
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_values,
            name="Historical",
            mode="lines",
            line=dict(color=config.PRIMARY_COLOR, width=2),
            hovertemplate=(
                "<b>Historical</b><br>%{x}<br>"
                "Interest: %{y:.1f}<extra></extra>"
            ),
        ))

        # Prediction band
        fd_list = list(forecast_dates)
        fig.add_trace(go.Scatter(
            x=fd_list + fd_list[::-1],
            y=list(upper_ci) + list(lower_ci[::-1]),
            fill="toself",
            fillcolor=_rgba(config.SECONDARY_COLOR, 0.10),
            line=dict(color="rgba(0,0,0,0)"),
            name="95 % prediction interval",
            hoverinfo="skip",
        ))

        # Forecast line. Prepend the final historical point so the dashed
        # forecast visually continues the solid history instead of starting
        # one period adrift with a gap between them.
        fc_x = fd_list
        fc_y = list(forecast_values)
        if hist_dates:
            fc_x = [hist_dates[-1]] + fc_x
            fc_y = [hist_values[-1]] + fc_y

        fig.add_trace(go.Scatter(
            x=fc_x, y=fc_y,
            name="Forecast",
            mode="lines+markers",
            line=dict(color=config.SECONDARY_COLOR, width=2.5, dash="dash"),
            marker=dict(size=6, color=config.SECONDARY_COLOR,
                        line=dict(color="white", width=1)),
            hovertemplate=(
                "<b>Forecast</b><br>%{x}<br>"
                "Predicted: %{y:.1f}<extra></extra>"
            ),
        ))

        # Divider
        if hist_dates:
            fig.add_vline(
                x=hist_dates[-1],
                line_dash="dot",
                line_color="rgba(255,255,255,0.25)",
                annotation_text="Forecast →",
                annotation_position="top left",
                annotation_font=dict(color="#aaa", size=11),
            )

        fig.update_layout(
            template=config.PLOTLY_TEMPLATE,
            paper_bgcolor=_PAPER,
            plot_bgcolor=_PAPER,
            font=_FONT,
            legend=_LEGEND,
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor=_GRID),
            yaxis=dict(
                title="Search interest (% of peak)",
                showgrid=True, gridwidth=0.5, gridcolor=_GRID,
                range=[0, 110],
            ),
            margin=dict(l=10, r=10, t=30, b=10),
            annotations=[dict(
                x=0.01, y=0.97, xref="paper", yref="paper",
                text=f"Method: {method}",
                showarrow=False,
                font=dict(size=11, color="#888"),
                bgcolor="rgba(255,255,255,0.05)",
                bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1, borderpad=5,
            )],
        )
        return fig

    # ── Related queries bar chart ─────────────────────────────────────────────

    @staticmethod
    def related_queries_chart(
        related_data: dict,
        keyword: str,
        query_type: str = "top",
    ) -> go.Figure:
        """Horizontal bar chart for top or rising related queries."""
        kw_data = related_data.get(keyword)
        if not kw_data:
            return go.Figure()
        df = kw_data.get(query_type)
        if df is None or df.empty:
            return go.Figure()

        df   = df.head(15)
        base = config.WARNING_COLOR if query_type == "rising" else config.PRIMARY_COLOR

        fig = go.Figure(go.Bar(
            x=df["value"],
            y=df["query"],
            orientation="h",
            marker=dict(
                color=df["value"],
                colorscale=[[0, "#1a1040"], [1, base]],
                showscale=False,
            ),
            text=df["value"].apply(
                lambda v: f"{v}%" if query_type == "rising" else str(v)
            ),
            textposition="outside",
            textfont=dict(color="#e0e0e0", size=11),
            hovertemplate="<b>%{y}</b><br>Value: %{x}<extra></extra>",
        ))
        fig.update_layout(
            template=config.PLOTLY_TEMPLATE,
            paper_bgcolor=_PAPER,
            plot_bgcolor=_PAPER,
            font=_FONT,
            xaxis=dict(
                title="Breakout %" if query_type == "rising" else "Interest",
                showgrid=True, gridwidth=0.5, gridcolor=_GRID,
            ),
            yaxis=dict(showgrid=False, autorange="reversed"),
            margin=dict(l=10, r=55, t=10, b=10),
        )
        return fig

    # ── Radar comparison chart ────────────────────────────────────────────────

    @staticmethod
    def comparison_radar(df: pd.DataFrame, keywords: List[str]) -> go.Figure:
        """
        Polar/radar chart comparing keywords across five dimensions.

        Every axis is oriented so that "further out is better", which a radar
        chart implies visually. "Volatility" used to break that rule — a wildly
        erratic keyword drew the largest shape — so it is plotted as its
        inverse, Stability.

        The axes are still not commensurate (an interest index and a
        dispersion measure share one 0–100 radial scale), so read this as a
        shape comparison rather than as a total score.
        """
        AXES = ["Peak Interest", "Current Interest", "Avg Interest", "Stability", "Trend Score"]

        fig = go.Figure()
        for i, kw in enumerate(keywords):
            if kw not in df.columns:
                continue
            s = df[kw].dropna()
            if s.empty:
                continue
            mid   = len(s) // 2
            trend = float(np.clip(s.iloc[mid:].mean() - s.iloc[:mid].mean() + 50, 0, 100))
            stability = float(100.0 - np.clip(s.std(), 0, 100)) if len(s) > 1 else 100.0
            vals  = [
                float(s.max()),
                float(s.iloc[-1]),
                float(s.mean()),
                stability,
                trend,
            ]
            color = PALETTE[i % len(PALETTE)]
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=AXES + [AXES[0]],
                fill="toself",
                name=kw,
                line=dict(color=color, width=2),
                fillcolor=_rgba(color, 0.12),
            ))

        fig.update_layout(
            template=config.PLOTLY_TEMPLATE,
            paper_bgcolor=_PAPER,
            polar=dict(
                bgcolor=_PAPER,
                radialaxis=dict(visible=True, range=[0, 100], gridcolor=_GRID),
                angularaxis=dict(gridcolor=_GRID),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.18,
                xanchor="center",  x=0.5,
            ),
            font=_FONT,
            margin=dict(l=30, r=30, t=30, b=60),
        )
        return fig

    # ── Stacked area comparison ───────────────────────────────────────────────

    @staticmethod
    def stacked_area(df: pd.DataFrame, keywords: List[str]) -> go.Figure:
        """Normalised stacked area chart for multi-keyword share-of-voice view."""
        avail = [k for k in keywords if k in df.columns]
        if not avail:
            return go.Figure()

        # Compute row-wise share
        total = df[avail].sum(axis=1).replace(0, np.nan)
        norm  = (df[avail].div(total, axis=0) * 100).fillna(0)

        fig = go.Figure()
        for i, kw in enumerate(avail):
            color = PALETTE[i % len(PALETTE)]
            fig.add_trace(go.Scatter(
                x=norm.index, y=norm[kw],
                name=kw,
                mode="lines",
                stackgroup="one",
                line=dict(color=color, width=0.5),
                fillcolor=_rgba(color, 0.55),
                hovertemplate=f"<b>{kw}</b><br>%{{x}}<br>Share: %{{y:.1f}}%<extra></extra>",
            ))

        fig.update_layout(
            template=config.PLOTLY_TEMPLATE,
            paper_bgcolor=_PAPER,
            plot_bgcolor=_PAPER,
            font=_FONT,
            legend=_LEGEND,
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor=_GRID),
            yaxis=dict(
                title="Share of Interest (%)",
                showgrid=True, gridwidth=0.5, gridcolor=_GRID,
                range=[0, 100],
            ),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        return fig
