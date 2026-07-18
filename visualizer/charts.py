"""
Interactive Plotly chart builders for Google Search Analysis.

All charts use the dark plotly_dark template and a consistent
purple → teal color palette to match the Streamlit dashboard theme.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config

# ─── Palette ──────────────────────────────────────────────────────────────────
PALETTE = [
    "#6c63ff", "#00d4aa", "#ffa726", "#ff6b6b",
    "#42a5f5", "#ab47bc", "#66bb6a", "#ef5350",
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
                series.rolling(window=4, center=True, min_periods=1).mean()
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
                    f"<b>{kw}</b><br>%{{x|%b %d, %Y}}<br>"
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
                title="Search Interest (0–100)",
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
        loc_col  = "geoCode" if "geoCode" in df_r.columns else "geoName"
        loc_mode = None if loc_col == "geoCode" else "country names"

        fig = px.choropleth(
            df_r,
            locations=loc_col,
            locationmode=loc_mode,
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
    def top_regions_bar(df: pd.DataFrame, keyword: str, n: int = 15) -> go.Figure:
        """Horizontal gradient bar chart ranking regions by interest."""
        if keyword not in df.columns:
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
                title="Search Interest",
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
        Combined historical + forecast chart with a shaded 95 % CI band
        and a vertical dashed divider marking where the forecast begins.
        """
        fig = go.Figure()

        # Historical
        fig.add_trace(go.Scatter(
            x=historical_dates, y=historical_values,
            name="Historical",
            mode="lines",
            line=dict(color="#6c63ff", width=2),
            hovertemplate=(
                "<b>Historical</b><br>%{x|%b %d, %Y}<br>"
                "Interest: %{y:.1f}<extra></extra>"
            ),
        ))

        # CI band
        fd_list = list(forecast_dates)
        fig.add_trace(go.Scatter(
            x=fd_list + fd_list[::-1],
            y=list(upper_ci) + list(lower_ci[::-1]),
            fill="toself",
            fillcolor=_rgba("#00d4aa", 0.10),
            line=dict(color="rgba(0,0,0,0)"),
            name="95 % Confidence",
            hoverinfo="skip",
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_values,
            name="Forecast",
            mode="lines+markers",
            line=dict(color="#00d4aa", width=2.5, dash="dash"),
            marker=dict(size=6, color="#00d4aa", line=dict(color="white", width=1)),
            hovertemplate=(
                "<b>Forecast</b><br>%{x|%b %d, %Y}<br>"
                "Predicted: %{y:.1f}<extra></extra>"
            ),
        ))

        # Divider
        if len(historical_dates):
            fig.add_vline(
                x=historical_dates[-1],
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
                title="Search Interest (0–100)",
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
        base = "#ffa726" if query_type == "rising" else "#6c63ff"

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
        """Polar/radar chart comparing keywords across five performance dimensions."""
        AXES = ["Peak Interest", "Current Interest", "Avg Interest", "Volatility", "Trend Score"]

        fig = go.Figure()
        for i, kw in enumerate(keywords):
            if kw not in df.columns:
                continue
            s = df[kw]
            mid   = len(s) // 2
            trend = float(np.clip(s.iloc[mid:].mean() - s.iloc[:mid].mean() + 50, 0, 100))
            vals  = [
                float(s.max()),
                float(s.iloc[-1]),
                float(s.mean()),
                float(np.clip(s.std(), 0, 100)),
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
                hovertemplate=f"<b>{kw}</b><br>%{{x|%b %Y}}<br>Share: %{{y:.1f}}%<extra></extra>",
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
