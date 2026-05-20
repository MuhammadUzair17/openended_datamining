# ============================================================
# LAB #08 – Streamlit Dashboard
# Dataset : KSE100_5Years_2019_2024.csv
# Run     : streamlit run lab08_dashboard.py
# Requires: pip install streamlit pandas numpy plotly scipy
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os

# ── Page config  (must be the FIRST Streamlit call) ──────────
st.set_page_config(
    page_title="PSX KSE-100 Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown(
    """
<style>
.stApp { background-color: #0a0e1a; color: #e0e0e0; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1224, #0a0e1a);
    border-right: 1px solid #1e2a3a;
}
.kpi-card {
    background: linear-gradient(135deg, #0d1224, #111827);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 4px 0;
}
.kpi-card:hover { border-color: #00d4ff; }
.kpi-label {
    font-size: 11px; color: #8899aa;
    letter-spacing: 1.2px; text-transform: uppercase;
    margin-bottom: 6px;
}
.kpi-value  { font-size: 26px; font-weight: 700; color: #00d4ff; }
.kpi-pos    { color: #00ff88; font-size: 12px; }
.kpi-neg    { color: #ff4757; font-size: 12px; }
.sec-head {
    font-size: 17px; font-weight: 700; color: #00d4ff;
    border-left: 4px solid #00d4ff;
    padding-left: 10px; margin: 20px 0 10px 0;
}
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================
CSV_FILE = "KSE100_5Years_2019_2024.csv"


@st.cache_data
def load_data(path: str):
    """Load, auto-detect columns, engineer features, and return cleaned df."""
    if not os.path.isfile(path):
        return None, {}, None

    df = pd.read_csv(path)

    # ── Auto-detect columns ───────────────────────────────────
    col_map: dict[str, str] = {}
    date_col = None
    for col in df.columns:
        cl = col.lower()
        if "date"    in cl or "time"   in cl: date_col           = col
        if "open"    in cl:                   col_map["OPEN"]    = col
        if "high"    in cl:                   col_map["HIGH"]    = col
        if "low"     in cl:                   col_map["LOW"]     = col
        if "close"   in cl:                   col_map["CLOSE"]   = col
        if "volume"  in cl:                   col_map["VOLUME"]  = col
        if "symbol"  in cl or "ticker"  in cl: col_map["SYMBOL"]  = col
        if "company" in cl or "name"    in cl: col_map["COMPANY"] = col
        if "sector"  in cl:                   col_map["SECTOR"]  = col

    # ── Parse date ────────────────────────────────────────────
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df["Year"]   = df[date_col].dt.year
        df["Month"]  = df[date_col].dt.month
        df["Date"]   = df[date_col]

    # ── Feature engineering ───────────────────────────────────
    if "CLOSE" in col_map and "SYMBOL" in col_map:
        df["Daily_Return"] = (
            df.groupby(col_map["SYMBOL"])[col_map["CLOSE"]].pct_change() * 100
        )
    if "HIGH" in col_map and "LOW" in col_map:
        df["Price_Range"] = df[col_map["HIGH"]] - df[col_map["LOW"]]

    # ── Rename detected columns to standard names ─────────────
    rename = {v: k for k, v in col_map.items() if v != k}
    df.rename(columns=rename, inplace=True)

    return df, col_map, date_col


# ── Load  ─────────────────────────────────────────────────────
df, col_map, date_col = load_data(CSV_FILE)

if df is None:
    st.error(
        f"❌  **'{CSV_FILE}' not found.**\n\n"
        "Place the CSV in the **same folder** as this script, then re-run:\n\n"
        "```\nstreamlit run lab08_dashboard.py\n```"
    )
    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.markdown("## 🎛️ Dashboard Filters")
st.sidebar.markdown("---")

sym_col  = "SYMBOL" if "SYMBOL" in df.columns else df.columns[0]
all_syms = sorted(df[sym_col].astype(str).unique())

selected_syms = st.sidebar.multiselect(
    "📌 Companies", options=all_syms, default=all_syms[:6]
)
if not selected_syms:
    selected_syms = all_syms[:6]

yr_min = int(df["Year"].min()) if "Year" in df.columns else 2019
yr_max = int(df["Year"].max()) if "Year" in df.columns else 2024
year_range = st.sidebar.slider(
    "📅 Year Range", min_value=yr_min, max_value=yr_max,
    value=(yr_min, yr_max),
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**KSE-100 Dashboard**\n\n"
    f"Companies : {df[sym_col].nunique()}\n\n"
    "Period    : 2019 – 2024\n\n"
    "Lab #08 – Data Visualization"
)

# ── Apply filter ──────────────────────────────────────────────
mask = df[sym_col].isin(selected_syms)
if "Year" in df.columns:
    mask &= df["Year"].between(year_range[0], year_range[1])
dff = df[mask].copy()


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
<div style='background:linear-gradient(135deg,#0d1224,#0a1628);
border:1px solid #1e3a5f;border-radius:14px;
padding:22px 30px;margin-bottom:18px;'>
<h1 style='color:#00d4ff;margin:0;font-size:30px;'>
📈 KSE-100 Pakistan Stock Exchange Dashboard
</h1>
<p style='color:#8899aa;margin:6px 0 0;font-size:14px;'>
Interactive Analytics · 2019 – 2024 · KSE-100 Listed Companies
</p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PART B – KPI CARDS
# ============================================================
st.markdown(
    '<div class="sec-head">📊 Key Performance Indicators (KPIs)</div>',
    unsafe_allow_html=True,
)

close_col = "CLOSE"  if "CLOSE"  in dff.columns else list(col_map.values())[0]
vol_col   = "VOLUME" if "VOLUME" in dff.columns else None

# ── Compute KPIs ─────────────────────────────────────────────
avg_close  = dff[close_col].mean()
max_price  = dff["HIGH"].max()    if "HIGH"         in dff.columns else dff[close_col].max()
min_price  = dff["LOW"].min()     if "LOW"          in dff.columns else dff[close_col].min()
avg_vol    = dff[vol_col].mean()  if vol_col                       else 0
avg_ret    = dff["Daily_Return"].mean() if "Daily_Return" in dff.columns else 0
volatility = dff["Daily_Return"].std()  if "Daily_Return" in dff.columns else 0
avg_range  = dff["Price_Range"].mean()  if "Price_Range"  in dff.columns else 0
num_cos    = dff[sym_col].nunique()
total_recs = len(dff)

best_sym  = (dff.groupby(sym_col)["Daily_Return"].mean().idxmax()
             if "Daily_Return" in dff.columns else "N/A")
worst_sym = (dff.groupby(sym_col)["Daily_Return"].mean().idxmin()
             if "Daily_Return" in dff.columns else "N/A")

# ── Row 1 ─────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-label">Avg Close Price</div>
        <div class="kpi-value">PKR {avg_close:,.1f}</div>
        <div class="kpi-pos">Filtered selection</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-label">Avg Daily Volume</div>
        <div class="kpi-value">{avg_vol / 1e6:.2f}M</div>
        <div class="kpi-pos">Shares per day</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k3:
    rc = "kpi-pos" if avg_ret >= 0 else "kpi-neg"
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-label">Avg Daily Return</div>
        <div class="kpi-value">{avg_ret:.3f}%</div>
        <div class="{rc}">Volatility: ±{volatility:.2f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-label">Best / Worst Stock</div>
        <div class="kpi-value" style="font-size:18px;">🟢 {best_sym} / 🔴 {worst_sym}</div>
        <div class="kpi-pos">By avg daily return</div>
        </div>""",
        unsafe_allow_html=True,
    )

# ── Row 2 ─────────────────────────────────────────────────────
k5, k6, k7, k8 = st.columns(4)
with k5:
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-label">All-Time High</div>
        <div class="kpi-value">PKR {max_price:,.0f}</div>
        <div class="kpi-pos">In selection</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k6:
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-label">All-Time Low</div>
        <div class="kpi-value" style="color:#ff4757;">PKR {min_price:,.2f}</div>
        <div class="kpi-neg">In selection</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k7:
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-label">Companies / Records</div>
        <div class="kpi-value">{num_cos} / {total_recs:,}</div>
        <div class="kpi-pos">Current filter</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k8:
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-label">Avg Daily Price Range</div>
        <div class="kpi-value">PKR {avg_range:.2f}</div>
        <div class="kpi-pos">HIGH − LOW</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ============================================================
# CHART HELPERS
# ============================================================
DARK_LAYOUT = dict(
    paper_bgcolor="#0d1224",
    plot_bgcolor="#0a0e1a",
    title_font_color="#00d4ff",
)
GRID = dict(gridcolor="#1e2a3a")


def apply_dark(fig, height: int = 400) -> go.Figure:
    fig.update_layout(**DARK_LAYOUT, height=height,
                      legend=dict(bgcolor="#0d1224"))
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return fig


# ============================================================
# CHARTS
# ============================================================

# ── Chart 1: Price trend ─────────────────────────────────────
st.markdown('<div class="sec-head">📉 Stock Price Trends</div>',
            unsafe_allow_html=True)

fig1 = px.line(
    dff.sort_values("Date"), x="Date", y=close_col, color=sym_col,
    title="Closing Price Over Time (2019–2024)",
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Vivid,
)
st.plotly_chart(apply_dark(fig1), use_container_width=True)

# ── Charts 2 & 3 ─────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="sec-head">📊 Avg Close by Stock</div>',
                unsafe_allow_html=True)
    avg_df = (dff.groupby(sym_col)[close_col].mean()
              .sort_values(ascending=False).reset_index())
    fig2 = px.bar(
        avg_df, x=sym_col, y=close_col,
        title="Average Closing Price per Stock",
        template="plotly_dark", color=close_col,
        color_continuous_scale="Blues",
    )
    st.plotly_chart(apply_dark(fig2, 370), use_container_width=True)

with c2:
    st.markdown('<div class="sec-head">🥧 Volume Share</div>',
                unsafe_allow_html=True)
    if vol_col:
        vol_df = dff.groupby(sym_col)[vol_col].sum().reset_index()
        fig3 = px.pie(
            vol_df, names=sym_col, values=vol_col,
            title="Volume Distribution by Company",
            template="plotly_dark", hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig3.update_layout(**DARK_LAYOUT, height=370,
                           legend=dict(bgcolor="#0d1224"))
        st.plotly_chart(fig3, use_container_width=True)

# ── Chart 4: Correlation heatmap ─────────────────────────────
st.markdown('<div class="sec-head">🔥 Correlation Heatmap</div>',
            unsafe_allow_html=True)

num_cols_list = [
    c for c in dff.select_dtypes(include=np.number).columns
    if c not in ("Year", "Month")
]
corr = dff[num_cols_list].corr().round(2)
fig4 = go.Figure(
    go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0,
        text=corr.values, texttemplate="%{text:.2f}", showscale=True,
    )
)
fig4.update_layout(title="Feature Correlation Matrix",
                   **DARK_LAYOUT, height=420)
st.plotly_chart(fig4, use_container_width=True)

# ── Charts 5 & 6 ─────────────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.markdown('<div class="sec-head">📦 Volume Over Time</div>',
                unsafe_allow_html=True)
    if vol_col and "Date" in dff.columns:
        vt = dff.groupby("Date")[vol_col].sum().reset_index()
        fig5 = px.area(
            vt, x="Date", y=vol_col,
            title="Total Daily Volume",
            template="plotly_dark",
            color_discrete_sequence=["#00d4ff"],
        )
        st.plotly_chart(apply_dark(fig5, 360), use_container_width=True)

with c4:
    st.markdown('<div class="sec-head">📅 Yearly Avg Close</div>',
                unsafe_allow_html=True)
    if "Year" in dff.columns:
        yr_df = (dff.groupby(["Year", sym_col])[close_col]
                 .mean().reset_index())
        fig6 = px.bar(
            yr_df, x="Year", y=close_col, color=sym_col, barmode="group",
            title="Yearly Average Close Price",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        st.plotly_chart(apply_dark(fig6, 360), use_container_width=True)

# ── Charts 7 & 8: Return distribution + Volatility ───────────
if "Daily_Return" in dff.columns:
    st.markdown(
        '<div class="sec-head">📐 Return Distribution & Volatility</div>',
        unsafe_allow_html=True,
    )
    c5, c6 = st.columns(2)

    with c5:
        fig7 = px.histogram(
            dff.dropna(subset=["Daily_Return"]),
            x="Daily_Return", color=sym_col,
            nbins=80, barmode="overlay", opacity=0.6,
            title="Distribution of Daily Returns (%)",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig7.update_xaxes(range=[-10, 10], **GRID)
        fig7.update_yaxes(**GRID)
        fig7.update_layout(**DARK_LAYOUT, height=370,
                           legend=dict(bgcolor="#0d1224"))
        st.plotly_chart(fig7, use_container_width=True)

    with c6:
        vol_s = (
            dff.dropna(subset=["Daily_Return"])
            .groupby(sym_col)["Daily_Return"].std()
            .reset_index()
            .rename(columns={"Daily_Return": "Volatility"})
            .sort_values("Volatility")
        )
        fig8 = px.bar(
            vol_s, x="Volatility", y=sym_col, orientation="h",
            title="Volatility by Stock (Std Dev of Daily Return)",
            template="plotly_dark", color="Volatility",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(apply_dark(fig8, 370), use_container_width=True)

# ── Chart 9: Candlestick ──────────────────────────────────────
st.markdown('<div class="sec-head">🕯️ Candlestick Chart</div>',
            unsafe_allow_html=True)

candle_sym = st.selectbox("Select stock:", selected_syms)
cdf = dff[dff[sym_col] == candle_sym].sort_values("Date").tail(120)

if all(c in cdf.columns for c in ["OPEN", "HIGH", "LOW", "CLOSE"]):
    fig9 = go.Figure(
        go.Candlestick(
            x=cdf["Date"],
            open=cdf["OPEN"], high=cdf["HIGH"],
            low=cdf["LOW"],   close=cdf["CLOSE"],
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff4757",
        )
    )
    fig9.update_layout(
        title=f"{candle_sym} – Last 120 Trading Days (OHLC)",
        **DARK_LAYOUT, height=420,
        xaxis_rangeslider_visible=True,
    )
    fig9.update_xaxes(**GRID)
    fig9.update_yaxes(**GRID)
    st.plotly_chart(fig9, use_container_width=True)
else:
    st.info("OHLC columns not found — candlestick unavailable.")

# ── Raw data expander ─────────────────────────────────────────
st.markdown("---")
with st.expander("📋 View Raw Data Table"):
    display_df = (dff.sort_values("Date", ascending=False).head(500)
                  if "Date" in dff.columns else dff.head(500))
    st.dataframe(display_df, use_container_width=True)
    st.caption(f"Showing up to 500 of {len(dff):,} filtered records")

# ── Footer ────────────────────────────────────────────────────
st.markdown(
    """
<div style='text-align:center;color:#3a4a5a;padding:24px 0 8px;font-size:12px;'>
  KSE-100 Stock Analytics Dashboard · Lab #08 · Streamlit + Plotly
</div>
""",
    unsafe_allow_html=True,
)