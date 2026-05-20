# ============================================================
# LAB #07 – Data Cleaning & EDA
# Dataset : KSE100_5Years_2019_2024.csv
# Run     : python lab07_eda.py   (inside VS Code terminal)
# Requires: pip install pandas numpy matplotlib seaborn scipy
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

# ── Use a non-interactive backend when running as a plain script
# ── (VS Code's Python extension handles display automatically)
matplotlib.use("Agg")        # saves PNGs; remove this line if you
                             # want pop-up windows instead

# ── Output folder ────────────────────────────────────────────
OUT_DIR = "lab07_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def save(name: str) -> None:
    """Save the current figure to the output folder and close it."""
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"   → Saved: {path}")


# ── Dark-theme style ─────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2d3a",
})

ACCENT = "#00d4ff"
GREEN  = "#00ff88"
RED    = "#ff4757"
GOLD   = "#ffd700"


# ============================================================
# PART A – DATASET UNDERSTANDING
# ============================================================
print("=" * 60)
print("  PART A – DATASET UNDERSTANDING")
print("=" * 60)

# A1 ── Load dataset
CSV_FILE = "KSE100_5Years_2019_2024.csv"

if not os.path.isfile(CSV_FILE):
    raise FileNotFoundError(
        f"\n❌  '{CSV_FILE}' not found.\n"
        "    Place the CSV in the same folder as this script and re-run."
    )

df = pd.read_csv(CSV_FILE)

print("\n[A1] First 5 records:")
print(df.head().to_string())

# A2 ── Shape
print(f"\n[A2] Rows    : {df.shape[0]:,}")
print(f"     Columns : {df.shape[1]}")
print(f"     Names   : {list(df.columns)}")

# A3 ── Data types
print("\n[A3] Data Types:")
print(df.dtypes.to_string())

# A4 ── Numerical vs Categorical
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
print(f"\n[A4] Numerical   : {num_cols}")
print(f"     Categorical : {cat_cols}")

# A5 ── Descriptive statistics
print("\n[A5] Descriptive Statistics:")
print(df.describe().round(2).to_string())

print("""
─────────────────────────────────────────────────────────────
DATASET:  KSE-100 Pakistan Stock Exchange Historical Data
PERIOD :  2019 – 2024 (5 Years)
CONTENT:  Daily OHLCV data for KSE-100 listed companies
GOAL   :  Analyse stock price trends, volatility, volume
          patterns, and sector performance over 5 years.
─────────────────────────────────────────────────────────────
""")


# ============================================================
# PART B – DATA CLEANING
# ============================================================
print("=" * 60)
print("  PART B – DATA CLEANING")
print("=" * 60)

df_clean = df.copy()

# ── Auto-detect date column ──────────────────────────────────
date_col = next(
    (c for c in df_clean.columns if "date" in c.lower() or "time" in c.lower()),
    None,
)

# ── Auto-detect OHLCV + symbol columns ──────────────────────
col_map: dict[str, str] = {}
for col in df_clean.columns:
    cl = col.lower()
    if "open"                        in cl: col_map["OPEN"]    = col
    if "high"                        in cl: col_map["HIGH"]    = col
    if "low"                         in cl: col_map["LOW"]     = col
    if "close"                       in cl: col_map["CLOSE"]   = col
    if "volume" in cl or "vol" in cl      : col_map["VOLUME"]  = col
    if "symbol" in cl or "ticker" in cl   : col_map["SYMBOL"]  = col
    if "company" in cl or "name" in cl    : col_map["COMPANY"] = col

print(f"\nDetected columns : {col_map}")
print(f"Date column      : {date_col}")

# B1 ── Missing values
print("\n[B1] Missing Values per Column:")
missing = df_clean.isnull().sum()
print(missing.to_string())
total_missing = int(missing.sum())

if total_missing == 0:
    print("✅ No missing values detected.")
else:
    print(f"⚠️  Total missing: {total_missing}")
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                print(f"   → '{col}' filled with median")
            else:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                print(f"   → '{col}' filled with mode")

# B2 ── Duplicates
dups = int(df_clean.duplicated().sum())
print(f"\n[B2] Duplicate rows: {dups}")
df_clean.drop_duplicates(inplace=True)
print(f"     Rows after removal: {df_clean.shape[0]:,}")

# B3 ── Type formatting
if date_col:
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
    print(f"\n[B3] '{date_col}' → datetime ✅")
if "SYMBOL" in col_map:
    df_clean[col_map["SYMBOL"]] = df_clean[col_map["SYMBOL"]].astype("category")
    print(f"     '{col_map['SYMBOL']}' → category ✅")

# B4 ── Zero-volume rows
if "VOLUME" in col_map:
    vcol = col_map["VOLUME"]
    zero_vol = int((df_clean[vcol] == 0).sum())
    print(f"\n[B4] Zero-volume rows: {zero_vol}")
    if zero_vol > 0:
        df_clean = df_clean[df_clean[vcol] > 0]
        print(f"     Removed. Remaining rows: {df_clean.shape[0]:,}")
    else:
        print("     None found.")

# B5 ── IQR outlier detection on VOLUME
if "VOLUME" in col_map:
    vcol  = col_map["VOLUME"]
    Q1    = df_clean[vcol].quantile(0.25)
    Q3    = df_clean[vcol].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    out_cnt = int(((df_clean[vcol] < lower) | (df_clean[vcol] > upper)).sum())
    print(f"\n[B5] IQR Outliers in '{vcol}':")
    print(f"     Q1={Q1:,.0f}  Q3={Q3:,.0f}  IQR={IQR:,.0f}")
    print(f"     Bounds=[{lower:,.0f}, {upper:,.0f}]  Outliers={out_cnt:,}")
    print("     → Retained (extreme volumes are real market events)")

# B6 ── Z-score outliers on CLOSE
if "CLOSE" in col_map:
    ccol = col_map["CLOSE"]
    z_scores = np.abs(stats.zscore(df_clean[ccol].dropna()))
    z_out = int((z_scores > 3).sum())
    print(f"\n[B6] Z-score Outliers in '{ccol}' (|z|>3): {z_out:,}")
    print("     → Retained as valid extreme prices")

# B7 ── Feature engineering
sym  = col_map.get("SYMBOL")
ccol = col_map.get("CLOSE")
vcol = col_map.get("VOLUME")

if ccol and sym:
    df_clean["Daily_Return"] = (
        df_clean.groupby(sym)[ccol].pct_change() * 100
    )
if "HIGH" in col_map and "LOW" in col_map:
    df_clean["Price_Range"] = (
        df_clean[col_map["HIGH"]] - df_clean[col_map["LOW"]]
    ).round(2)
if date_col:
    df_clean["Year"]  = df_clean[date_col].dt.year
    df_clean["Month"] = df_clean[date_col].dt.month

print("\n[B7] New features: Daily_Return, Price_Range, Year, Month ✅")
print(f"\n✅ Cleaned dataset shape: {df_clean.shape}")

cleaned_csv = os.path.join(OUT_DIR, "psx_cleaned.csv")
df_clean.to_csv(cleaned_csv, index=False)
print(f"✅ Saved: {cleaned_csv}")


# ============================================================
# PART C – EDA VISUALISATIONS
# ============================================================
print("\n" + "=" * 60)
print("  PART C – EDA VISUALISATIONS")
print("=" * 60)

# Resolve column references (with safe fallbacks)
ccol = col_map.get("CLOSE",  df_clean.select_dtypes(np.number).columns[0])
vcol = col_map.get("VOLUME", df_clean.select_dtypes(np.number).columns[-1])

# ── Plot 1: Histogram of Close prices ────────────────────────
print("\n[Plot 1] Close Price Distribution …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Plot 1 – Close Price Distribution",
             color=ACCENT, fontsize=14, fontweight="bold")

axes[0].hist(df_clean[ccol].dropna(), bins=60,
             color=ACCENT, alpha=0.8, edgecolor="none")
axes[0].set_title("Histogram – Close Price", color="white")
axes[0].set_xlabel("Close Price (PKR)")
axes[0].set_ylabel("Frequency")

axes[1].hist(np.log1p(df_clean[ccol].dropna()), bins=60,
             color=GREEN, alpha=0.8, edgecolor="none")
axes[1].set_title("Log-transformed Close Price", color="white")
axes[1].set_xlabel("log(Close)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
save("plot1_histogram.png")

# ── Plot 2: Boxplot – Close by Year ──────────────────────────
print("[Plot 2] Boxplot – Close Price by Year …")
fig, ax = plt.subplots(figsize=(12, 6))
years         = sorted(df_clean["Year"].dropna().unique())
data_by_year  = [df_clean[df_clean["Year"] == y][ccol].dropna() for y in years]
bp = ax.boxplot(
    data_by_year,
    labels=[int(y) for y in years],
    patch_artist=True,
    medianprops=dict(color=GOLD, linewidth=2),
    flierprops=dict(marker="o", markerfacecolor=RED, markersize=3, alpha=0.4),
)
colors = plt.cm.cool(np.linspace(0.2, 0.9, len(years)))  # type: ignore[attr-defined]
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_title("Plot 2 – Close Price by Year", color=ACCENT, fontsize=13)
ax.set_xlabel("Year")
ax.set_ylabel("Close Price (PKR)")
plt.tight_layout()
save("plot2_boxplot_year.png")

# ── Plot 3: Line chart – top-5 stocks ────────────────────────
if sym and date_col:
    print("[Plot 3] Price Trend – Top 5 Stocks …")
    top5    = df_clean.groupby(sym)[ccol].mean().nlargest(5).index.tolist()
    palette = [ACCENT, GREEN, GOLD, "#ff6b81", "#a29bfe"]
    fig, ax = plt.subplots(figsize=(14, 6))
    for s, color in zip(top5, palette):
        d = df_clean[df_clean[sym] == s].sort_values(date_col)
        ax.plot(d[date_col], d[ccol], label=s, color=color,
                linewidth=1.5, alpha=0.9)
    ax.set_title("Plot 3 – Close Price Trend (Top 5 Stocks)",
                 color=ACCENT, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (PKR)")
    ax.legend(facecolor="#1a1d27", labelcolor="white")
    plt.tight_layout()
    save("plot3_price_trend.png")

# ── Plot 4: Correlation Heatmap ───────────────────────────────
print("[Plot 4] Correlation Heatmap …")
num_df = df_clean.select_dtypes(include=np.number)
corr   = num_df.corr()
mask   = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0,
            linewidths=0.5, linecolor="#0f1117", ax=ax)
ax.set_title("Plot 4 – Correlation Heatmap", color=ACCENT, fontsize=13)
plt.tight_layout()
save("plot4_heatmap.png")

# ── Plot 5: Bar – Avg Close by Company (top 15) ──────────────
if sym:
    print("[Plot 5] Avg Close Price by Company …")
    avg_c = (df_clean.groupby(sym)[ccol].mean()
             .sort_values(ascending=False).head(15))
    fig, ax = plt.subplots(figsize=(14, 6))
    bar_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(avg_c)))  # type: ignore[attr-defined]
    bars = ax.bar(avg_c.index, avg_c.values,
                  color=bar_colors, edgecolor="none", alpha=0.85)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{bar.get_height():.0f}",
            ha="center", va="bottom", fontsize=8, color="white",
        )
    ax.set_title("Plot 5 – Avg Closing Price by Company (Top 15)",
                 color=ACCENT, fontsize=13)
    ax.set_xlabel("Symbol")
    ax.set_ylabel("Avg Close (PKR)")
    plt.xticks(rotation=40)
    plt.tight_layout()
    save("plot5_avg_close_bar.png")

# ── Plot 6: Scatter – Daily Return vs Volume ─────────────────
if "Daily_Return" in df_clean.columns:
    print("[Plot 6] Daily Return vs log(Volume) …")
    sample = df_clean.dropna(subset=["Daily_Return", vcol]).sample(
        min(3000, len(df_clean)), random_state=42
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        np.log1p(sample[vcol]), sample["Daily_Return"],
        alpha=0.3, s=10,
        c=sample["Daily_Return"], cmap="RdYlGn",
        vmin=-5, vmax=5,
    )
    ax.axhline(0, color=GOLD, linewidth=1, linestyle="--", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Daily Return (%)")
    ax.set_title("Plot 6 – Daily Return vs log(Volume)",
                 color=ACCENT, fontsize=13)
    ax.set_xlabel("log(Volume)")
    ax.set_ylabel("Daily Return (%)")
    plt.tight_layout()
    save("plot6_scatter_return_vol.png")

# ── Plot 7 (Open-Ended): Yearly Avg Return Heatmap ───────────
if "Daily_Return" in df_clean.columns and sym:
    print("[Plot 7] Yearly Avg Daily Return Heatmap …")
    pivot  = (df_clean.dropna(subset=["Daily_Return"])
              .groupby([sym, "Year"])["Daily_Return"]
              .mean().unstack())
    top12  = pivot.abs().mean(axis=1).nlargest(12).index
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot.loc[top12], annot=True, fmt=".2f",
                cmap="RdYlGn", center=0,
                linewidths=0.5, linecolor="#0f1117", ax=ax)
    ax.set_title(
        "Plot 7 (Open-Ended) – Avg Daily Return by Stock & Year",
        color=GOLD, fontsize=13,
    )
    plt.tight_layout()
    save("plot7_yearly_return_heatmap.png")

print(f"\n✅ LAB #07 COMPLETE — All plots saved to '{OUT_DIR}/'")