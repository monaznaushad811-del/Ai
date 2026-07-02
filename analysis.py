"""
Trader Performance vs. Market Sentiment — Analysis Script
Primetrade.ai Data Science Assignment

Inputs (expected in the same folder, or edit the paths below):
    historical_data.csv     - Hyperliquid trade-level data
    fear_greed_index.csv    - Bitcoin Fear & Greed Index (daily)

Outputs:
    merged_trades_sentiment.csv  - trade-level data joined with daily sentiment
    daily_summary.csv            - daily aggregated PnL / volume / sentiment
    charts/*.png                 - all figures used in the report

Run:
    pip install pandas matplotlib numpy
    python analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

HIST_PATH = "historical_data.csv"
FG_PATH = "fear_greed_index.csv"
OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

ORDER5 = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
COLORS5 = ["#8B0000", "#E07B39", "#999999", "#5CA05C", "#1B5E20"]
ORDER3 = ["Fear", "Neutral", "Greed"]
COLORS3 = ["#C0392B", "#95A5A6", "#27AE60"]


def simplify(classification: str) -> str:
    """Collapse the 5-way Fear&Greed label into Fear / Neutral / Greed."""
    if "Fear" in classification:
        return "Fear"
    if "Greed" in classification:
        return "Greed"
    return "Neutral"


def load_and_merge():
    df = pd.read_csv(HIST_PATH)
    fg = pd.read_csv(FG_PATH)

    df["date"] = pd.to_datetime(df["Timestamp IST"], format="%d-%m-%Y %H:%M").dt.date
    fg["date"] = pd.to_datetime(fg["date"]).dt.date

    merged = df.merge(fg[["date", "classification", "value"]], on="date", how="left")
    merged = merged.dropna(subset=["classification"])
    merged["sentiment_simple"] = merged["classification"].apply(simplify)
    return merged


def summary_tables(m: pd.DataFrame):
    g = m.groupby("classification").agg(
        trades=("Closed PnL", "count"),
        total_pnl=("Closed PnL", "sum"),
        avg_pnl=("Closed PnL", "mean"),
        median_pnl=("Closed PnL", "median"),
        avg_size_usd=("Size USD", "mean"),
        total_volume=("Size USD", "sum"),
    ).reindex(ORDER5).round(2)
    g["pnl_pct_of_volume"] = (g["total_pnl"] / g["total_volume"] * 100).round(4)

    closed = m[m["Closed PnL"] != 0].copy()
    closed["is_win"] = closed["Closed PnL"] > 0
    win_rate = closed.groupby("classification")["is_win"].mean().reindex(ORDER5).round(4)

    print("=== Summary by sentiment classification ===")
    print(g.to_string())
    print("\n=== Win rate by sentiment classification ===")
    print(win_rate.to_string())
    return g, win_rate


def make_charts(m: pd.DataFrame):
    g = m.groupby("classification").agg(
        total_pnl=("Closed PnL", "sum"), avg_pnl=("Closed PnL", "mean")
    ).reindex(ORDER5)

    # Fig 1: PnL by sentiment
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(g.index, g["total_pnl"] / 1e6, color=COLORS5)
    axes[0].set_title("Total Closed PnL by Market Sentiment", fontweight="bold")
    axes[0].set_ylabel("Total Closed PnL ($ millions)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(g.index, g["avg_pnl"], color=COLORS5)
    axes[1].set_title("Average PnL per Trade by Sentiment", fontweight="bold")
    axes[1].set_ylabel("Average Closed PnL ($)")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_pnl_by_sentiment.png", dpi=150)
    plt.close()

    # Fig 2: Win rate
    closed = m[m["Closed PnL"] != 0].copy()
    closed["is_win"] = closed["Closed PnL"] > 0
    wr = closed.groupby("classification")["is_win"].mean().reindex(ORDER5) * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(wr.index, wr.values, color=COLORS5)
    ax.set_title("Win Rate by Market Sentiment", fontweight="bold")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_winrate_by_sentiment.png", dpi=150)
    plt.close()

    # Fig 3: PnL as % of volume
    gg = m.groupby("classification").agg(
        total_pnl=("Closed PnL", "sum"), total_volume=("Size USD", "sum")
    ).reindex(ORDER5)
    gg["pnl_pct"] = gg["total_pnl"] / gg["total_volume"] * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(gg.index, gg["pnl_pct"], color=COLORS5)
    ax.set_title("PnL as % of Traded Volume", fontweight="bold")
    ax.set_ylabel("PnL / Volume (%)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_pnl_pct_volume.png", dpi=150)
    plt.close()

    # Fig 4: Activity
    g2 = m.groupby("sentiment_simple").agg(
        trades=("Closed PnL", "count"), avg_size=("Size USD", "mean")
    ).reindex(ORDER3)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(g2.index, g2["trades"], color=COLORS3)
    axes[0].set_title("Trade Count by Sentiment", fontweight="bold")
    axes[1].bar(g2.index, g2["avg_size"], color=COLORS3)
    axes[1].set_title("Average Trade Size by Sentiment", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_activity_by_sentiment.png", dpi=150)
    plt.close()

    # Fig 5: Volatility
    vol = m.groupby("sentiment_simple")["Closed PnL"].std().reindex(ORDER3)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(vol.index, vol.values, color=COLORS3)
    ax.set_title("PnL Volatility (Std. Dev.) by Sentiment", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_volatility_by_sentiment.png", dpi=150)
    plt.close()

    # Fig 6: Cumulative PnL timeline
    daily = m.groupby("date").agg(
        daily_pnl=("Closed PnL", "sum"), daily_volume=("Size USD", "sum"),
        n_trades=("Closed PnL", "count"), fg_value=("value", "first"),
        sentiment_simple=("sentiment_simple", "first"),
    ).reset_index().sort_values("date")
    fig, ax1 = plt.subplots(figsize=(13, 5.5))
    ax1.plot(daily["date"], daily["daily_pnl"].cumsum(), color="#2C3E50", linewidth=1.8)
    ax1.set_title("Cumulative Trader PnL Over Time vs. Market Sentiment", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/06_cumulative_pnl_timeline.png", dpi=150)
    plt.close()
    daily.to_csv("daily_summary.csv", index=False)

    # Fig 7: Coin-level
    top_coins = m.groupby("Coin")["Size USD"].sum().sort_values(ascending=False).head(5).index.tolist()
    sub = m[m["Coin"].isin(top_coins)]
    piv = sub.groupby(["Coin", "sentiment_simple"])["Closed PnL"].mean().unstack().reindex(columns=ORDER3).reindex(top_coins)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    piv.plot(kind="bar", ax=ax, color=COLORS3)
    ax.set_title("Average PnL per Trade by Coin and Sentiment (Top 5 Coins by Volume)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/07_coin_sentiment_pnl.png", dpi=150)
    plt.close()

    # Correlations
    corr_pnl = daily["daily_pnl"].corr(daily["fg_value"])
    corr_vol = daily["daily_volume"].corr(daily["fg_value"])
    corr_trades = daily["n_trades"].corr(daily["fg_value"])
    print(f"\nCorrelation daily PnL vs FG score:    {corr_pnl:.3f}")
    print(f"Correlation daily volume vs FG score: {corr_vol:.3f}")
    print(f"Correlation daily trades vs FG score: {corr_trades:.3f}")


if __name__ == "__main__":
    merged = load_and_merge()
    merged.to_csv("merged_trades_sentiment.csv", index=False)
    summary_tables(merged)
    make_charts(merged)
    print("\nDone. Charts saved to ./charts, cleaned data saved to ./merged_trades_sentiment.csv and ./daily_summary.csv")
