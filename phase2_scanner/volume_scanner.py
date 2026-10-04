"""
Phase 2: Volume Anomaly Scanner
--------------------------------
This is the core "institutional detection" tool.

The insight:
  When big players (FIIs, hedge funds, HFT algos) are accumulating a stock,
  they CANNOT hide their volume footprint. Even if they split orders,
  the total daily volume spikes above the historical average.

  We use Z-score statistics to flag these spikes:
    Z = (today's volume - 20-day avg volume) / 20-day std dev
    Z > 2.0  →  Unusual  (top ~2.5% of days)
    Z > 3.0  →  Very unusual  (top ~0.1% of days)

What you'll learn:
  - Z-score anomaly detection
  - Rolling window statistics
  - Combining price + volume signals
  - Classifying signals: accumulation vs distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase1_data.fetcher import fetch_stock_data, load_data, WATCHLIST


# ── Config ────────────────────────────────────────────────────────────────────
VOLUME_WINDOW   = 20    # rolling days to compute average volume
Z_THRESHOLD     = 2.0   # flag if Z-score exceeds this
STRONG_SIGNAL   = 3.0   # "strong" signal threshold


# ── Core Logic ────────────────────────────────────────────────────────────────

def compute_volume_zscore(df: pd.DataFrame, window: int = VOLUME_WINDOW) -> pd.DataFrame:
    """
    Add volume Z-score column to dataframe.

    Z-score tells you: "How many standard deviations above normal is today's volume?"
    Normal days: Z between -1 and +1
    Suspicious days: Z > 2 or Z < -2
    """
    df = df.copy()
    df["vol_ma"]     = df["volume"].rolling(window).mean()   # 20-day average
    df["vol_std"]    = df["volume"].rolling(window).std()    # 20-day std dev
    df["vol_zscore"] = (df["volume"] - df["vol_ma"]) / df["vol_std"]
    return df


def classify_signal(row) -> str:
    """
    Classify a volume spike as accumulation or distribution.

    Accumulation = high volume + price went UP  → institutions BUYING
    Distribution = high volume + price went DOWN → institutions SELLING
    """
    if abs(row["vol_zscore"]) < Z_THRESHOLD:
        return "normal"

    price_change = row["close"] - row["open"]

    if row["vol_zscore"] > Z_THRESHOLD:
        return "accumulation" if price_change > 0 else "distribution"
    else:
        return "low_volume"  # unusually quiet day


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Run full anomaly detection pipeline on a stock's data."""
    df = compute_volume_zscore(df)

    # Daily price change %
    df["price_change_pct"] = ((df["close"] - df["open"]) / df["open"]) * 100

    # Signal classification
    df["signal"] = df.apply(classify_signal, axis=1)

    # Flag only anomaly rows
    anomalies = df[df["signal"].isin(["accumulation", "distribution"])].copy()
    return df, anomalies


def scan_watchlist(symbols: list = None) -> pd.DataFrame:
    """
    Scan multiple stocks and return a ranked list of anomalies.
    Returns the most recent anomaly per stock, sorted by Z-score.
    """
    if symbols is None:
        symbols = WATCHLIST

    results = []

    for symbol in symbols:
        try:
            df = load_data(symbol, "1d")
        except FileNotFoundError:
            print(f"  {symbol}: no local data, fetching...")
            df = fetch_stock_data(symbol, period="1y", interval="1d")

        if df.empty or len(df) < VOLUME_WINDOW + 5:
            continue

        _, anomalies = detect_anomalies(df)

        if anomalies.empty:
            continue

        # Most recent anomaly for this stock
        latest = anomalies.iloc[-1]
        results.append({
            "symbol"         : symbol.replace(".NS", ""),
            "date"           : latest.name.date(),
            "close"          : round(latest["close"], 2),
            "volume"         : int(latest["volume"]),
            "vol_zscore"     : round(latest["vol_zscore"], 2),
            "price_chg_pct"  : round(latest["price_change_pct"], 2),
            "signal"         : latest["signal"],
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("vol_zscore", ascending=False)

    return result_df


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_volume_anomalies(df: pd.DataFrame, symbol: str, last_n_days: int = 120):
    """
    Plot price + volume chart with anomaly flags highlighted.
    This is the chart that shows you EXACTLY what institutions were doing.
    """
    df_plot = df.tail(last_n_days).copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={"height_ratios": [2, 1]},
                                   sharex=True)
    fig.suptitle(f"{symbol} — Volume Anomaly Detection (last {last_n_days} days)",
                 fontsize=14, fontweight="bold")

    # ── Price chart ───────────────────────────────────────────────────────────
    ax1.plot(df_plot.index, df_plot["close"], color="#2196F3", linewidth=1.5,
             label="Close Price")
    ax1.set_ylabel("Price (₹)", fontsize=11)
    ax1.grid(alpha=0.3)

    # Mark accumulation days on price chart
    acc = df_plot[df_plot["signal"] == "accumulation"]
    dist = df_plot[df_plot["signal"] == "distribution"]

    ax1.scatter(acc.index, acc["close"], color="green", marker="^",
                s=80, zorder=5, label="Accumulation (Z>2, price up)")
    ax1.scatter(dist.index, dist["close"], color="red", marker="v",
                s=80, zorder=5, label="Distribution (Z>2, price down)")
    ax1.legend(fontsize=9)

    # ── Volume chart ──────────────────────────────────────────────────────────
    colors = []
    for _, row in df_plot.iterrows():
        if row["signal"] == "accumulation":
            colors.append("green")
        elif row["signal"] == "distribution":
            colors.append("red")
        else:
            colors.append("#90CAF9")

    ax2.bar(df_plot.index, df_plot["volume"], color=colors, alpha=0.8)
    ax2.plot(df_plot.index, df_plot["vol_ma"], color="orange",
             linewidth=1.5, label="20-day avg volume")
    ax2.set_ylabel("Volume", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "..",
                            "data", f"{symbol.replace('.NS','')}_anomaly.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {out_path}")
    plt.show()


def print_anomaly_report(symbol: str, anomalies: pd.DataFrame):
    """Print a clean anomaly report for a stock."""
    clean = symbol.replace(".NS", "")
    print(f"\n── {clean} Anomaly Report ──────────────────────────────")
    print(f"  Total anomaly days found: {len(anomalies)}\n")

    acc  = anomalies[anomalies["signal"] == "accumulation"]
    dist = anomalies[anomalies["signal"] == "distribution"]

    print(f"  Accumulation days (institutions BUYING): {len(acc)}")
    print(f"  Distribution days (institutions SELLING): {len(dist)}\n")

    print("  Most recent 5 anomalies:")
    cols = ["close", "volume", "vol_zscore", "price_change_pct", "signal"]
    print(anomalies[cols].tail(5).to_string())

    if not acc.empty:
        print(f"\n  Strongest accumulation day:")
        strongest = acc.loc[acc["vol_zscore"].idxmax()]
        print(f"    Date    : {strongest.name.date()}")
        print(f"    Close   : ₹{strongest['close']:,.2f}")
        print(f"    Volume  : {strongest['volume']:,.0f}")
        print(f"    Z-score : {strongest['vol_zscore']:.2f}x above average")
        print(f"    Price Δ : +{strongest['price_change_pct']:.2f}%")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  VOLUME ANOMALY SCANNER")
    print("  Detecting institutional activity on NSE stocks")
    print("="*60 + "\n")

    # ── Single stock deep dive ────────────────────────────────────────────────
    TARGET = "RELIANCE.NS"
    print(f"[ STEP 1 ] Deep dive: {TARGET}\n")

    try:
        df = load_data(TARGET, "1d")
    except FileNotFoundError:
        df = fetch_stock_data(TARGET, period="1y", interval="1d")

    df_analyzed, anomalies = detect_anomalies(df)

    print_anomaly_report(TARGET, anomalies)
    plot_volume_anomalies(df_analyzed, TARGET)

    # ── Scan full watchlist ───────────────────────────────────────────────────
    print(f"\n[ STEP 2 ] Scanning full watchlist for anomalies...\n")
    scan_results = scan_watchlist()

    if scan_results.empty:
        print("  No anomalies found. Run phase1_data/fetcher.py first.")
    else:
        print("  ANOMALY SCAN RESULTS (sorted by Z-score):")
        print("  " + "-"*75)
        print(scan_results.to_string(index=False))
        print("\n  Legend:")
        print("  accumulation = high volume + price up   → institutions BUYING")
        print("  distribution = high volume + price down → institutions SELLING")

    print("\nPhase 2 complete.")
    print("Next → Run phase3_backtest/backtester.py\n")
