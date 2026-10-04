"""
Phase 1: NSE Data Fetcher
--------------------------
Fetches historical OHLCV data for NSE stocks using yfinance.
NSE tickers on yfinance use the format: SYMBOL.NS (e.g., RELIANCE.NS)

What you'll learn:
- How market data is structured (OHLCV)
- How to fetch and store it locally
- Basic data quality checks
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta


# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Popular NSE stocks to track
WATCHLIST = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "WIPRO.NS",
    "TATAMOTORS.NS",
    "ADANIENT.NS",
    "BAJFINANCE.NS",
]


# ── Core Functions ────────────────────────────────────────────────────────────

def fetch_stock_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data for a single NSE stock.

    Args:
        symbol  : NSE ticker e.g. "RELIANCE.NS"
        period  : how far back — "1d","5d","1mo","3mo","6mo","1y","2y","5y","10y"
        interval: candle size  — "1m","5m","15m","1h","1d","1wk","1mo"

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    print(f"Fetching {symbol} | period={period} interval={interval} ...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        print(f"  WARNING: No data returned for {symbol}")
        return df

    # Clean up — drop timezone info, keep only OHLCV
    df.index = df.index.tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]

    print(f"  Got {len(df)} rows | {df.index[0].date()} to {df.index[-1].date()}")
    return df


def save_data(df: pd.DataFrame, symbol: str, interval: str = "1d"):
    """Save fetched data to CSV in the data/ directory."""
    clean_symbol = symbol.replace(".NS", "")
    filename = f"{clean_symbol}_{interval}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath)
    print(f"  Saved → {filepath}")


def load_data(symbol: str, interval: str = "1d") -> pd.DataFrame:
    """Load previously saved data from CSV."""
    clean_symbol = symbol.replace(".NS", "")
    filepath = os.path.join(DATA_DIR, f"{clean_symbol}_{interval}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved data for {symbol}. Run fetch first.")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def fetch_and_save_watchlist(period: str = "1y", interval: str = "1d"):
    """Fetch and save data for all stocks in the watchlist."""
    print(f"\n{'='*50}")
    print(f"Fetching watchlist: {len(WATCHLIST)} stocks")
    print(f"{'='*50}\n")

    results = {}
    for symbol in WATCHLIST:
        df = fetch_stock_data(symbol, period=period, interval=interval)
        if not df.empty:
            save_data(df, symbol, interval)
            results[symbol] = df
        print()

    print(f"Done. Fetched {len(results)}/{len(WATCHLIST)} stocks successfully.\n")
    return results


def show_data_summary(df: pd.DataFrame, symbol: str):
    """Print a human-readable summary of the fetched data."""
    print(f"\n── {symbol} Data Summary ──────────────────────")
    print(f"  Rows       : {len(df)}")
    print(f"  Date range : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Latest close : ₹{df['close'].iloc[-1]:,.2f}")
    print(f"  52w High   : ₹{df['high'].max():,.2f}")
    print(f"  52w Low    : ₹{df['low'].min():,.2f}")
    print(f"  Avg Volume : {df['volume'].mean():,.0f} shares/day")
    print(f"\n  Last 5 days:")
    print(df.tail(5)[["open", "high", "low", "close", "volume"]].to_string())
    print()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Fetch a single stock and inspect it
    print("\n[ STEP 1 ] Fetch single stock — RELIANCE\n")
    df = fetch_stock_data("RELIANCE.NS", period="1y", interval="1d")
    show_data_summary(df, "RELIANCE.NS")
    save_data(df, "RELIANCE.NS", "1d")

    # Step 2: Load it back and confirm it works
    print("[ STEP 2 ] Load saved data back from disk\n")
    df_loaded = load_data("RELIANCE.NS", "1d")
    print(f"  Loaded {len(df_loaded)} rows from disk. ✓\n")

    # Step 3: Fetch the full watchlist
    print("[ STEP 3 ] Fetch full watchlist\n")
    all_data = fetch_and_save_watchlist(period="1y", interval="1d")

    print("Phase 1 complete. Data saved to /data directory.")
    print("Next → Run phase2_scanner/volume_scanner.py\n")
