"""
Demo Data Generator
--------------------
Generates realistic NSE-like stock data for offline demonstration.
Uses statistical properties of real Indian stocks (volatility, volume patterns).

Run this instead of phase1_data/fetcher.py when internet is unavailable.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

np.random.seed(42)

# Realistic starting prices for NSE stocks (approximate 2024 levels)
STOCK_CONFIG = {
    "RELIANCE"   : {"price": 2800, "volatility": 0.015, "avg_volume": 8_000_000},
    "TCS"        : {"price": 4100, "volatility": 0.012, "avg_volume": 3_500_000},
    "INFY"       : {"price": 1820, "volatility": 0.014, "avg_volume": 6_000_000},
    "HDFCBANK"   : {"price": 1720, "volatility": 0.013, "avg_volume": 12_000_000},
    "ICICIBANK"  : {"price": 1290, "volatility": 0.016, "avg_volume": 10_000_000},
    "SBIN"       : {"price": 810,  "volatility": 0.018, "avg_volume": 25_000_000},
    "WIPRO"      : {"price": 545,  "volatility": 0.015, "avg_volume": 7_000_000},
    "TATAMOTORS" : {"price": 990,  "volatility": 0.022, "avg_volume": 15_000_000},
    "ADANIENT"   : {"price": 2450, "volatility": 0.025, "avg_volume": 5_000_000},
    "BAJFINANCE" : {"price": 7200, "volatility": 0.017, "avg_volume": 2_500_000},
}

def generate_ohlcv(symbol: str, config: dict, days: int = 365) -> pd.DataFrame:
    """
    Generate realistic OHLCV data using Geometric Brownian Motion (GBM).
    GBM is the same model used in Black-Scholes options pricing.

    Adds realistic features:
    - Institutional volume spikes (3-5 per year)
    - Trending phases and mean reversion
    - Intraday high/low range proportional to volatility
    """
    vol   = config["volatility"]
    price = config["price"]
    avg_v = config["avg_volume"]

    # Business days only (no weekends)
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=days * 1.5)
    dates = pd.bdate_range(start=start_date, end=end_date)[-days:]

    closes = [price]
    for _ in range(days - 1):
        daily_return = np.random.normal(0.0003, vol)   # slight upward drift
        closes.append(closes[-1] * (1 + daily_return))

    closes = np.array(closes)

    # Generate OHLCV from close prices
    opens  = np.roll(closes, 1); opens[0] = closes[0]
    gaps   = np.random.normal(0, vol * 0.3, days)       # overnight gaps
    opens  = opens * (1 + gaps)

    daily_range = np.abs(np.random.normal(0, vol * 1.2, days))
    highs  = np.maximum(opens, closes) * (1 + daily_range)
    lows   = np.minimum(opens, closes) * (1 - daily_range)

    # Base volume with realistic noise
    volumes = np.random.lognormal(np.log(avg_v), 0.4, days).astype(int)

    # Inject 4-6 institutional volume spikes (the "anomalies" we want to detect)
    n_spikes = np.random.randint(4, 7)
    spike_idx = np.random.choice(range(30, days - 5), n_spikes, replace=False)

    for idx in spike_idx:
        spike_mult   = np.random.uniform(3.0, 6.0)       # 3x to 6x volume spike
        volumes[idx] = int(avg_v * spike_mult)

        # 60% chance spike is accumulation (price up), 40% distribution (price down)
        if np.random.random() < 0.6:
            move = np.random.uniform(0.015, 0.04)         # +1.5% to +4%
            closes[idx] = opens[idx] * (1 + move)
            highs[idx]  = closes[idx] * (1 + move * 0.3)
        else:
            move = np.random.uniform(0.015, 0.04)
            closes[idx] = opens[idx] * (1 - move)
            lows[idx]   = closes[idx] * (1 - move * 0.3)

    df = pd.DataFrame({
        "open"   : np.round(opens,  2),
        "high"   : np.round(highs,  2),
        "low"    : np.round(lows,   2),
        "close"  : np.round(closes, 2),
        "volume" : volumes,
    }, index=dates)

    return df


def generate_all(days: int = 365):
    print("\n" + "="*55)
    print("  DEMO DATA GENERATOR")
    print("  Creating realistic NSE stock data (offline mode)")
    print("="*55 + "\n")

    for symbol, config in STOCK_CONFIG.items():
        df = generate_ohlcv(symbol, config, days=days)
        path = os.path.join(DATA_DIR, f"{symbol}_1d.csv")
        df.to_csv(path)
        print(f"  {symbol:12s} | {len(df)} days | "
              f"Price: {df['close'].iloc[0]:.0f} -> {df['close'].iloc[-1]:.0f} | "
              f"Spikes injected: {(df['volume'] > df['volume'].mean()*2.5).sum()}")

    print(f"\n  Data saved to /data  ({len(STOCK_CONFIG)} stocks x {days} days)")
    print("  Run phase2_scanner/volume_scanner.py next.\n")


if __name__ == "__main__":
    generate_all(days=365)
