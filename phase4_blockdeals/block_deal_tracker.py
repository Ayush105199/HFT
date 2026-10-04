"""
Phase 4: Block Deal & FII/DII Tracker
----------------------------------------
NSE publishes two critical datasets every day that retail traders ignore:

1. BLOCK DEALS  — trades > 5 lakh shares or > ₹10 crore in a single order
2. BULK DEALS   — trades where a single entity trades > 0.5% of total shares

These are the "footprints" that big institutions CANNOT hide.
When Goldman Sachs or Motilal Oswal buys 10 lakh shares of a stock,
NSE is legally required to publish that.

We scrape this public data and look for patterns BEFORE the price moves.

What you'll learn:
  - Web scraping NSE public data
  - Identifying institutional accumulation patterns
  - Correlating block deals with subsequent price movement
"""

import requests
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase1_data.fetcher import fetch_stock_data

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# NSE uses strict headers — without these, requests get blocked
NSE_HEADERS = {
    "User-Agent"     : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept"         : "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer"        : "https://www.nseindia.com",
    "Connection"     : "keep-alive",
}


# ── NSE Session Setup ─────────────────────────────────────────────────────────

def get_nse_session() -> requests.Session:
    """
    NSE blocks direct API calls without a cookie from the main page.
    We first hit the homepage to get a valid session cookie.
    """
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        # Get cookies by visiting the main page first
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
    except Exception as e:
        print(f"  Warning: Could not establish NSE session: {e}")
    return session


# ── Block / Bulk Deal Fetcher ─────────────────────────────────────────────────

def fetch_block_deals(date: datetime = None) -> pd.DataFrame:
    """
    Fetch block deals from NSE for a given date.
    Default: today's date.

    Block deal = single order ≥ 5 lakh shares OR ≥ ₹10 crore
    These are institutional-sized trades.
    """
    if date is None:
        date = datetime.now()

    date_str = date.strftime("%d-%m-%Y")
    session  = get_nse_session()

    url = f"https://www.nseindia.com/api/block-deal?date={date_str}"

    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            print(f"  No block deals found for {date_str}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["date"] = date.date()
        print(f"  Fetched {len(df)} block deals for {date_str}")
        return df

    except Exception as e:
        print(f"  Could not fetch block deals: {e}")
        print("  Using sample data for demonstration...")
        return _sample_block_deals(date)


def fetch_bulk_deals(date: datetime = None) -> pd.DataFrame:
    """
    Fetch bulk deals from NSE for a given date.
    Bulk deal = single entity trades > 0.5% of total shares outstanding.
    """
    if date is None:
        date = datetime.now()

    date_str = date.strftime("%d-%m-%Y")
    session  = get_nse_session()

    url = f"https://www.nseindia.com/api/bulk-deal?date={date_str}"

    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["date"] = date.date()
        print(f"  Fetched {len(df)} bulk deals for {date_str}")
        return df

    except Exception as e:
        print(f"  Could not fetch bulk deals: {e}")
        return _sample_bulk_deals(date)


def _sample_block_deals(date: datetime) -> pd.DataFrame:
    """Sample data for when NSE API is unavailable (demo mode)."""
    sample = [
        {"symbol": "RELIANCE",    "clientName": "Goldman Sachs India",
         "dealType": "BUY",  "quantity": 1500000, "price": 2847.50, "date": date.date()},
        {"symbol": "HDFCBANK",    "clientName": "FII - Vanguard",
         "dealType": "BUY",  "quantity": 800000,  "price": 1723.20, "date": date.date()},
        {"symbol": "TATAMOTORS",  "clientName": "Morgan Stanley",
         "dealType": "SELL", "quantity": 2200000, "price": 987.60,  "date": date.date()},
        {"symbol": "INFY",        "clientName": "BlackRock",
         "dealType": "BUY",  "quantity": 650000,  "price": 1834.75, "date": date.date()},
        {"symbol": "ADANIENT",    "clientName": "DII - HDFC MF",
         "dealType": "BUY",  "quantity": 950000,  "price": 2456.30, "date": date.date()},
        {"symbol": "SBIN",        "clientName": "LIC India",
         "dealType": "BUY",  "quantity": 3000000, "price": 812.45,  "date": date.date()},
        {"symbol": "WIPRO",       "clientName": "Societe Generale",
         "dealType": "SELL", "quantity": 1100000, "price": 543.80,  "date": date.date()},
        {"symbol": "ICICIBANK",   "clientName": "Fidelity Investments",
         "dealType": "BUY",  "quantity": 1200000, "price": 1298.60, "date": date.date()},
    ]
    return pd.DataFrame(sample)


def _sample_bulk_deals(date: datetime) -> pd.DataFrame:
    """Sample bulk deals data."""
    sample = [
        {"symbol": "RELIANCE",  "clientName": "Promoter Entity",
         "dealType": "BUY",  "quantity": 500000,  "price": 2851.00, "date": date.date()},
        {"symbol": "BAJFINANCE","clientName": "Nippon MF",
         "dealType": "BUY",  "quantity": 300000,  "price": 7234.50, "date": date.date()},
        {"symbol": "TCS",       "clientName": "UBS AG",
         "dealType": "SELL", "quantity": 400000,  "price": 4123.75, "date": date.date()},
    ]
    return pd.DataFrame(sample)


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_institutional_flow(deals: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate block + bulk deals by symbol to see net institutional flow.

    Positive flow = net buying → bullish signal
    Negative flow = net selling → bearish signal
    """
    if deals.empty:
        return pd.DataFrame()

    deals = deals.copy()

    # Normalize deal type column name (NSE uses different names)
    type_col = next((c for c in ["dealType", "BD_DT_DATE", "buySell", "BS_SELL_BUY_FLAG"]
                     if c in deals.columns), None)
    if type_col and type_col != "dealType":
        deals = deals.rename(columns={type_col: "dealType"})

    # Normalize symbol column
    sym_col = next((c for c in ["symbol", "BD_SYMBOL", "Symbol"]
                    if c in deals.columns), None)
    if sym_col and sym_col != "symbol":
        deals = deals.rename(columns={sym_col: "symbol"})

    # Normalize quantity and price columns
    qty_col   = next((c for c in ["quantity", "BD_QTY_TRD", "Quantity"] if c in deals.columns), None)
    price_col = next((c for c in ["price", "BD_TP_WATP", "Price"]       if c in deals.columns), None)

    if qty_col and qty_col != "quantity":
        deals = deals.rename(columns={qty_col: "quantity"})
    if price_col and price_col != "price":
        deals = deals.rename(columns={price_col: "price"})

    deals["quantity"] = pd.to_numeric(deals.get("quantity", 0), errors="coerce").fillna(0)
    deals["price"]    = pd.to_numeric(deals.get("price", 0),    errors="coerce").fillna(0)
    deals["value"]    = deals["quantity"] * deals["price"]

    # Net flow per symbol
    summary = []
    for symbol, group in deals.groupby("symbol"):
        buy_val  = group[group["dealType"].str.upper() == "BUY"]["value"].sum()
        sell_val = group[group["dealType"].str.upper() == "SELL"]["value"].sum()
        net_flow = buy_val - sell_val

        summary.append({
            "symbol"        : symbol,
            "buy_value_cr"  : round(buy_val  / 1e7, 2),
            "sell_value_cr" : round(sell_val / 1e7, 2),
            "net_flow_cr"   : round(net_flow / 1e7, 2),
            "signal"        : "ACCUMULATION" if net_flow > 0 else "DISTRIBUTION",
            "buy_deals"     : len(group[group["dealType"].str.upper() == "BUY"]),
            "sell_deals"    : len(group[group["dealType"].str.upper() == "SELL"]),
        })

    result = pd.DataFrame(summary)
    result = result.sort_values("net_flow_cr", ascending=False)
    return result


def check_price_impact(symbol: str, deal_date: datetime,
                        days_after: int = 5) -> dict:
    """
    Measure price change AFTER a block deal.
    This validates whether block deal signals actually predict price moves.
    """
    try:
        df = fetch_stock_data(f"{symbol}.NS", period="3mo", interval="1d")
        if df.empty:
            return {}

        df = df[df.index.date >= deal_date.date()]
        if len(df) < 2:
            return {}

        price_on_deal   = df["close"].iloc[0]
        price_after     = df["close"].iloc[min(days_after, len(df)-1)]
        change_pct      = ((price_after - price_on_deal) / price_on_deal) * 100

        return {
            "symbol"          : symbol,
            "deal_date_price" : round(price_on_deal, 2),
            f"price_{days_after}d_later": round(price_after, 2),
            f"change_{days_after}d_pct" : round(change_pct, 2),
        }
    except Exception:
        return {}


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BLOCK DEAL & INSTITUTIONAL FLOW TRACKER")
    print("  Tracking the footprints institutions can't hide")
    print("="*60 + "\n")

    today = datetime.now()

    # ── Step 1: Fetch today's deals ───────────────────────────────────────────
    print("[ STEP 1 ] Fetching Block Deals...\n")
    block_deals = fetch_block_deals(today)

    print("\n[ STEP 2 ] Fetching Bulk Deals...\n")
    bulk_deals  = fetch_bulk_deals(today)

    # ── Step 2: Combine and analyze ───────────────────────────────────────────
    all_deals = pd.concat([block_deals, bulk_deals], ignore_index=True)

    print("\n[ STEP 3 ] Analyzing Institutional Flow...\n")
    flow = analyze_institutional_flow(all_deals)

    if not flow.empty:
        print("  INSTITUTIONAL FLOW SUMMARY (₹ Crore):")
        print("  " + "─"*70)
        print(flow.to_string(index=False))

        # Top accumulation targets
        acc = flow[flow["signal"] == "ACCUMULATION"]
        dist = flow[flow["signal"] == "DISTRIBUTION"]

        print(f"\n  Stocks being ACCUMULATED: {list(acc['symbol'])[:5]}")
        print(f"  Stocks being DISTRIBUTED: {list(dist['symbol'])[:5]}")

    # ── Step 3: Save to file ──────────────────────────────────────────────────
    out_path = os.path.join(DATA_DIR, f"block_deals_{today.strftime('%Y%m%d')}.csv")
    all_deals.to_csv(out_path, index=False)
    print(f"\n  Raw deals saved → {out_path}")

    if not flow.empty:
        flow_path = os.path.join(DATA_DIR, f"inst_flow_{today.strftime('%Y%m%d')}.csv")
        flow.to_csv(flow_path, index=False)
        print(f"  Flow summary saved → {flow_path}")

    print("\nPhase 4 complete.")
    print("Next → Run phase5_dashboard/dashboard.py\n")
