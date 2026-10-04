"""
Phase 3: Strategy Backtester
------------------------------
Tests your trading strategy on HISTORICAL data before risking real money.

Strategy we're testing: "Volume Spike Momentum"
  → BUY  when Z-score > 2.0 AND price closed UP on that day (accumulation)
  → SELL when Z-score > 2.0 AND price closed DOWN (distribution) OR after N days

What you'll learn:
  - How to simulate trades on historical data
  - Key performance metrics: Sharpe ratio, max drawdown, win rate
  - Why backtesting is critical (and its limitations)

Metrics explained:
  Sharpe Ratio   : Return per unit of risk. > 1.0 is decent, > 2.0 is excellent
  Max Drawdown   : Worst peak-to-trough loss. Tells you the worst pain you'd feel
  Win Rate       : % of trades that were profitable
  CAGR           : Compound Annual Growth Rate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase1_data.fetcher import fetch_stock_data, load_data
from phase2_scanner.volume_scanner import compute_volume_zscore, classify_signal


# ── Strategy Parameters (tune these) ─────────────────────────────────────────
INITIAL_CAPITAL  = 100_000   # ₹1 lakh starting capital
POSITION_SIZE    = 0.2       # Use 20% of capital per trade
HOLD_DAYS        = 10        # Max days to hold a position
Z_ENTRY          = 2.0       # Z-score threshold to enter
BROKERAGE_PCT    = 0.0003    # 0.03% per trade (Zerodha approx)
SLIPPAGE_PCT     = 0.001     # 0.1% slippage (realistic for retail)


# ── Trade Simulation ──────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, symbol: str = "STOCK") -> dict:
    """
    Simulate the volume-spike momentum strategy on historical data.

    Returns a dict with:
      - trades     : DataFrame of all executed trades
      - equity     : Daily equity curve
      - metrics    : Summary performance stats
    """
    df = compute_volume_zscore(df).copy()
    df["signal"] = df.apply(classify_signal, axis=1)
    df = df.dropna()

    capital    = INITIAL_CAPITAL
    equity     = [capital]
    dates      = [df.index[0]]
    trades     = []

    in_position    = False
    entry_price    = 0.0
    entry_date     = None
    shares_held    = 0
    days_in_trade  = 0

    for i in range(1, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        date = df.index[i]

        # ── Check EXIT conditions ─────────────────────────────────────────────
        if in_position:
            days_in_trade += 1
            exit_triggered = False
            exit_reason    = ""

            # Exit 1: Distribution signal (institutions selling)
            if row["signal"] == "distribution":
                exit_triggered = True
                exit_reason    = "distribution_signal"

            # Exit 2: Max holding period reached
            elif days_in_trade >= HOLD_DAYS:
                exit_triggered = True
                exit_reason    = "max_hold_period"

            if exit_triggered:
                exit_price  = row["open"] * (1 - SLIPPAGE_PCT)
                brokerage   = exit_price * shares_held * BROKERAGE_PCT
                proceeds    = exit_price * shares_held - brokerage
                pnl         = proceeds - (entry_price * shares_held)
                pnl_pct     = (pnl / (entry_price * shares_held)) * 100

                capital    += proceeds
                in_position = False

                trades.append({
                    "symbol"       : symbol,
                    "entry_date"   : entry_date,
                    "exit_date"    : date,
                    "entry_price"  : round(entry_price, 2),
                    "exit_price"   : round(exit_price, 2),
                    "shares"       : shares_held,
                    "pnl"          : round(pnl, 2),
                    "pnl_pct"      : round(pnl_pct, 2),
                    "days_held"    : days_in_trade,
                    "exit_reason"  : exit_reason,
                })

        # ── Check ENTRY conditions ────────────────────────────────────────────
        if not in_position and row["signal"] == "accumulation":
            entry_price   = row["close"] * (1 + SLIPPAGE_PCT)
            trade_capital = capital * POSITION_SIZE
            shares_held   = int(trade_capital / entry_price)

            if shares_held > 0:
                brokerage     = entry_price * shares_held * BROKERAGE_PCT
                capital      -= (entry_price * shares_held + brokerage)
                in_position   = True
                entry_date    = date
                days_in_trade = 0

        # Track equity (capital + current position value)
        position_value = (row["close"] * shares_held) if in_position else 0
        equity.append(capital + position_value)
        dates.append(date)

    # Build equity curve
    equity_series = pd.Series(equity, index=dates)

    # Build trades dataframe
    trades_df = pd.DataFrame(trades)

    # Compute metrics
    metrics = compute_metrics(equity_series, trades_df, INITIAL_CAPITAL)

    return {
        "equity"  : equity_series,
        "trades"  : trades_df,
        "metrics" : metrics,
    }


def compute_metrics(equity: pd.Series, trades: pd.DataFrame,
                    initial_capital: float) -> dict:
    """Compute standard trading performance metrics."""
    if equity.empty:
        return {}

    # Returns
    daily_returns = equity.pct_change().dropna()
    total_return  = (equity.iloc[-1] - initial_capital) / initial_capital * 100

    # Annualized metrics
    n_days   = (equity.index[-1] - equity.index[0]).days
    n_years  = n_days / 365.25
    cagr     = ((equity.iloc[-1] / initial_capital) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # Sharpe ratio (annualized, assuming 0 risk-free rate for simplicity)
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
              if daily_returns.std() > 0 else 0)

    # Max drawdown
    rolling_max  = equity.cummax()
    drawdown     = (equity - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    # Trade stats
    if not trades.empty:
        win_rate      = (trades["pnl"] > 0).mean() * 100
        avg_win       = trades[trades["pnl"] > 0]["pnl_pct"].mean() if (trades["pnl"] > 0).any() else 0
        avg_loss      = trades[trades["pnl"] < 0]["pnl_pct"].mean() if (trades["pnl"] < 0).any() else 0
        profit_factor = (trades[trades["pnl"] > 0]["pnl"].sum() /
                         abs(trades[trades["pnl"] < 0]["pnl"].sum())
                         if (trades["pnl"] < 0).any() else float("inf"))
        total_trades  = len(trades)
    else:
        win_rate = avg_win = avg_loss = profit_factor = total_trades = 0

    return {
        "total_return_pct" : round(total_return, 2),
        "cagr_pct"         : round(cagr, 2),
        "sharpe_ratio"     : round(sharpe, 2),
        "max_drawdown_pct" : round(max_drawdown, 2),
        "win_rate_pct"     : round(win_rate, 2),
        "avg_win_pct"      : round(avg_win, 2),
        "avg_loss_pct"     : round(avg_loss, 2),
        "profit_factor"    : round(profit_factor, 2),
        "total_trades"     : total_trades,
        "final_capital"    : round(equity.iloc[-1], 2),
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_backtest(result: dict, symbol: str):
    """Plot equity curve with drawdown and trade markers."""
    equity  = result["equity"]
    trades  = result["trades"]
    metrics = result["metrics"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)
    fig.suptitle(
        f"{symbol} — Volume Spike Strategy Backtest\n"
        f"Return: {metrics.get('total_return_pct',0):.1f}%  |  "
        f"Sharpe: {metrics.get('sharpe_ratio',0):.2f}  |  "
        f"Max DD: {metrics.get('max_drawdown_pct',0):.1f}%  |  "
        f"Win Rate: {metrics.get('win_rate_pct',0):.1f}%  |  "
        f"Trades: {metrics.get('total_trades',0)}",
        fontsize=12, fontweight="bold"
    )

    # ── Equity curve ──────────────────────────────────────────────────────────
    ax1.plot(equity.index, equity.values, color="#2196F3", linewidth=1.5,
             label="Portfolio Value")
    ax1.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5,
                label=f"Starting capital ₹{INITIAL_CAPITAL:,}")
    ax1.fill_between(equity.index, INITIAL_CAPITAL, equity.values,
                     where=equity.values >= INITIAL_CAPITAL,
                     alpha=0.2, color="green")
    ax1.fill_between(equity.index, INITIAL_CAPITAL, equity.values,
                     where=equity.values < INITIAL_CAPITAL,
                     alpha=0.2, color="red")
    ax1.set_ylabel("Portfolio Value (₹)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))

    # Mark entry/exit on equity curve
    if not trades.empty:
        for _, trade in trades.iterrows():
            if trade["entry_date"] in equity.index:
                ax1.axvline(trade["entry_date"], color="green", alpha=0.3, linewidth=0.8)
            if trade["exit_date"] in equity.index:
                ax1.axvline(trade["exit_date"], color="red", alpha=0.3, linewidth=0.8)

    # ── Drawdown chart ────────────────────────────────────────────────────────
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.5)
    ax2.set_ylabel("Drawdown %", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..",
                            "data", f"{symbol.replace('.NS','')}_backtest.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {out_path}")
    plt.show()


def print_metrics(metrics: dict, symbol: str):
    """Print a formatted performance report."""
    print(f"\n── Backtest Results: {symbol} ────────────────────────────────")
    print(f"  Starting capital  : ₹{INITIAL_CAPITAL:,}")
    print(f"  Final capital     : ₹{metrics.get('final_capital', 0):,.2f}")
    print(f"  Total return      : {metrics.get('total_return_pct', 0):+.2f}%")
    print(f"  CAGR              : {metrics.get('cagr_pct', 0):+.2f}%")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Sharpe ratio      : {metrics.get('sharpe_ratio', 0):.2f}  (>1.0 is good)")
    print(f"  Max drawdown      : {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Total trades      : {metrics.get('total_trades', 0)}")
    print(f"  Win rate          : {metrics.get('win_rate_pct', 0):.1f}%")
    print(f"  Avg win           : +{metrics.get('avg_win_pct', 0):.2f}%")
    print(f"  Avg loss          : {metrics.get('avg_loss_pct', 0):.2f}%")
    print(f"  Profit factor     : {metrics.get('profit_factor', 0):.2f}  (>1.5 is decent)")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  STRATEGY BACKTESTER")
    print("  Volume Spike Momentum Strategy on NSE")
    print("="*60 + "\n")

    TARGET = "RELIANCE.NS"

    print(f"[ STEP 1 ] Loading data for {TARGET}...\n")
    try:
        df = load_data(TARGET, "1d")
    except FileNotFoundError:
        df = fetch_stock_data(TARGET, period="2y", interval="1d")

    print(f"[ STEP 2 ] Running backtest on {len(df)} days of data...\n")
    result = run_backtest(df, TARGET)

    print_metrics(result["metrics"], TARGET)

    if not result["trades"].empty:
        print(f"\n  Last 5 trades:")
        cols = ["entry_date", "exit_date", "entry_price",
                "exit_price", "pnl", "pnl_pct", "exit_reason"]
        print(result["trades"][cols].tail(5).to_string(index=False))

    print(f"\n[ STEP 3 ] Plotting equity curve...\n")
    plot_backtest(result, TARGET)

    # ── Compare with Buy & Hold ───────────────────────────────────────────────
    bnh_return = ((df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]) * 100
    strat_return = result["metrics"].get("total_return_pct", 0)

    print(f"\n  Strategy vs Buy & Hold comparison:")
    print(f"  Buy & Hold return  : {bnh_return:+.2f}%")
    print(f"  Strategy return    : {strat_return:+.2f}%")
    print(f"  Alpha generated    : {strat_return - bnh_return:+.2f}%")

    print("\nPhase 3 complete.")
    print("Next → Run phase4_blockdeals/block_deal_tracker.py\n")
