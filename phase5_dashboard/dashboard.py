"""
Phase 5: Streamlit Live Dashboard
------------------------------------
Ties all 4 phases together into one interactive web dashboard.

Run with:
  streamlit run phase5_dashboard/dashboard.py

What you'll see:
  - Live stock data with volume anomaly flags
  - Institutional flow from block/bulk deals
  - Strategy backtest results per stock
  - Watchlist scanner with signal summary
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase1_data.fetcher       import fetch_stock_data, WATCHLIST
from phase2_scanner.volume_scanner import detect_anomalies, scan_watchlist
from phase3_backtest.backtester    import run_backtest
from phase4_blockdeals.block_deal_tracker import (
    fetch_block_deals, fetch_bulk_deals, analyze_institutional_flow
)


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Retail Edge Toolkit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .big-metric { font-size: 28px; font-weight: bold; }
    .green      { color: #00C853; }
    .red        { color: #FF1744; }
    .neutral    { color: #90CAF9; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("NSE Retail Edge Toolkit")
st.sidebar.markdown("*Level the playing field with data*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Watchlist Scanner", "Stock Deep Dive", "Backtest", "Institutional Flow"]
)

st.sidebar.divider()
selected_symbol = st.sidebar.selectbox(
    "Select Stock",
    [s.replace(".NS", "") for s in WATCHLIST],
    index=0
)
symbol_ns = f"{selected_symbol}.NS"


# ── Helper ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)   # cache for 5 minutes
def get_stock_data(symbol: str, period: str = "1y"):
    return fetch_stock_data(symbol, period=period, interval="1d")


@st.cache_data(ttl=300)
def get_scan_results():
    return scan_watchlist()


# ── Page: Watchlist Scanner ───────────────────────────────────────────────────
if page == "Watchlist Scanner":
    st.title("Watchlist Anomaly Scanner")
    st.markdown("**Scanning for institutional footprints across your watchlist**")
    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Stocks Tracked", len(WATCHLIST))
    col2.metric("Z-Score Threshold", "2.0σ")
    col3.metric("Rolling Window", "20 days")

    st.divider()

    with st.spinner("Scanning watchlist for anomalies..."):
        scan_df = get_scan_results()

    if scan_df.empty:
        st.warning("No data found. Fetching data now...")
        for sym in WATCHLIST[:3]:
            df = get_stock_data(sym)
            st.write(f"Fetched {sym}: {len(df)} rows")
    else:
        # Color rows by signal
        acc_stocks  = scan_df[scan_df["signal"] == "accumulation"]
        dist_stocks = scan_df[scan_df["signal"] == "distribution"]

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Accumulation Signals: {len(acc_stocks)}**")
            if not acc_stocks.empty:
                st.dataframe(acc_stocks, use_container_width=True)

        with col2:
            st.error(f"**Distribution Signals: {len(dist_stocks)}**")
            if not dist_stocks.empty:
                st.dataframe(dist_stocks, use_container_width=True)

        st.divider()
        st.subheader("All Scan Results (sorted by Z-score)")
        st.dataframe(scan_df, use_container_width=True)


# ── Page: Stock Deep Dive ─────────────────────────────────────────────────────
elif page == "Stock Deep Dive":
    st.title(f"{selected_symbol} — Deep Dive")
    st.markdown("**Price action + volume anomaly detection**")
    st.divider()

    with st.spinner(f"Loading {symbol_ns}..."):
        df = get_stock_data(symbol_ns)

    if df.empty:
        st.error("Could not fetch data. Check your connection.")
    else:
        df_analyzed, anomalies = detect_anomalies(df)

        # ── KPI Row ───────────────────────────────────────────────────────────
        latest       = df_analyzed.iloc[-1]
        prev         = df_analyzed.iloc[-2]
        price_change = latest["close"] - prev["close"]
        price_chg_p  = (price_change / prev["close"]) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"₹{latest['close']:,.2f}",
                  f"{price_chg_p:+.2f}%")
        c2.metric("Volume Today",  f"{latest['volume']:,.0f}",
                  f"Z: {latest.get('vol_zscore', 0):.2f}σ")
        c3.metric("52W High", f"₹{df['high'].max():,.2f}")
        c4.metric("Anomaly Days",  len(anomalies))

        st.divider()

        # ── Chart ─────────────────────────────────────────────────────────────
        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            vertical_spacing=0.05)

        plot_df = df_analyzed.tail(120)

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=plot_df.index,
            open=plot_df["open"], high=plot_df["high"],
            low=plot_df["low"],   close=plot_df["close"],
            name="OHLC", increasing_line_color="#00C853",
            decreasing_line_color="#FF1744"
        ), row=1, col=1)

        # Accumulation markers
        acc_df = plot_df[plot_df["signal"] == "accumulation"]
        dist_df = plot_df[plot_df["signal"] == "distribution"]

        fig.add_trace(go.Scatter(
            x=acc_df.index, y=acc_df["low"] * 0.99,
            mode="markers", marker=dict(symbol="triangle-up", size=12, color="#00C853"),
            name="Accumulation"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=dist_df.index, y=dist_df["high"] * 1.01,
            mode="markers", marker=dict(symbol="triangle-down", size=12, color="#FF1744"),
            name="Distribution"
        ), row=1, col=1)

        # Volume bars
        vol_colors = ["#00C853" if s == "accumulation"
                      else "#FF1744" if s == "distribution"
                      else "#546E7A" for s in plot_df["signal"]]
        fig.add_trace(go.Bar(
            x=plot_df.index, y=plot_df["volume"],
            marker_color=vol_colors, name="Volume"
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["vol_ma"],
            line=dict(color="orange", width=1.5),
            name="20d Avg Volume"
        ), row=2, col=1)

        fig.update_layout(
            height=600, template="plotly_dark",
            title=f"{selected_symbol} — Volume Anomaly Chart",
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Anomaly Table ─────────────────────────────────────────────────────
        if not anomalies.empty:
            st.subheader("Recent Anomaly Days")
            cols = ["close", "volume", "vol_zscore", "price_change_pct", "signal"]
            st.dataframe(anomalies[cols].tail(10), use_container_width=True)


# ── Page: Backtest ────────────────────────────────────────────────────────────
elif page == "Backtest":
    st.title(f"{selected_symbol} — Strategy Backtest")
    st.markdown("**Volume Spike Momentum Strategy**")
    st.divider()

    col1, col2 = st.columns(2)
    initial_cap  = col1.number_input("Initial Capital (₹)", value=100000, step=10000)
    position_sz  = col2.slider("Position Size %", 10, 50, 20) / 100

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            df = get_stock_data(symbol_ns, period="2y")

            if df.empty:
                st.error("No data available.")
            else:
                result = run_backtest(df, symbol_ns)
                metrics = result["metrics"]
                equity  = result["equity"]
                trades  = result["trades"]

                # ── Metrics ───────────────────────────────────────────────────
                c1, c2, c3, c4 = st.columns(4)
                ret_color = "normal" if metrics["total_return_pct"] >= 0 else "inverse"
                c1.metric("Total Return",   f"{metrics['total_return_pct']:+.1f}%")
                c2.metric("Sharpe Ratio",   f"{metrics['sharpe_ratio']:.2f}")
                c3.metric("Max Drawdown",   f"{metrics['max_drawdown_pct']:.1f}%")
                c4.metric("Win Rate",       f"{metrics['win_rate_pct']:.1f}%")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("CAGR",          f"{metrics['cagr_pct']:+.1f}%")
                c6.metric("Total Trades",   metrics["total_trades"])
                c7.metric("Profit Factor",  f"{metrics['profit_factor']:.2f}")
                c8.metric("Final Capital",  f"₹{metrics['final_capital']:,.0f}")

                st.divider()

                # ── Equity Curve ──────────────────────────────────────────────
                bnh_final = initial_cap * (1 + (df["close"].iloc[-1] - df["close"].iloc[0])
                                           / df["close"].iloc[0])
                bnh_equity = (df["close"] / df["close"].iloc[0]) * initial_cap

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity.index, y=equity.values,
                    fill="tozeroy", fillcolor="rgba(0,200,83,0.1)",
                    line=dict(color="#00C853", width=2),
                    name="Strategy"
                ))
                fig.add_trace(go.Scatter(
                    x=bnh_equity.index, y=bnh_equity.values,
                    line=dict(color="orange", width=1.5, dash="dash"),
                    name="Buy & Hold"
                ))
                fig.update_layout(
                    height=400, template="plotly_dark",
                    title="Equity Curve: Strategy vs Buy & Hold",
                    yaxis_title="Portfolio Value (₹)"
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Trades Table ──────────────────────────────────────────────
                if not trades.empty:
                    st.subheader("Trade Log")
                    st.dataframe(trades, use_container_width=True)


# ── Page: Institutional Flow ──────────────────────────────────────────────────
elif page == "Institutional Flow":
    st.title("Institutional Flow Tracker")
    st.markdown("**Block deals & bulk deals — the footprints institutions can't hide**")
    st.divider()

    st.info("""
    **What is this?**
    NSE is legally required to publish every trade where:
    - A single order is ≥ 5 lakh shares or ≥ ₹10 crore (**Block Deal**)
    - A single entity trades > 0.5% of total shares (**Bulk Deal**)

    These are the only trades where institutions CANNOT hide their activity.
    """)

    if st.button("Fetch Today's Institutional Activity", type="primary"):
        with st.spinner("Fetching block & bulk deals from NSE..."):
            today       = datetime.now()
            block_deals = fetch_block_deals(today)
            bulk_deals  = fetch_bulk_deals(today)
            all_deals   = pd.concat([block_deals, bulk_deals], ignore_index=True)

        if all_deals.empty:
            st.warning("No deals found for today (market may be closed or data unavailable).")
        else:
            flow = analyze_institutional_flow(all_deals)

            if not flow.empty:
                # Summary bar chart
                fig = px.bar(
                    flow, x="symbol",
                    y=["buy_value_cr", "sell_value_cr"],
                    title="Institutional Buy vs Sell (₹ Crore)",
                    color_discrete_map={"buy_value_cr": "#00C853", "sell_value_cr": "#FF1744"},
                    template="plotly_dark",
                    barmode="group"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Net Flow Summary")
                st.dataframe(flow, use_container_width=True)

            st.subheader("Raw Deal Data")
            st.dataframe(all_deals, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("""
**How to run each phase:**
```bash
# Phase 1: Fetch data
python phase1_data/fetcher.py

# Phase 2: Scan anomalies
python phase2_scanner/volume_scanner.py

# Phase 3: Backtest
python phase3_backtest/backtester.py

# Phase 4: Block deals
python phase4_blockdeals/block_deal_tracker.py

# Phase 5: Dashboard
streamlit run phase5_dashboard/dashboard.py
```
""")
