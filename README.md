# MarketLens — NSE Retail Intelligence Toolkit

> "The machine will keep taking from people who don't understand it. I choose to understand it."

MarketLens is a Python-based algorithmic trading research toolkit built for retail traders
on the Indian stock market (NSE). It detects institutional activity, flags volume anomalies,
backtests trading strategies, and visualizes everything in a live dashboard — giving retail
traders the same data-driven edge that institutions have.

---

## Why This Project Exists

Right now, 47+ algorithm trading firms operate inside NSE's own building (co-location).
Their trades execute in microseconds. Yours executes in 3-5 seconds on Zerodha.
In that gap, HFT algorithms detect your order, front-run it, and sell back to you at a
higher price — legally, thousands of times a day.

The only counter to an algorithm is your own algorithm.
MarketLens doesn't give you co-location. But it gives you:
- Visibility into what institutions are actually doing (block deals, volume spikes)
- A statistical edge through data — not emotions
- A way to test your strategy on years of data before risking a single rupee

---

## Project Name

**MarketLens**
Because it gives retail traders a clear lens into what's really happening in the market —
the activity that's usually invisible to the naked eye.

---

## Tech Stack

| Layer            | Technology          | Why it's used                                          |
|------------------|---------------------|--------------------------------------------------------|
| Language         | Python 3.12         | Industry standard for quant finance & data science     |
| Data Fetching    | yfinance, requests  | Free NSE/Yahoo Finance OHLCV + NSE web scraping        |
| Data Processing  | pandas, numpy       | Time-series manipulation, rolling statistics           |
| Statistics       | scipy, numpy        | Z-score, standard deviation, anomaly detection         |
| Visualization    | matplotlib, plotly  | Static charts (Phase 2-3) + interactive charts (Phase 5)|
| Dashboard        | Streamlit           | Build web dashboards in pure Python, no frontend needed |
| HTML Parsing     | beautifulsoup4      | Scraping NSE block deal pages                          |
| Technical Indicators | ta              | Pre-built RSI, MACD, Bollinger Bands etc.              |
| Data Storage     | CSV files           | Simple local persistence, no database needed           |

---

## Project Structure

```
MarketLens/
|
|-- phase1_data/
|   └── fetcher.py                  # Step 1: Fetch & store NSE stock data
|
|-- phase2_scanner/
|   └── volume_scanner.py           # Step 2: Detect institutional volume spikes
|
|-- phase3_backtest/
|   └── backtester.py               # Step 3: Backtest trading strategies
|
|-- phase4_blockdeals/
|   └── block_deal_tracker.py       # Step 4: Track block/bulk deals from NSE
|
|-- phase5_dashboard/
|   └── dashboard.py                # Step 5: Streamlit live dashboard
|
|-- demo_data_generator.py          # Generate offline demo data (no internet needed)
|-- requirements.txt                # All Python dependencies
|-- README.md                       # This file
|
└── data/                           # Auto-created — stores all CSVs and charts
    |-- RELIANCE_1d.csv
    |-- TCS_1d.csv
    |-- RELIANCE_anomaly.png
    |-- RELIANCE_backtest.png
    └── ...
```

---

## Full Workflow — How Everything Connects

```
                         MarketLens Workflow
                         ===================

  [NSE / Yahoo Finance]          [NSE Website]
         |                             |
         v                             v
  +--------------+           +------------------+
  |  Phase 1     |           |  Phase 4          |
  |  Data Fetcher|           |  Block Deal       |
  |              |           |  Tracker          |
  |  OHLCV data  |           |  Who bought what  |
  |  per stock   |           |  today (crores)   |
  +--------------+           +------------------+
         |                             |
         v                             |
  +--------------+                     |
  |  Phase 2     |                     |
  |  Volume      |                     |
  |  Anomaly     |                     |
  |  Scanner     |                     |
  |              |                     |
  |  Z-score >2? |                     |
  |  Accumulate  |                     |
  |  or Distribute|                    |
  +--------------+                     |
         |                             |
         v                             |
  +--------------+                     |
  |  Phase 3     |                     |
  |  Backtester  |                     |
  |              |                     |
  |  Simulate    |                     |
  |  trades on   |                     |
  |  history     |                     |
  |  Sharpe,DD,  |                     |
  |  Win Rate    |                     |
  +--------------+                     |
         |                             |
         +-----------------------------+
                       |
                       v
              +-----------------+
              |   Phase 5       |
              |   Dashboard     |
              |                 |
              |  Everything in  |
              |  one browser UI |
              +-----------------+
                       |
                       v
              http://localhost:8501
```

---

## Step-by-Step Explanation

### Phase 1 — Data Fetcher (`phase1_data/fetcher.py`)

**What it does:**
Connects to Yahoo Finance (which mirrors NSE data) and downloads 1 year of daily
OHLCV (Open, High, Low, Close, Volume) data for 10 major NSE stocks.
Saves each stock as a CSV file in the `/data` folder for offline use.

**What OHLCV means:**
- Open   = price when market opened that day
- High   = highest price touched that day
- Low    = lowest price touched that day
- Close  = price when market closed
- Volume = total shares traded that day

**Key concept learned:** Data pipelines, time-series data, API usage

**Output:** `/data/RELIANCE_1d.csv`, `/data/TCS_1d.csv` ... (10 files)

---

### Phase 2 — Volume Anomaly Scanner (`phase2_scanner/volume_scanner.py`)

**What it does:**
This is the core "institutional detection" engine. It uses Z-score statistics to find
days where trading volume was abnormally high — which is the footprint institutions
leave when they are buying or selling large quantities.

**How Z-score works:**
```
Z = (Today's Volume - 20-day Average Volume) / 20-day Standard Deviation

Z between -1 and +1  = Normal day, nothing unusual
Z > 2.0              = Top 2.5% of days — something big happened
Z > 3.0              = Top 0.1% of days — very likely institutional activity
```

**Signal Classification:**
```
High Volume (Z > 2) + Price went UP   = ACCUMULATION (institutions buying)
High Volume (Z > 2) + Price went DOWN = DISTRIBUTION  (institutions selling)
```

**Key concept learned:** Statistical anomaly detection, rolling windows, signal classification

**Output:** Chart with green triangles (buy pressure) and red triangles (sell pressure)
marked directly on the price chart

---

### Phase 3 — Backtest Engine (`phase3_backtest/backtester.py`)

**What it does:**
Takes the signals from Phase 2 and simulates what would have happened if you had
actually traded on them — using 1-2 years of historical data. It calculates realistic
results including brokerage fees and slippage.

**Strategy tested:** Volume Spike Momentum
- BUY  when Z-score > 2.0 AND price closed UP (accumulation signal)
- SELL when Z-score > 2.0 AND price closed DOWN, OR after 10 days max

**Performance Metrics explained:**
```
Sharpe Ratio   : Return divided by risk. > 1.0 is good, > 2.0 is excellent
Max Drawdown   : Worst loss from peak to bottom (e.g. -15% means at worst
                 your Rs.1 lakh dropped to Rs.85,000 before recovering)
Win Rate       : % of trades that made money
CAGR           : Compound Annual Growth Rate (annualized return)
Profit Factor  : Total profit / Total loss. > 1.5 is decent
Alpha          : Return above what Buy & Hold would have given you
```

**Key concept learned:** Strategy simulation, performance measurement, why backtesting
matters before risking real capital

**Output:** Equity curve chart showing portfolio value over time vs Buy & Hold benchmark

---

### Phase 4 — Block Deal Tracker (`phase4_blockdeals/block_deal_tracker.py`)

**What it does:**
NSE is legally required to publish every trade where:
- A single order is >= 5 lakh shares OR >= Rs.10 crore  (Block Deal)
- A single entity trades > 0.5% of total shares outstanding (Bulk Deal)

These are the only trades where institutions like Goldman Sachs, LIC, FIIs, and
mutual funds CANNOT hide. MarketLens scrapes this public data from NSE and
calculates the net institutional flow per stock.

**Example output:**
```
RELIANCE   BUY: Rs.570 Cr   SELL: Rs.0 Cr   NET: +Rs.570 Cr  → ACCUMULATION
TCS        BUY: Rs.0 Cr     SELL: Rs.165 Cr  NET: -Rs.165 Cr  → DISTRIBUTION
```

**Key concept learned:** Web scraping, HTTP sessions, data aggregation, reading
institutional intent from public data

**Output:** CSV file + flow summary showing which stocks institutions are
accumulating vs distributing today

---

### Phase 5 — Live Dashboard (`phase5_dashboard/dashboard.py`)

**What it does:**
Ties all 4 phases together into one interactive web dashboard that you can open
in any browser. Built with Streamlit — no HTML/CSS/JavaScript needed.

**4 pages in the dashboard:**

| Page | What you see |
|------|-------------|
| Watchlist Scanner | All 10 stocks ranked by Z-score with latest signal |
| Stock Deep Dive | Candlestick chart + volume bars with anomaly markers |
| Backtest | Equity curve, performance metrics, full trade log |
| Institutional Flow | Block deal bar chart + net buy/sell per institution |

**How to launch:**
```bash
streamlit run phase5_dashboard/dashboard.py
```
Then open: http://localhost:8501

**Key concept learned:** Building data applications, caching, interactive UI,
connecting a full pipeline end-to-end

---

## How to Run the Full Project

```bash
# Step 0: Install all dependencies (one time only)
pip install -r requirements.txt

# Step 1: Generate demo data (offline) OR fetch real NSE data (needs internet)
python demo_data_generator.py        # offline mode
python phase1_data/fetcher.py        # live mode (needs internet)

# Step 2: Run the volume scanner (see anomaly charts)
python phase2_scanner/volume_scanner.py

# Step 3: Run the backtest (see strategy performance)
python phase3_backtest/backtester.py

# Step 4: Check today's institutional activity
python phase4_blockdeals/block_deal_tracker.py

# Step 5: Launch the full dashboard (all phases in one UI)
streamlit run phase5_dashboard/dashboard.py
# Open browser: http://localhost:8501
```

---

## What This Project Demonstrates (for Interviews / Resume)

| Skill | Where it's shown |
|-------|-----------------|
| Data Engineering | Phase 1 — building a data pipeline from an API |
| Statistical Analysis | Phase 2 — Z-score anomaly detection |
| Quantitative Research | Phase 3 — strategy backtesting with risk metrics |
| Web Scraping | Phase 4 — scraping NSE public data |
| Full-Stack Data App | Phase 5 — Streamlit dashboard |
| Financial Domain Knowledge | All phases — OHLCV, Sharpe, drawdown, block deals |
| Python Proficiency | Entire codebase — pandas, numpy, matplotlib, plotly |

---

## Dependencies

```
yfinance          # NSE/Yahoo Finance data
pandas            # Data manipulation
numpy             # Numerical computing
scipy             # Statistical functions
matplotlib        # Static charts
seaborn           # Statistical visualization
plotly            # Interactive charts
streamlit         # Web dashboard
requests          # HTTP calls to NSE
beautifulsoup4    # HTML parsing
ta                # Technical indicators
nsepy             # NSE-specific data library
```

Install all: `pip install -r requirements.txt`

---

Built from scratch. No paid data. No Bloomberg terminal. Just Python.
