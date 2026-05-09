# Inside-Outside Day Prediction

Walk-forward volatility forecasting for ES and NQ futures. Predicts next-day log range ratio and classifies days as **inside**, **outside**, or **neither** relative to the prior day's high/low.

## Overview

An "inside day" occurs when the day's high-low range is contained within the prior day's range. These days represent volatility compression and are relatively rare (~10–12% base rate). This project builds a machine learning pipeline to predict them using a broad set of market microstructure features.

**Primary target:** `y = log(range_{t+1} / range_t)` — next-day log range ratio
**Sessions:** ETH/Globex and RTH are modeled as separate targets. ETH remains the original baseline; RTH has its own inside/outside labels and range-ratio target.

## Results (OOS, ~792 test days)

| Symbol | OOS R² (Ridge) | AUC inside | Brier Skill |
|--------|---------------|-----------|-------------|
| ES     | ~0.42         | 0.732     | +0.046      |
| NQ     | ~0.45         | 0.748     | +0.079      |

## Pipeline

```
data/{es,nq}_1m.parquet + vix_cboe.parquet + economic_events.parquet
         │
         ▼
feature_engineering.py     → output/features_{es,nq}_{eth,rth}.parquet
         │
         ▼
model.py (walk-forward)    → output/predictions_{es,nq}_{eth,rth}_{har,ridge}.parquet
         │
         ▼
evaluate.py                → output/metrics_summary.csv + output/plots/
```

## Usage

Run the full pipeline in order:

```bash
python3 feature_engineering.py   # ~5 min
python3 model.py                  # ~10 min
python3 evaluate.py               # ~1 min
```

Baseline stats (not part of the main pipeline):

```bash
python3 baseline_stats.py
```

Run safe tests (default skips parquet/pipeline-heavy tests):

```bash
python3 -m pytest tests/ -v
```

Run data-heavy regression tests only when local memory budget allows:

```bash
RUN_DATA_HEAVY=1 python3 -m pytest tests/ -m data_heavy -v
```

> Data-heavy tests require `data/` parquets and generated `output/` parquets.

## Models

**HAR (OLS)** — Heterogeneous Autoregressive model using only RV features (rv_1d, rv_5d, rv_22d). Classical volatility baseline with analytic prediction intervals.

**Ridge** — Regularised regression on the full 40+ feature set. Expanding walk-forward window with initial training period of 252 trading days.

Both models share the same **class-balanced logistic regression classifier** trained on `FEATURE_COLS_ALL` for inside/outside/neither probabilities. `HistGradientBoostingClassifier` is available as an experimental classifier option.

## Features (40+)

| Group | Features |
|-------|---------|
| RV | rv_1d, rv_5d, rv_22d, rv_ratio_1_5, rv_percentile_252, parkinson_vol |
| Range | range_pct_of_prev, atr_ratio, range_ma_5/22, close_location, overnight_gap |
| Volume | volume_prev, volume_zscore_22, volume_rth_vs_globex, volume_first_hour_pct |
| Session | nyam/london/asia_range_pct, session_vol_entropy |
| ETH/RTH | rth_pct_of_eth, rth_inside/outside_flag, eth_rth_divergence |
| Calendar | day_of_week, fomc, nfp, high_impact_* |
| VIX | vix_close, vix_change_1d, vix_rv_spread, vix_percentile_252 |
| Cross-instrument | es_nq_rv_ratio, es_nq_range_ratio |
| Pattern | hl_containment, range_vs_max_{3,5,10}d, contraction_streak, inside_lag1, range_percentile_22, nr4/nr7, wr4/wr7, inside/outside streaks |

## Data

| File | Contents |
|------|----------|
| `data/es_1m.parquet` | ES 1-min OHLCV + session labels, 2020–2025 |
| `data/nq_1m.parquet` | NQ 1-min OHLCV + session labels, 2020–2025 |
| `data/vix_cboe.parquet` | CBOE VIX daily OHLC, 1990–2025 |
| `data/economic_events.parquet` | High-impact USD events with datetime and impact level |

## Project Structure

```
├── feature_engineering.py   # Feature computation pipeline
├── model.py                 # Walk-forward HAR + Ridge models
├── evaluate.py              # OOS metrics and plots
├── baseline_stats.py        # Inside/outside base rates by weekday/month
├── inside.py                # Standalone exploratory script (CSV data)
├── tests/
│   ├── test_features.py     # Regression tests on known parquet values
│   └── test_model.py        # PI coverage and walk-forward structure tests
├── data/                    # Input parquets (not tracked)
└── output/                  # Generated predictions, metrics, plots
```
