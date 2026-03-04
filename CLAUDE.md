# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Full pipeline (run in order when starting fresh or after data changes)
python3 feature_engineering.py   # ~5 min — regenerates output/features_{es,nq}_eth.parquet
python3 model.py                  # ~10 min — regenerates output/predictions_*.parquet
python3 evaluate.py               # ~1 min — prints OOS metrics, saves metrics_summary.csv + plots

# Baseline stats (outside main pipeline)
python3 baseline_stats.py        # prints inside/outside base rates by weekday/month

# Tests (must run from repo root; test_features requires data/ parquets, test_model requires output/ parquets)
python3 -m pytest tests/ -v
python3 -m pytest tests/test_features.py::test_rv_1d_known_date -v   # single test
```

Tests in `test_features.py` hit the actual parquet data files and serve as regression tests for known values (e.g. ES range on 2024-01-03 = 49.75). `test_model.py` tests PI coverage on synthetic data and walk-forward structure.

## Architecture

### Pipeline DAG

```
data/{es,nq}_1m.parquet          (1-min OHLCV, 2020-2025)
data/vix_cboe.parquet
data/economic_events.parquet
         │
         ▼
feature_engineering.py
  ├── build_eth_daily()           ETH = full Globex session, trade date rolls at 18:00 ET
  ├── build_rth_daily()           RTH = 09:30–16:14 ET, used only as feature source
  ├── compute_rv_features()       Group 1: HAR components (rv_1d, rv_5d, rv_22d, ...)
  ├── compute_range_features()    Group 2: atr_ratio, close_location, range_ma_*, ...
  ├── compute_volume_features()   Group 3: volume_zscore_22, first_hour_pct, ...
  ├── compute_session_features()  Group 4: nyam/london/asia_range_pct, session entropy
  ├── compute_eth_rth_cross_features()  Group 4b: rth_inside/outside_flag, eth_rth_divergence
  ├── compute_calendar_features() Group 5: fomc, nfp, high_impact_*
  ├── compute_vix_features()      Group 6: vix_close, vix_rv_spread, vix_percentile_252
  ├── compute_cross_instrument_features()  Group 7: es_nq_rv_ratio, es_nq_range_ratio
  ├── add_target()                y = log(range_{t+1}/range_t); inside/outside/neither labels
  └── compute_pattern_features()  Group 8: hl_containment, range_vs_max_{3,5,10}d,
                                  contraction_streak, inside_lag1, range_percentile_22, ...
         │
         ▼ output/features_{es,nq}_eth.parquet
         │
model.py — walk_forward()
  ├── Regression path: OLS (HAR) or Ridge on feature_cols → y_hat, PI (sigma)
  └── Classification path: LogisticRegression on class_feature_cols (always FEATURE_COLS_ALL)
       → p_inside, p_outside, p_neither
         │
         ▼ output/predictions_{es,nq}_{har,ridge}.parquet
         │
evaluate.py → metrics_summary.csv, output/plots/
```

### Key design decisions

**Two separate prediction heads.** The regression model (Ridge/OLS on `y`) and the classification model (logistic on `p_inside/p_outside`) are trained separately. `walk_forward()` takes `feature_cols` (for regression) and `class_feature_cols` (for classification, defaults to `FEATURE_COLS_ALL`). This lets HAR use only RV features for its continuous forecast while still benefiting from the full feature set for classification.

**Target = next-day log range ratio.** `y_t = log(range_{t+1}/range_t)`. All labels (`true_inside`, `true_outside`) in prediction outputs are **next-day labels** (`inside_{t+1}`), not same-day. Next-day labels are computed by shifting on the full `df` *before* `dropna`, to avoid calendar misalignment when Ridge drops more rows than HAR due to NaN features.

**ETH is primary.** Inside/outside classification is defined on ETH (Globex) H-L. RTH is used only as a feature dimension. Trade dates follow CME convention: bars from 18:00–23:59 ET roll to the next calendar day.

**Session labels in 1-min data.** The parquets have a `session` column (`ASIA`, `LONDON`, `NYAM`, `LUNCH`, `PM`, `OTHER`) used by `compute_session_features()`.

### `compute_pattern_features()` dependency
This function requires `inside` and `outside` columns from `add_target()`, so it must be called after `add_target()` in `main()`. It is not called inside `_build_features_for()`.

### Walk-forward structure
- Initial training window: 252 days (1 trading year)
- Expanding window: trains on all prior data at each step
- HAR test set: 1087 days (only needs rv_1d/5d/22d)
- Ridge/Full test set: 792 days (requires all 40+ features to be non-NaN)

When `class_feature_cols=FEATURE_COLS_ALL` is passed to HAR, `walk_forward` unions the feature sets for `dropna`, making HAR's test set also 792 days and the classification metrics identical between HAR and Ridge.

## Data

| File | Contents |
|------|----------|
| `data/es_1m.parquet` | ES 1-min OHLCV + `session` labels, 2020-08-31–2025-11-21 |
| `data/nq_1m.parquet` | NQ 1-min OHLCV + `session` labels, same range |
| `data/vix_cboe.parquet` | CBOE VIX daily OHLC, 1990–2025 |
| `data/economic_events.parquet` | High-impact USD events with `datetime_utc` and `impact` columns |

`inside.py` is a standalone exploratory script for CSV data (not part of the main pipeline).

## Current OOS performance (792 test days)

| Symbol | OOS R² (Ridge) | AUC inside | Brier skill inside |
|--------|---------------|-----------|-------------------|
| ES     | ~0.42         | 0.732     | +0.046            |
| NQ     | ~0.45         | 0.748     | +0.079            |

Inside day base rate ~10–12%. AUC measures rank discrimination; Brier skill is improvement over always-predicting the base rate.
