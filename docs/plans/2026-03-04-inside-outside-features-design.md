# Inside / Outside Day Prediction — Feature & Model Design

**Date:** 2026-03-04 (revised: ETH-primary)
**Instruments:** ES, NQ (1-min OHLCV, Sep 2020 – Nov 2025)
**External data:** CBOE VIX daily (`vix_cboe.parquet`), USD economic calendar (`economic_events.parquet`)

---

## 1. Problem Framing

Inside and outside days are range-contraction and range-expansion events respectively. Rather than a direct binary classifier, we model the **continuous log-ratio of tomorrow's range to today's range**:

```
y = log( range_t+1 / range_t )
```

This is approximately normally distributed, which enables:

- **Point forecast** + **OLS prediction interval** (frequentist CI) directly from the regression
- **P(inside) / P(outside)** by integrating the Gaussian predictive distribution against the inside/outside thresholds

### Session definitions

**Primary: ETH (Extended Trading Hours) = full Globex session**
- 18:00 ET (prev calendar day) → 17:00 ET, excluding 17:00–17:59 maintenance break
- Trade date convention: bars from 18:00–23:59 belong to the *next* calendar day
- ES ETH: ~1,380 bars/day (full), min threshold 100 bars
- Inside/outside defined on ETH H-L range (base rates: ~11.5% / ~12.9%)

**Secondary: RTH (Regular Trading Hours) = 09:30–16:14 ET**
- Remains available as a *feature dimension* within ETH days, not as a separate prediction target
- RTH contributes ~81% of the ETH range on average (median 84%), overnight the remaining ~19%
- 81 days (out of ~1,350) where ETH was an outside day but RTH alone was "neither" — pure overnight-driven moves invisible to RTH-only models

Both sessions' inside/outside classifications are output, but **ETH is the prediction target**.

---

## 2. Baseline Model (Literature Benchmark)

**HAR-RV** (Corsi 2009 — Heterogeneous Autoregressive Realized Variance):

```
RV_t+1 = α + β_d·RV_t + β_w·RV̄_{t-5} + β_m·RV̄_{t-22} + ε
```

Where `RV_t` = daily realized variance computed from 1-min returns: `Σ r_i²`.

This is the standard benchmark in realized volatility forecasting literature. All additional feature groups are evaluated as incremental improvements over this baseline.

---

## 3. Feature Groups

Features are engineered on day `t` to predict day `t+1`. All features must be computable at market close with no lookahead.

### Group 1 — Realized Volatility (HAR components)

| Feature | Formula / Description |
|---|---|
| `rv_1d` | `Σ r_i²` for all 1-min returns in RTH session |
| `rv_5d` | 5-day rolling mean of `rv_1d` |
| `rv_22d` | 22-day rolling mean of `rv_1d` |
| `rv_ratio_1_5` | `rv_1d / rv_5d` — current vol elevated vs recent? |
| `rv_percentile_252` | Percentile rank of `rv_1d` within trailing 252-day window |
| `parkinson_vol` | `(ln(H) - ln(L))² / (4·ln2)` — range-based vol, lower noise |

**Theoretical basis:** Volatility clustering (vol today predicts vol tomorrow). HAR captures heterogeneous memory across daily/weekly/monthly horizons.

### Group 2 — Range Structure

| Feature | Formula / Description |
|---|---|
| `range_pct_of_prev` | `(H_t - L_t) / (H_{t-1} - L_{t-1})` × 100 |
| `atr_ratio` | `range_t / ATR(14)_t` — range normalised to recent average |
| `range_ma_5` | 5-day rolling mean of daily range |
| `range_ma_22` | 22-day rolling mean of daily range |
| `close_location` | `(C_t - L_t) / (H_t - L_t)` — close position in today's range |
| `overnight_gap` | `(O_{t+1} - C_t) / range_t` — size of overnight gap (computed next open, use as lagged feature) |

**Theoretical basis:** Range contraction/expansion is autocorrelated. Close location signals directional conviction — closes near extremes tend to precede trending (outside) days.

### Group 3 — Volume

| Feature | Formula / Description |
|---|---|
| `volume_prev` | Total RTH volume on day `t` |
| `volume_zscore_22` | `(volume_t - mean_22) / std_22` — volume vs recent norm |
| `volume_rth_vs_globex` | RTH volume / total globex volume for same session-date |
| `volume_first_hour_pct` | NYAM (09:30–10:29) volume as % of total RTH volume |

**Theoretical basis:** Low volume days tend to produce inside days (thin market, no conviction). High volume with range expansion = outside day conditions.

### Group 4 — Intraday Session Structure

Unique advantage of 1-min data. Computed per session using `session` column labels.
All range percentages are relative to the **ETH daily range** (primary session).

| Feature | Formula / Description |
|---|---|
| `nyam_range_pct` | NYAM H-L / ETH daily H-L |
| `london_range_pct` | LONDON session H-L / ETH daily H-L |
| `asia_range_pct` | ASIA session H-L / ETH daily H-L |
| `session_vol_entropy` | Shannon entropy of per-session RV shares — high = vol spread evenly, low = concentrated burst |

**Theoretical basis:** A day where the morning session already consumed most of the ETH range leaves little room for expansion. Session entropy captures whether vol was a single burst or distributed.

### Group 4b — ETH / RTH Cross-session Features *(new)*

These features capture the *relationship* between the overnight move and the RTH session, which is the core of why ETH matters beyond RTH.

| Feature | Formula / Description |
|---|---|
| `rth_pct_of_eth` | RTH H-L / ETH H-L — how much of the ETH range occurred during RTH (mean ~81%) |
| `overnight_pct_of_eth` | 1 − rth_pct_of_eth — overnight share of ETH range |
| `overnight_gap_eth` | (RTH_Open − prev_ETH_Close) / prev_ETH_range — gap at RTH open relative to ETH context |
| `rth_inside_flag` | Binary: was today's RTH session an inside day within RTH context |
| `rth_outside_flag` | Binary: was today's RTH session an outside day within RTH context |
| `eth_rth_divergence` | ETH_inside XOR RTH_inside (or outside) — days where overnight flipped the classification |

**Theoretical basis:** Days where RTH was quiet (inside) but ETH had a large overnight move are fundamentally different from days where RTH drove all the expansion. The ETH/RTH divergence flag directly captures the 81 cases in our dataset where the overnight session caused an ETH outside day that would have been invisible to RTH-only analysis.

### Group 5 — Calendar & Macro Events

| Feature | Formula / Description |
|---|---|
| `day_of_week` | Categorical 0–4 (Monday = ~17% inside rate; Tuesday elevated outside) |
| `high_impact_today` | Binary: ≥1 high-impact USD event today |
| `high_impact_tomorrow` | Binary: ≥1 high-impact USD event tomorrow (anticipation effect) |
| `n_events_next_2d` | Count of high-impact events in next 2 calendar days |
| `is_fomc_day` | Binary: FOMC Statement or Federal Funds Rate day |
| `is_nfp_day` | Binary: Non-Farm Employment Change day (first Friday of month) |
| `days_since_fomc` | Days since last FOMC decision — regime stability proxy |

**Theoretical basis:** Scheduled macro events systematically affect realised volatility. Pre-event days tend to see compression (inside days) as participants await the catalyst; event days often produce outside days.

### Group 6 — VIX / Implied Volatility

Source: `vix_cboe.parquet` (CBOE VIX Index, 1990–present).

| Feature | Formula / Description |
|---|---|
| `vix_close` | VIX index close on day `t` |
| `vix_change_1d` | `VIX_t - VIX_{t-1}` — vol expanding or contracting? |
| `vix_rv_spread` | `VIX_t - sqrt(rv_22d · 252)` — implied vs realised vol premium |
| `vix_percentile_252` | Percentile rank of VIX within trailing year |

**Theoretical basis:** VIX prices the market's expectation of near-term realised vol. A VIX that is elevated relative to realised vol (high risk premium) is a strong signal for range expansion. Low VIX relative to recent history is a contraction signal.

### Group 7 — Cross-instrument (ES ↔ NQ)

| Feature | Formula / Description |
|---|---|
| `es_nq_rv_ratio` | `rv_1d_ES / rv_1d_NQ` — relative vol between indices |
| `es_nq_range_ratio` | `range_ES / range_NQ` — structural divergence |

**Theoretical basis:** When ES and NQ decouple (one contracts while the other expands), it signals sector rotation rather than broad market vol — a useful regime indicator.

---

## 4. Modelling Architecture

### Pipeline

```
1-min parquet  →  feature_engineering.py  →  features_es_eth.parquet  ← PRIMARY
                                          →  features_nq_eth.parquet
                                          (RTH bars computed internally as features,
                                           not output separately)
                         ↓
               model.py  →  predictions_es_{har,ridge}.parquet
                         →  predictions_nq_{har,ridge}.parquet
                         ↓
               evaluate.py  →  OOS R², PI coverage, Brier, DM test,
                                feature importance, calibration plots
```

### Validation

- **Walk-forward expanding window**: train on first 252 days, test on day 253; expand by 1 day each step
- **No lookahead**: all features computed strictly from data available at day `t` close
- **Metrics**: OOS R², RMSE, prediction interval coverage (should be ~90% for 90% PI), Brier score for P(inside)/P(outside)

### Regularisation

When adding all 30+ features, use **Ridge regression** (L2) to handle collinearity between RV components. Compare against plain OLS HAR baseline.

---

## 5. Output per Day

Each row in the output dataset contains:

```
trade_date | features (30+) | y_true | y_hat | pi_lower_90 | pi_upper_90 | p_inside | p_outside | p_neither
```

---

## 6. Implementation Plan

Steps (to be detailed in writing-plans):

1. `feature_engineering.py` — compute all feature groups, output one parquet per instrument/session
2. `model.py` — HAR baseline + full Ridge model, walk-forward validation
3. `evaluate.py` — OOS metrics, calibration plots, feature importance

---

## 7. Data Sources Summary

| File | Content | Date range |
|---|---|---|
| `data/es_1m.parquet` | ES 1-min OHLCV + session labels | 2020-08-31 – 2025-11-21 |
| `data/nq_1m.parquet` | NQ 1-min OHLCV + session labels | 2020-08-31 – 2025-11-21 |
| `data/vix_cboe.parquet` | CBOE VIX daily OHLC | 1990-01-02 – 2025-11-21 |
| `data/economic_events.parquet` | High-impact USD events | 2020-09-01 – 2025-11-27 |
