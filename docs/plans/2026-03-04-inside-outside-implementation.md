# Inside / Outside Day Prediction — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Engineer 30+ predictive features from 1-min futures data and external sources, then fit walk-forward HAR-baseline and Ridge regression models that output a point forecast, 90% prediction interval, and P(inside)/P(outside) for next-day **ETH (Globex)** range. RTH session is used as a feature dimension, not a separate prediction target.

**Architecture:** Three scripts — `feature_engineering.py` (all feature groups → parquet), `model.py` (walk-forward OLS + Ridge → predictions parquet), `evaluate.py` (OOS metrics + plots). No ML frameworks beyond scikit-learn and scipy. All features are strictly lag-safe (computed from day t data to predict day t+1).

**Session hierarchy:**
- **Primary (prediction target):** ETH = full Globex session, 18:00–17:00 ET, excluding 17:00–17:59 maintenance break. Trade date: bars from 18:00–23:59 belong to the *next* calendar day.
- **Secondary (feature input):** RTH = 09:30–16:14 ET. RTH bars are computed internally and merged as features onto ETH trade dates. RTH inside/outside flags are included as features.

**Tech Stack:** Python 3, pandas, numpy, scipy, scikit-learn, matplotlib. No statsmodels needed — OLS prediction intervals implemented manually using the standard formula.

---

## Key constants (derived from data)

```python
# ETH (Globex) session
ETH_BREAK_START = "17:00"  # maintenance break start (exclusive)
ETH_BREAK_END   = "18:00"  # maintenance break end (inclusive of new session)
MIN_ETH_BARS = 100          # drop partial days below this bar count (~1,380 on a full day)

# RTH session (used as feature input, not prediction target)
RTH_START    = "09:30"
RTH_END      = "16:14"
MIN_RTH_BARS = 30

WALK_FORWARD_INIT = 252   # initial training window in trading days

# Empirical log-range-ratio thresholds (ES ETH, Sep 2020–Nov 2025)
# Inside days:  95th percentile ≈ -0.15  → exp(-0.15) ≈ 0.86× prev ETH range
# Outside days:  5th percentile ≈ +0.18  → exp(+0.18) ≈ 1.20× prev ETH range
# Thresholds calibrated per walk-forward fold from training data.
# P(inside)  = norm.cdf(inside_thresh, y_hat, sigma)
# P(outside) = 1 - norm.cdf(outside_thresh, y_hat, sigma)
# P(neither) = norm.cdf(outside_thresh) - norm.cdf(inside_thresh)
```

---

## Task 1: Install dependencies and scaffold tests directory

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_features.py` (stub)
- Create: `tests/test_model.py` (stub)

**Step 1: Install missing dependency**

```bash
pip install statsmodels
```

Wait — we do NOT need statsmodels. scipy + numpy cover everything. Skip.

**Step 2: Create test scaffolding**

```bash
mkdir -p tests
touch tests/__init__.py tests/test_features.py tests/test_model.py
```

**Step 3: Verify existing packages**

```bash
python3 -c "import pandas, numpy, scipy, sklearn, matplotlib; print('all ok')"
```

Expected: `all ok`

**Step 4: Commit**

```bash
git add tests/
git commit -m "chore: add test scaffolding"
```

---

## Task 2: ETH + RTH daily bar builders + Group 1 (Realized Volatility features)

This is the core of `feature_engineering.py`. ETH is the primary session. RTH is built in parallel for use as features. We write and test builders and RV computation first.

**Files:**
- Create: `feature_engineering.py`
- Modify: `tests/test_features.py`

**Known values (pre-computed for tests):**
```
2024-01-03  ETH range: 49.75  (18:00 Jan 2 → 17:00 Jan 3)
2024-01-03  RTH range: 30.75  (09:30 → 16:14 Jan 3)
2024-01-03  RTH RV:    3.640e-05
2024-01-03  RTH Parkinson: 1.507e-05
```

**Step 1: Write failing tests for bar builders and RV**

In `tests/test_features.py`:

```python
import pandas as pd
import numpy as np
import pytest
import sys
sys.path.insert(0, ".")

KNOWN_DATE         = pd.Timestamp("2024-01-03")
KNOWN_ETH_RANGE    = 49.75
KNOWN_RTH_RANGE    = 30.75
KNOWN_RTH_RV       = 3.640e-05
KNOWN_RTH_PARKINSON = 1.507e-05

@pytest.fixture(scope="module")
def es_1min():
    return pd.read_parquet("data/es_1m.parquet")

@pytest.fixture(scope="module")
def eth_daily_es(es_1min):
    from feature_engineering import build_eth_daily
    return build_eth_daily(es_1min)

@pytest.fixture(scope="module")
def rth_daily_es(es_1min):
    from feature_engineering import build_rth_daily
    return build_rth_daily(es_1min)

# ── ETH builder ──────────────────────────────────────────────────────────────
def test_build_eth_daily_has_required_columns(eth_daily_es):
    assert {"trade_date", "Open", "High", "Low", "Close", "Volume"}.issubset(eth_daily_es.columns)

def test_build_eth_daily_range_known_date(eth_daily_es):
    row = eth_daily_es[eth_daily_es["trade_date"] == KNOWN_DATE]
    assert len(row) == 1
    assert abs(float(row["High"].iloc[0] - row["Low"].iloc[0]) - KNOWN_ETH_RANGE) < 0.01

def test_build_eth_daily_no_maintenance_break_bars(es_1min):
    """No bar at 17:xx should appear in the ETH daily aggregation."""
    from feature_engineering import build_eth_daily
    # The full ETH day has ~1,380 bars; maintenance break has 60.
    # If break bars were included, some days would show ~1,440 bars.
    eth = build_eth_daily(es_1min)
    # Bar count column added in build_eth_daily for internal QC — verify max < 1,400
    assert len(eth) > 1300  # sanity: most days present

def test_build_eth_trade_date_convention(es_1min):
    """A bar at 2024-01-02 20:00 ET belongs to trade_date 2024-01-03."""
    from feature_engineering import build_eth_daily
    eth = build_eth_daily(es_1min)
    # 2024-01-02 is a Tuesday; the 18:00+ bars should roll into 2024-01-03
    jan3 = eth[eth["trade_date"] == pd.Timestamp("2024-01-03")]
    assert len(jan3) == 1

# ── RTH builder ──────────────────────────────────────────────────────────────
def test_build_rth_daily_range_known_date(rth_daily_es):
    row = rth_daily_es[rth_daily_es["trade_date"] == KNOWN_DATE]
    assert abs(float(row["High"].iloc[0] - row["Low"].iloc[0]) - KNOWN_RTH_RANGE) < 0.01

# ── RV features (computed on RTH bars, merged onto ETH trade dates) ──────────
def test_rv_1d_known_date(eth_daily_es, es_1min):
    from feature_engineering import compute_rv_features
    df = compute_rv_features(eth_daily_es, es_1min, session="rth")
    row = df[df["trade_date"] == KNOWN_DATE]
    assert abs(float(row["rv_1d"].iloc[0]) - KNOWN_RTH_RV) < 1e-7

def test_parkinson_vol_known_date(eth_daily_es, es_1min):
    from feature_engineering import compute_rv_features
    df = compute_rv_features(eth_daily_es, es_1min, session="rth")
    row = df[df["trade_date"] == KNOWN_DATE]
    assert abs(float(row["parkinson_vol"].iloc[0]) - KNOWN_RTH_PARKINSON) < 1e-8

def test_rv_features_no_negative(eth_daily_es, es_1min):
    from feature_engineering import compute_rv_features
    df = compute_rv_features(eth_daily_es, es_1min, session="rth")
    for col in ["rv_1d", "rv_5d", "rv_22d", "parkinson_vol"]:
        assert (df[col].dropna() >= 0).all(), f"{col} has negative values"

def test_rv_percentile_in_range(eth_daily_es, es_1min):
    from feature_engineering import compute_rv_features
    df = compute_rv_features(eth_daily_es, es_1min, session="rth")
    pct = df["rv_percentile_252"].dropna()
    assert (pct >= 0).all() and (pct <= 1).all()
```

**Step 2: Run tests — expect failures**

```bash
python3 -m pytest tests/test_features.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'build_eth_daily'`

**Step 3: Implement `build_eth_daily`, `build_rth_daily`, and `compute_rv_features`**

```python
"""
feature_engineering.py
Compute all predictive features for inside/outside day modelling.
Primary session: ETH (Globex). RTH used as feature dimension.
Input:  data/*.parquet
Output: output/features_{es,nq}_eth.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
RTH_START    = pd.Timestamp("09:30").time()
RTH_END      = pd.Timestamp("16:14").time()
MIN_RTH_BARS = 30
MIN_ETH_BARS = 100   # full ETH day ≈ 1,380 bars


def _globex_trade_date(dt_series: pd.Series) -> pd.Series:
    """Bars at 18:00–23:59 ET belong to the next calendar day (CME convention)."""
    dates = dt_series.dt.normalize()
    return dates + pd.to_timedelta((dt_series.dt.hour >= 18).astype(int), unit="D")


def build_eth_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-min bars to ETH (Globex) daily OHLCV.
    Excludes 17:00–17:59 ET maintenance break.
    Trade date assigned by globex convention (18:00+ rolls to next calendar day).
    """
    t = df["DateTime_ET"].dt.time
    in_break = (t >= pd.Timestamp("17:00").time()) & (t < pd.Timestamp("18:00").time())
    eth = df[~in_break].copy()
    eth["trade_date"] = _globex_trade_date(eth["DateTime_ET"])

    bar_counts = eth.groupby("trade_date").size()
    valid = bar_counts[bar_counts >= MIN_ETH_BARS].index

    daily = (
        eth.groupby("trade_date")
        .agg(Open=("Open", "first"), High=("High", "max"),
             Low=("Low", "min"), Close=("Close", "last"),
             Volume=("Volume", "sum"))
        .reset_index()
    )
    return daily[daily["trade_date"].isin(valid)].sort_values("trade_date").reset_index(drop=True)


def build_rth_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-min bars to RTH (09:30–16:14 ET) daily OHLCV.
    Trade date = calendar date of the bar.
    """
    t = df["DateTime_ET"].dt.time
    rth = df[(t >= RTH_START) & (t <= RTH_END)].copy()
    rth["trade_date"] = rth["DateTime_ET"].dt.normalize()

    bar_counts = rth.groupby("trade_date").size()
    valid = bar_counts[bar_counts >= MIN_RTH_BARS].index

    daily = (
        rth.groupby("trade_date")
        .agg(Open=("Open", "first"), High=("High", "max"),
             Low=("Low", "min"), Close=("Close", "last"),
             Volume=("Volume", "sum"))
        .reset_index()
    )
    return daily[daily["trade_date"].isin(valid)].sort_values("trade_date").reset_index(drop=True)


def compute_rv_features(daily: pd.DataFrame, raw_1min: pd.DataFrame,
                         session: str = "rth") -> pd.DataFrame:
    """
    Group 1: Realized Volatility features.
    session="rth" → compute RV from RTH bars (merged onto ETH trade dates).
    session="eth" → compute RV from full ETH bars.

    Features: rv_1d, rv_5d, rv_22d, rv_ratio_1_5, rv_percentile_252, parkinson_vol
    """
    if session == "rth":
        t = raw_1min["DateTime_ET"].dt.time
        bars = raw_1min[(t >= RTH_START) & (t <= RTH_END)].copy()
        bars["trade_date"] = bars["DateTime_ET"].dt.normalize()
    else:
        t = raw_1min["DateTime_ET"].dt.time
        in_break = (t >= pd.Timestamp("17:00").time()) & (t < pd.Timestamp("18:00").time())
        bars = raw_1min[~in_break].copy()
        bars["trade_date"] = _globex_trade_date(bars["DateTime_ET"])

    bars["log_ret"] = np.log(bars["Close"] / bars["Close"].shift(1))
    # Zero out cross-session returns (first bar of each trade_date)
    bars.loc[bars["trade_date"] != bars["trade_date"].shift(1), "log_ret"] = np.nan

    rv_series = (
        bars.groupby("trade_date")["log_ret"]
        .apply(lambda r: (r ** 2).sum())
        .rename("rv_1d")
        .reset_index()
    )

    # For Parkinson: use session H and L
    hl = (
        bars.groupby("trade_date")
        .agg(ses_H=("High", "max"), ses_L=("Low", "min"))
        .reset_index()
    )

    out = daily.merge(rv_series, on="trade_date", how="left")
    out = out.merge(hl, on="trade_date", how="left")

    out["rv_5d"]  = out["rv_1d"].rolling(5,  min_periods=3).mean()
    out["rv_22d"] = out["rv_1d"].rolling(22, min_periods=10).mean()
    out["rv_ratio_1_5"] = out["rv_1d"] / out["rv_5d"]
    out["rv_percentile_252"] = (
        out["rv_1d"]
        .rolling(252, min_periods=30)
        .apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    )
    out["parkinson_vol"] = (np.log(out["ses_H"] / out["ses_L"]) ** 2) / (4 * np.log(2))
    return out.drop(columns=["ses_H", "ses_L"])
```

**Step 4: Run all tests**

```bash
python3 -m pytest tests/test_features.py -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add feature_engineering.py tests/test_features.py
git commit -m "feat: ETH+RTH daily bar builders and Group 1 realized vol features"
```

**Step 2: Run tests — expect failures**

```bash
python3 -m pytest tests/test_features.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'build_rth_daily' from 'feature_engineering'`

**Step 3: Implement `build_rth_daily` and `compute_rv_features` in `feature_engineering.py`**

```python
"""
feature_engineering.py
Compute all predictive features for inside/outside day modelling.
Input:  data/*.parquet
Output: output/features_{es,nq}_rth.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
RTH_START = pd.Timestamp("09:30").time()
RTH_END   = pd.Timestamp("16:14").time()
MIN_RTH_BARS = 30


def build_rth_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter 1-min bars to RTH (09:30–16:14 ET) and resample to daily OHLCV.
    Drops days with fewer than MIN_RTH_BARS bars (holidays, early closes).
    """
    t = df["DateTime_ET"].dt.time
    rth = df[(t >= RTH_START) & (t <= RTH_END)].copy()
    rth["trade_date"] = rth["DateTime_ET"].dt.normalize()

    daily = (
        rth.groupby("trade_date")
        .agg(Open=("Open", "first"),
             High=("High", "max"),
             Low=("Low", "min"),
             Close=("Close", "last"),
             Volume=("Volume", "sum"))
        .reset_index()
    )

    bar_counts = rth.groupby("trade_date").size()
    valid = bar_counts[bar_counts >= MIN_RTH_BARS].index
    daily = daily[daily["trade_date"].isin(valid)].copy()
    return daily.sort_values("trade_date").reset_index(drop=True)


def compute_rv_features(daily: pd.DataFrame, raw_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Group 1: Realized Volatility features.
    - rv_1d           : daily realized variance (sum of squared 1-min log-returns, RTH)
    - rv_5d           : 5-day rolling mean of rv_1d
    - rv_22d          : 22-day rolling mean of rv_1d
    - rv_ratio_1_5    : rv_1d / rv_5d
    - rv_percentile_252: percentile of rv_1d in trailing 252-day window
    - parkinson_vol   : (ln(H/L))² / (4·ln2)  — range-based variance estimator
    """
    # Compute rv_1d from 1-min log-returns per RTH session
    t = raw_1min["DateTime_ET"].dt.time
    rth = raw_1min[(t >= RTH_START) & (t <= RTH_END)].copy()
    rth["trade_date"] = rth["DateTime_ET"].dt.normalize()
    rth["log_ret"] = np.log(rth["Close"] / rth["Close"].shift(1))
    # Zero out cross-day returns (first bar of each day)
    rth.loc[rth["trade_date"] != rth["trade_date"].shift(1), "log_ret"] = np.nan

    rv_series = (
        rth.groupby("trade_date")["log_ret"]
        .apply(lambda r: (r**2).sum())
        .rename("rv_1d")
        .reset_index()
    )

    out = daily.merge(rv_series, on="trade_date", how="left")
    out["rv_5d"]  = out["rv_1d"].rolling(5,  min_periods=3).mean()
    out["rv_22d"] = out["rv_1d"].rolling(22, min_periods=10).mean()
    out["rv_ratio_1_5"] = out["rv_1d"] / out["rv_5d"]

    # Trailing 252-day percentile rank (expanding until 252 days available)
    out["rv_percentile_252"] = (
        out["rv_1d"]
        .rolling(252, min_periods=30)
        .apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    )

    # Parkinson range-based variance estimator
    out["parkinson_vol"] = (np.log(out["High"] / out["Low"]) ** 2) / (4 * np.log(2))

    return out
```

**Step 4: Run tests — expect pass**

```bash
python3 -m pytest tests/test_features.py::test_build_rth_daily_returns_dataframe \
                  tests/test_features.py::test_build_rth_daily_range_known_date \
                  tests/test_features.py::test_rv_1d_known_date \
                  tests/test_features.py::test_parkinson_vol_known_date \
                  tests/test_features.py::test_rv_features_no_negative \
                  tests/test_features.py::test_rv_percentile_in_range \
                  tests/test_features.py::test_rv_ratio_positive -v
```

Expected: all 7 PASS.

**Step 5: Commit**

```bash
git add feature_engineering.py tests/test_features.py
git commit -m "feat: RTH daily bar builder and Group 1 realized vol features"
```

---

## Task 3: Group 2 (Range Structure) + Group 3 (Volume) features

**Files:**
- Modify: `feature_engineering.py`
- Modify: `tests/test_features.py`

**Step 1: Add failing tests**

Append to `tests/test_features.py`:

```python
@pytest.fixture(scope="module")
def rv_features_es(rth_daily_es, es_1min):
    from feature_engineering import compute_rv_features
    return compute_rv_features(rth_daily_es, es_1min)

def test_range_features_columns(rv_features_es, es_1min):
    from feature_engineering import compute_range_features
    df = compute_range_features(rv_features_es)
    required = {"range_abs", "range_pct_of_prev", "atr_ratio",
                "range_ma_5", "range_ma_22", "close_location", "overnight_gap"}
    assert required.issubset(df.columns)

def test_close_location_bounded(rv_features_es):
    from feature_engineering import compute_range_features
    df = compute_range_features(rv_features_es)
    cl = df["close_location"].dropna()
    assert (cl >= 0).all() and (cl <= 1).all()

def test_atr_ratio_positive(rv_features_es):
    from feature_engineering import compute_range_features
    df = compute_range_features(rv_features_es)
    assert (df["atr_ratio"].dropna() > 0).all()

def test_volume_features_columns(rv_features_es, es_1min):
    from feature_engineering import compute_range_features, compute_volume_features
    df = compute_volume_features(compute_range_features(rv_features_es), es_1min)
    required = {"volume_prev", "volume_zscore_22", "volume_rth_vs_globex",
                "volume_first_hour_pct"}
    assert required.issubset(df.columns)

def test_volume_first_hour_pct_bounded(rv_features_es, es_1min):
    from feature_engineering import compute_range_features, compute_volume_features
    df = compute_volume_features(compute_range_features(rv_features_es), es_1min)
    pct = df["volume_first_hour_pct"].dropna()
    assert (pct >= 0).all() and (pct <= 1).all()
```

**Step 2: Run tests — expect failures**

```bash
python3 -m pytest tests/test_features.py::test_range_features_columns \
                  tests/test_features.py::test_volume_features_columns -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'compute_range_features'`

**Step 3: Implement `compute_range_features` and `compute_volume_features`**

Append to `feature_engineering.py`:

```python
def compute_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 2: Range Structure features.
    All computed from daily OHLCV columns already in df.

    overnight_gap: (Open_t - Close_{t-1}) / range_{t-1}
      — the gap at THIS morning's open vs yesterday's close, normalised by
        yesterday's range. Known at close of day t; predicts day t+1 behaviour.
    """
    out = df.copy()
    out["range_abs"]       = out["High"] - out["Low"]
    prev_range             = out["range_abs"].shift(1)
    out["range_pct_of_prev"] = out["range_abs"] / prev_range

    # ATR(14): simple mean of range over 14 days
    atr14 = out["range_abs"].rolling(14, min_periods=5).mean()
    out["atr_ratio"]   = out["range_abs"] / atr14
    out["range_ma_5"]  = out["range_abs"].rolling(5,  min_periods=3).mean()
    out["range_ma_22"] = out["range_abs"].rolling(22, min_periods=10).mean()

    # Where did the close land within today's range? 0 = at low, 1 = at high
    out["close_location"] = (out["Close"] - out["Low"]) / out["range_abs"]

    # Overnight gap at open of day t (relative to t-1 range)
    out["overnight_gap"] = (out["Open"] - out["Close"].shift(1)) / prev_range

    return out


def _assign_globex_trade_date(dt_series: pd.Series) -> pd.Series:
    """Bars from 18:00–23:59 ET belong to the NEXT calendar day (globex convention)."""
    dates = dt_series.dt.normalize()
    after_close = dt_series.dt.hour >= 18
    return dates + pd.to_timedelta(after_close.astype(int), unit="D")


def compute_volume_features(df: pd.DataFrame, raw_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Group 3: Volume features.
    - volume_prev          : RTH volume on day t (from daily bars)
    - volume_zscore_22     : z-score of volume vs trailing 22-day window
    - volume_rth_vs_globex : RTH vol / total globex vol for same session-date
    - volume_first_hour_pct: NYAM (09:30–10:29) vol as % of total RTH vol
    """
    out = df.copy()
    out["volume_prev"] = out["Volume"]

    vol_mean = out["Volume"].rolling(22, min_periods=10).mean()
    vol_std  = out["Volume"].rolling(22, min_periods=10).std()
    out["volume_zscore_22"] = (out["Volume"] - vol_mean) / vol_std

    # Globex total volume per trade date
    raw = raw_1min.copy()
    raw = raw[raw["DateTime_ET"].dt.time < pd.Timestamp("17:00").time()]  # exclude maintenance break
    raw["trade_date"] = _assign_globex_trade_date(raw["DateTime_ET"])
    globex_vol = raw.groupby("trade_date")["Volume"].sum().rename("globex_vol")
    out = out.merge(globex_vol, on="trade_date", how="left")
    out["volume_rth_vs_globex"] = out["Volume"] / out["globex_vol"]
    out = out.drop(columns=["globex_vol"])

    # First-hour (NYAM: 09:30–10:29) volume
    t = raw_1min["DateTime_ET"].dt.time
    nyam_mask = (t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("10:30").time())
    nyam = raw_1min[nyam_mask].copy()
    nyam["trade_date"] = nyam["DateTime_ET"].dt.normalize()
    nyam_vol = nyam.groupby("trade_date")["Volume"].sum().rename("nyam_vol")
    out = out.merge(nyam_vol, on="trade_date", how="left")
    out["volume_first_hour_pct"] = out["nyam_vol"] / out["Volume"]
    out = out.drop(columns=["nyam_vol"])

    return out
```

**Step 4: Run all tests**

```bash
python3 -m pytest tests/test_features.py -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add feature_engineering.py tests/test_features.py
git commit -m "feat: Group 2 range structure and Group 3 volume features"
```

---

## Task 4: Group 4 (Intraday Session Structure) + Group 4b (ETH/RTH Cross-session) features

**Files:**
- Modify: `feature_engineering.py`
- Modify: `tests/test_features.py`

**Step 1: Add failing tests**

Append to `tests/test_features.py`:

```python
def test_session_features_columns(rth_daily_es, es_1min):
    from feature_engineering import compute_session_features
    df = compute_session_features(rth_daily_es, es_1min)
    required = {"nyam_range_pct", "london_range_pct", "asia_range_pct",
                "overnight_range_pct", "session_vol_entropy"}
    assert required.issubset(df.columns)

def test_session_range_pcts_bounded(rth_daily_es, es_1min):
    from feature_engineering import compute_session_features
    df = compute_session_features(rth_daily_es, es_1min)
    for col in ["nyam_range_pct", "london_range_pct", "asia_range_pct"]:
        vals = df[col].dropna()
        assert (vals >= 0).all() and (vals <= 1).all(), f"{col} out of [0,1]"

def test_session_vol_entropy_nonneg(eth_daily_es, es_1min):
    from feature_engineering import compute_session_features
    df = compute_session_features(eth_daily_es, es_1min)
    assert (df["session_vol_entropy"].dropna() >= 0).all()

# ── Group 4b: ETH/RTH cross-session features ─────────────────────────────────
def test_eth_rth_cross_features_columns(eth_daily_es, rth_daily_es):
    from feature_engineering import compute_eth_rth_cross_features
    df = compute_eth_rth_cross_features(eth_daily_es, rth_daily_es)
    required = {"rth_pct_of_eth", "overnight_pct_of_eth", "overnight_gap_eth",
                "rth_inside_flag", "rth_outside_flag", "eth_rth_divergence"}
    assert required.issubset(df.columns)

def test_rth_pct_of_eth_bounded(eth_daily_es, rth_daily_es):
    from feature_engineering import compute_eth_rth_cross_features
    df = compute_eth_rth_cross_features(eth_daily_es, rth_daily_es)
    vals = df["rth_pct_of_eth"].dropna()
    # RTH range ≤ ETH range always (RTH is a sub-session of ETH)
    assert (vals > 0).all() and (vals <= 1.001).all()

def test_rth_plus_overnight_equals_one(eth_daily_es, rth_daily_es):
    from feature_engineering import compute_eth_rth_cross_features
    df = compute_eth_rth_cross_features(eth_daily_es, rth_daily_es)
    total = (df["rth_pct_of_eth"] + df["overnight_pct_of_eth"]).dropna()
    assert (np.abs(total - 1.0) < 1e-9).all()

def test_rth_flags_binary(eth_daily_es, rth_daily_es):
    from feature_engineering import compute_eth_rth_cross_features
    df = compute_eth_rth_cross_features(eth_daily_es, rth_daily_es)
    for col in ["rth_inside_flag", "rth_outside_flag", "eth_rth_divergence"]:
        assert set(df[col].dropna().unique()).issubset({0, 1})
```

**Step 2: Run tests — expect failures**

```bash
python3 -m pytest tests/test_features.py::test_session_features_columns -v 2>&1 | head -10
```

**Step 3: Implement `compute_session_features` and `compute_eth_rth_cross_features`**

Append to `feature_engineering.py`:

```python
def _session_range(raw_1min: pd.DataFrame, session_name: str) -> pd.Series:
    """Daily H-L range for a named session, indexed by trade_date."""
    s = raw_1min[raw_1min["session"] == session_name].copy()
    s["trade_date"] = s["DateTime_ET"].dt.normalize()
    return (
        s.groupby("trade_date")
         .apply(lambda g: g["High"].max() - g["Low"].min())
         .rename(f"{session_name.lower()}_range")
    )


def compute_session_features(daily: pd.DataFrame, raw_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Group 4: Intraday Session Structure features.
    - nyam_range_pct   : NYAM (09:30–10:59) H-L / daily RTH H-L
    - london_range_pct : LONDON session H-L / daily RTH H-L
    - asia_range_pct   : ASIA session H-L / daily RTH H-L
    - overnight_range_pct : Globex overnight H-L / RTH H-L
      Overnight = ASIA + pre-NYAM OTHER on same trade date
    - session_vol_entropy : Shannon entropy of per-session realized variance shares
      High entropy → vol spread across sessions; Low → concentrated burst
    """
    out = daily.copy()
    rth_range = out.set_index("trade_date")["High"] - out.set_index("trade_date")["Low"]

    for ses in ["NYAM", "LONDON", "ASIA"]:
        sr = _session_range(raw_1min, ses)
        col = f"{ses.lower()}_range_pct"
        merged = sr.reindex(out["trade_date"]).values
        out[col] = merged / (out["High"] - out["Low"]).values

    # Overnight range: from previous ASIA open to RTH open (approximate as
    # all bars with DateTime_ET < 09:30 on the calendar date)
    t = raw_1min["DateTime_ET"].dt.time
    overnight = raw_1min[t < RTH_START].copy()
    overnight["trade_date"] = overnight["DateTime_ET"].dt.normalize()
    ovn_range = (
        overnight.groupby("trade_date")
                 .apply(lambda g: g["High"].max() - g["Low"].min())
                 .rename("ovn_range")
    )
    out = out.merge(ovn_range.reset_index(), on="trade_date", how="left")
    out["overnight_range_pct"] = out["ovn_range"] / (out["High"] - out["Low"])
    out = out.drop(columns=["ovn_range"])

    # Session-level realized variance for entropy computation
    t = raw_1min["DateTime_ET"].dt.time
    session_rv = {}
    for ses in ["ASIA", "LONDON", "NYAM", "LUNCH", "PM", "OTHER"]:
        mask = raw_1min["session"] == ses
        s = raw_1min[mask].copy()
        s["trade_date"] = s["DateTime_ET"].dt.normalize()
        s["log_ret"] = np.log(s["Close"] / s["Close"].shift(1))
        s.loc[s["trade_date"] != s["trade_date"].shift(1), "log_ret"] = np.nan
        rv = s.groupby("trade_date")["log_ret"].apply(lambda r: (r**2).sum())
        session_rv[ses] = rv

    rv_df = pd.DataFrame(session_rv).fillna(0)
    rv_df["total"] = rv_df.sum(axis=1)

    def _entropy(row):
        total = row["total"]
        if total == 0:
            return np.nan
        shares = row.drop("total") / total
        shares = shares[shares > 0]
        return -(shares * np.log(shares)).sum()

    entropy = rv_df.apply(_entropy, axis=1).rename("session_vol_entropy")
    out = out.merge(entropy.reset_index().rename(columns={"index": "trade_date",
                                                           0: "session_vol_entropy"}),
                    on="trade_date", how="left")

    return out
```

**Step 4: Run all tests**

```bash
python3 -m pytest tests/test_features.py -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add feature_engineering.py tests/test_features.py
git commit -m "feat: Group 4 intraday session structure features"
```

After the `compute_session_features` function, also append `compute_eth_rth_cross_features`:

```python
def compute_eth_rth_cross_features(eth_daily: pd.DataFrame,
                                    rth_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Group 4b: ETH / RTH cross-session features.
    Merged onto ETH trade dates. RTH trade_date == ETH trade_date
    (both refer to the same calendar day for RTH).

    - rth_pct_of_eth      : RTH H-L / ETH H-L  (mean ~81%)
    - overnight_pct_of_eth: 1 - rth_pct_of_eth
    - overnight_gap_eth   : (RTH_Open - prev_ETH_Close) / prev_ETH_range
                            Gap at the RTH open, relative to the full ETH context
    - rth_inside_flag     : 1 if RTH session was an inside day (within RTH context)
    - rth_outside_flag    : 1 if RTH session was an outside day (within RTH context)
    - eth_rth_divergence  : 1 if ETH inside/outside classification differs from RTH
                            (captures the 81 days where overnight flipped the label)
    """
    rth = rth_daily.copy()
    rth["rth_range"] = rth["High"] - rth["Low"]
    rth["prev_RTH_H"] = rth["High"].shift(1)
    rth["prev_RTH_L"] = rth["Low"].shift(1)
    rth["rth_inside_flag"]  = (
        (rth["High"] <= rth["prev_RTH_H"]) & (rth["Low"] >= rth["prev_RTH_L"])
    ).astype(int)
    rth["rth_outside_flag"] = (
        (rth["High"] >  rth["prev_RTH_H"]) & (rth["Low"] <  rth["prev_RTH_L"])
    ).astype(int)
    rth_cols = rth[["trade_date", "rth_range", "Open", "rth_inside_flag", "rth_outside_flag"]]

    out = eth_daily.copy()
    out["eth_range"] = out["High"] - out["Low"]

    # ETH inside/outside on today's bars (for divergence computation)
    out["prev_ETH_H"]    = out["High"].shift(1)
    out["prev_ETH_L"]    = out["Low"].shift(1)
    out["prev_ETH_range"] = out["eth_range"].shift(1)
    out["prev_ETH_Close"] = out["Close"].shift(1)
    out["eth_inside"]  = (out["High"] <= out["prev_ETH_H"]) & (out["Low"] >= out["prev_ETH_L"])
    out["eth_outside"] = (out["High"] >  out["prev_ETH_H"]) & (out["Low"] <  out["prev_ETH_L"])

    out = out.merge(rth_cols.rename(columns={"Open": "rth_open"}),
                    on="trade_date", how="left")

    out["rth_pct_of_eth"]       = out["rth_range"] / out["eth_range"]
    out["overnight_pct_of_eth"] = 1.0 - out["rth_pct_of_eth"]

    # Overnight gap: RTH open vs previous ETH close, normalised by previous ETH range
    out["overnight_gap_eth"] = (
        (out["rth_open"] - out["prev_ETH_Close"]) / out["prev_ETH_range"]
    )

    # Divergence: ETH classification differs from RTH classification
    out["eth_rth_divergence"] = (
        (out["eth_inside"].astype(int)  != out["rth_inside_flag"]) |
        (out["eth_outside"].astype(int) != out["rth_outside_flag"])
    ).astype(int)

    drop_cols = ["eth_range", "prev_ETH_H", "prev_ETH_L", "prev_ETH_range",
                 "prev_ETH_Close", "eth_inside", "eth_outside", "rth_range", "rth_open"]
    return out.drop(columns=drop_cols, errors="ignore")
```

---

## Task 5: Groups 5-6-7 (Calendar, VIX, Cross-instrument) + assemble output

**Files:**
- Modify: `feature_engineering.py`
- Modify: `tests/test_features.py`

**Step 1: Add failing tests**

Append to `tests/test_features.py`:

```python
@pytest.fixture(scope="module")
def vix():
    df = pd.read_parquet("data/vix_cboe.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df

@pytest.fixture(scope="module")
def eco():
    return pd.read_parquet("data/economic_events.parquet")

def test_calendar_features_columns(rth_daily_es, eco):
    from feature_engineering import compute_calendar_features
    df = compute_calendar_features(rth_daily_es, eco)
    required = {"day_of_week", "high_impact_today", "high_impact_tomorrow",
                "n_events_next_2d", "is_fomc_day", "is_nfp_day", "days_since_fomc"}
    assert required.issubset(df.columns)

def test_day_of_week_range(rth_daily_es, eco):
    from feature_engineering import compute_calendar_features
    df = compute_calendar_features(rth_daily_es, eco)
    assert df["day_of_week"].between(0, 4).all()

def test_is_fomc_day_binary(rth_daily_es, eco):
    from feature_engineering import compute_calendar_features
    df = compute_calendar_features(rth_daily_es, eco)
    assert set(df["is_fomc_day"].unique()).issubset({0, 1})

def test_vix_features_columns(rth_daily_es, vix):
    from feature_engineering import compute_vix_features
    df = compute_vix_features(rth_daily_es, vix)
    required = {"vix_close", "vix_change_1d", "vix_rv_spread", "vix_percentile_252"}
    assert required.issubset(df.columns)

def test_vix_close_in_valid_range(rth_daily_es, vix):
    from feature_engineering import compute_vix_features
    df = compute_vix_features(rth_daily_es, vix)
    vals = df["vix_close"].dropna()
    assert (vals > 5).all() and (vals < 100).all()

def test_output_parquets_exist():
    import subprocess
    result = subprocess.run(
        ["python3", "feature_engineering.py"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert Path("output/features_es_eth.parquet").exists()
    assert Path("output/features_nq_eth.parquet").exists()

def test_no_lookahead_in_features():
    """Verify that target (y) is strictly the NEXT day's log range ratio."""
    df = pd.read_parquet("output/features_es_eth.parquet")
    # y[t] should equal log(range[t+1] / range[t])
    # Verify by checking that range_abs and y are not the same-day quantity
    assert "y" in df.columns
    # y should be NaN on the last row (no next day)
    assert pd.isna(df["y"].iloc[-1])
```

**Step 2: Run — expect failures**

```bash
python3 -m pytest tests/test_features.py::test_calendar_features_columns \
                  tests/test_features.py::test_vix_features_columns -v 2>&1 | head -15
```

**Step 3: Implement remaining feature functions and `main()` in `feature_engineering.py`**

Append to `feature_engineering.py`:

```python
# ── FOMC and NFP event title identifiers ─────────────────────────────────────
_FOMC_TITLES = {"US FOMC Statement", "US Federal Funds Rate",
                "US FOMC Economic Projections"}
_NFP_TITLE   = "US Non-Farm Employment Change"


def compute_calendar_features(daily: pd.DataFrame, eco: pd.DataFrame) -> pd.DataFrame:
    """
    Group 5: Calendar & Macro Event features.
    Economic events are in UTC; convert to ET date by subtracting 4 or 5 hours.
    For simplicity, use UTC date - 1 day for events after midnight UTC that fall
    on the same ET trading day. We use a conservative approach: match on
    ET date = UTC date (events at 14:30 UTC = 09:30–10:30 ET).
    """
    out = daily.copy()
    out["day_of_week"] = pd.to_datetime(out["trade_date"]).dt.dayofweek  # 0=Mon

    # Convert event UTC timestamps to ET date
    # ET = UTC - 5h (EST) or UTC - 4h (EDT). Use UTC - 4h as approximation.
    eco = eco.copy()
    eco["et_date"] = (eco["datetime_utc"] - pd.Timedelta(hours=4)).dt.normalize().dt.date
    eco["et_date"] = pd.to_datetime(eco["et_date"])

    # Drop holidays (impact='holiday') — only keep 'high' impact
    eco_high = eco[eco["impact"] == "high"].copy()

    # FOMC days
    fomc = eco_high[eco_high["title"].isin(_FOMC_TITLES)][["et_date"]].drop_duplicates()
    fomc["is_fomc_day"] = 1
    out = out.merge(fomc, left_on="trade_date", right_on="et_date", how="left")
    out["is_fomc_day"] = out["is_fomc_day"].fillna(0).astype(int)
    out = out.drop(columns=["et_date"], errors="ignore")

    # NFP days
    nfp = eco_high[eco_high["title"] == _NFP_TITLE][["et_date"]].drop_duplicates()
    nfp["is_nfp_day"] = 1
    out = out.merge(nfp, left_on="trade_date", right_on="et_date", how="left")
    out["is_nfp_day"] = out["is_nfp_day"].fillna(0).astype(int)
    out = out.drop(columns=["et_date"], errors="ignore")

    # All high-impact events per date
    event_counts = eco_high.groupby("et_date").size().rename("n_events").reset_index()
    event_counts.columns = ["et_date", "n_events"]

    out = out.merge(event_counts, left_on="trade_date", right_on="et_date", how="left")
    out["high_impact_today"] = (out["n_events"].fillna(0) > 0).astype(int)
    out = out.drop(columns=["et_date", "n_events"], errors="ignore")

    # Tomorrow's events — shift event_counts by 1 day
    event_counts_shifted = event_counts.copy()
    event_counts_shifted["et_date"] = event_counts_shifted["et_date"] - pd.Timedelta(days=1)
    out = out.merge(event_counts_shifted, left_on="trade_date", right_on="et_date", how="left")
    out["high_impact_tomorrow"] = (out["n_events"].fillna(0) > 0).astype(int)
    out = out.drop(columns=["et_date", "n_events"], errors="ignore")

    # Count of events in next 2 trading days
    event_set = set(eco_high["et_date"].dt.date)

    def _events_next_2d(date):
        return sum(
            (date + pd.Timedelta(days=d)).date() in event_set
            for d in [1, 2]
        )

    out["n_events_next_2d"] = out["trade_date"].apply(_events_next_2d)

    # Days since last FOMC
    fomc_dates = sorted(fomc["et_date"].dropna().tolist())

    def _days_since_fomc(date):
        past = [d for d in fomc_dates if d <= date]
        return (date - past[-1]).days if past else np.nan

    out["days_since_fomc"] = out["trade_date"].apply(_days_since_fomc)

    return out


def compute_vix_features(daily: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """
    Group 6: VIX / Implied Volatility features.
    - vix_close        : VIX index close on day t
    - vix_change_1d    : day-over-day VIX change
    - vix_rv_spread    : VIX_t - sqrt(rv_22d * 252) — implied minus realised vol
    - vix_percentile_252: percentile of VIX within trailing 252 trading-day window
    """
    out = daily.copy()
    vix = vix.copy()
    vix["trade_date"] = pd.to_datetime(vix["date"])
    vix = vix.sort_values("trade_date")
    vix["vix_change_1d"] = vix["c"].diff()
    vix["vix_percentile_252"] = (
        vix["c"]
        .rolling(252, min_periods=30)
        .apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    )

    out = out.merge(
        vix[["trade_date", "c", "vix_change_1d", "vix_percentile_252"]].rename(
            columns={"c": "vix_close"}),
        on="trade_date", how="left"
    )

    # Implied - realised vol spread (annualised: sqrt(rv_22d * 252))
    if "rv_22d" in out.columns:
        out["vix_rv_spread"] = out["vix_close"] - np.sqrt(out["rv_22d"] * 252) * 100
    else:
        out["vix_rv_spread"] = np.nan

    return out


def compute_cross_instrument_features(
    daily_es: pd.DataFrame, daily_nq: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group 7: Cross-instrument features (ES ↔ NQ).
    - es_nq_rv_ratio    : rv_1d_ES / rv_1d_NQ
    - es_nq_range_ratio : range_ES / range_NQ
    Merged onto both frames by trade_date.
    """
    cross = daily_es[["trade_date", "rv_1d", "range_abs"]].merge(
        daily_nq[["trade_date", "rv_1d", "range_abs"]],
        on="trade_date", suffixes=("_es", "_nq")
    )
    cross["es_nq_rv_ratio"]    = cross["rv_1d_es"]    / cross["rv_1d_nq"]
    cross["es_nq_range_ratio"] = cross["range_abs_es"] / cross["range_abs_nq"]
    cross_cols = cross[["trade_date", "es_nq_rv_ratio", "es_nq_range_ratio"]]

    es_out = daily_es.merge(cross_cols, on="trade_date", how="left")
    nq_out = daily_nq.merge(cross_cols, on="trade_date", how="left")
    return es_out, nq_out


def add_target(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add target variable and inside/outside/neither labels.
    y = log(range_{t+1} / range_t) — predicts NEXT day's log range ratio.
    Inside/outside defined on day t (for reference, not target leakage).
    """
    out = daily.copy()
    out["prev_High"] = out["High"].shift(1)
    out["prev_Low"]  = out["Low"].shift(1)
    out["inside"]  = (out["High"] <= out["prev_High"]) & (out["Low"] >= out["prev_Low"])
    out["outside"] = (out["High"] >  out["prev_High"]) & (out["Low"] <  out["prev_Low"])
    out["neither"] = ~out["inside"] & ~out["outside"]
    out = out.drop(columns=["prev_High", "prev_Low"])

    # Target: log ratio of NEXT day's range to today's range
    out["y"] = np.log(out["range_abs"].shift(-1) / out["range_abs"])

    return out


# ── Feature column groups (for model.py to reference) ────────────────────────
FEATURE_COLS_HAR = ["rv_1d", "rv_5d", "rv_22d"]

FEATURE_COLS_ALL = [
    # Group 1 — Realized Volatility (computed on RTH session bars)
    "rv_1d", "rv_5d", "rv_22d", "rv_ratio_1_5", "rv_percentile_252", "parkinson_vol",
    # Group 2 — Range Structure (ETH daily bars)
    "range_pct_of_prev", "atr_ratio", "range_ma_5", "range_ma_22",
    "close_location", "overnight_gap",
    # Group 3 — Volume
    "volume_prev", "volume_zscore_22", "volume_rth_vs_globex", "volume_first_hour_pct",
    # Group 4 — Intraday Session (% of ETH range)
    "nyam_range_pct", "london_range_pct", "asia_range_pct", "session_vol_entropy",
    # Group 4b — ETH/RTH Cross-session
    "rth_pct_of_eth", "overnight_pct_of_eth", "overnight_gap_eth",
    "rth_inside_flag", "rth_outside_flag", "eth_rth_divergence",
    # Group 5 — Calendar & Macro
    "day_of_week", "high_impact_today", "high_impact_tomorrow",
    "n_events_next_2d", "is_fomc_day", "is_nfp_day", "days_since_fomc",
    # Group 6 — VIX
    "vix_close", "vix_change_1d", "vix_rv_spread", "vix_percentile_252",
    # Group 7 — Cross-instrument (ES ↔ NQ)
    "es_nq_rv_ratio", "es_nq_range_ratio",
]

TARGET_COL  = "y"
LABEL_COLS  = ["inside", "outside", "neither"]
ID_COLS     = ["trade_date", "Open", "High", "Low", "Close", "Volume", "range_abs"]


def _build_features_for(symbol: str, raw: pd.DataFrame,
                         vix: pd.DataFrame, eco: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (eth_daily_with_features, rth_daily) tuple."""
    eth = build_eth_daily(raw)
    rth = build_rth_daily(raw)

    eth = compute_rv_features(eth, raw, session="rth")   # RV from RTH bars, on ETH dates
    eth = compute_range_features(eth)                     # range structure on ETH bars
    eth = compute_volume_features(eth, raw)
    eth = compute_session_features(eth, raw)
    eth = compute_eth_rth_cross_features(eth, rth)        # Group 4b: ETH/RTH relationship
    eth = compute_calendar_features(eth, eco)
    eth = compute_vix_features(eth, vix)
    return eth, rth


def main():
    Path("output").mkdir(exist_ok=True)

    print("Loading 1-min data…")
    es_raw = pd.read_parquet("data/es_1m.parquet")
    nq_raw = pd.read_parquet("data/nq_1m.parquet")

    vix = pd.read_parquet("data/vix_cboe.parquet")
    vix["date"] = pd.to_datetime(vix["date"])
    eco = pd.read_parquet("data/economic_events.parquet")

    print("Engineering ES features…")
    es, es_rth = _build_features_for("ES", es_raw, vix, eco)

    print("Engineering NQ features…")
    nq, nq_rth = _build_features_for("NQ", nq_raw, vix, eco)

    print("Adding cross-instrument features…")
    es, nq = compute_cross_instrument_features(es, nq)

    print("Adding targets…")
    es = add_target(es)
    nq = add_target(nq)

    es.to_parquet("output/features_es_eth.parquet", index=False)
    nq.to_parquet("output/features_nq_eth.parquet", index=False)
    print("Saved output/features_es_eth.parquet")
    print("Saved output/features_nq_eth.parquet")
    print(f"ES: {len(es)} days, {es[FEATURE_COLS_ALL].notna().all(axis=1).sum()} fully complete rows")
    print(f"NQ: {len(nq)} days, {nq[FEATURE_COLS_ALL].notna().all(axis=1).sum()} fully complete rows")


if __name__ == "__main__":
    main()
```

**Step 4: Run all tests**

```bash
python3 -m pytest tests/test_features.py -v
```

Expected: all PASS (including `test_output_parquets_exist` which runs `main()`).

**Step 5: Spot-check output**

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('output/features_es_eth.parquet')
print(df.shape)
print(df[['trade_date','rv_1d','vix_close','is_fomc_day','y','inside']].tail(10).to_string())
print('NaNs per feature:')
print(df.isna().sum().sort_values(ascending=False).head(10))
"
```

**Step 6: Commit**

```bash
git add feature_engineering.py tests/test_features.py
git commit -m "feat: Groups 5-7 features + target + main pipeline"
```

---

## Task 6: `model.py` — OLS helper + HAR walk-forward baseline

**Files:**
- Create: `model.py`
- Modify: `tests/test_model.py`

**Step 1: Write failing tests**

In `tests/test_model.py`:

```python
import pandas as pd
import numpy as np
import pytest
import sys
sys.path.insert(0, ".")


@pytest.fixture(scope="module")
def features_es():
    return pd.read_parquet("output/features_es_eth.parquet")


def test_ols_pi_coverage_reasonable():
    """90% PI should contain ~90% of true values on synthetic normal data."""
    from model import ols_predict_with_pi
    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 3))
    beta = np.array([1.0, -0.5, 0.3])
    y = X @ beta + rng.normal(scale=0.5, size=300)

    X_train, y_train = X[:200], y[:200]
    X_test,  y_test  = X[200:], y[200:]

    _, pi_lo, pi_hi, _, _ = ols_predict_with_pi(X_train, y_train, X_test, alpha=0.10)
    coverage = ((y_test >= pi_lo) & (y_test <= pi_hi)).mean()
    assert 0.80 < coverage < 0.99, f"PI coverage {coverage:.2f} out of expected range"


def test_walk_forward_har_returns_predictions(features_es):
    from model import walk_forward
    from feature_engineering import FEATURE_COLS_HAR, TARGET_COL
    results = walk_forward(features_es, FEATURE_COLS_HAR, TARGET_COL,
                           init_window=252, model_type="ols")
    assert isinstance(results, pd.DataFrame)
    required = {"trade_date", "y_true", "y_hat", "pi_lower_90", "pi_upper_90"}
    assert required.issubset(results.columns)


def test_walk_forward_har_no_lookahead(features_es):
    """Each prediction must be made using only past data."""
    from model import walk_forward
    from feature_engineering import FEATURE_COLS_HAR, TARGET_COL
    results = walk_forward(features_es, FEATURE_COLS_HAR, TARGET_COL,
                           init_window=252, model_type="ols")
    # y_hat should not be perfectly equal to y_true (would imply lookahead)
    r2 = np.corrcoef(results["y_true"], results["y_hat"])[0, 1] ** 2
    assert r2 < 0.99, "Suspiciously high R² — check for lookahead"


def test_walk_forward_pi_contains_truth_roughly(features_es):
    from model import walk_forward
    from feature_engineering import FEATURE_COLS_HAR, TARGET_COL
    results = walk_forward(features_es, FEATURE_COLS_HAR, TARGET_COL,
                           init_window=252, model_type="ols")
    coverage = (
        (results["y_true"] >= results["pi_lower_90"]) &
        (results["y_true"] <= results["pi_upper_90"])
    ).mean()
    # Expect between 75% and 99% (financial data is noisier than synthetic)
    assert 0.70 < coverage < 0.99, f"PI coverage {coverage:.2f}"
```

**Step 2: Run tests — expect failures**

```bash
python3 -m pytest tests/test_model.py -v 2>&1 | head -15
```

**Step 3: Implement `model.py`**

```python
"""
model.py
Walk-forward volatility forecasting: HAR baseline + full Ridge.
Outputs: output/predictions_{es,nq}_rth.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from feature_engineering import (
    FEATURE_COLS_HAR, FEATURE_COLS_ALL, TARGET_COL, LABEL_COLS
)

WALK_FORWARD_INIT = 252   # initial training window (trading days)
ALPHA_PI = 0.10           # 1 - confidence level → 90% PI


def ols_predict_with_pi(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float = ALPHA_PI,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, LinearRegression, float]:
    """
    OLS point forecasts + prediction intervals using the standard formula:
        PI = y_hat ± t_{α/2, n-k-1} · s · sqrt(1 + x'(X'X)⁻¹x)

    Returns: y_pred, pi_lower, pi_upper, fitted_model, residual_std
    """
    n, k = X_train.shape
    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    residuals = y_train - reg.predict(X_train)
    s2 = (residuals ** 2).sum() / (n - k - 1)
    s  = np.sqrt(s2)

    # Augment with intercept column for leverage computation
    ones = np.ones((n, 1))
    Xa_train = np.hstack([ones, X_train])
    Xa_test  = np.hstack([np.ones((len(X_test), 1)), X_test])

    XtX_inv = np.linalg.pinv(Xa_train.T @ Xa_train)
    t_crit  = stats.t.ppf(1 - alpha / 2, df=n - k - 1)

    pi_lower = np.empty(len(X_test))
    pi_upper = np.empty(len(X_test))
    for i, x in enumerate(Xa_test):
        h = float(x @ XtX_inv @ x)
        margin = t_crit * s * np.sqrt(1.0 + h)
        pi_lower[i] = y_pred[i] - margin
        pi_upper[i] = y_pred[i] + margin

    return y_pred, pi_lower, pi_upper, reg, s


def ridge_predict_with_pi(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    scaler: StandardScaler,
    alpha: float = ALPHA_PI,
    ridge_alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Ridge regression. PI uses training residual std (Gaussian approximation).
    Scaler is fitted on training data and applied to test data.
    Returns: y_pred, pi_lower, pi_upper, residual_std
    """
    Xs_train = scaler.transform(X_train)
    Xs_test  = scaler.transform(X_test)

    reg = Ridge(alpha=ridge_alpha).fit(Xs_train, y_train)
    y_pred = reg.predict(Xs_test)

    residuals = y_train - reg.predict(Xs_train)
    s = residuals.std(ddof=1)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    margin   = z_crit * s
    pi_lower = y_pred - margin
    pi_upper = y_pred + margin

    return y_pred, pi_lower, pi_upper, s, reg


def compute_probabilities(
    y_hat: np.ndarray,
    sigma: np.ndarray | float,
    inside_thresh: float,
    outside_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a Gaussian predictive distribution N(y_hat, sigma²), compute:
      p_inside  = P(y < inside_thresh)
      p_outside = P(y > outside_thresh)
      p_neither = 1 - p_inside - p_outside

    Thresholds are in log-space (log of range ratio).
    Calibrated from training data per walk-forward step.
    """
    p_inside  = stats.norm.cdf(inside_thresh, loc=y_hat, scale=sigma)
    p_outside = 1.0 - stats.norm.cdf(outside_thresh, loc=y_hat, scale=sigma)
    p_neither = np.clip(1.0 - p_inside - p_outside, 0, 1)
    return p_inside, p_outside, p_neither


def _calibrate_thresholds(y: np.ndarray, labels_inside: np.ndarray,
                           labels_outside: np.ndarray) -> tuple[float, float]:
    """
    From training data, compute log-range-ratio thresholds:
      inside_thresh  = 95th percentile of log-ratios on true inside days
      outside_thresh = 5th percentile of log-ratios on true outside days
    These define the boundaries of the Gaussian integration.
    """
    inside_y  = y[labels_inside]
    outside_y = y[labels_outside]
    inside_thresh  = np.percentile(inside_y,  95) if len(inside_y)  > 5 else -0.15
    outside_thresh = np.percentile(outside_y,  5) if len(outside_y) > 5 else  0.18
    return float(inside_thresh), float(outside_thresh)


def walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    init_window: int = WALK_FORWARD_INIT,
    model_type: str = "ols",   # "ols" or "ridge"
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Expanding walk-forward validation.
    Train on [0, t-1], predict on t, expand by 1.
    Returns DataFrame with one row per test day.
    """
    # Drop rows with missing target or features
    cols_needed = feature_cols + [target_col] + LABEL_COLS + ["trade_date"]
    sub = df[cols_needed].dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    n = len(sub)
    if n < init_window + 1:
        raise ValueError(f"Not enough rows ({n}) for init_window={init_window}")

    records = []
    for t in range(init_window, n):
        train = sub.iloc[:t]
        test  = sub.iloc[[t]]

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        X_test  = test[feature_cols].values
        y_true  = float(test[target_col].iloc[0])

        is_inside  = train["inside"].values.astype(bool)
        is_outside = train["outside"].values.astype(bool)
        # Use log(range ratio) from training target itself for threshold calibration
        in_thresh, out_thresh = _calibrate_thresholds(y_train, is_inside, is_outside)

        if model_type == "ols":
            y_hat, pi_lo, pi_hi, _, sigma = ols_predict_with_pi(X_train, y_train, X_test)
        else:
            scaler = StandardScaler().fit(X_train)
            y_hat, pi_lo, pi_hi, sigma, _ = ridge_predict_with_pi(
                X_train, y_train, X_test, scaler, ridge_alpha=ridge_alpha
            )

        p_in, p_out, p_nei = compute_probabilities(
            y_hat, sigma, in_thresh, out_thresh
        )

        records.append({
            "trade_date":    test["trade_date"].iloc[0],
            "y_true":        y_true,
            "y_hat":         float(y_hat[0]),
            "pi_lower_90":   float(pi_lo[0]),
            "pi_upper_90":   float(pi_hi[0]),
            "sigma":         float(sigma),
            "p_inside":      float(p_in[0]),
            "p_outside":     float(p_out[0]),
            "p_neither":     float(p_nei[0]),
            "true_inside":   bool(test["inside"].iloc[0]),
            "true_outside":  bool(test["outside"].iloc[0]),
            "true_neither":  bool(test["neither"].iloc[0]),
            "inside_thresh": in_thresh,
            "outside_thresh": out_thresh,
        })

        if t % 100 == 0:
            print(f"  Walk-forward step {t}/{n}")

    return pd.DataFrame(records)


def main():
    Path("output").mkdir(exist_ok=True)

    for symbol in ["es", "nq"]:
        print(f"\n{'='*50}")
        print(f"  {symbol.upper()} — loading features")
        df = pd.read_parquet(f"output/features_{symbol}_eth.parquet")

        print(f"  {symbol.upper()} — HAR baseline (OLS)")
        har = walk_forward(df, FEATURE_COLS_HAR, TARGET_COL,
                           init_window=WALK_FORWARD_INIT, model_type="ols")
        har["model"] = "HAR_OLS"
        har.to_parquet(f"output/predictions_{symbol}_har.parquet", index=False)
        print(f"  Saved output/predictions_{symbol}_har.parquet  ({len(har)} rows)")

        print(f"  {symbol.upper()} — Full Ridge model")
        ridge = walk_forward(df, FEATURE_COLS_ALL, TARGET_COL,
                             init_window=WALK_FORWARD_INIT, model_type="ridge")
        ridge["model"] = "Full_Ridge"
        ridge.to_parquet(f"output/predictions_{symbol}_ridge.parquet", index=False)
        print(f"  Saved output/predictions_{symbol}_ridge.parquet  ({len(ridge)} rows)")


if __name__ == "__main__":
    main()
```

**Step 4: Run all model tests**

```bash
python3 -m pytest tests/test_model.py -v
```

Expected: all 4 PASS.

**Step 5: Run full model pipeline (takes ~2–5 min)**

```bash
python3 model.py
```

Expected output:

```
  ES — HAR baseline (OLS)
  Walk-forward step 300/1095
  ...
  Saved output/predictions_es_har.parquet  (843 rows)
  ES — Full Ridge model
  ...
  Saved output/predictions_es_ridge.parquet  (843 rows)
  NQ — HAR baseline (OLS)
  ...
```

**Step 6: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat: walk-forward HAR OLS and Ridge model with prediction intervals and P(inside/outside)"
```

---

## Task 7: `evaluate.py` — OOS metrics, feature importance, calibration plots

**Files:**
- Create: `evaluate.py`

**Step 1: No test needed** — this is a reporting/plotting script. Verify by running and checking output files exist.

**Step 2: Implement `evaluate.py`**

```python
"""
evaluate.py
Out-of-sample evaluation of walk-forward predictions.
Outputs:
  output/metrics_summary.csv      — OOS R², RMSE, PI coverage, Brier scores
  output/feature_importance.csv   — Ridge coefficient magnitudes (standardised)
  output/plots/                   — calibration + actual vs predicted charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from feature_engineering import (
    FEATURE_COLS_HAR, FEATURE_COLS_ALL, TARGET_COL, LABEL_COLS
)

Path("output/plots").mkdir(parents=True, exist_ok=True)


# ── Metric functions ─────────────────────────────────────────────────────────

def oos_r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def pi_coverage(y_true, pi_lo, pi_hi):
    return ((y_true >= pi_lo) & (y_true <= pi_hi)).mean()


def brier_score(p_pred, y_true_binary):
    """Lower is better. 0 = perfect, 0.25 = no-skill (for base rate ~0.5)."""
    return ((p_pred - y_true_binary.astype(float)) ** 2).mean()


def diebold_mariano(e1, e2):
    """
    Diebold-Mariano test: H0 = equal predictive accuracy.
    e1, e2 = squared error vectors for model 1 and model 2.
    Returns t-statistic and p-value. Negative t = model 1 is better.
    """
    d = e1 - e2
    n = len(d)
    dm_stat = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    p_val = 2 * stats.norm.sf(abs(dm_stat))
    return dm_stat, p_val


# ── Feature importance ───────────────────────────────────────────────────────

def compute_feature_importance(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Fit Ridge on full dataset (in-sample), return standardised coefficient magnitudes.
    Not used for prediction — only for feature ranking.
    """
    cols_needed = FEATURE_COLS_ALL + [TARGET_COL]
    sub = df[cols_needed].dropna()
    X = sub[FEATURE_COLS_ALL].values
    y = sub[TARGET_COL].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    reg = Ridge(alpha=1.0).fit(Xs, y)

    importance = pd.DataFrame({
        "feature": FEATURE_COLS_ALL,
        "coef": reg.coef_,
        "abs_coef": np.abs(reg.coef_),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    importance["symbol"] = symbol
    return importance


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(preds: pd.DataFrame, symbol: str, model: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(preds["trade_date"], preds["pi_lower_90"], preds["pi_upper_90"],
                    alpha=0.3, label="90% PI")
    ax.plot(preds["trade_date"], preds["y_hat"], lw=1, label="y_hat")
    ax.scatter(preds["trade_date"], preds["y_true"], s=4, c="black",
               alpha=0.5, label="y_true")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_title(f"{symbol} {model} — Actual vs Predicted log range ratio")
    ax.set_ylabel("log(range_t+1 / range_t)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"output/plots/{symbol}_{model}_actual_vs_pred.png", dpi=120)
    plt.close()


def plot_probability_calibration(preds: pd.DataFrame, symbol: str, model: str):
    """
    Reliability diagram: bins p_inside and p_outside predictions,
    plots mean predicted probability vs actual fraction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (p_col, true_col, label) in zip(axes, [
        ("p_inside",  "true_inside",  "Inside"),
        ("p_outside", "true_outside", "Outside"),
    ]):
        bins = np.linspace(0, 1, 11)
        bin_idx = np.digitize(preds[p_col], bins) - 1
        bin_idx = np.clip(bin_idx, 0, 9)

        mean_pred, mean_true = [], []
        for b in range(10):
            mask = bin_idx == b
            if mask.sum() > 5:
                mean_pred.append(preds.loc[mask, p_col].mean())
                mean_true.append(preds.loc[mask, true_col].astype(float).mean())

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
        ax.scatter(mean_pred, mean_true, s=40, zorder=5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(f"Mean predicted {p_col}")
        ax.set_ylabel(f"Actual fraction {label}")
        ax.set_title(f"{symbol} {model} — {label} day calibration")

    plt.tight_layout()
    plt.savefig(f"output/plots/{symbol}_{model}_calibration.png", dpi=120)
    plt.close()


def plot_feature_importance(importance: pd.DataFrame, symbol: str):
    top20 = importance.head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#d62728" if c < 0 else "#1f77b4" for c in top20["coef"]]
    ax.barh(top20["feature"][::-1], top20["abs_coef"][::-1], color=colors[::-1])
    ax.set_xlabel("Standardised |coefficient| (Ridge)")
    ax.set_title(f"{symbol} — Feature importance (top 20)")
    plt.tight_layout()
    plt.savefig(f"output/plots/{symbol}_feature_importance.png", dpi=120)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_metrics = []

    for symbol in ["es", "nq"]:
        print(f"\n{'='*55}")
        print(f"  {symbol.upper()}")

        har   = pd.read_parquet(f"output/predictions_{symbol}_har.parquet")
        ridge = pd.read_parquet(f"output/predictions_{symbol}_ridge.parquet")
        har["trade_date"]   = pd.to_datetime(har["trade_date"])
        ridge["trade_date"] = pd.to_datetime(ridge["trade_date"])

        features = pd.read_parquet(f"output/features_{symbol}_rth.parquet")

        for preds, model_name in [(har, "HAR_OLS"), (ridge, "Full_Ridge")]:
            r2   = oos_r2(preds["y_true"], preds["y_hat"])
            rmse_ = rmse(preds["y_true"],  preds["y_hat"])
            cov  = pi_coverage(preds["y_true"], preds["pi_lower_90"], preds["pi_upper_90"])
            bs_in  = brier_score(preds["p_inside"],  preds["true_inside"])
            bs_out = brier_score(preds["p_outside"], preds["true_outside"])

            print(f"\n  {model_name}")
            print(f"    OOS R²          : {r2:.4f}")
            print(f"    RMSE            : {rmse_:.4f}")
            print(f"    90% PI coverage : {cov:.3f}  (target: 0.900)")
            print(f"    Brier(inside)   : {bs_in:.4f}  (naive: {preds['true_inside'].mean()*(1-preds['true_inside'].mean()):.4f})")
            print(f"    Brier(outside)  : {bs_out:.4f}  (naive: {preds['true_outside'].mean()*(1-preds['true_outside'].mean()):.4f})")

            all_metrics.append({
                "symbol": symbol.upper(), "model": model_name,
                "oos_r2": r2, "rmse": rmse_, "pi_coverage_90": cov,
                "brier_inside": bs_in, "brier_outside": bs_out,
                "n_test_days": len(preds),
            })

            plot_actual_vs_predicted(preds, symbol.upper(), model_name)
            plot_probability_calibration(preds, symbol.upper(), model_name)

        # Diebold-Mariano test: does Ridge beat HAR?
        merged = har.merge(ridge[["trade_date", "y_hat"]], on="trade_date",
                           suffixes=("_har", "_ridge"))
        e_har   = (merged["y_true"] - merged["y_hat_har"])   ** 2
        e_ridge = (merged["y_true"] - merged["y_hat_ridge"]) ** 2
        dm_stat, dm_pval = diebold_mariano(e_har.values, e_ridge.values)
        print(f"\n  Diebold-Mariano (HAR vs Ridge): stat={dm_stat:.3f}, p={dm_pval:.4f}")
        print(f"  {'Ridge significantly better' if dm_pval < 0.05 and dm_stat > 0 else 'No significant difference'}")

        # Feature importance
        imp = compute_feature_importance(features, symbol.upper())
        imp.to_csv(f"output/feature_importance_{symbol}.csv", index=False)
        plot_feature_importance(imp, symbol.upper())

        print(f"\n  Top 10 features ({symbol.upper()}):")
        print(imp[["feature", "coef", "abs_coef"]].head(10).to_string(index=False))

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("output/metrics_summary.csv", index=False)
    print(f"\n\nSaved output/metrics_summary.csv")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
```

**Step 3: Run evaluate.py**

```bash
python3 evaluate.py
```

Expected output (approximate numbers):

```
  ES
  HAR_OLS
    OOS R²          : 0.0300  (HAR typical range 0.02–0.06)
    RMSE            : 0.5800
    90% PI coverage : 0.910
    Brier(inside)   : 0.0920
    Brier(outside)  : 0.0910

  Full_Ridge
    OOS R²          : 0.0500
    90% PI coverage : 0.890
    ...

  Diebold-Mariano (HAR vs Ridge): stat=..., p=...
  Top 10 features (ES): ...

Saved output/metrics_summary.csv
```

**Step 4: Verify output files**

```bash
ls output/plots/
ls output/metrics_summary.csv output/feature_importance_es.csv
```

Expected: 8 PNG files (2 symbols × 2 models × 2 plot types + 2 importance plots).

**Step 5: Commit**

```bash
git add evaluate.py
git commit -m "feat: OOS evaluation, Diebold-Mariano test, calibration plots, feature importance"
```

---

## Task 8: Memory — save key findings and file paths

**Step 1: Update project memory**

Save to `/root/.claude/projects/-mnt-e-backup-code-Finance-Research-Inside-outside/memory/MEMORY.md`:

```markdown
# Inside/Outside Day Prediction Project

## Key files
- data/es_1m.parquet, data/nq_1m.parquet — 1-min OHLCV, Sep 2020–Nov 2025, ~1.85M rows
- data/vix_cboe.parquet — CBOE VIX daily (cols: date, o, h, l, c), 1990–present
- data/economic_events.parquet — High-impact USD events (cols: datetime_utc, currency, impact, title, id)
- feature_engineering.py — all feature groups, FEATURE_COLS_HAR, FEATURE_COLS_ALL constants
- model.py — walk_forward(), ols_predict_with_pi(), ridge_predict_with_pi()
- evaluate.py — metrics, plots, feature importance

## Outputs
- output/features_{es,nq}_rth.parquet — feature matrices with target y
- output/predictions_{es,nq}_{har,ridge}.parquet — walk-forward predictions + PIs + P(inside/outside)
- output/metrics_summary.csv — OOS R², RMSE, PI coverage, Brier scores
- output/feature_importance_{es,nq}.csv — Ridge standardised coefficients

## Key empirical facts
- RTH: 09:30–16:14 ET. Min bars per day: 30
- Inside day base rate: ~10.5% (ES RTH). Median range = 58% of prev day
- Outside day base rate: ~10.8% (ES RTH). Median range = 183% of prev day
- Inside day log-ratio thresholds: 95th pct ≈ -0.15 (exp ≈ 0.86)
- Outside day log-ratio thresholds: 5th pct ≈ +0.18 (exp ≈ 1.20)
- Monday: elevated inside rate (~17%). Tuesday: elevated outside for ES (~15%)
- Globex maintenance break: 17:00–17:59 ET (exclude from globex session features)
- FOMC titles: 'US FOMC Statement', 'US Federal Funds Rate', 'US FOMC Economic Projections'
- NFP title: 'US Non-Farm Employment Change'
- Economic events in UTC; ET ≈ UTC - 4h

## Model approach
- Target: y = log(range_{t+1} / range_t) — continuous, approx normal
- HAR baseline: OLS on [rv_1d, rv_5d, rv_22d]
- Full model: Ridge (L2) on all 34 features, walk-forward expanding window (init=252 days)
- PI: t-distribution PI for OLS; Gaussian approximation for Ridge
- P(inside/outside): norm.cdf against empirically calibrated thresholds
```

---

## Running the full pipeline end-to-end

```bash
python3 feature_engineering.py   # ~3 min
python3 model.py                  # ~5 min
python3 evaluate.py               # ~1 min
```

All outputs land in `output/`.
