import pandas as pd
import numpy as np
import pytest
import sys
sys.path.insert(0, ".")

KNOWN_DATE          = pd.Timestamp("2024-01-03")
KNOWN_ETH_RANGE     = 49.75
KNOWN_RTH_RANGE     = 30.75
KNOWN_RTH_RV        = 3.640e-05
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
    from feature_engineering import build_eth_daily
    eth = build_eth_daily(es_1min)
    assert len(eth) > 1300

def test_build_eth_trade_date_convention(es_1min):
    from feature_engineering import build_eth_daily
    eth = build_eth_daily(es_1min)
    jan3 = eth[eth["trade_date"] == pd.Timestamp("2024-01-03")]
    assert len(jan3) == 1

# ── RTH builder ──────────────────────────────────────────────────────────────
def test_build_rth_daily_range_known_date(rth_daily_es):
    row = rth_daily_es[rth_daily_es["trade_date"] == KNOWN_DATE]
    assert abs(float(row["High"].iloc[0] - row["Low"].iloc[0]) - KNOWN_RTH_RANGE) < 0.01

# ── RV features ───────────────────────────────────────────────────────────────
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

# ── Range + Volume features ───────────────────────────────────────────────────
@pytest.fixture(scope="module")
def rv_features_es(eth_daily_es, es_1min):
    from feature_engineering import compute_rv_features
    return compute_rv_features(eth_daily_es, es_1min, session="rth")

def test_range_features_columns(rv_features_es):
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

# ── Session features ──────────────────────────────────────────────────────────
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
        assert (vals >= 0).all() and (vals <= 1.01).all(), f"{col} out of [0,1]"

def test_session_vol_entropy_nonneg(eth_daily_es, es_1min):
    from feature_engineering import compute_session_features
    df = compute_session_features(eth_daily_es, es_1min)
    assert (df["session_vol_entropy"].dropna() >= 0).all()

# ── ETH/RTH cross-session features ───────────────────────────────────────────
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
    assert (vals > 0).all() and (vals <= 1.001).all()

def test_rth_flags_binary(eth_daily_es, rth_daily_es):
    from feature_engineering import compute_eth_rth_cross_features
    df = compute_eth_rth_cross_features(eth_daily_es, rth_daily_es)
    for col in ["rth_inside_flag", "rth_outside_flag", "eth_rth_divergence"]:
        assert set(df[col].dropna().unique()).issubset({0, 1})

# ── Calendar + VIX features ───────────────────────────────────────────────────
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
    import subprocess, os
    result = subprocess.run(
        ["python3", "feature_engineering.py"],
        capture_output=True, text=True, cwd="/mnt/e/backup/code/Finance/Research/Inside-outside"
    )
    assert result.returncode == 0, result.stderr
    assert os.path.exists("output/features_es_eth.parquet")
    assert os.path.exists("output/features_nq_eth.parquet")

def test_no_lookahead_in_features():
    df = pd.read_parquet("output/features_es_eth.parquet")
    assert "y" in df.columns
    assert pd.isna(df["y"].iloc[-1])
