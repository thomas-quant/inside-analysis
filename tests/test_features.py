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


def test_session_features_use_globex_trade_date_for_asia(es_1min, eth_daily_es):
    from feature_engineering import compute_session_features
    df = compute_session_features(eth_daily_es, es_1min)
    row = df[df["trade_date"] == KNOWN_DATE].iloc[0]

    prev_evening_asia = es_1min[
        (es_1min["session"] == "ASIA")
        & (es_1min["DateTime_ET"] >= pd.Timestamp("2024-01-02 18:00"))
        & (es_1min["DateTime_ET"] < pd.Timestamp("2024-01-03 00:00"))
    ]
    expected_range = prev_evening_asia["High"].max() - prev_evening_asia["Low"].min()
    expected_pct = expected_range / float(row["High"] - row["Low"])

    assert abs(row["asia_range_pct"] - expected_pct) < 1e-12

def test_calendar_features_ignore_events_after_market_data_end(rth_daily_es, eco):
    from feature_engineering import compute_calendar_features
    df = compute_calendar_features(rth_daily_es, eco)
    last = df.iloc[-1]
    assert last["trade_date"] == rth_daily_es["trade_date"].max()
    assert int(last["high_impact_tomorrow"]) == 0
    assert int(last["n_events_next_2d"]) == 0

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

# ── Target builder ────────────────────────────────────────────────────────────
def test_add_target_uses_session_daily_high_low():
    import pandas as pd
    import pytest
    import numpy as np
    from feature_engineering import add_target

    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "Open": [100.0, 101.0, 102.0],
        "High": [110.0, 108.0, 112.0],
        "Low": [90.0, 92.0, 88.0],
        "Close": [105.0, 103.0, 100.0],
        "Volume": [1, 1, 1],
        "range_abs": [20.0, 16.0, 24.0],
    })

    out = add_target(df)

    assert bool(out.loc[1, "inside"]) is True
    assert bool(out.loc[1, "outside"]) is False
    assert bool(out.loc[2, "inside"]) is False
    assert bool(out.loc[2, "outside"]) is True
    assert out.loc[0, "y"] == pytest.approx(np.log(16.0 / 20.0))

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
        capture_output=True, text=True, cwd="."
    )
    assert result.returncode == 0, result.stderr
    assert os.path.exists("output/features_es_eth.parquet")
    assert os.path.exists("output/features_nq_eth.parquet")

def test_no_lookahead_in_features():
    df = pd.read_parquet("output/features_es_eth.parquet")
    assert "y" in df.columns
    assert pd.isna(df["y"].iloc[-1])

def test_build_features_for_returns_eth_and_rth_feature_frames():
    import pandas as pd
    import feature_engineering as fe

    raw = pd.read_parquet("data/es_1m.parquet")
    vix = pd.read_parquet("data/vix_cboe.parquet")
    vix["date"] = pd.to_datetime(vix["date"])
    eco = pd.read_parquet("data/economic_events.parquet")

    eth, rth = fe._build_features_for("ES", raw, vix, eco)

    assert "range_abs" in eth.columns
    assert "range_abs" in rth.columns
    assert len(eth) > 1000
    assert len(rth) > 1000
    assert rth["trade_date"].is_monotonic_increasing

    rth_pct = rth["rth_pct_of_eth"].dropna()
    overnight_pct = rth["overnight_pct_of_eth"].dropna()
    assert 0 < rth_pct.median() < 1
    assert not (rth_pct == 1).all()
    assert not (overnight_pct == 0).all()


def test_finalize_feature_frames_returns_eth_and_rth_outputs():
    import pandas as pd
    from feature_engineering import finalize_feature_frames

    def frame(offset):
        return pd.DataFrame({
            "trade_date": pd.date_range("2024-01-02", periods=4, freq="D"),
            "Open": [100 + offset, 101 + offset, 102 + offset, 103 + offset],
            "High": [110 + offset, 108 + offset, 112 + offset, 111 + offset],
            "Low": [90 + offset, 92 + offset, 88 + offset, 91 + offset],
            "Close": [105 + offset, 103 + offset, 100 + offset, 109 + offset],
            "Volume": [1, 1, 1, 1],
            "range_abs": [20.0, 16.0, 24.0, 20.0],
            "close_location": [0.75, 0.6875, 0.5, 0.9],
            "rv_1d": [0.1, 0.2, 0.3, 0.4],
        })

    outputs = finalize_feature_frames(
        es=frame(0),
        nq=frame(100),
        es_rth=frame(1),
        nq_rth=frame(101),
    )

    assert set(outputs) == {
        "output/features_es_eth.parquet",
        "output/features_nq_eth.parquet",
        "output/features_es_rth.parquet",
        "output/features_nq_rth.parquet",
    }
    for result in outputs.values():
        assert {"inside", "outside", "neither", "y", "range_percentile_22"}.issubset(result.columns)
        assert "es_nq_rv_ratio" in result.columns
        assert "es_nq_outside_divergence" in result.columns
        assert "one_side_break" in result.columns
        assert pd.isna(result["y"].iloc[-1])


def test_pattern_features_include_nr_wr_and_streaks():
    import pandas as pd
    from feature_engineering import compute_range_features, add_target, compute_pattern_features

    df = pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=8, freq="D"),
        "Open": [10, 10, 10, 10, 10, 10, 10, 10],
        "High": [20, 19, 18, 17, 16, 30, 29, 28],
        "Low": [10, 11, 12, 13, 14, 5, 6, 7],
        "Close": [15, 15, 15, 15, 15, 20, 20, 20],
        "Volume": [1]*8,
    })
    df = compute_range_features(df)
    df = add_target(df)
    out = compute_pattern_features(df)

    for col in ["nr4_flag", "nr7_flag", "wr4_flag", "wr7_flag", "inside_streak", "outside_streak"]:
        assert col in out.columns

    assert out.loc[4, "nr4_flag"] == 1
    assert out.loc[5, "wr4_flag"] == 1


def test_pattern_features_include_breakout_context():
    import pandas as pd
    from feature_engineering import compute_range_features, add_target, compute_pattern_features

    df = pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=4, freq="D"),
        "Open": [100, 101, 101, 101],
        "High": [110, 112, 111, 113],
        "Low": [90, 91, 89, 88],
        "Close": [105, 111, 90, 100],
        "Volume": [1, 1, 1, 1],
    })

    out = compute_pattern_features(add_target(compute_range_features(df)))

    assert bool(out.loc[1, "break_high"]) is True
    assert bool(out.loc[1, "break_low"]) is False
    assert bool(out.loc[1, "high_only_break"]) is True
    assert bool(out.loc[1, "one_side_break"]) is True
    assert bool(out.loc[2, "low_only_break"]) is True
    assert bool(out.loc[3, "outside"]) is True
    assert bool(out.loc[3, "one_side_break"]) is False
    assert out.loc[1, "dist_to_prev_high"] == 0.0
    assert out.loc[2, "dist_to_prev_low"] == 0.0


def test_cross_instrument_features_include_observed_divergence_without_future_rows():
    import pandas as pd
    from feature_engineering import compute_cross_instrument_features

    es = pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "High": [110, 112, 113, 114, 113],
        "Low": [90, 91, 89, 88, 89],
        "range_abs": [20, 21, 24, 26, 24],
        "rv_1d": [1, 2, 3, 4, 5],
    })
    nq = pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "High": [210, 211, 212, 211, 213],
        "Low": [190, 189, 188, 189, 187],
        "range_abs": [20, 22, 24, 22, 26],
        "rv_1d": [2, 2, 2, 2, 2],
    })

    es_out, nq_out = compute_cross_instrument_features(es, nq)

    required = {
        "both_outside",
        "es_nq_outside_divergence",
        "nq_outside_es_one_side",
        "es_outside_nq_one_side",
        "cross_outside_divergence_rate_5",
    }
    assert required.issubset(es_out.columns)
    assert required.issubset(nq_out.columns)
    assert int(es_out.loc[1, "es_nq_outside_divergence"]) == 1
    assert int(es_out.loc[1, "nq_outside_es_one_side"]) == 1
    assert int(es_out.loc[2, "both_outside"]) == 1
    assert es_out.loc[1, "cross_outside_divergence_rate_5"] == 1.0
