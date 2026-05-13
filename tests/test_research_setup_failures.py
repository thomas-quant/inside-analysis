import numpy as np
import pandas as pd


def _daily_features(n=90):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "trade_date": dates,
        "Open": np.linspace(100, 110, n),
        "High": np.linspace(101, 111, n),
        "Low": np.linspace(99, 109, n),
        "Close": np.linspace(100.5, 110.5, n),
        "Volume": np.arange(n) + 1000,
        "range_abs": np.linspace(2.0, 3.0, n),
        "inside": [(i % 5) == 0 for i in range(n)],
        "outside": [(i % 7) == 0 for i in range(n)],
        "neither": [True] * n,
        "rv_1d": np.linspace(0.01, 0.03, n),
        "rv_5d": np.linspace(0.02, 0.04, n),
        "rv_22d": np.linspace(0.03, 0.05, n),
        "range_percentile_22": np.linspace(0.1, 0.9, n),
        "vix_close": np.linspace(12, 20, n),
        "vix_change_1d": np.linspace(-1, 1, n),
    })


def _signals(n=80):
    dates = pd.date_range("2024-01-03", periods=n, freq="D")
    rows = []
    for i, date in enumerate(dates):
        direction = "LONG" if i % 2 == 0 else "SHORT"
        rows.append({
            "date": date,
            "signal_date": date - pd.Timedelta(days=1),
            "direction": direction,
            "hit": i % 4 != 0,
            "wick_filtered": True,
            "ict_bias": 1 if direction == "LONG" else -1,
            "cisd_direction": 1 if direction == "LONG" else -1,
            "signal_body_to_range": 0.2 + i / 500,
            "prior_body_to_range": 0.1 + i / 500,
            "signal_close_location": 0.5,
            "prior_close_location": 0.4,
            "signal_close_to_target_extreme": 0.05 + i / 1000,
            "prior_close_to_target_extreme": 0.2,
            "signal_close_through": 0.1,
            "prior_close_through": -0.1,
            "signal_range_vs_prior_range": 1.0,
        })
    return pd.DataFrame(rows)


def test_build_setup_frame_uses_signal_date_features_and_target_date_labels():
    from research_setup_failures import build_setup_frame

    daily = _daily_features(10)
    signals = _signals(3)
    signals.loc[0, "hit"] = False
    target_date = pd.Timestamp(signals.loc[0, "date"])
    signal_date = pd.Timestamp(signals.loc[0, "signal_date"])
    daily.loc[daily["trade_date"].eq(target_date), "inside"] = True
    daily.loc[daily["trade_date"].eq(signal_date), "rv_1d"] = 0.123
    daily.loc[daily["trade_date"].eq(target_date), "rv_1d"] = 0.999

    frame, feature_cols = build_setup_frame(daily, signals, setup="pcx_ict")
    row = frame[frame["trade_date"].eq(target_date)].iloc[0]

    assert bool(row["failure_any"]) is True
    assert bool(row["inside_failure"]) is True
    assert row["rv_1d"] == 0.123
    assert "inside_next" not in feature_cols


def test_build_setup_frame_computes_missing_ict_bias_from_signal_date(monkeypatch):
    import research_setup_failures
    from research_setup_failures import build_setup_frame

    daily = _daily_features(10)
    signals = _signals(3).drop(columns=["ict_bias"])
    target_date = pd.Timestamp(signals.loc[0, "date"])
    signal_date = pd.Timestamp(signals.loc[0, "signal_date"])

    def fake_ict(daily_features, markovian_root):
        values = pd.Series(-1, index=daily_features["trade_date"].values)
        values.loc[signal_date] = 1
        values.loc[target_date] = -1
        return values.rename("ict_bias")

    monkeypatch.setattr(research_setup_failures, "_compute_ict_bias_with_markovian", fake_ict)

    frame, _ = build_setup_frame(daily, signals, setup="pcx_ict")

    assert target_date in set(frame["trade_date"])


def test_build_setup_frame_drops_missing_target_labels():
    from research_setup_failures import build_setup_frame

    daily = _daily_features(5)
    signals = _signals(5)

    frame, _ = build_setup_frame(daily, signals, setup="pcx_ict")

    assert frame["trade_date"].max() <= daily["trade_date"].max()


def test_blocked_setup_failure_scores_only_scores_future_blocks():
    from research_setup_failures import build_setup_frame, blocked_setup_failure_scores

    frame, feature_cols = build_setup_frame(_daily_features(90), _signals(80), setup="pcx_ict")
    scores = blocked_setup_failure_scores(
        frame,
        feature_cols[:4],
        target_col="failure_any",
        model_name="logistic",
        init_window=30,
        n_splits=5,
        threshold_fractions=[0.10],
    )

    assert len(scores) == len(frame.dropna(subset=feature_cols[:4] + ["failure_any"])) - 30
    assert scores["block_id"].nunique() == 5
    assert scores["trade_date"].min() > frame["trade_date"].iloc[29]
    assert "remove_top_10" in scores.columns


def test_blocked_setup_failure_scores_does_not_remove_all_when_train_labels_degenerate():
    from research_setup_failures import blocked_setup_failure_scores

    frame = pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=40),
        "direction": ["LONG"] * 40,
        "hit": [True] * 40,
        "inside_failure": [False] * 40,
        "x": np.linspace(0, 1, 40),
    })

    scores = blocked_setup_failure_scores(
        frame,
        ["x"],
        target_col="inside_failure",
        model_name="logistic",
        init_window=30,
        n_splits=2,
        threshold_fractions=[0.20],
    )

    assert scores["remove_top_20"].sum() == 0


def test_summarize_setup_filter_reports_kept_removed_trade_value():
    from research_setup_failures import summarize_setup_filter

    scores = pd.DataFrame({
        "hit": [True, False, True, False],
        "remove_top_10": [False, True, False, True],
        "trade_date": pd.date_range("2024-01-01", periods=4),
    })

    summary = summarize_setup_filter(scores, setup="pcx_ict", target="failure_any", model_name="logistic")

    row = summary.iloc[0]
    assert row["base_hit_rate"] == 0.5
    assert row["kept_hit_rate"] == 1.0
    assert row["removed_hit_rate"] == 0.0
    assert row["removed_n"] == 2


def test_select_frozen_candidate_returns_exact_config_only():
    from research_setup_failures import select_frozen_candidate

    summary = pd.DataFrame({
        "setup": ["pcx_ict", "pcx_ict", "pcx_ict"],
        "target": ["inside_failure", "failure_any", "inside_failure"],
        "candidate_model": ["hgb", "hgb", "logistic"],
        "filter": ["remove_top_20", "remove_top_20", "remove_top_20"],
        "delta_kept_vs_base": [0.05, 0.08, 0.02],
    })

    frozen = select_frozen_candidate(summary)

    assert len(frozen) == 1
    row = frozen.iloc[0]
    assert row["target"] == "inside_failure"
    assert row["candidate_model"] == "hgb"
    assert row["filter"] == "remove_top_20"
    assert bool(row["is_frozen_candidate"]) is True


def test_build_yearly_stability_reports_base_kept_removed_by_year():
    from research_setup_failures import build_yearly_stability

    scores = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-01-01", "2023-02-01", "2024-01-01", "2024-02-01"]),
        "hit": [True, False, True, True],
        "remove_top_20": [False, True, False, True],
        "setup": ["pcx_ict"] * 4,
        "target": ["inside_failure"] * 4,
        "candidate_model": ["hgb"] * 4,
    })

    out = build_yearly_stability(scores, filter_col="remove_top_20")

    assert set(out["year"]) == {2023, 2024}
    row_2023 = out[out["year"] == 2023].iloc[0]
    assert row_2023["base_n"] == 2
    assert row_2023["kept_n"] == 1
    assert row_2023["removed_n"] == 1
    assert row_2023["removed_hit_rate"] == 0.0


def test_same_date_comparison_uses_identical_trade_dates():
    from research_setup_failures import build_same_date_comparison

    setup_scores = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "hit": [True, False, True],
        "remove_top_20": [False, True, False],
    })
    generic_scores = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]),
        "inside_score": [0.1, 0.9, 0.8],
    })

    out = build_same_date_comparison(setup_scores, generic_scores, fraction=0.5)

    assert out.iloc[0]["shared_n"] == 2
    assert out.iloc[0]["setup_removed_n"] == 1
    assert out.iloc[0]["generic_removed_n"] == 1


def test_slice_eval_applies_scores_to_named_slices():
    from research_setup_failures import build_slice_eval

    scores = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "hit": [True, False, True],
        "remove_top_20": [False, True, False],
        "setup": ["pcx_wick"] * 3,
    })
    slice_membership = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "pcx_ict": [True, True, False],
        "pcx_ict_cisd": [False, True, False],
    })

    out = build_slice_eval(scores, slice_membership, filter_col="remove_top_20")

    assert set(out["eval_setup"]) == {"pcx_ict", "pcx_ict_cisd"}
    assert out[out["eval_setup"] == "pcx_ict"].iloc[0]["base_n"] == 2


def test_permutation_uses_same_test_hits_and_shuffled_train_targets():
    from research_setup_failures import permutation_delta_test

    frame = pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=60),
        "direction": ["LONG"] * 60,
        "hit": [i % 3 != 0 for i in range(60)],
        "inside_failure": [i % 5 == 0 for i in range(60)],
        "x": np.linspace(0, 1, 60),
    })

    out = permutation_delta_test(
        frame,
        feature_cols=["x"],
        target_col="inside_failure",
        model_name="logistic",
        init_window=30,
        n_splits=3,
        runs=5,
        random_state=42,
    )

    assert len(out) == 5
    assert {"run", "delta_kept_vs_base"}.issubset(out.columns)

def test_failure_mode_feature_columns_include_plausible_groups_and_exclude_leaky_columns():
    from research_setup_failures import failure_mode_feature_columns

    frame = pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=3),
        "hit": [True, False, True],
        "inside_failure": [False, True, False],
        "failure_any": [False, True, False],
        "inside_next": [True, False, True],
        "signal_body_to_range": [0.2, 0.3, 0.4],
        "signal_close_location": [0.6, 0.5, 0.4],
        "signal_close_through": [0.1, -0.1, 0.2],
        "range_percentile_22": [0.1, 0.5, 0.9],
        "rv_1d": [0.01, 0.02, 0.03],
        "inside_lag1": [0, 1, 0],
        "ict_match": [True, False, True],
        "cisd_match": [True, True, False],
        "side": [1.0, -1.0, 1.0],
    })

    cols = failure_mode_feature_columns(frame)

    assert "signal_body_to_range" in cols
    assert "signal_close_location" in cols
    assert "signal_close_through" in cols
    assert "range_percentile_22" in cols
    assert "rv_1d" in cols
    assert "inside_lag1" in cols
    assert "ict_match" in cols
    assert "cisd_match" in cols
    assert "side" in cols
    assert "inside_next" not in cols
    assert "hit" not in cols
    assert "inside_failure" not in cols
    assert "failure_any" not in cols


def test_build_slice_eval_preserves_target_model_filter_and_train_setup():
    from research_setup_failures import build_slice_eval

    scores = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "hit": [True, False, True],
        "remove_top_20": [False, True, False],
        "setup": ["pcx_wick"] * 3,
        "target": ["failure_any"] * 3,
        "candidate_model": ["hgb"] * 3,
    })
    slice_membership = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "pcx_wick": [True, True, True],
        "pcx_ict": [True, True, False],
        "pcx_ict_cisd": [False, True, False],
    })

    out = build_slice_eval(scores, slice_membership, filter_col="remove_top_20")
    row = out[out["eval_setup"].eq("pcx_ict")].iloc[0]

    assert row["train_setup"] == "pcx_wick"
    assert row["target"] == "failure_any"
    assert row["candidate_model"] == "hgb"
    assert row["filter"] == "remove_top_20"
    assert row["base_n"] == 2
