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


def test_xgb_gpu_params_use_cuda_device_api_and_depth_variants():
    from research_setup_failures import xgb_gpu_params

    params = xgb_gpu_params(scale_pos_weight=2.0, model_name="xgb_gpu_depth2")
    depth1 = xgb_gpu_params(scale_pos_weight=2.0, model_name="xgb_gpu_depth1")
    l1 = xgb_gpu_params(scale_pos_weight=2.0, model_name="xgb_gpu_depth2_l1")
    subsample = xgb_gpu_params(scale_pos_weight=2.0, model_name="xgb_gpu_depth2_subsample")

    assert params["device"] == "cuda"
    assert params["tree_method"] == "hist"
    assert params["max_depth"] == 2
    assert params["scale_pos_weight"] == 2.0
    assert "gpu_id" not in params
    assert "predictor" not in params
    assert depth1["max_depth"] == 1
    assert l1["reg_alpha"] > 0
    assert subsample["subsample"] < params["subsample"]


def test_fit_scores_routes_xgb_gpu_depth_variant_to_xgboost(monkeypatch):
    import sys
    import types
    import research_setup_failures

    captured = {}

    class FakeXGBClassifier:
        def __init__(self, **params):
            captured.update(params)

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            p = np.linspace(0.2, 0.8, len(X))
            return np.column_stack([1.0 - p, p])

    fake_module = types.SimpleNamespace(XGBClassifier=FakeXGBClassifier)
    monkeypatch.setitem(sys.modules, "xgboost", fake_module)

    X_train = np.arange(40, dtype=float).reshape(20, 2)
    y_train = np.array([False, True] * 10)
    X_test = np.arange(8, dtype=float).reshape(4, 2)

    train_scores, test_scores = research_setup_failures._fit_scores(
        "xgb_gpu_depth4", X_train, y_train, X_test
    )

    assert captured["device"] == "cuda"
    assert captured["max_depth"] == 4
    assert len(train_scores) == len(X_train)
    assert len(test_scores) == len(X_test)


def test_run_setup_failure_research_skips_unavailable_xgb_gpu(monkeypatch, tmp_path):
    import research_setup_failures
    from research_setup_failures import run_pcx_failure_mode_research

    daily = _daily_features(80)
    signals = _signals(70)
    feature_path = tmp_path / "features.parquet"
    signal_path = tmp_path / "signals.csv"
    daily.to_parquet(feature_path, index=False)
    signals.to_csv(signal_path, index=False)

    original = research_setup_failures._fit_scores

    def fake_fit(model_name, X_train, y_train, X_test):
        if model_name.startswith("xgb_gpu"):
            raise RuntimeError("xgboost cuda unavailable")
        return original(model_name, X_train, y_train, X_test)

    monkeypatch.setattr(research_setup_failures, "_fit_scores", fake_fit)

    summary, scores, slice_eval = run_pcx_failure_mode_research(
        feature_path=feature_path,
        signal_path=signal_path,
        markovian_root=None,
        train_setup="pcx_wick",
        eval_setups=["pcx_wick"],
        targets=["failure_any"],
        model_names=["logistic", "xgb_gpu_depth2"],
        init_window=30,
        n_splits=2,
    )

    assert set(summary["candidate_model"]) == {"logistic"}
    assert set(scores["candidate_model"]) == {"logistic"}
    assert set(slice_eval["candidate_model"].dropna()) == {"logistic"}


def test_fit_scores_supports_ensemble_models(monkeypatch):
    import research_setup_failures

    def fake_fit(model_name, X_train, y_train, X_test):
        if model_name == "logistic":
            return np.linspace(0.1, 0.7, len(X_train)), np.linspace(0.2, 0.8, len(X_test))
        if model_name == "hgb":
            return np.linspace(0.3, 0.9, len(X_train)), np.linspace(0.4, 0.9, len(X_test))
        if model_name == "xgb_gpu_depth2":
            raise RuntimeError("cuda unavailable")
        raise AssertionError(model_name)

    monkeypatch.setattr(research_setup_failures, "_fit_single_model_scores", fake_fit)
    X_train = np.arange(40, dtype=float).reshape(20, 2)
    y_train = np.array([False, True] * 10)
    X_test = np.arange(8, dtype=float).reshape(4, 2)

    mean_train, mean_test = research_setup_failures._fit_scores("ensemble_mean", X_train, y_train, X_test)
    rank_train, rank_test = research_setup_failures._fit_scores("ensemble_rank_mean", X_train, y_train, X_test)

    assert len(mean_train) == len(X_train)
    assert len(mean_test) == len(X_test)
    assert np.all((rank_train >= 0) & (rank_train <= 1))
    assert np.all((rank_test >= 0) & (rank_test <= 1))


def test_parse_thresholds_and_runner_emit_extra_threshold_flags(tmp_path):
    from research_setup_failures import parse_thresholds, run_pcx_failure_mode_research

    assert parse_thresholds("0.05,0.15,0.30") == [0.05, 0.15, 0.30]

    daily = _daily_features(80)
    signals = _signals(70)
    feature_path = tmp_path / "features.parquet"
    signal_path = tmp_path / "signals.csv"
    daily.to_parquet(feature_path, index=False)
    signals.to_csv(signal_path, index=False)

    summary, scores, slice_eval = run_pcx_failure_mode_research(
        feature_path=feature_path,
        signal_path=signal_path,
        markovian_root=None,
        train_setup="pcx_wick",
        eval_setups=["pcx_wick"],
        targets=["failure_any"],
        model_names=["logistic"],
        init_window=30,
        n_splits=2,
        threshold_fractions=[0.05, 0.15, 0.30],
    )

    assert {"remove_top_05", "remove_top_15", "remove_top_30"}.issubset(scores.columns)
    assert {"remove_top_05", "remove_top_15", "remove_top_30"}.issubset(set(summary["filter"]))
    assert {"remove_top_05", "remove_top_15", "remove_top_30"}.issubset(set(slice_eval["filter"]))


def test_build_side_slice_eval_reports_long_and_short_rows():
    from research_setup_failures import build_side_slice_eval

    scores = pd.DataFrame({
        "trade_date": pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
        ]),
        "direction": ["LONG", "SHORT", "LONG", "SHORT"] * 2,
        "hit": [True, False, True, True] * 2,
        "remove_top_20": [False, True, True, False] * 2,
        "setup": ["pcx_wick"] * 8,
        "target": ["failure_any"] * 8,
        "candidate_model": ["logistic"] * 4 + ["hgb"] * 4,
    })
    membership = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "pcx_wick": [True, True, True, True],
    })

    out = build_side_slice_eval(scores, membership, filter_col="remove_top_20")

    assert set(out["side"]) == {"LONG", "SHORT"}
    assert set(out["eval_setup"]) == {"pcx_wick"}
    assert set(out["candidate_model"]) == {"logistic", "hgb"}
    assert out[out["side"].eq("LONG")].iloc[0]["base_n"] == 2


def test_build_selection_report_ranks_transfer_and_penalizes_small_removed():
    from research_setup_failures import build_selection_report

    slice_eval = pd.DataFrame({
        "target": ["failure_any", "failure_any", "failure_any", "failure_any"],
        "candidate_model": ["a", "a", "b", "b"],
        "filter": ["remove_top_20"] * 4,
        "eval_setup": ["pcx_ict", "pcx_ict_cisd", "pcx_ict", "pcx_ict_cisd"],
        "delta_kept_vs_base": [0.04, 0.03, 0.10, 0.10],
        "removed_n": [30, 25, 4, 3],
    })

    out = build_selection_report(slice_eval, min_removed_n=20)

    assert out.iloc[0]["candidate_model"] == "a"
    assert out.iloc[0]["selection_score"] > out.iloc[1]["selection_score"]


def test_select_ship_config_returns_best_selection_row():
    from research_setup_failures import select_ship_config

    selection = pd.DataFrame({
        "target": ["failure_any", "inside_failure"],
        "candidate_model": ["logistic", "hgb"],
        "filter": ["remove_top_30", "remove_top_20"],
        "selection_score": [0.10, 0.05],
    })

    out = select_ship_config(selection)

    assert len(out) == 1
    assert out.iloc[0]["target"] == "failure_any"
    assert bool(out.iloc[0]["selected_for_ship"]) is True


def test_build_ship_skip_list_filters_selected_scores():
    from research_setup_failures import build_ship_skip_list

    scores = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "direction": ["LONG", "SHORT", "LONG"],
        "score": [0.1, 0.9, 0.8],
        "hit": [True, False, True],
        "setup": ["pcx_wick"] * 3,
        "target": ["failure_any"] * 3,
        "candidate_model": ["logistic"] * 3,
        "remove_top_30": [False, True, True],
    })
    ship_config = pd.DataFrame({
        "target": ["failure_any"],
        "candidate_model": ["logistic"],
        "filter": ["remove_top_30"],
    })

    out = build_ship_skip_list(scores, ship_config)

    assert list(out["trade_date"]) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert set(out["action"]) == {"skip"}
    assert set(out["ship_filter"]) == {"remove_top_30"}


def test_evaluate_ship_holdout_reports_recent_slice_metrics():
    from research_setup_failures import evaluate_ship_holdout

    slice_eval = pd.DataFrame({
        "eval_setup": ["pcx_ict", "pcx_ict_cisd", "pcx_ict"],
        "target": ["failure_any", "failure_any", "inside_failure"],
        "candidate_model": ["logistic", "logistic", "logistic"],
        "filter": ["remove_top_30", "remove_top_30", "remove_top_30"],
        "base_n": [100, 50, 100],
        "kept_hit_rate": [0.85, 0.86, 0.82],
        "delta_kept_vs_base": [0.08, 0.09, 0.02],
        "removed_n": [30, 20, 30],
    })
    ship_config = pd.DataFrame({
        "target": ["failure_any"],
        "candidate_model": ["logistic"],
        "filter": ["remove_top_30"],
    })

    out = evaluate_ship_holdout(slice_eval, ship_config)

    assert set(out["eval_setup"]) == {"pcx_ict", "pcx_ict_cisd"}
    assert (out["holdout_pass"] == True).all()


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


def test_run_pcx_failure_mode_research_trains_wick_and_evaluates_slices(tmp_path):
    from research_setup_failures import run_pcx_failure_mode_research

    daily = _daily_features(140)
    signals = _signals(130)
    for i in signals.index:
        signals.loc[i, "wick_filtered"] = True
        signals.loc[i, "ict_bias"] = 1 if signals.loc[i, "direction"] == "LONG" else -1
        signals.loc[i, "cisd_direction"] = signals.loc[i, "ict_bias"] if i % 3 != 0 else -signals.loc[i, "ict_bias"]

    feature_path = tmp_path / "features.parquet"
    signal_path = tmp_path / "signals.csv"
    daily.to_parquet(feature_path, index=False)
    signals.to_csv(signal_path, index=False)

    summary, scores, slice_eval = run_pcx_failure_mode_research(
        feature_path=feature_path,
        signal_path=signal_path,
        markovian_root=None,
        train_setup="pcx_wick",
        eval_setups=["pcx_wick", "pcx_ict", "pcx_ict_cisd"],
        targets=["failure_any"],
        model_names=["logistic"],
        init_window=40,
        n_splits=3,
    )

    assert not summary.empty
    assert not scores.empty
    assert set(slice_eval["eval_setup"]) == {"pcx_wick", "pcx_ict", "pcx_ict_cisd"}
    assert set(slice_eval["train_setup"]) == {"pcx_wick"}
    assert set(slice_eval["target"].dropna()) == {"failure_any"}


def test_parse_args_accepts_pcx_failure_mode_flags(monkeypatch):
    from pathlib import Path
    from research_setup_failures import parse_args

    monkeypatch.setattr("sys.argv", [
        "research_setup_failures.py",
        "--pcx-failure-mode",
        "--train-setup", "pcx_wick",
        "--eval-setups", "pcx_wick,pcx_ict,pcx_ict_cisd",
    ])

    args = parse_args()

    assert args.pcx_failure_mode is True
    assert args.train_setup == "pcx_wick"
    assert args.eval_setups == "pcx_wick,pcx_ict,pcx_ict_cisd"
    assert args.pcx_failure_summary_output == Path("output/pcx_failure_mode_summary.csv")
    assert args.pcx_failure_scores_output == Path("output/pcx_failure_mode_scores.parquet")
    assert args.pcx_failure_slice_output == Path("output/pcx_failure_mode_slice_eval.csv")
    assert args.pcx_failure_yearly_by_slice_output == Path("output/pcx_failure_mode_yearly_by_slice.csv")


def test_research_setup_failures_uses_safe_inputs_only():
    import ast
    from pathlib import Path

    source = Path("research_setup_failures.py").read_text()
    tree = ast.parse(source)
    allowed_reader_args = {
        "summary_path",
        "feature_path",
        "signal_path",
        "args.feature_path",
        "args.signal_path",
    }

    assert "Does not read raw 1-minute data" in source
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {"glob", "rglob"}:
            raise AssertionError("research_setup_failures.py must not discover raw input files")
        if node.func.attr not in {"read_csv", "read_parquet"}:
            continue
        arg_source = ast.get_source_segment(source, node.args[0]) if node.args else ""
        if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            raise AssertionError(f"reader uses literal path: {node.args[0].value}")
        assert arg_source in allowed_reader_args


def test_build_setup_frame_computes_missing_cisd_direction_from_signal_date(monkeypatch):
    import research_setup_failures
    from research_setup_failures import build_setup_frame

    daily = _daily_features(10)
    signals = _signals(1).drop(columns=["cisd_direction"])
    target_date = pd.Timestamp(signals.loc[0, "date"])
    signal_date = pd.Timestamp(signals.loc[0, "signal_date"])
    earlier_signal_date = signal_date - pd.Timedelta(days=1)
    signals.loc[0, "signal_date"] = earlier_signal_date

    daily.loc[daily["trade_date"].eq(target_date), "inside"] = True
    daily.loc[daily["trade_date"].eq(earlier_signal_date), "Close"] = 999
    daily.loc[daily["trade_date"].eq(signal_date), "Close"] = 555
    daily.loc[daily["trade_date"].eq(target_date), "Close"] = 111

    def fake_cisd(daily_features):
        values = pd.Series(-1, index=daily_features.index)
        values.loc[daily_features["trade_date"].eq(earlier_signal_date)] = 1
        values.loc[daily_features["trade_date"].eq(signal_date)] = -1
        values.loc[daily_features["trade_date"].eq(target_date)] = -1
        return values

    monkeypatch.setattr(research_setup_failures, "_cisd_direction", fake_cisd)

    frame, _ = build_setup_frame(daily, signals, setup="pcx_ict_cisd")

    assert target_date in set(frame["trade_date"])
