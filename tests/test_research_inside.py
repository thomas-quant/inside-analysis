import numpy as np
import pandas as pd


def _sample_features(n=80):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    base = np.linspace(100, 110, n)
    ranges = 5 + np.sin(np.arange(n) / 3)
    df = pd.DataFrame({
        "trade_date": dates,
        "Open": base,
        "High": base + ranges,
        "Low": base - ranges,
        "Close": base + rng.normal(scale=0.5, size=n),
        "Volume": 1000 + rng.normal(scale=10, size=n),
        "range_abs": ranges * 2,
        "inside": [i % 7 == 0 for i in range(n)],
        "outside": [i % 11 == 0 for i in range(n)],
        "rv_1d": np.linspace(0.01, 0.03, n),
        "rv_5d": np.linspace(0.011, 0.031, n),
        "rv_22d": np.linspace(0.012, 0.032, n),
        "range_percentile_22": np.linspace(0.1, 0.9, n),
        "vix_close": np.linspace(12, 24, n),
        "vix_change_1d": rng.normal(size=n),
        "high_impact_tomorrow": [i % 13 == 0 for i in range(n)],
        "is_fomc_day": [i % 31 == 0 for i in range(n)],
        "is_nfp_day": [i % 23 == 0 for i in range(n)],
    })
    df["neither"] = ~(df["inside"] | df["outside"])
    return df


def test_add_inside_research_features_creates_compression_candle_and_event_fields():
    from research_inside import add_inside_research_features

    out = add_inside_research_features(_sample_features())

    required = {
        "inside_next",
        "body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "wick_balance",
        "range_percentile_5",
        "range_percentile_10",
        "range_zscore_22",
        "range_zscore_63",
        "is_pre_fomc",
        "is_pre_nfp",
        "has_major_event_tomorrow",
        "rv_regime_252",
        "vix_regime_252",
    }
    assert required.issubset(out.columns)
    assert out["body_pct"].dropna().between(0, 1).all()
    assert out["inside_next"].iloc[0] == bool(out["inside"].iloc[1])


def test_walk_forward_inside_research_scores_multiple_models_and_feature_sets():
    from research_inside import add_inside_research_features, run_inside_research_for_frame

    df = add_inside_research_features(_sample_features(90))
    summary = run_inside_research_for_frame(
        df,
        symbol="ES",
        session="ETH",
        init_window=30,
        model_names=["logistic", "hgb"],
        feature_set_names=["range_pattern", "range_pattern_event"],
    )

    assert {"logistic", "hgb"} == set(summary["candidate_model"])
    assert {"range_pattern", "range_pattern_event"} == set(summary["feature_set"])
    assert {"auc", "brier_skill", "precision_top_5", "precision_top_10", "yearly_top10_min"}.issubset(summary.columns)
    assert (summary["n_test_days"] > 0).all()


def test_blocked_inside_scores_scores_test_blocks_with_few_refits(monkeypatch):
    import research_inside

    df = research_inside.add_inside_research_features(_sample_features(90))
    calls = []
    original = research_inside._fit_predict_many

    def wrapped(model_name, X_train, y_train, X_test):
        calls.append((len(X_train), len(X_test)))
        return original(model_name, X_train, y_train, X_test)

    monkeypatch.setattr(research_inside, "_fit_predict_many", wrapped)

    scores = research_inside.blocked_inside_scores(
        df,
        ["rv_1d", "range_percentile_22", "body_pct"],
        "logistic",
        init_window=30,
        n_splits=5,
    )

    assert len(scores) > 0
    assert len(calls) == 5
    assert sum(test_len for _, test_len in calls) == len(scores)


def test_default_research_grid_avoids_slow_elasticnet_and_unavailable_gpu():
    import research_inside

    models = research_inside.default_model_names()

    assert models == ["logistic", "hgb"]
    assert "elasticnet" not in models
    assert "xgb_gpu" not in models


def test_gpu_research_grid_uses_only_xgb_gpu_when_available(monkeypatch):
    import research_inside

    monkeypatch.setattr(research_inside.importlib.util, "find_spec", lambda name: object())

    assert research_inside.default_model_names(include_gpu=True) == ["xgb_gpu"]


def test_xgb_gpu_params_use_xgboost_31_device_api():
    import research_inside

    params = research_inside.xgb_gpu_params(scale_pos_weight=2.0)

    assert params["device"] == "cuda"
    assert params["tree_method"] == "hist"
    assert "gpu_id" not in params
    assert "predictor" not in params


def test_select_best_inside_candidates_prefers_eth_top10_then_auc():
    from research_inside import select_best_inside_candidates

    summary = pd.DataFrame({
        "symbol": ["ES", "ES", "NQ"],
        "session": ["ETH", "ETH", "ETH"],
        "candidate_model": ["logistic", "hgb", "hgb"],
        "feature_set": ["a", "b", "b"],
        "base_rate": [0.10, 0.10, 0.10],
        "precision_top_10": [0.25, 0.30, 0.20],
        "auc": [0.72, 0.70, 0.74],
        "brier_skill": [0.01, 0.02, 0.03],
        "yearly_top10_min": [0.12, 0.13, 0.11],
    })

    best = select_best_inside_candidates(summary)

    es = best[(best["symbol"] == "ES") & (best["session"] == "ETH")].iloc[0]
    assert es["candidate_model"] == "hgb"
    assert bool(es["selected"]) is True


def test_summarize_inside_scores_uses_calibrated_prob_for_brier():
    from research_inside import summarize_inside_scores

    scores = pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=160),
        "score": np.r_[np.linspace(0.0, 1.0, 80), np.linspace(0.0, 1.0, 80)],
        "true_inside": [False] * 60 + [True] * 20 + [False] * 60 + [True] * 20,
    })

    summary = summarize_inside_scores(scores)

    assert "brier_raw" in summary
    assert "brier_calibrated" in summary
    assert summary["brier"] == summary["brier_calibrated"]
    assert summary["brier_skill"] > -0.1


def test_selection_rejects_negative_brier_or_unstable_yearly_min():
    from research_inside import select_best_inside_candidates

    summary = pd.DataFrame({
        "symbol": ["ES", "ES"],
        "session": ["ETH", "ETH"],
        "candidate_model": ["xgb_gpu", "logistic"],
        "feature_set": ["rank_good_bad_prob", "stable"],
        "base_rate": [0.10, 0.10],
        "precision_top_10": [0.30, 0.24],
        "auc": [0.70, 0.69],
        "brier_skill": [-0.20, 0.02],
        "yearly_top10_min": [0.20, 0.12],
    })

    best = select_best_inside_candidates(summary)
    selected = best[best["selected"]].iloc[0]

    assert selected["feature_set"] == "stable"
    assert selected["selection_status"] == "eligible"


def test_selection_marks_no_selected_row_when_all_candidates_rejected():
    from research_inside import select_best_inside_candidates

    summary = pd.DataFrame({
        "symbol": ["ES", "ES"],
        "session": ["ETH", "ETH"],
        "candidate_model": ["xgb_gpu", "logistic"],
        "feature_set": ["unstable", "bad_brier"],
        "base_rate": [0.10, 0.10],
        "precision_top_10": [0.30, 0.24],
        "auc": [0.70, 0.69],
        "brier_skill": [0.02, -0.01],
        "yearly_top10_min": [0.00, 0.12],
    })

    selected = select_best_inside_candidates(summary)

    assert selected["selection_status"].eq("rejected").all()
    assert not selected["selected"].any()
    assert selected["best_rejected"].sum() == 1


def test_export_selected_models_only_exports_selected_xgb(monkeypatch, tmp_path):
    import research_inside

    summary = pd.DataFrame({
        "symbol": ["NQ", "ES", "NQ"],
        "session": ["ETH", "ETH", "ETH"],
        "candidate_model": ["xgb_gpu", "xgb_gpu", "logistic"],
        "feature_set": ["full_available", "range_pattern", "range_pattern"],
        "selected": [True, False, True],
    })
    frames = {
        "output/features_nq_eth.parquet": _sample_features(80),
        "output/features_es_eth.parquet": _sample_features(80),
    }
    calls = []

    monkeypatch.setattr(pd, "read_parquet", lambda path: frames[path])

    def fake_export(df, symbol, session, feature_set, output_dir):
        calls.append((symbol, session, feature_set, str(output_dir)))
        return {"model": output_dir / "fake.json"}

    monkeypatch.setattr(research_inside, "export_final_xgb_model", fake_export)

    artifacts = research_inside.export_selected_models(summary, output_dir=tmp_path)

    assert calls == [("NQ", "ETH", "full_available", str(tmp_path))]
    assert artifacts == [{"model": tmp_path / "fake.json"}]
