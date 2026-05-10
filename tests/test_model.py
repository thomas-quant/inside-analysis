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
    assert 0.70 < coverage < 0.99, f"PI coverage {coverage:.2f}"


def test_run_session_models_writes_session_prediction_outputs(monkeypatch, tmp_path):
    import pandas as pd
    import model

    feature_paths = {}
    for symbol in ["es", "nq"]:
        for session in ["eth", "rth"]:
            feature_paths[(symbol, session)] = pd.DataFrame({"trade_date": pd.date_range("2024-01-01", periods=3)})

    def fake_read_parquet(path):
        stem = path.replace("output/features_", "").replace(".parquet", "")
        symbol, session = stem.split("_")
        return feature_paths[(symbol, session)]

    def fake_walk_forward(df, feature_cols, target_col, **kwargs):
        return pd.DataFrame({
            "trade_date": [df["trade_date"].iloc[-1]],
            "y_true": [0.0],
            "y_hat": [0.1],
            "pi_lower_90": [-1.0],
            "pi_upper_90": [1.0],
            "sigma": [0.2],
            "p_inside": [0.1],
            "p_outside": [0.2],
            "p_neither": [0.7],
            "true_inside": [False],
            "true_outside": [False],
            "true_neither": [True],
        })

    written = {}

    def fake_to_parquet(self, path, index=False):
        written[path] = self.copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(model, "walk_forward", fake_walk_forward)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    model.run_session_models()

    expected = {
        "output/predictions_es_eth_har.parquet",
        "output/predictions_es_eth_ridge.parquet",
        "output/predictions_es_rth_har.parquet",
        "output/predictions_es_rth_ridge.parquet",
        "output/predictions_nq_eth_har.parquet",
        "output/predictions_nq_eth_ridge.parquet",
        "output/predictions_nq_rth_har.parquet",
        "output/predictions_nq_rth_ridge.parquet",
        "output/predictions_es_har.parquet",
        "output/predictions_es_ridge.parquet",
        "output/predictions_nq_har.parquet",
        "output/predictions_nq_ridge.parquet",
    }
    assert set(written) == expected
    assert written["output/predictions_es_rth_har.parquet"]["session"].iloc[0] == "RTH"
    assert written["output/predictions_es_rth_har.parquet"]["model"].iloc[0] == "HAR_OLS"


def test_compute_probabilities_accepts_balanced_class_weight():
    import numpy as np
    from model import compute_probabilities

    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(80, 4))
    X_test = rng.normal(size=(1, 4))
    inside = np.zeros(80, dtype=bool)
    inside[:8] = True
    outside = np.zeros(80, dtype=bool)
    outside[8:18] = True

    p_in, p_out, p_nei = compute_probabilities(
        X_test, X_train, inside, outside, class_weight="balanced"
    )

    assert 0.0 <= p_in <= 1.0
    assert 0.0 <= p_out <= 1.0
    assert 0.0 <= p_nei <= 1.0


def test_compute_probabilities_supports_hist_gradient_boosting():
    import numpy as np
    from model import compute_probabilities

    rng = np.random.default_rng(11)
    X_train = rng.normal(size=(120, 5))
    X_test = rng.normal(size=(1, 5))
    inside = X_train[:, 0] > 1.0
    outside = X_train[:, 1] < -1.0

    p_in, p_out, p_nei = compute_probabilities(
        X_test, X_train, inside, outside, clf_type="hgb"
    )

    assert 0.0 <= p_in <= 1.0
    assert 0.0 <= p_out <= 1.0
    assert 0.0 <= p_nei <= 1.0


def test_compute_probabilities_rejects_unknown_classifier_before_label_fallback():
    import numpy as np
    import pytest
    from model import compute_probabilities

    X_train = np.zeros((20, 2))
    X_test = np.zeros((1, 2))
    labels = np.zeros(20, dtype=bool)

    with pytest.raises(ValueError, match="Unknown clf_type"):
        compute_probabilities(X_test, X_train, labels, labels, clf_type="bad")


def test_rolling_platt_calibration_uses_only_prior_rows():
    from model import rolling_platt_calibrate

    scores = np.array([0.2, 0.8, 0.7, 0.9, 0.1, 0.95, 0.85])
    labels_a = np.array([0, 1, 1, 1, 0, 1, 0])
    labels_b = np.array([0, 1, 1, 1, 0, 1, 1])

    cal_a = rolling_platt_calibrate(scores, labels_a, min_samples=4, min_pos=1, min_neg=1)
    cal_b = rolling_platt_calibrate(scores, labels_b, min_samples=4, min_pos=1, min_neg=1)

    assert np.all((0.0 <= cal_a) & (cal_a <= 1.0))
    assert cal_a[-1] == cal_b[-1], "current row label must not calibrate its own probability"


def test_normalize_event_probabilities_clips_and_rescales():
    from model import normalize_event_probabilities

    p_inside, p_outside, p_neither = normalize_event_probabilities([0.8, -1.0], [0.7, 0.4])

    assert np.allclose(p_inside + p_outside + p_neither, 1.0)
    assert np.all((0.0 <= p_inside) & (p_inside <= 1.0))
    assert np.all((0.0 <= p_outside) & (p_outside <= 1.0))
    assert np.all((0.0 <= p_neither) & (p_neither <= 1.0))


def test_walk_forward_emits_raw_scores_and_calibrated_probabilities():
    from model import walk_forward

    n = 40
    rng = np.random.default_rng(123)
    df = pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "rv_1d": rng.normal(size=n),
        "y": rng.normal(size=n),
        "inside": [i % 7 == 0 for i in range(n)],
        "outside": [i % 11 == 0 for i in range(n)],
    })
    df["neither"] = ~(df["inside"] | df["outside"])

    out = walk_forward(
        df,
        ["rv_1d"],
        "y",
        init_window=12,
        model_type="ridge",
        class_feature_cols=["rv_1d"],
    )

    assert {"score_inside_raw", "score_outside_raw", "score_neither_raw"}.issubset(out.columns)
    assert {"p_inside", "p_outside", "p_neither"}.issubset(out.columns)
    assert np.allclose(out[["p_inside", "p_outside", "p_neither"]].sum(axis=1), 1.0)
