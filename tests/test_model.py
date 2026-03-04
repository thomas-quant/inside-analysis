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
