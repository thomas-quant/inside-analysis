"""
model.py
Walk-forward volatility forecasting: HAR baseline + full Ridge.
Outputs: output/predictions_{es,nq}_{har,ridge}.parquet
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
) -> tuple:
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
) -> tuple:
    """
    Ridge regression. PI uses training residual std (Gaussian approximation).
    Scaler is fitted on training data and applied to test data.
    Returns: y_pred, pi_lower, pi_upper, residual_std, fitted_model
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
    X_test: np.ndarray,
    X_train: np.ndarray,
    labels_inside_next: np.ndarray,
    labels_outside_next: np.ndarray,
    scaler: "StandardScaler | None" = None,
) -> tuple:
    """
    Calibrated classification probabilities via logistic regression on the full
    feature matrix. Using all features directly (rather than the compressed y_hat
    scalar) gives substantially better AUC for inside/outside day classification.

    X_train / X_test must be on the same scale; pass scaler to reuse the
    regression scaler when available (avoids fitting a second scaler).

    Returns p_inside, p_outside, p_neither.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler as SS

    if scaler is not None:
        Xs_tr = scaler.transform(X_train)
        Xs_te = scaler.transform(X_test)
    else:
        sc = SS().fit(X_train)
        Xs_tr = sc.transform(X_train)
        Xs_te = sc.transform(X_test)

    def _logistic_prob(labels):
        if labels.sum() < 5:
            return float(labels.mean())
        lr = LogisticRegression(C=0.1, max_iter=300, solver="lbfgs")
        lr.fit(Xs_tr, labels.astype(int))
        return float(lr.predict_proba(Xs_te)[0, 1])

    p_in  = _logistic_prob(labels_inside_next)
    p_out = _logistic_prob(labels_outside_next)
    p_nei = max(0.0, 1.0 - p_in - p_out)
    return p_in, p_out, p_nei



def walk_forward(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    init_window: int = WALK_FORWARD_INIT,
    model_type: str = "ols",
    ridge_alpha: float = 1.0,
    class_feature_cols: list | None = None,
) -> pd.DataFrame:
    """
    Expanding walk-forward validation.
    Train on [0, t-1], predict on t, expand by 1.
    Returns DataFrame with one row per test day.

    class_feature_cols: features used for the classification logistic regression.
    Defaults to feature_cols, but can be set to FEATURE_COLS_ALL to give HAR's
    classification the full feature set while keeping its regression on RV only.
    """
    clf_cols = class_feature_cols if class_feature_cols is not None else feature_cols

    # Compute next-day labels on the full df (all calendar days present) before
    # any dropna, so the shift aligns by calendar date rather than by row index
    # in the filtered subset (which would mislabel gaps where features are NaN).
    df = df.copy()
    df["inside_next"]  = df["inside"].shift(-1).fillna(False).astype(bool)
    df["outside_next"] = df["outside"].shift(-1).fillna(False).astype(bool)
    df["neither_next"] = (~df["inside_next"]) & (~df["outside_next"])

    next_label_cols = ["inside_next", "outside_next", "neither_next"]
    all_feat_cols = list(dict.fromkeys(feature_cols + clf_cols))  # deduplicated union
    cols_needed = all_feat_cols + [target_col] + LABEL_COLS + next_label_cols + ["trade_date"]
    sub = df[cols_needed].dropna(subset=all_feat_cols + [target_col]).reset_index(drop=True)

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

        X_train_clf = train[clf_cols].values
        X_test_clf  = test[clf_cols].values

        is_inside_next  = train["inside_next"].values
        is_outside_next = train["outside_next"].values

        if model_type == "ols":
            y_hat, pi_lo, pi_hi, _, sigma = ols_predict_with_pi(X_train, y_train, X_test)
        else:
            scaler = StandardScaler().fit(X_train)
            y_hat, pi_lo, pi_hi, sigma, _ = ridge_predict_with_pi(
                X_train, y_train, X_test, scaler, ridge_alpha=ridge_alpha
            )

        p_in, p_out, p_nei = compute_probabilities(
            X_test_clf, X_train_clf, is_inside_next, is_outside_next
        )

        records.append({
            "trade_date":     test["trade_date"].iloc[0],
            "y_true":         y_true,
            "y_hat":          float(y_hat[0]),
            "pi_lower_90":    float(pi_lo[0]),
            "pi_upper_90":    float(pi_hi[0]),
            "sigma":          float(sigma),
            "p_inside":       float(p_in),
            "p_outside":      float(p_out),
            "p_neither":      float(p_nei),
            "true_inside":    bool(test["inside_next"].iloc[0]),
            "true_outside":   bool(test["outside_next"].iloc[0]),
            "true_neither":   bool(test["neither_next"].iloc[0]),
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
                           init_window=WALK_FORWARD_INIT, model_type="ols",
                           class_feature_cols=FEATURE_COLS_ALL)
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
