"""
model.py
Walk-forward volatility forecasting: HAR baseline + full Ridge.
Outputs: output/predictions_{es,nq}_{har,ridge}.parquet
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from feature_engineering import (
    FEATURE_COLS_HAR, FEATURE_COLS_ALL, TARGET_COL, LABEL_COLS
)

WALK_FORWARD_INIT = 252   # initial training window (trading days)
ALPHA_PI = 0.10           # 1 - confidence level → 90% PI


def normalize_event_probabilities(p_inside, p_outside) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip event probabilities, preserving p_inside when outside must be reduced."""
    p_in = np.asarray(p_inside, dtype=float)
    p_out = np.asarray(p_outside, dtype=float)
    p_in = np.nan_to_num(p_in, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    p_out = np.nan_to_num(p_out, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    p_out = np.minimum(p_out, 1.0 - p_in)
    p_nei = 1.0 - p_in - p_out
    return p_in, p_out, p_nei


def rolling_platt_calibrate(
    scores: np.ndarray,
    labels: np.ndarray,
    min_samples: int = 100,
    min_pos: int = 10,
    min_neg: int = 10,
) -> np.ndarray:
    """
    Convert raw ranking scores into probabilities using only prior OOS rows.

    Row i is calibrated on rows [:i], so the current label can never influence
    its own probability. Falls back to the prior base rate until enough prior
    positives and negatives exist.
    """
    from sklearn.linear_model import LogisticRegression

    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    out = np.zeros(len(s), dtype=float)

    for i in range(len(s)):
        prior_scores = s[:i]
        prior_labels = y[:i]
        valid = np.isfinite(prior_scores)
        prior_scores = prior_scores[valid]
        prior_labels = prior_labels[valid]

        n_pos = int(prior_labels.sum())
        n_neg = int(len(prior_labels) - n_pos)
        if len(prior_labels) >= min_samples and n_pos >= min_pos and n_neg >= min_neg:
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300)
            clf.fit(prior_scores.reshape(-1, 1), prior_labels.astype(int))
            out[i] = float(clf.predict_proba(np.array([[s[i]]]))[0, 1])
        elif len(prior_labels) > 0:
            out[i] = float(prior_labels.mean())
        else:
            out[i] = 0.0

    return out.clip(0.0, 1.0)


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
    class_weight: "str | dict | None" = None,
    clf_type: str = "logistic",
) -> tuple:
    """
    Raw inside/outside ranking scores via a classifier on the full feature matrix.
    Rolling calibration is applied later to keep ranking separate from probability
    estimation.

    X_train / X_test must be on the same scale; pass scaler to reuse the
    regression scaler when available (avoids fitting a second scaler).

    Returns score_inside_raw, score_outside_raw, score_neither_raw.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler as SS

    if clf_type not in {"logistic", "hgb"}:
        raise ValueError(f"Unknown clf_type={clf_type}")

    if scaler is not None:
        Xs_tr = scaler.transform(X_train)
        Xs_te = scaler.transform(X_test)
    else:
        sc = SS().fit(X_train)
        Xs_tr = sc.transform(X_train)
        Xs_te = sc.transform(X_test)

    def _clf_prob(labels):
        n_pos = int(labels.sum())
        n_neg = int(len(labels) - n_pos)
        if n_pos < 5 or n_neg < 5:
            return float(labels.mean())
        if clf_type == "logistic":
            clf = LogisticRegression(C=0.1, max_iter=300, solver="lbfgs", class_weight=class_weight)
        elif clf_type == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(
                max_iter=30, learning_rate=0.05, max_leaf_nodes=15, random_state=42
            )
        fit_kwargs = {}
        if clf_type == "hgb" and class_weight == "balanced":
            pos_weight = len(labels) / (2.0 * n_pos)
            neg_weight = len(labels) / (2.0 * n_neg)
            fit_kwargs["sample_weight"] = np.where(labels, pos_weight, neg_weight)
        clf.fit(Xs_tr, labels.astype(int), **fit_kwargs)
        return float(clf.predict_proba(Xs_te)[0, 1])

    p_in  = _clf_prob(labels_inside_next)
    p_out = _clf_prob(labels_outside_next)
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
    add_regression_context_to_classifier: bool = True,
    include_inside_hgb_candidate: bool = False,
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
            y_hat, pi_lo, pi_hi, fitted_reg, sigma = ols_predict_with_pi(X_train, y_train, X_test)
            y_hat_train = fitted_reg.predict(X_train)
        else:
            scaler = StandardScaler().fit(X_train)
            y_hat, pi_lo, pi_hi, sigma, fitted_reg = ridge_predict_with_pi(
                X_train, y_train, X_test, scaler, ridge_alpha=ridge_alpha
            )
            y_hat_train = fitted_reg.predict(scaler.transform(X_train))

        if add_regression_context_to_classifier:
            train_context = np.column_stack([
                y_hat_train,
                np.full(len(train), float(sigma)),
                np.full(len(train), float(pi_hi[0] - pi_lo[0])),
            ])
            test_context = np.array([[float(y_hat[0]), float(sigma), float(pi_hi[0] - pi_lo[0])]])
            X_train_clf = np.hstack([X_train_clf, train_context])
            X_test_clf = np.hstack([X_test_clf, test_context])

        score_in, score_out, score_nei = compute_probabilities(
            X_test_clf, X_train_clf, is_inside_next, is_outside_next,
            class_weight="balanced",
        )
        score_in_hgb = np.nan
        if include_inside_hgb_candidate:
            score_in_hgb, _, _ = compute_probabilities(
                X_test_clf, X_train_clf, is_inside_next, is_outside_next,
                class_weight="balanced", clf_type="hgb",
            )

        records.append({
            "trade_date":     test["trade_date"].iloc[0],
            "y_true":         y_true,
            "y_hat":          float(y_hat[0]),
            "pi_lower_90":    float(pi_lo[0]),
            "pi_upper_90":    float(pi_hi[0]),
            "sigma":          float(sigma),
            "score_inside_raw":  float(score_in),
            "score_inside_raw_hgb": float(score_in_hgb),
            "score_outside_raw": float(score_out),
            "score_neither_raw": float(score_nei),
            "true_inside":    bool(test["inside_next"].iloc[0]),
            "true_outside":   bool(test["outside_next"].iloc[0]),
            "true_neither":   bool(test["neither_next"].iloc[0]),
        })

        if t % 100 == 0:
            print(f"  Walk-forward step {t}/{n}")

    result = pd.DataFrame(records)
    p_in = rolling_platt_calibrate(result["score_inside_raw"].values, result["true_inside"].values)
    p_out = rolling_platt_calibrate(result["score_outside_raw"].values, result["true_outside"].values)
    p_in, p_out, p_nei = normalize_event_probabilities(p_in, p_out)
    result["p_inside"] = p_in
    result["p_outside"] = p_out
    result["p_neither"] = p_nei
    return result


def run_session_models() -> None:
    """Run HAR/Ridge predictions for ES/NQ across ETH and RTH feature targets."""
    Path("output").mkdir(exist_ok=True)

    for symbol in ["es", "nq"]:
        for session in ["eth", "rth"]:
            print(f"\n{'='*50}")
            print(f"  {symbol.upper()} {session.upper()} — loading features")
            df = pd.read_parquet(f"output/features_{symbol}_{session}.parquet")

            print(f"  {symbol.upper()} {session.upper()} — HAR baseline (OLS)")
            har = walk_forward(df, FEATURE_COLS_HAR, TARGET_COL,
                               init_window=WALK_FORWARD_INIT, model_type="ols",
                               class_feature_cols=FEATURE_COLS_ALL)
            har["model"] = "HAR_OLS"
            har["session"] = session.upper()
            har_path = f"output/predictions_{symbol}_{session}_har.parquet"
            har.to_parquet(har_path, index=False)
            print(f"  Saved {har_path}  ({len(har)} rows)")

            print(f"  {symbol.upper()} {session.upper()} — Full Ridge model")
            ridge = walk_forward(
                df, FEATURE_COLS_ALL, TARGET_COL,
                init_window=WALK_FORWARD_INIT, model_type="ridge",
                include_inside_hgb_candidate=os.environ.get("RUN_INSIDE_HGB") == "1",
            )
            ridge["model"] = "Full_Ridge"
            ridge["session"] = session.upper()
            ridge_path = f"output/predictions_{symbol}_{session}_ridge.parquet"
            ridge.to_parquet(ridge_path, index=False)
            print(f"  Saved {ridge_path}  ({len(ridge)} rows)")

            if session == "eth":
                har.to_parquet(f"output/predictions_{symbol}_har.parquet", index=False)
                ridge.to_parquet(f"output/predictions_{symbol}_ridge.parquet", index=False)


def main():
    run_session_models()


if __name__ == "__main__":
    main()
