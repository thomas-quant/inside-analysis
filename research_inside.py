"""
Inside-day research harness.

Uses existing output/features_{symbol}_{session}.parquet files only.
Does not read 1-minute market data and does not regenerate pipeline outputs.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


OUTPUT_PATH = Path("output/inside_research_summary.csv")


BASE_FEATURE_SETS = {
    "range_pattern": [
        "range_abs", "range_pct_of_prev", "atr_ratio", "range_ma_5", "range_ma_22",
        "range_percentile_22", "inside_lag1", "outside_lag1", "inside_streak",
        "outside_streak", "nr4_flag", "nr7_flag", "wr4_flag", "wr7_flag",
        "body_pct", "upper_wick_pct", "lower_wick_pct", "wick_balance",
        "range_percentile_5", "range_percentile_10", "range_zscore_22", "range_zscore_63",
    ],
    "range_pattern_event": [
        "range_abs", "range_pct_of_prev", "atr_ratio", "range_ma_5", "range_ma_22",
        "range_percentile_22", "inside_lag1", "outside_lag1", "inside_streak",
        "outside_streak", "nr4_flag", "nr7_flag", "wr4_flag", "wr7_flag",
        "body_pct", "upper_wick_pct", "lower_wick_pct", "wick_balance",
        "range_percentile_5", "range_percentile_10", "range_zscore_22", "range_zscore_63",
        "high_impact_tomorrow", "is_pre_fomc", "is_pre_nfp", "has_major_event_tomorrow",
    ],
    "range_vix_regime": [
        "range_abs", "range_percentile_22", "range_percentile_5", "range_percentile_10",
        "range_zscore_22", "body_pct", "wick_balance",
        "rv_1d", "rv_5d", "rv_22d", "rv_regime_252",
        "vix_close", "vix_change_1d", "vix_regime_252",
    ],
    "full_available": [],
}


def default_model_names(include_gpu: bool = False) -> list[str]:
    """Fast default grid. xgb_gpu only when requested and xgboost installed."""
    if include_gpu:
        if importlib.util.find_spec("xgboost") is None:
            raise RuntimeError("xgboost not installed; run `pip install xgboost` or omit --gpu")
        return ["xgb_gpu"]
    return ["logistic", "hgb"]


def xgb_gpu_params(scale_pos_weight: float) -> dict:
    """XGBoost >=3.1 CUDA params: use device, no removed gpu_id/predictor."""
    return {
        "n_estimators": 80,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
    }


def _rolling_percentile_current(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    return s.rolling(window, min_periods=min_periods).apply(
        lambda x: (x[:-1] < x[-1]).mean(), raw=True
    )


def add_inside_research_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add no-new-data inside-day research features from daily feature parquets."""
    out = df.copy().sort_values("trade_date").reset_index(drop=True)
    out["inside_next"] = out["inside"].shift(-1).astype("boolean").fillna(False).astype(bool)

    range_abs = out["range_abs"] if "range_abs" in out else (out["High"] - out["Low"])
    safe_range = range_abs.replace(0, np.nan)
    body = (out["Close"] - out["Open"]).abs()
    upper = out["High"] - out[["Open", "Close"]].max(axis=1)
    lower = out[["Open", "Close"]].min(axis=1) - out["Low"]

    out["body_pct"] = (body / safe_range).clip(0.0, 1.0)
    out["upper_wick_pct"] = (upper / safe_range).clip(0.0, 1.0)
    out["lower_wick_pct"] = (lower / safe_range).clip(0.0, 1.0)
    out["wick_balance"] = ((upper - lower) / safe_range).clip(-1.0, 1.0)

    out["range_percentile_5"] = _rolling_percentile_current(range_abs, 5, 3)
    out["range_percentile_10"] = _rolling_percentile_current(range_abs, 10, 5)
    for window in [22, 63]:
        mean = range_abs.rolling(window, min_periods=max(5, window // 3)).mean()
        std = range_abs.rolling(window, min_periods=max(5, window // 3)).std()
        out[f"range_zscore_{window}"] = ((range_abs - mean) / std).replace([np.inf, -np.inf], np.nan)

    out["is_pre_fomc"] = out.get("is_fomc_day", pd.Series(False, index=out.index)).shift(-1).fillna(0).astype(int)
    out["is_pre_nfp"] = out.get("is_nfp_day", pd.Series(False, index=out.index)).shift(-1).fillna(0).astype(int)
    out["has_major_event_tomorrow"] = out.get(
        "high_impact_tomorrow", pd.Series(False, index=out.index)
    ).fillna(0).astype(int)

    rv = out.get("rv_1d", pd.Series(np.nan, index=out.index))
    vix = out.get("vix_close", pd.Series(np.nan, index=out.index))
    out["rv_regime_252"] = (rv > rv.rolling(252, min_periods=30).median()).astype(float)
    out["vix_regime_252"] = (vix > vix.rolling(252, min_periods=30).median()).astype(float)
    return out


def _safe_auc(y_true, score) -> float:
    y = pd.Series(y_true).astype(int)
    if y.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y, score))


def _precision_at_fraction(y_true, score, fraction: float) -> float:
    y = pd.Series(y_true).astype(float).reset_index(drop=True)
    s = pd.Series(score).astype(float).reset_index(drop=True)
    if len(y) == 0:
        return np.nan
    k = max(1, int(np.ceil(len(y) * fraction)))
    idx = s.sort_values(ascending=False).index[:k]
    return float(y.loc[idx].mean())


def _brier_score(p_pred, y_true) -> float:
    y = pd.Series(y_true).astype(float)
    p = pd.Series(p_pred).astype(float)
    return float(((p - y) ** 2).mean())


def rolling_platt_calibrate(
    scores: np.ndarray,
    labels: np.ndarray,
    min_samples: int = 100,
    min_pos: int = 10,
    min_neg: int = 10,
) -> np.ndarray:
    """Use prior rows only to calibrate rank scores into probabilities."""
    out = np.zeros(len(scores), dtype=float)
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    for i in range(len(s)):
        prior_s = s[:i]
        prior_y = y[:i]
        valid = np.isfinite(prior_s)
        prior_s = prior_s[valid]
        prior_y = prior_y[valid]
        n_pos = int(prior_y.sum())
        n_neg = int(len(prior_y) - n_pos)
        if len(prior_y) >= min_samples and n_pos >= min_pos and n_neg >= min_neg:
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300)
            clf.fit(prior_s.reshape(-1, 1), prior_y.astype(int))
            out[i] = float(clf.predict_proba(np.array([[s[i]]]))[0, 1])
        elif len(prior_y) > 0:
            out[i] = float(prior_y.mean())
        else:
            out[i] = 0.0
    return out.clip(0.0, 1.0)


def _feature_columns(df: pd.DataFrame, feature_set_name: str) -> list[str]:
    if feature_set_name == "full_available":
        excluded = {
            "trade_date", "inside", "outside", "neither", "inside_next",
            "y", "Open", "High", "Low", "Close", "Volume",
        }
        cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    else:
        cols = [c for c in BASE_FEATURE_SETS[feature_set_name] if c in df.columns]
    return cols


def _fit_predict(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> float:
    return float(_fit_predict_many(model_name, X_train, y_train, X_test)[0])


def _fit_predict_many(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if n_pos < 5 or n_neg < 5:
        return np.full(len(X_test), float(y_train.mean()))
    if model_name == "logistic":
        clf = LogisticRegression(C=0.1, max_iter=300, solver="lbfgs", class_weight="balanced")
        clf.fit(X_train, y_train.astype(int))
    elif model_name == "elasticnet":
        clf = LogisticRegression(
            C=0.1, penalty="elasticnet", solver="saga", l1_ratio=0.5,
            max_iter=1000, class_weight="balanced", random_state=42,
        )
        clf.fit(X_train, y_train.astype(int))
    elif model_name == "hgb":
        pos_weight = len(y_train) / (2.0 * n_pos)
        neg_weight = len(y_train) / (2.0 * n_neg)
        weights = np.where(y_train, pos_weight, neg_weight)
        clf = HistGradientBoostingClassifier(
            max_iter=30, learning_rate=0.05, max_leaf_nodes=15, random_state=42
        )
        clf.fit(X_train, y_train.astype(int), sample_weight=weights)
    elif model_name == "xgb_gpu":
        try:
            from xgboost import DMatrix, XGBClassifier
        except ModuleNotFoundError as exc:
            raise RuntimeError("xgboost not installed; cannot use xgb_gpu") from exc
        scale_pos_weight = n_neg / n_pos
        clf = XGBClassifier(**xgb_gpu_params(scale_pos_weight))
        clf.fit(X_train, y_train.astype(int))
        return clf.get_booster().predict(DMatrix(X_test)).astype(float)
    else:
        raise ValueError(f"Unknown model_name={model_name}")
    return clf.predict_proba(X_test)[:, 1].astype(float)


def walk_forward_inside_scores(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    init_window: int = 252,
) -> pd.DataFrame:
    """Expanding walk-forward scores for inside_next; feature parquets only."""
    sub = df[["trade_date", "inside_next"] + feature_cols].dropna().reset_index(drop=True)
    records = []
    for t in range(init_window, len(sub)):
        train = sub.iloc[:t]
        test = sub.iloc[[t]]
        scaler = StandardScaler().fit(train[feature_cols].values)
        X_train = scaler.transform(train[feature_cols].values)
        X_test = scaler.transform(test[feature_cols].values)
        y_train = train["inside_next"].values.astype(bool)
        score = _fit_predict(model_name, X_train, y_train, X_test)
        records.append({
            "trade_date": test["trade_date"].iloc[0],
            "score": score,
            "true_inside": bool(test["inside_next"].iloc[0]),
        })
    return pd.DataFrame(records)


def blocked_inside_scores(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    init_window: int = 252,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Blocked time-series CV: few refits, contiguous future test blocks."""
    sub = df[["trade_date", "inside_next"] + feature_cols].dropna().reset_index(drop=True)
    if len(sub) <= init_window:
        return pd.DataFrame(columns=["trade_date", "score", "true_inside"])
    test_positions = np.arange(init_window, len(sub))
    blocks = [b for b in np.array_split(test_positions, n_splits) if len(b)]
    records = []
    for block in blocks:
        train = sub.iloc[: int(block[0])]
        test = sub.iloc[block]
        scaler = StandardScaler().fit(train[feature_cols].values)
        X_train = scaler.transform(train[feature_cols].values)
        X_test = scaler.transform(test[feature_cols].values)
        y_train = train["inside_next"].values.astype(bool)
        scores = _fit_predict_many(model_name, X_train, y_train, X_test)
        for trade_date, score, true_inside in zip(test["trade_date"], scores, test["inside_next"]):
            records.append({
                "trade_date": trade_date,
                "score": float(score),
                "true_inside": bool(true_inside),
            })
    return pd.DataFrame(records)


def summarize_inside_scores(scores: pd.DataFrame) -> dict:
    y = scores["true_inside"].astype(bool)
    s = scores["score"].astype(float)
    base = float(y.mean())
    p_cal = rolling_platt_calibrate(s.values, y.values)
    brier_raw = _brier_score(s, y)
    brier = _brier_score(p_cal, y)
    naive = base * (1.0 - base)
    yearly = []
    years = pd.to_datetime(scores["trade_date"]).dt.year
    for _, g in scores.groupby(years):
        if len(g) >= 20 and g["true_inside"].nunique() > 1:
            yearly.append(_precision_at_fraction(g["true_inside"], g["score"], 0.10))
    return {
        "base_rate": base,
        "auc": _safe_auc(y, s),
        "brier": brier,
        "brier_raw": brier_raw,
        "brier_calibrated": brier,
        "brier_naive": naive,
        "brier_skill": np.nan if naive == 0 else 1.0 - brier / naive,
        "precision_top_5": _precision_at_fraction(y, s, 0.05),
        "precision_top_10": _precision_at_fraction(y, s, 0.10),
        "precision_top_20": _precision_at_fraction(y, s, 0.20),
        "yearly_top10_min": np.nan if not yearly else float(np.nanmin(yearly)),
        "yearly_top10_mean": np.nan if not yearly else float(np.nanmean(yearly)),
        "n_test_days": len(scores),
    }


def run_inside_research_for_frame(
    df: pd.DataFrame,
    symbol: str,
    session: str,
    init_window: int = 252,
    model_names: list[str] | None = None,
    feature_set_names: list[str] | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    model_names = model_names or default_model_names()
    feature_set_names = feature_set_names or list(BASE_FEATURE_SETS)
    rows = []
    for feature_set in feature_set_names:
        cols = _feature_columns(df, feature_set)
        if len(cols) == 0:
            continue
        for model_name in model_names:
            scores = blocked_inside_scores(df, cols, model_name, init_window=init_window, n_splits=n_splits)
            if scores.empty:
                continue
            row = {
                "symbol": symbol,
                "session": session,
                "candidate_model": model_name,
                "feature_set": feature_set,
                "n_features": len(cols),
            }
            row.update(summarize_inside_scores(scores))
            rows.append(row)
    return pd.DataFrame(rows)


def select_best_inside_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    """Select best row per symbol/session by top10, AUC, then Brier skill."""
    out = summary.copy()
    out["selection_status"] = np.where(
        (out["brier_skill"] > 0) & (out["yearly_top10_min"] >= out["base_rate"]),
        "eligible",
        "rejected",
    )
    out["_eligible_rank"] = (out["selection_status"] == "eligible").astype(int)
    sort_cols = ["symbol", "session", "_eligible_rank", "precision_top_10", "auc", "brier_skill"]
    out = out.sort_values(sort_cols, ascending=[True, True, False, False, False, False]).copy()
    out["selected"] = False
    out["best_rejected"] = False
    eligible = out[out["selection_status"] == "eligible"]
    if not eligible.empty:
        best_idx = eligible.groupby(["symbol", "session"], sort=False).head(1).index
        out.loc[best_idx, "selected"] = True
    rejected = out[out["selection_status"] == "rejected"]
    if not rejected.empty:
        best_rejected_idx = rejected.groupby(["symbol", "session"], sort=False).head(1).index
        out.loc[best_rejected_idx, "best_rejected"] = True
    return out.drop(columns=["_eligible_rank"])


def _artifact_stem(symbol: str, session: str, model_name: str, feature_set: str) -> str:
    return f"{symbol.upper()}_{session.upper()}_{model_name}_{feature_set}"


def export_final_xgb_model(
    df: pd.DataFrame,
    symbol: str,
    session: str,
    feature_set: str,
    output_dir: Path = Path("output/inside_models"),
) -> dict[str, Path]:
    """Train selected XGB GPU model on all available rows and export research artifacts."""
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        raise RuntimeError("xgboost not installed; cannot export xgb_gpu model") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    cols = _feature_columns(df, feature_set)
    sub = df[["inside_next"] + cols].dropna().reset_index(drop=True)
    X = sub[cols].values
    y = sub["inside_next"].values.astype(bool)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos < 5 or n_neg < 5:
        raise ValueError(f"Not enough class balance to export {symbol} {session} {feature_set}")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    params = xgb_gpu_params(scale_pos_weight=n_neg / n_pos)
    clf = XGBClassifier(**params)
    clf.fit(Xs, y.astype(int))
    booster = clf.get_booster()
    booster.feature_names = cols

    stem = _artifact_stem(symbol, session, "xgb_gpu", feature_set)
    model_path = output_dir / f"{stem}.model.json"
    config_path = output_dir / f"{stem}.config.json"
    importance_path = output_dir / f"{stem}.feature_importance.csv"
    trees_path = output_dir / f"{stem}.trees.txt"

    booster.save_model(model_path)
    config = {
        "symbol": symbol.upper(),
        "session": session.upper(),
        "candidate_model": "xgb_gpu",
        "feature_set": feature_set,
        "n_features": len(cols),
        "n_train_rows": len(sub),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "xgb_params": params,
        "feature_columns": cols,
        "booster_config": json.loads(booster.save_config()),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    gains = booster.get_score(importance_type="gain")
    weights = booster.get_score(importance_type="weight")
    covers = booster.get_score(importance_type="cover")
    importance = pd.DataFrame({
        "feature": cols,
        "gain": [gains.get(c, 0.0) for c in cols],
        "weight": [weights.get(c, 0.0) for c in cols],
        "cover": [covers.get(c, 0.0) for c in cols],
    }).sort_values(["gain", "weight"], ascending=False)
    importance.to_csv(importance_path, index=False)
    trees_path.write_text("\n".join(booster.get_dump(with_stats=True)), encoding="utf-8")

    return {
        "model": model_path,
        "config": config_path,
        "feature_importance": importance_path,
        "trees": trees_path,
    }


def export_selected_models(
    summary: pd.DataFrame,
    output_dir: Path = Path("output/inside_models"),
) -> list[dict[str, Path]]:
    """Export final artifacts for selected eligible XGB rows only."""
    artifacts = []
    if summary.empty:
        return artifacts
    selected = summary[
        summary.get("selected", False).astype(bool) &
        summary["candidate_model"].eq("xgb_gpu")
    ]
    for _, row in selected.iterrows():
        symbol = row["symbol"].lower()
        session = row["session"].lower()
        df = add_inside_research_features(pd.read_parquet(f"output/features_{symbol}_{session}.parquet"))
        artifacts.append(export_final_xgb_model(
            df, row["symbol"], row["session"], row["feature_set"], output_dir=output_dir
        ))
    return artifacts


def run_inside_research(
    sessions: list[str] | None = None,
    symbols: list[str] | None = None,
    init_window: int = 252,
    model_names: list[str] | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    sessions = sessions or ["eth"]
    symbols = symbols or ["es", "nq"]
    rows = []
    for symbol in symbols:
        for session in sessions:
            path = Path(f"output/features_{symbol}_{session}.parquet")
            if not path.exists():
                continue
            df = add_inside_research_features(pd.read_parquet(path))
            rows.append(run_inside_research_for_frame(
                df, symbol.upper(), session.upper(), init_window=init_window,
                model_names=model_names, n_splits=n_splits,
            ))
    summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not summary.empty:
        summary = select_best_inside_candidates(summary)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)
    export_selected_models(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inside-day research harness.")
    parser.add_argument("--gpu", action="store_true", help="Include xgb_gpu if xgboost with CUDA is installed.")
    parser.add_argument("--models", default=None, help="Comma-separated models, e.g. logistic,hgb,xgb_gpu.")
    parser.add_argument("--sessions", default="eth", help="Comma-separated sessions, default eth.")
    parser.add_argument("--symbols", default="es,nq", help="Comma-separated symbols, default es,nq.")
    parser.add_argument("--init-window", type=int, default=252)
    parser.add_argument("--splits", type=int, default=5, help="Blocked CV splits; 5 means 5 fits per combo.")
    args = parser.parse_args()

    models = args.models.split(",") if args.models else default_model_names(include_gpu=args.gpu)
    summary = run_inside_research(
        sessions=args.sessions.split(","),
        symbols=args.symbols.split(","),
        init_window=args.init_window,
        model_names=models,
        n_splits=args.splits,
    )
    print(f"Saved {OUTPUT_PATH}")
    if not summary.empty:
        print(summary[summary["selected"]].to_string(index=False))


if __name__ == "__main__":
    main()
