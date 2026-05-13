"""
Setup-conditioned failure research harness.

Safe path: reads existing daily feature parquets and markovian-ms signal_features.csv.
Does not read raw 1-minute data or regenerate feature pipelines.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from research_inside import add_inside_research_features, _feature_columns


def _default_markovian_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "markovian-ms"
        if candidate.exists():
            return candidate
    return here.parent.parent / "markovian-ms"


DEFAULT_MARKOVIAN_ROOT = _default_markovian_root()
DEFAULT_FEATURE_PATH = Path("output/features_nq_eth.parquet")
DEFAULT_SIGNAL_PATH = DEFAULT_MARKOVIAN_ROOT / "experiments/close_magnitude_analysis/logs/signal_features.csv"
SUMMARY_PATH = Path("output/setup_failure_research_summary.csv")
SCORES_PATH = Path("output/setup_failure_scores_nq_eth.parquet")
YEARLY_PATH = Path("output/setup_failure_yearly_stability.csv")
COMPARISON_PATH = Path("output/setup_failure_same_date_comparison.csv")
PERMUTATION_PATH = Path("output/setup_failure_permutation.csv")
SLICE_PATH = Path("output/setup_failure_slice_eval.csv")
FROZEN_PATH = Path("output/setup_failure_frozen_candidate.csv")
PCX_FAILURE_SUMMARY_PATH = Path("output/pcx_failure_mode_summary.csv")
PCX_FAILURE_SCORES_PATH = Path("output/pcx_failure_mode_scores.parquet")
PCX_FAILURE_SLICE_PATH = Path("output/pcx_failure_mode_slice_eval.csv")
PCX_FAILURE_YEARLY_BY_SLICE_PATH = Path("output/pcx_failure_mode_yearly_by_slice.csv")
PCX_FAILURE_SIDE_SLICE_PATH = Path("output/pcx_failure_mode_side_slice_eval.csv")
PCX_FAILURE_SELECTION_PATH = Path("output/pcx_failure_mode_selection.csv")

FROZEN_SETUP = "pcx_ict"
FROZEN_TARGET = "inside_failure"
FROZEN_MODEL = "hgb"
FROZEN_FILTER = "remove_top_20"

PCX_FEATURE_COLUMNS = [
    "signal_body_to_range",
    "prior_body_to_range",
    "signal_close_location",
    "prior_close_location",
    "signal_close_to_target_extreme",
    "prior_close_to_target_extreme",
    "signal_close_through",
    "prior_close_through",
    "signal_range_vs_prior_range",
]

FAILURE_MODE_FEATURE_CANDIDATES = [
    # PCX candle quality
    "signal_body_to_range",
    "prior_body_to_range",
    "signal_close_location",
    "prior_close_location",
    "signal_close_to_target_extreme",
    "prior_close_to_target_extreme",
    "signal_close_through",
    "prior_close_through",
    "signal_range_vs_prior_range",
    # inside/compression regime
    "inside_lag1",
    "outside_lag1",
    "inside_streak",
    "outside_streak",
    "nr4_flag",
    "nr7_flag",
    "wr4_flag",
    "wr7_flag",
    "range_percentile_5",
    "range_percentile_10",
    "range_percentile_22",
    "range_zscore_22",
    "range_zscore_63",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "wick_balance",
    # volatility/range context
    "range_abs",
    "range_pct_of_prev",
    "atr_ratio",
    "range_ma_5",
    "range_ma_22",
    "rv_1d",
    "rv_5d",
    "rv_22d",
    "rv_regime_252",
    "vix_close",
    "vix_change_1d",
    "vix_regime_252",
    # directional setup context
    "side",
    "ict_match",
    "cisd_match",
]

LEAKY_FAILURE_MODE_COLUMNS = {
    "trade_date",
    "signal_date",
    "hit",
    "failure_any",
    "inside_failure",
    "target_inside",
    "inside_next",
    "true_failure",
    "score",
}


def failure_mode_feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = []
    for col in FAILURE_MODE_FEATURE_CANDIDATES:
        if col in frame.columns and col not in LEAKY_FAILURE_MODE_COLUMNS:
            if pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col]):
                cols.append(col)
    return cols

EXCLUDED_SIGNAL_COLUMNS = {
    "date", "trade_date", "signal_date", "prior_date", "direction", "hit", "wick_filtered",
    "ran_high", "ran_low", "ict_bias", "cisd_direction", "cisd_direction_raw",
    "pcx_prediction_for_today", "signal_open", "signal_high", "signal_low", "signal_close",
    "prior_open", "prior_high", "prior_low", "prior_close",
}


def _normalize_dates(values) -> pd.Series:
    dates = pd.Series(pd.to_datetime(values))
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    return dates.dt.normalize()


def _rate(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else np.nan


def _direction_to_side(direction: pd.Series) -> pd.Series:
    return direction.map({"LONG": 1, "SHORT": -1}).astype(float)


def _cisd_direction(daily: pd.DataFrame) -> pd.Series:
    close = daily["Close"]
    open_ = daily["Open"]
    direction = np.where(close > open_, "bullish", np.where(close < open_, "bearish", "neutral"))
    prev_direction = pd.Series(direction, index=daily.index).shift(1)
    prev_close = close.shift(1)
    cisd = np.select(
        [
            (prev_direction == "bearish") & (close > prev_close),
            (prev_direction == "bullish") & (close < prev_close),
        ],
        [1, -1],
        default=0,
    )
    return pd.Series(cisd.astype(int), index=daily.index)


def _compute_ict_bias_with_markovian(daily_features: pd.DataFrame, markovian_root: Path | None) -> pd.Series:
    if markovian_root is None:
        raise ValueError("ict_bias missing from signals; provide markovian_root to compute it from daily OHLC")
    root = Path(markovian_root)
    sys.path.insert(0, str(root))
    try:
        from ict_daily_bias import BiasEngine
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass

    engine = BiasEngine()
    biases = []
    daily = daily_features.sort_values("trade_date").reset_index(drop=True)
    for i, row in daily.iterrows():
        _, bias = engine.update(i, {
            "Open": row["Open"],
            "High": row["High"],
            "Low": row["Low"],
            "Close": row["Close"],
        })
        biases.append(bias.value)
    return pd.Series(biases, index=daily["trade_date"].values, name="ict_bias")


def enrich_signal_context(
    daily_features: pd.DataFrame,
    signal_features: pd.DataFrame,
    markovian_root: str | Path | None = DEFAULT_MARKOVIAN_ROOT,
) -> pd.DataFrame:
    """Add ict_bias/cisd_direction to signal rows when absent, using daily feature OHLC only."""
    signals = signal_features.copy()
    if "date" in signals.columns and "trade_date" not in signals.columns:
        signals = signals.rename(columns={"date": "trade_date"})
    signals["trade_date"] = _normalize_dates(signals["trade_date"]).to_numpy()
    signals["signal_date"] = _normalize_dates(signals["signal_date"]).to_numpy()

    daily = daily_features.copy()
    daily["trade_date"] = _normalize_dates(daily["trade_date"]).to_numpy()
    daily = daily.sort_values("trade_date").reset_index(drop=True)

    if "ict_bias" not in signals.columns:
        ict = _compute_ict_bias_with_markovian(daily, Path(markovian_root) if markovian_root else None)
        ict_frame = ict.rename("ict_bias").reset_index().rename(columns={"index": "signal_date"})
        ict_frame["signal_date"] = _normalize_dates(ict_frame["signal_date"]).to_numpy()
        signals = signals.merge(ict_frame, on="signal_date", how="left")

    if "cisd_direction" not in signals.columns:
        cisd = _cisd_direction(daily).rename("cisd_direction_raw")
        cisd_frame = pd.DataFrame({
            "signal_date": daily["trade_date"],
            "cisd_direction": cisd.to_numpy(),
        })
        signals = signals.merge(cisd_frame, on="signal_date", how="left")

    return signals


def build_setup_frame(
    daily_features: pd.DataFrame,
    signal_features: pd.DataFrame,
    setup: str = "pcx_ict",
    markovian_root: str | Path | None = DEFAULT_MARKOVIAN_ROOT,
) -> tuple[pd.DataFrame, list[str]]:
    """Build setup rows with labels and leak-safe signal-date features."""
    daily = add_inside_research_features(daily_features)
    daily["trade_date"] = _normalize_dates(daily["trade_date"]).to_numpy()
    daily = daily.sort_values("trade_date").reset_index(drop=True)

    signals = enrich_signal_context(daily, signal_features, markovian_root=markovian_root)
    signals["side"] = _direction_to_side(signals["direction"])
    signals["ict_match"] = signals["ict_bias"].astype(float).eq(signals["side"])
    signals["cisd_match"] = signals["cisd_direction"].astype(float).eq(signals["side"])

    if setup == "pcx_wick":
        setup_mask = signals["wick_filtered"].astype(bool)
    elif setup == "pcx_ict":
        setup_mask = signals["wick_filtered"].astype(bool) & signals["ict_match"]
    elif setup == "pcx_ict_cisd":
        setup_mask = signals["wick_filtered"].astype(bool) & signals["ict_match"] & signals["cisd_match"]
    else:
        raise ValueError(f"Unknown setup={setup}")
    signals = signals[setup_mask].copy()

    target_labels = daily[["trade_date", "inside"]].rename(columns={"inside": "target_inside"})
    frame = signals.merge(target_labels, on="trade_date", how="left")
    frame = frame[frame["target_inside"].notna()].copy()

    daily_feature_cols = _feature_columns(daily, "full_available")
    signal_daily = daily[["trade_date"] + daily_feature_cols].rename(columns={"trade_date": "signal_date"})
    frame = frame.merge(signal_daily, on="signal_date", how="left")

    frame["hit"] = frame["hit"].astype(bool)
    frame["target_inside"] = frame["target_inside"].fillna(False).astype(bool)
    frame["failure_any"] = ~frame["hit"]
    frame["inside_failure"] = frame["failure_any"] & frame["target_inside"]

    pcx_cols = [c for c in PCX_FEATURE_COLUMNS if c in frame.columns]
    feature_cols = daily_feature_cols + pcx_cols + ["side"]
    feature_cols = [c for c in feature_cols if c in frame.columns and c != "inside_next"]
    return frame.sort_values("trade_date").reset_index(drop=True), feature_cols


def _fit_single_model_scores(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if model_name == "logistic":
        clf = LogisticRegression(C=0.1, max_iter=300, solver="lbfgs", class_weight="balanced")
        clf.fit(X_train, y_train.astype(int))
    elif model_name == "hgb":
        pos_weight = len(y_train) / (2.0 * n_pos)
        neg_weight = len(y_train) / (2.0 * n_neg)
        weights = np.where(y_train, pos_weight, neg_weight)
        clf = HistGradientBoostingClassifier(max_iter=30, learning_rate=0.05, max_leaf_nodes=15, random_state=42)
        clf.fit(X_train, y_train.astype(int), sample_weight=weights)
    elif model_name.startswith("xgb_gpu"):
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError as exc:
            raise RuntimeError("xgboost not installed; cannot use xgb_gpu models") from exc
        clf = XGBClassifier(**xgb_gpu_params(scale_pos_weight=n_neg / n_pos, model_name=model_name))
        try:
            clf.fit(X_train, y_train.astype(int))
        except Exception as exc:
            raise RuntimeError(f"{model_name} failed; CUDA-enabled xgboost may be unavailable: {exc}") from exc
    else:
        raise ValueError(f"Unknown model_name={model_name}")
    return clf.predict_proba(X_train)[:, 1].astype(float), clf.predict_proba(X_test)[:, 1].astype(float)


def _rank_against_train(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    train = np.sort(np.asarray(train_scores, dtype=float))
    if len(train) == 0:
        return np.zeros(len(scores), dtype=float)
    return np.searchsorted(train, np.asarray(scores, dtype=float), side="right") / len(train)


def _fit_ensemble_scores(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_models = ["logistic", "hgb", "xgb_gpu_depth2"]
    train_parts = []
    test_parts = []
    for base_model in base_models:
        try:
            train_scores, test_scores = _fit_single_model_scores(base_model, X_train, y_train, X_test)
        except RuntimeError:
            if base_model.startswith("xgb_gpu"):
                continue
            raise
        if model_name == "ensemble_rank_mean":
            raw_train_scores = train_scores
            train_scores = _rank_against_train(raw_train_scores, raw_train_scores)
            test_scores = _rank_against_train(raw_train_scores, test_scores)
        train_parts.append(train_scores)
        test_parts.append(test_scores)
    if not train_parts:
        raise RuntimeError(f"{model_name} has no available base models")
    return np.mean(train_parts, axis=0), np.mean(test_parts, axis=0)


def _fit_scores(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if n_pos < 3 or n_neg < 3:
        p = float(y_train.mean()) if len(y_train) else 0.0
        return np.full(len(y_train), p), np.full(len(X_test), p)
    if model_name in {"ensemble_mean", "ensemble_rank_mean"}:
        return _fit_ensemble_scores(model_name, X_train, y_train, X_test)
    return _fit_single_model_scores(model_name, X_train, y_train, X_test)


def xgb_gpu_params(scale_pos_weight: float, model_name: str = "xgb_gpu") -> dict:
    """XGBoost >=3.1 CUDA params: use device, no removed gpu_id/predictor."""
    variants = {
        "xgb_gpu": {"max_depth": 3, "n_estimators": 160, "learning_rate": 0.04},
        "xgb_gpu_depth1": {"max_depth": 1, "n_estimators": 260, "learning_rate": 0.035},
        "xgb_gpu_depth2": {"max_depth": 2, "n_estimators": 220, "learning_rate": 0.035},
        "xgb_gpu_depth2_l1": {"max_depth": 2, "n_estimators": 220, "learning_rate": 0.035, "reg_alpha": 1.0},
        "xgb_gpu_depth2_subsample": {
            "max_depth": 2,
            "n_estimators": 260,
            "learning_rate": 0.03,
            "subsample": 0.65,
            "colsample_bytree": 0.65,
        },
        "xgb_gpu_depth3": {"max_depth": 3, "n_estimators": 220, "learning_rate": 0.035},
        "xgb_gpu_depth4": {"max_depth": 4, "n_estimators": 180, "learning_rate": 0.03},
        "xgb_gpu_tuned": {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.025},
    }
    if model_name not in variants:
        raise ValueError(f"Unknown xgb gpu model_name={model_name}")
    params = {
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_lambda": 5.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "verbosity": 0,
    }
    params.update(variants[model_name])
    return params


def _should_skip_model_error(model_name: str, exc: Exception) -> bool:
    return model_name.startswith("xgb_gpu") and isinstance(exc, RuntimeError)


def _flag_name(fraction: float) -> str:
    return f"remove_top_{int(round(fraction * 100)):02d}"


def parse_thresholds(value: str | None) -> list[float] | None:
    if value is None or str(value).strip() == "":
        return None
    thresholds = [float(x.strip()) for x in str(value).split(",") if x.strip()]
    if any(x <= 0 or x >= 1 for x in thresholds):
        raise ValueError("thresholds must be fractions between 0 and 1")
    return thresholds


def _thresholds_from_train_scores(train_scores: np.ndarray, threshold_fractions: list[float]) -> dict[str, float]:
    s = pd.Series(train_scores)
    if s.nunique(dropna=True) < 2:
        return {_flag_name(frac): np.inf for frac in threshold_fractions}
    return {
        _flag_name(frac): float(s.quantile(1.0 - frac))
        for frac in threshold_fractions
    }


def _usable_feature_columns(frame: pd.DataFrame, feature_cols: list[str], min_non_na_fraction: float = 0.80) -> list[str]:
    """Drop sparse columns before model fitting to preserve setup sample size."""
    if frame.empty:
        return []
    return [c for c in feature_cols if c in frame.columns and frame[c].notna().mean() >= min_non_na_fraction]


def blocked_setup_failure_scores(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "inside_failure",
    model_name: str = "logistic",
    init_window: int = 200,
    n_splits: int = 3,
    threshold_fractions: list[float] | None = None,
) -> pd.DataFrame:
    """Blocked time-series CV. Thresholds derive from train scores only."""
    threshold_fractions = threshold_fractions or [0.05, 0.10, 0.20]
    cols = ["trade_date", "direction", "hit", target_col] + feature_cols
    sub = frame[cols].dropna().sort_values("trade_date").reset_index(drop=True)
    if len(sub) <= init_window:
        return pd.DataFrame()

    test_positions = np.arange(init_window, len(sub))
    blocks = [b for b in np.array_split(test_positions, n_splits) if len(b)]
    records = []
    for block_id, block in enumerate(blocks):
        train = sub.iloc[: int(block[0])]
        test = sub.iloc[block]
        scaler = StandardScaler().fit(train[feature_cols].to_numpy(dtype=float))
        X_train = scaler.transform(train[feature_cols].to_numpy(dtype=float))
        X_test = scaler.transform(test[feature_cols].to_numpy(dtype=float))
        y_train = train[target_col].to_numpy(dtype=bool)
        train_scores, test_scores = _fit_scores(model_name, X_train, y_train, X_test)
        thresholds = _thresholds_from_train_scores(train_scores, threshold_fractions)
        for i, (_, row) in enumerate(test.iterrows()):
            rec = {
                "trade_date": row["trade_date"],
                "direction": row["direction"],
                "hit": bool(row["hit"]),
                "target": target_col,
                "true_failure": bool(row[target_col]),
                "score": float(test_scores[i]),
                "block_id": block_id,
            }
            for flag, threshold in thresholds.items():
                rec[flag] = bool(test_scores[i] >= threshold)
            records.append(rec)
    return pd.DataFrame(records)


def _blocked_setup_failure_scores_with_permuted_train(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_name: str,
    init_window: int,
    n_splits: int,
    rng: np.random.Generator,
    threshold_fractions: list[float] | None = None,
) -> pd.DataFrame:
    """Blocked CV null: shuffle train labels in each block; keep test hit/labels fixed."""
    threshold_fractions = threshold_fractions or [0.20]
    cols = ["trade_date", "direction", "hit", target_col] + feature_cols
    sub = frame[cols].dropna().sort_values("trade_date").reset_index(drop=True)
    if len(sub) <= init_window:
        return pd.DataFrame()

    test_positions = np.arange(init_window, len(sub))
    blocks = [b for b in np.array_split(test_positions, n_splits) if len(b)]
    records = []
    for block_id, block in enumerate(blocks):
        train = sub.iloc[: int(block[0])]
        test = sub.iloc[block]
        scaler = StandardScaler().fit(train[feature_cols].to_numpy(dtype=float))
        X_train = scaler.transform(train[feature_cols].to_numpy(dtype=float))
        X_test = scaler.transform(test[feature_cols].to_numpy(dtype=float))
        y_train = train[target_col].to_numpy(dtype=bool).copy()
        rng.shuffle(y_train)
        train_scores, test_scores = _fit_scores(model_name, X_train, y_train, X_test)
        thresholds = _thresholds_from_train_scores(train_scores, threshold_fractions)
        for i, (_, row) in enumerate(test.iterrows()):
            rec = {
                "trade_date": row["trade_date"],
                "direction": row["direction"],
                "hit": bool(row["hit"]),
                "target": target_col,
                "true_failure": bool(row[target_col]),
                "score": float(test_scores[i]),
                "block_id": block_id,
            }
            for flag, threshold in thresholds.items():
                rec[flag] = bool(test_scores[i] >= threshold)
            records.append(rec)
    return pd.DataFrame(records)


def _monthly_frequency(dates: pd.Series) -> float:
    if dates.empty:
        return 0.0
    d = pd.to_datetime(dates)
    months = pd.period_range(d.min().to_period("M"), d.max().to_period("M"), freq="M")
    return len(d) / len(months) if len(months) else 0.0


def summarize_setup_filter(scores: pd.DataFrame, setup: str, target: str, model_name: str) -> pd.DataFrame:
    """Summarize kept vs removed hit rate for every remove_top_* flag."""
    rows = []
    flag_cols = [c for c in scores.columns if c.startswith("remove_top_")]
    failure = scores["true_failure"] if "true_failure" in scores.columns else ~scores["hit"].astype(bool)
    for flag in flag_cols:
        mask = scores[flag].astype(bool)
        removed = scores[mask]
        kept = scores[~mask]
        removed_failure = failure[mask]
        rows.append({
            "setup": setup,
            "target": target,
            "candidate_model": model_name,
            "filter": flag,
            "base_n": len(scores),
            "base_hit_rate": float(scores["hit"].mean()) if len(scores) else np.nan,
            "kept_n": len(kept),
            "kept_hit_rate": float(kept["hit"].mean()) if len(kept) else np.nan,
            "removed_n": len(removed),
            "removed_hit_rate": float(removed["hit"].mean()) if len(removed) else np.nan,
            "removed_failure_rate": float(removed_failure.mean()) if len(removed) else np.nan,
            "delta_kept_vs_base": (float(kept["hit"].mean()) - float(scores["hit"].mean())) if len(kept) and len(scores) else np.nan,
            "monthly_frequency_removed": _monthly_frequency(removed["trade_date"]),
            "monthly_frequency_kept": _monthly_frequency(kept["trade_date"]),
        })
    return pd.DataFrame(rows)


def select_frozen_candidate(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary[
        summary["setup"].eq(FROZEN_SETUP)
        & summary["target"].eq(FROZEN_TARGET)
        & summary["candidate_model"].eq(FROZEN_MODEL)
        & summary["filter"].eq(FROZEN_FILTER)
    ].copy()
    out["is_frozen_candidate"] = True
    return out


def build_yearly_stability(scores: pd.DataFrame, filter_col: str = FROZEN_FILTER) -> pd.DataFrame:
    rows = []
    if scores.empty:
        return pd.DataFrame()
    tmp = scores.copy()
    tmp["trade_date"] = _normalize_dates(tmp["trade_date"]).to_numpy()
    tmp["year"] = pd.to_datetime(tmp["trade_date"]).dt.year
    for year, group in tmp.groupby("year"):
        removed = group[group[filter_col].astype(bool)]
        kept = group[~group[filter_col].astype(bool)]
        rows.append({
            "year": int(year),
            "setup": group["setup"].iloc[0] if "setup" in group else FROZEN_SETUP,
            "target": group["target"].iloc[0] if "target" in group else FROZEN_TARGET,
            "candidate_model": group["candidate_model"].iloc[0] if "candidate_model" in group else FROZEN_MODEL,
            "filter": filter_col,
            "base_n": len(group),
            "base_hit_rate": _rate(group["hit"]),
            "kept_n": len(kept),
            "kept_hit_rate": _rate(kept["hit"]),
            "removed_n": len(removed),
            "removed_hit_rate": _rate(removed["hit"]),
            "monthly_frequency_removed": _monthly_frequency(removed["trade_date"]),
        })
    return pd.DataFrame(rows)


def build_same_date_comparison(
    setup_scores: pd.DataFrame,
    generic_scores: pd.DataFrame,
    fraction: float = 0.20,
) -> pd.DataFrame:
    setup = setup_scores.copy()
    generic = generic_scores.copy()
    setup["trade_date"] = _normalize_dates(setup["trade_date"]).to_numpy()
    generic["trade_date"] = _normalize_dates(generic["trade_date"]).to_numpy()
    joined = setup.merge(generic[["trade_date", "inside_score"]], on="trade_date", how="inner")
    if joined.empty:
        return pd.DataFrame([{"shared_n": 0}])
    k = max(1, math.ceil(len(joined) * fraction))
    generic_removed_idx = joined["inside_score"].sort_values(ascending=False).index[:k]
    joined["generic_remove"] = False
    joined.loc[generic_removed_idx, "generic_remove"] = True
    setup_removed = joined[joined[FROZEN_FILTER].astype(bool)] if FROZEN_FILTER in joined else joined.iloc[0:0]
    generic_removed = joined[joined["generic_remove"]]
    return pd.DataFrame([{
        "shared_n": len(joined),
        "setup_removed_n": len(setup_removed),
        "setup_removed_hit_rate": _rate(setup_removed["hit"]),
        "generic_removed_n": len(generic_removed),
        "generic_removed_hit_rate": _rate(generic_removed["hit"]),
        "base_hit_rate": _rate(joined["hit"]),
    }])


def load_generic_inside_scores(
    feature_path: Path = DEFAULT_FEATURE_PATH,
    init_window: int = 252,
    n_splits: int = 5,
    model_name: str = "hgb",
    feature_set: str = "full_available",
    summary_path: Path = Path("output/inside_research_summary.csv"),
) -> pd.DataFrame:
    from research_inside import blocked_inside_scores

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if not summary.empty:
            candidates = summary.copy()
            if "selected" in candidates:
                selected = candidates[candidates["selected"].astype(bool)]
                if not selected.empty:
                    candidates = selected
            sort_cols = [c for c in ["precision_top_10", "auc", "brier_skill"] if c in candidates]
            if sort_cols:
                candidates = candidates.sort_values(sort_cols, ascending=False)
            row = candidates.iloc[0]
            model_name = str(row.get("candidate_model", model_name))
            feature_set = str(row.get("feature_set", feature_set))

    daily = pd.read_parquet(feature_path)
    daily = add_inside_research_features(daily)
    cols = _feature_columns(daily, feature_set)
    scores = blocked_inside_scores(daily, cols, model_name=model_name, init_window=init_window, n_splits=n_splits)
    out = scores.rename(columns={"score": "inside_score"}).copy()
    out["trade_date"] = _normalize_dates(out["trade_date"]).to_numpy()
    return out[["trade_date", "inside_score"]]


def build_slice_membership(
    daily_features: pd.DataFrame,
    signal_features: pd.DataFrame,
    setups: list[str] | None = None,
    markovian_root: str | Path | None = DEFAULT_MARKOVIAN_ROOT,
) -> pd.DataFrame:
    setups = setups or ["pcx_wick", "pcx_ict", "pcx_ict_cisd"]
    dates = pd.DataFrame({
        "trade_date": _normalize_dates(
            signal_features["trade_date"] if "trade_date" in signal_features else signal_features["date"]
        ).drop_duplicates().sort_values().to_numpy()
    })
    out = dates.copy()
    for setup in setups:
        frame, _ = build_setup_frame(daily_features, signal_features, setup=setup, markovian_root=markovian_root)
        setup_dates = set(_normalize_dates(frame["trade_date"]))
        out[setup] = out["trade_date"].isin(setup_dates)
    return out


def _summary_row_for_scores(scores: pd.DataFrame, filter_col: str) -> dict:
    removed = scores[scores[filter_col].astype(bool)]
    kept = scores[~scores[filter_col].astype(bool)]
    return {
        "target": scores["target"].iloc[0] if "target" in scores and len(scores) else np.nan,
        "candidate_model": scores["candidate_model"].iloc[0] if "candidate_model" in scores and len(scores) else np.nan,
        "filter": filter_col,
        "base_n": len(scores),
        "base_hit_rate": _rate(scores["hit"]),
        "kept_n": len(kept),
        "kept_hit_rate": _rate(kept["hit"]),
        "removed_n": len(removed),
        "removed_hit_rate": _rate(removed["hit"]),
        "delta_kept_vs_base": (_rate(kept["hit"]) - _rate(scores["hit"])) if len(scores) and len(kept) else np.nan,
        "monthly_frequency_removed": _monthly_frequency(removed["trade_date"]),
    }


def build_slice_eval(
    scores: pd.DataFrame,
    slice_membership: pd.DataFrame,
    filter_col: str = FROZEN_FILTER,
) -> pd.DataFrame:
    scored = scores.copy()
    membership = slice_membership.copy()
    scored["trade_date"] = _normalize_dates(scored["trade_date"]).to_numpy()
    membership["trade_date"] = _normalize_dates(membership["trade_date"]).to_numpy()
    joined = scored.merge(membership, on="trade_date", how="inner")
    rows = []
    for col in [c for c in membership.columns if c != "trade_date"]:
        group = joined[joined[col].astype(bool)]
        row = _summary_row_for_scores(group, filter_col) if len(group) else {
            "target": scored["target"].iloc[0] if "target" in scored and len(scored) else np.nan,
            "candidate_model": scored["candidate_model"].iloc[0] if "candidate_model" in scored and len(scored) else np.nan,
            "filter": filter_col,
            "base_n": 0,
            "base_hit_rate": np.nan,
            "kept_n": 0,
            "kept_hit_rate": np.nan,
            "removed_n": 0,
            "removed_hit_rate": np.nan,
            "delta_kept_vs_base": np.nan,
            "monthly_frequency_removed": 0.0,
        }
        row["train_setup"] = scored["setup"].iloc[0] if "setup" in scored and len(scored) else np.nan
        row["eval_setup"] = col
        rows.append(row)
    return pd.DataFrame(rows)


def build_side_slice_eval(
    scores: pd.DataFrame,
    slice_membership: pd.DataFrame,
    filter_col: str = FROZEN_FILTER,
) -> pd.DataFrame:
    scored = scores.copy()
    membership = slice_membership.copy()
    scored["trade_date"] = _normalize_dates(scored["trade_date"]).to_numpy()
    membership["trade_date"] = _normalize_dates(membership["trade_date"]).to_numpy()
    joined = scored.merge(membership, on="trade_date", how="inner")
    rows = []
    group_cols = [c for c in ["setup", "target", "candidate_model"] if c in joined.columns]
    grouped = joined.groupby(group_cols, dropna=False) if group_cols else [((), joined)]
    for keys, score_group in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        meta = dict(zip(group_cols, key_values))
        for setup_col in [c for c in membership.columns if c != "trade_date"]:
            setup_group = score_group[score_group[setup_col].astype(bool)]
            for side in ["LONG", "SHORT"]:
                side_group = setup_group[setup_group["direction"].eq(side)]
                row = _summary_row_for_scores(side_group, filter_col) if len(side_group) else {
                    "target": meta.get("target", np.nan),
                    "candidate_model": meta.get("candidate_model", np.nan),
                    "filter": filter_col,
                    "base_n": 0,
                    "base_hit_rate": np.nan,
                    "kept_n": 0,
                    "kept_hit_rate": np.nan,
                    "removed_n": 0,
                    "removed_hit_rate": np.nan,
                    "delta_kept_vs_base": np.nan,
                    "monthly_frequency_removed": 0.0,
                }
                row["train_setup"] = meta.get("setup", np.nan)
                row["eval_setup"] = setup_col
                row["side"] = side
                rows.append(row)
    return pd.DataFrame(rows)


def build_selection_report(slice_eval: pd.DataFrame, min_removed_n: int = 20) -> pd.DataFrame:
    if slice_eval.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["target", "candidate_model", "filter"]
    for keys, group in slice_eval.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        ict = group[group["eval_setup"].eq("pcx_ict")]
        cisd = group[group["eval_setup"].eq("pcx_ict_cisd")]
        ict_delta = _rate(ict["delta_kept_vs_base"]) if len(ict) else np.nan
        cisd_delta = _rate(cisd["delta_kept_vs_base"]) if len(cisd) else np.nan
        ict_removed = int(ict["removed_n"].sum()) if len(ict) else 0
        cisd_removed = int(cisd["removed_n"].sum()) if len(cisd) else 0
        low_removed_penalty = 0.05 * int(ict_removed < min_removed_n) + 0.05 * int(cisd_removed < min_removed_n)
        selection_score = np.nanmean([ict_delta, cisd_delta]) - low_removed_penalty
        row.update({
            "pcx_ict_delta": ict_delta,
            "pcx_ict_removed_n": ict_removed,
            "pcx_ict_cisd_delta": cisd_delta,
            "pcx_ict_cisd_removed_n": cisd_removed,
            "low_removed_penalty": low_removed_penalty,
            "selection_score": selection_score,
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False).reset_index(drop=True)


def build_yearly_by_slice(
    scores: pd.DataFrame,
    slice_membership: pd.DataFrame,
    filter_col: str = FROZEN_FILTER,
) -> pd.DataFrame:
    if scores.empty or slice_membership.empty:
        return pd.DataFrame()
    scored = scores.copy()
    membership = slice_membership.copy()
    scored["trade_date"] = _normalize_dates(scored["trade_date"]).to_numpy()
    membership["trade_date"] = _normalize_dates(membership["trade_date"]).to_numpy()
    joined = scored.merge(membership, on="trade_date", how="inner")
    if joined.empty:
        return pd.DataFrame()
    joined["year"] = pd.to_datetime(joined["trade_date"]).dt.year
    rows = []
    slice_cols = [c for c in membership.columns if c != "trade_date"]
    group_cols = [c for c in ["setup", "target", "candidate_model"] if c in joined.columns]
    for keys, score_group in joined.groupby(group_cols, dropna=False) if group_cols else [((), joined)]:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        meta = dict(zip(group_cols, key_values))
        for slice_col in slice_cols:
            slice_group = score_group[score_group[slice_col].astype(bool)]
            for year, year_group in slice_group.groupby("year"):
                row = _summary_row_for_scores(year_group, filter_col)
                row["year"] = int(year)
                row["train_setup"] = meta.get("setup", np.nan)
                row["eval_setup"] = slice_col
                if "target" in meta:
                    row["target"] = meta["target"]
                if "candidate_model" in meta:
                    row["candidate_model"] = meta["candidate_model"]
                rows.append(row)
    return pd.DataFrame(rows)


def permutation_delta_test(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = FROZEN_TARGET,
    model_name: str = FROZEN_MODEL,
    init_window: int = 200,
    n_splits: int = 3,
    runs: int = 100,
    random_state: int = 42,
    setup: str = FROZEN_SETUP,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for run in range(runs):
        scores = _blocked_setup_failure_scores_with_permuted_train(
            frame,
            feature_cols,
            target_col=target_col,
            model_name=model_name,
            init_window=init_window,
            n_splits=n_splits,
            rng=rng,
            threshold_fractions=[0.20],
        )
        summary = summarize_setup_filter(scores, setup=setup, target=target_col, model_name=model_name)
        row = summary[summary["filter"].eq(FROZEN_FILTER)].iloc[0].to_dict() if not summary.empty else {}
        row["run"] = run
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_permutation(real_delta: float, permutation: pd.DataFrame) -> pd.DataFrame:
    p = float((permutation["delta_kept_vs_base"] >= real_delta).mean()) if len(permutation) else np.nan
    return pd.DataFrame([{"real_delta": real_delta, "permutation_runs": len(permutation), "p_perm": p}])


def run_setup_failure_research(
    feature_path: Path = DEFAULT_FEATURE_PATH,
    signal_path: Path = DEFAULT_SIGNAL_PATH,
    markovian_root: Path = DEFAULT_MARKOVIAN_ROOT,
    setup: str = "pcx_ict",
    targets: list[str] | None = None,
    model_names: list[str] | None = None,
    init_window: int = 200,
    n_splits: int = 3,
    threshold_fractions: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = targets or ["inside_failure", "failure_any"]
    model_names = model_names or ["logistic", "hgb"]
    daily = pd.read_parquet(feature_path)
    signals = pd.read_csv(signal_path, parse_dates=["date", "signal_date"])
    frame, feature_cols = build_setup_frame(daily, signals, setup=setup, markovian_root=markovian_root)
    feature_cols = _usable_feature_columns(frame, feature_cols)

    summaries = []
    score_frames = []
    for target in targets:
        for model_name in model_names:
            try:
                scores = blocked_setup_failure_scores(
                    frame,
                    feature_cols,
                    target_col=target,
                    model_name=model_name,
                    init_window=init_window,
                    n_splits=n_splits,
                    threshold_fractions=threshold_fractions,
                )
            except RuntimeError as exc:
                if _should_skip_model_error(model_name, exc):
                    print(f"Skipping {model_name}: {exc}")
                    continue
                raise
            if scores.empty:
                continue
            scores["setup"] = setup
            scores["candidate_model"] = model_name
            score_frames.append(scores)
            summaries.append(summarize_setup_filter(scores, setup=setup, target=target, model_name=model_name))
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    all_scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    return summary, all_scores


def run_pcx_failure_mode_research(
    feature_path: Path = DEFAULT_FEATURE_PATH,
    signal_path: Path = DEFAULT_SIGNAL_PATH,
    markovian_root: Path | None = DEFAULT_MARKOVIAN_ROOT,
    train_setup: str = "pcx_wick",
    eval_setups: list[str] | None = None,
    targets: list[str] | None = None,
    model_names: list[str] | None = None,
    init_window: int = 200,
    n_splits: int = 3,
    threshold_fractions: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eval_setups = eval_setups or ["pcx_wick", "pcx_ict", "pcx_ict_cisd"]
    targets = targets or ["failure_any", "inside_failure"]
    model_names = model_names or ["logistic", "hgb"]

    daily = pd.read_parquet(feature_path)
    signals = pd.read_csv(signal_path, parse_dates=["date", "signal_date"])
    frame, _ = build_setup_frame(daily, signals, setup=train_setup, markovian_root=markovian_root)
    feature_cols = _usable_feature_columns(frame, failure_mode_feature_columns(frame))
    membership = build_slice_membership(daily, signals, setups=eval_setups, markovian_root=markovian_root)

    summaries = []
    score_frames = []
    slice_frames = []
    for target in targets:
        for model_name in model_names:
            try:
                scores = blocked_setup_failure_scores(
                    frame,
                    feature_cols,
                    target_col=target,
                    model_name=model_name,
                    init_window=init_window,
                    n_splits=n_splits,
                    threshold_fractions=threshold_fractions,
                )
            except RuntimeError as exc:
                if _should_skip_model_error(model_name, exc):
                    print(f"Skipping {model_name}: {exc}")
                    continue
                raise
            if scores.empty:
                continue
            scores["setup"] = train_setup
            scores["candidate_model"] = model_name
            score_frames.append(scores)
            summaries.append(summarize_setup_filter(scores, setup=train_setup, target=target, model_name=model_name))
            for flag in [c for c in scores.columns if c.startswith("remove_top_")]:
                slice_frames.append(build_slice_eval(scores, membership, filter_col=flag))

    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    all_scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    slice_eval = pd.concat(slice_frames, ignore_index=True) if slice_frames else pd.DataFrame()
    return summary, all_scores, slice_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Setup-conditioned failure model research")
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--signal-path", type=Path, default=DEFAULT_SIGNAL_PATH)
    parser.add_argument("--markovian-root", type=Path, default=DEFAULT_MARKOVIAN_ROOT)
    parser.add_argument("--setup", default="pcx_ict", choices=["pcx_wick", "pcx_ict", "pcx_ict_cisd"])
    parser.add_argument("--targets", default="inside_failure,failure_any")
    parser.add_argument("--models", default="logistic,hgb")
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--init-window", type=int, default=200)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--scores-output", type=Path, default=SCORES_PATH)
    parser.add_argument("--frozen-candidate", action="store_true")
    parser.add_argument("--yearly", action="store_true")
    parser.add_argument("--compare-generic-inside", action="store_true")
    parser.add_argument("--permutation-runs", type=int, default=0)
    parser.add_argument("--train-setup", default=None)
    parser.add_argument("--eval-setups", default="")
    parser.add_argument("--pcx-failure-mode", action="store_true")
    parser.add_argument("--pcx-failure-summary-output", type=Path, default=PCX_FAILURE_SUMMARY_PATH)
    parser.add_argument("--pcx-failure-scores-output", type=Path, default=PCX_FAILURE_SCORES_PATH)
    parser.add_argument("--pcx-failure-slice-output", type=Path, default=PCX_FAILURE_SLICE_PATH)
    parser.add_argument("--pcx-failure-yearly-by-slice-output", type=Path, default=PCX_FAILURE_YEARLY_BY_SLICE_PATH)
    parser.add_argument("--pcx-failure-side-slice-output", type=Path, default=PCX_FAILURE_SIDE_SLICE_PATH)
    parser.add_argument("--pcx-failure-selection-output", type=Path, default=PCX_FAILURE_SELECTION_PATH)
    parser.add_argument("--yearly-output", type=Path, default=YEARLY_PATH)
    parser.add_argument("--comparison-output", type=Path, default=COMPARISON_PATH)
    parser.add_argument("--permutation-output", type=Path, default=PERMUTATION_PATH)
    parser.add_argument("--slice-output", type=Path, default=SLICE_PATH)
    return parser.parse_args()


def _frozen_score_rows(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return scores.copy()
    mask = (
        scores["setup"].eq(FROZEN_SETUP)
        & scores["target"].eq(FROZEN_TARGET)
        & scores["candidate_model"].eq(FROZEN_MODEL)
    )
    return scores[mask].copy()


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    if args.pcx_failure_mode:
        eval_setups = [x.strip() for x in args.eval_setups.split(",") if x.strip()]
        threshold_fractions = parse_thresholds(args.thresholds) or [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        summary, scores, slice_eval = run_pcx_failure_mode_research(
            feature_path=args.feature_path,
            signal_path=args.signal_path,
            markovian_root=args.markovian_root,
            train_setup=args.train_setup or "pcx_wick",
            eval_setups=eval_setups or ["pcx_wick", "pcx_ict", "pcx_ict_cisd"],
            targets=[x.strip() for x in args.targets.split(",") if x.strip()],
            model_names=[x.strip() for x in args.models.split(",") if x.strip()],
            init_window=args.init_window,
            n_splits=args.splits,
            threshold_fractions=threshold_fractions,
        )
        daily = pd.read_parquet(args.feature_path)
        signals = pd.read_csv(args.signal_path, parse_dates=["date", "signal_date"])
        membership = build_slice_membership(
            daily,
            signals,
            setups=eval_setups or ["pcx_wick", "pcx_ict", "pcx_ict_cisd"],
            markovian_root=args.markovian_root,
        )
        yearly_by_slice = build_yearly_by_slice(scores, membership, filter_col=FROZEN_FILTER)
        side_frames = [
            build_side_slice_eval(scores, membership, filter_col=flag)
            for flag in [c for c in scores.columns if c.startswith("remove_top_")]
        ]
        side_slice_eval = pd.concat(side_frames, ignore_index=True) if side_frames else pd.DataFrame()
        selection = build_selection_report(slice_eval)
        args.pcx_failure_summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.pcx_failure_scores_output.parent.mkdir(parents=True, exist_ok=True)
        args.pcx_failure_slice_output.parent.mkdir(parents=True, exist_ok=True)
        args.pcx_failure_yearly_by_slice_output.parent.mkdir(parents=True, exist_ok=True)
        args.pcx_failure_side_slice_output.parent.mkdir(parents=True, exist_ok=True)
        args.pcx_failure_selection_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.pcx_failure_summary_output, index=False)
        scores.to_parquet(args.pcx_failure_scores_output, index=False)
        slice_eval.to_csv(args.pcx_failure_slice_output, index=False)
        yearly_by_slice.to_csv(args.pcx_failure_yearly_by_slice_output, index=False)
        side_slice_eval.to_csv(args.pcx_failure_side_slice_output, index=False)
        selection.to_csv(args.pcx_failure_selection_output, index=False)
        print(summary.to_string(index=False))
        print(f"Wrote PCX failure summary -> {args.pcx_failure_summary_output}")
        print(f"Wrote PCX failure scores -> {args.pcx_failure_scores_output}")
        print(f"Wrote PCX failure slice eval -> {args.pcx_failure_slice_output}")
        print(f"Wrote PCX failure yearly by slice -> {args.pcx_failure_yearly_by_slice_output}")
        print(f"Wrote PCX failure side slice eval -> {args.pcx_failure_side_slice_output}")
        print(f"Wrote PCX failure selection -> {args.pcx_failure_selection_output}")
        return
    setup = args.train_setup or args.setup
    summary, scores = run_setup_failure_research(
        feature_path=args.feature_path,
        signal_path=args.signal_path,
        markovian_root=args.markovian_root,
        setup=setup,
        targets=[x.strip() for x in args.targets.split(",") if x.strip()],
        model_names=[x.strip() for x in args.models.split(",") if x.strip()],
        init_window=args.init_window,
        n_splits=args.splits,
        threshold_fractions=parse_thresholds(args.thresholds),
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    scores.to_parquet(args.scores_output, index=False)
    frozen_summary = select_frozen_candidate(summary)
    frozen_scores = _frozen_score_rows(scores)

    if args.frozen_candidate:
        _write_csv(frozen_summary, FROZEN_PATH)
        print("Frozen candidate:")
        print(frozen_summary.to_string(index=False))

    if args.yearly:
        yearly = build_yearly_stability(frozen_scores, filter_col=FROZEN_FILTER)
        _write_csv(yearly, args.yearly_output)
        print(f"Wrote yearly -> {args.yearly_output}")

    if args.compare_generic_inside:
        generic_scores = load_generic_inside_scores(
            feature_path=args.feature_path,
            init_window=args.init_window,
            n_splits=args.splits,
            model_name=FROZEN_MODEL,
        )
        comparison = build_same_date_comparison(frozen_scores, generic_scores, fraction=0.20)
        _write_csv(comparison, args.comparison_output)
        print(f"Wrote comparison -> {args.comparison_output}")

    if args.permutation_runs > 0:
        daily = pd.read_parquet(args.feature_path)
        signals = pd.read_csv(args.signal_path, parse_dates=["date", "signal_date"])
        frame, feature_cols = build_setup_frame(daily, signals, setup=setup, markovian_root=args.markovian_root)
        feature_cols = _usable_feature_columns(frame, feature_cols)
        permutation = permutation_delta_test(
            frame,
            feature_cols,
            target_col=FROZEN_TARGET,
            model_name=FROZEN_MODEL,
            init_window=args.init_window,
            n_splits=args.splits,
            runs=args.permutation_runs,
            setup=setup,
        )
        real_delta = (
            float(frozen_summary["delta_kept_vs_base"].iloc[0])
            if not frozen_summary.empty and "delta_kept_vs_base" in frozen_summary
            else np.nan
        )
        perm_summary = summarize_permutation(real_delta, permutation)
        permutation_out = pd.concat([permutation, perm_summary], ignore_index=True, sort=False)
        _write_csv(permutation_out, args.permutation_output)
        print(f"Wrote permutation -> {args.permutation_output}")

    eval_setups = [x.strip() for x in args.eval_setups.split(",") if x.strip()]
    if eval_setups:
        daily = pd.read_parquet(args.feature_path)
        signals = pd.read_csv(args.signal_path, parse_dates=["date", "signal_date"])
        membership = build_slice_membership(daily, signals, setups=eval_setups, markovian_root=args.markovian_root)
        slice_eval = build_slice_eval(scores, membership, filter_col=FROZEN_FILTER)
        _write_csv(slice_eval, args.slice_output)
        print(f"Wrote slice eval -> {args.slice_output}")

    print(summary.to_string(index=False))
    print(f"Wrote summary -> {args.summary_output}")
    print(f"Wrote scores -> {args.scores_output}")


if __name__ == "__main__":
    main()
