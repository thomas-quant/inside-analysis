"""
evaluate.py
Out-of-sample evaluation of walk-forward predictions.
Outputs:
  output/metrics_summary.csv           — OOS R², RMSE, PI coverage, Brier scores
  output/feature_importance_{es,nq}.csv — Ridge coefficient magnitudes (standardised)
  output/plots/                          — calibration + actual vs predicted charts
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from feature_engineering import (
    FEATURE_COLS_HAR, FEATURE_COLS_ALL, TARGET_COL, LABEL_COLS
)

Path("output/plots").mkdir(parents=True, exist_ok=True)


# ── Metric functions ──────────────────────────────────────────────────────────

def oos_r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def pi_coverage(y_true, pi_lo, pi_hi):
    return ((y_true >= pi_lo) & (y_true <= pi_hi)).mean()


def brier_score(p_pred, y_true_binary):
    """Lower is better. 0 = perfect, 0.25 = no-skill."""
    return ((p_pred - y_true_binary.astype(float)) ** 2).mean()


def diebold_mariano(e1, e2):
    """
    Diebold-Mariano test: H0 = equal predictive accuracy.
    e1, e2 = squared error vectors for model 1 and model 2.
    Returns t-statistic and p-value. Negative t = model 1 is better.
    """
    d = e1 - e2
    n = len(d)
    dm_stat = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    p_val = 2 * stats.norm.sf(abs(dm_stat))
    return dm_stat, p_val


# ── Feature importance ────────────────────────────────────────────────────────

def compute_feature_importance(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Fit Ridge on full dataset (in-sample), return standardised coefficient magnitudes.
    Not used for prediction — only for feature ranking.
    """
    cols_needed = FEATURE_COLS_ALL + [TARGET_COL]
    sub = df[cols_needed].dropna()
    X = sub[FEATURE_COLS_ALL].values
    y = sub[TARGET_COL].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    reg = Ridge(alpha=1.0).fit(Xs, y)

    importance = pd.DataFrame({
        "feature": FEATURE_COLS_ALL,
        "coef": reg.coef_,
        "abs_coef": np.abs(reg.coef_),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    importance["symbol"] = symbol
    return importance


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(preds: pd.DataFrame, symbol: str, model: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(preds["trade_date"], preds["pi_lower_90"], preds["pi_upper_90"],
                    alpha=0.3, label="90% PI")
    ax.plot(preds["trade_date"], preds["y_hat"], lw=1, label="y_hat")
    ax.scatter(preds["trade_date"], preds["y_true"], s=4, c="black",
               alpha=0.5, label="y_true")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_title(f"{symbol} {model} — Actual vs Predicted log range ratio")
    ax.set_ylabel("log(range_t+1 / range_t)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"output/plots/{symbol}_{model}_actual_vs_pred.png", dpi=120)
    plt.close()


def plot_probability_calibration(preds: pd.DataFrame, symbol: str, model: str):
    """
    Reliability diagram: bins p_inside and p_outside predictions,
    plots mean predicted probability vs actual fraction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (p_col, true_col, label) in zip(axes, [
        ("p_inside",  "true_inside",  "Inside"),
        ("p_outside", "true_outside", "Outside"),
    ]):
        bins = np.linspace(0, 1, 11)
        bin_idx = np.digitize(preds[p_col], bins) - 1
        bin_idx = np.clip(bin_idx, 0, 9)

        mean_pred, mean_true = [], []
        for b in range(10):
            mask = bin_idx == b
            if mask.sum() > 5:
                mean_pred.append(preds.loc[mask, p_col].mean())
                mean_true.append(preds.loc[mask, true_col].astype(float).mean())

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
        ax.scatter(mean_pred, mean_true, s=40, zorder=5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(f"Mean predicted {p_col}")
        ax.set_ylabel(f"Actual fraction {label}")
        ax.set_title(f"{symbol} {model} — {label} day calibration")

    plt.tight_layout()
    plt.savefig(f"output/plots/{symbol}_{model}_calibration.png", dpi=120)
    plt.close()


def plot_feature_importance(importance: pd.DataFrame, symbol: str):
    top20 = importance.head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#d62728" if c < 0 else "#1f77b4" for c in top20["coef"]]
    ax.barh(top20["feature"][::-1], top20["abs_coef"][::-1], color=colors[::-1])
    ax.set_xlabel("Standardised |coefficient| (Ridge)")
    ax.set_title(f"{symbol} — Feature importance (top 20)")
    plt.tight_layout()
    plt.savefig(f"output/plots/{symbol}_feature_importance.png", dpi=120)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_metrics = []

    for symbol in ["es", "nq"]:
        print(f"\n{'='*55}")
        print(f"  {symbol.upper()}")

        har   = pd.read_parquet(f"output/predictions_{symbol}_har.parquet")
        ridge = pd.read_parquet(f"output/predictions_{symbol}_ridge.parquet")
        har["trade_date"]   = pd.to_datetime(har["trade_date"])
        ridge["trade_date"] = pd.to_datetime(ridge["trade_date"])

        # Load feature file for importance computation (ETH parquet)
        features = pd.read_parquet(f"output/features_{symbol}_eth.parquet")

        for preds, model_name in [(har, "HAR_OLS"), (ridge, "Full_Ridge")]:
            r2    = oos_r2(preds["y_true"], preds["y_hat"])
            rmse_ = rmse(preds["y_true"],  preds["y_hat"])
            cov   = pi_coverage(preds["y_true"], preds["pi_lower_90"], preds["pi_upper_90"])
            bs_in  = brier_score(preds["p_inside"],  preds["true_inside"])
            bs_out = brier_score(preds["p_outside"], preds["true_outside"])

            print(f"\n  {model_name}")
            print(f"    OOS R²          : {r2:.4f}")
            print(f"    RMSE            : {rmse_:.4f}")
            print(f"    90% PI coverage : {cov:.3f}  (target: 0.900)")
            print(f"    Brier(inside)   : {bs_in:.4f}  (naive: {preds['true_inside'].mean()*(1-preds['true_inside'].mean()):.4f})")
            print(f"    Brier(outside)  : {bs_out:.4f}  (naive: {preds['true_outside'].mean()*(1-preds['true_outside'].mean()):.4f})")

            all_metrics.append({
                "symbol": symbol.upper(), "model": model_name,
                "oos_r2": r2, "rmse": rmse_, "pi_coverage_90": cov,
                "brier_inside": bs_in, "brier_outside": bs_out,
                "n_test_days": len(preds),
            })

            plot_actual_vs_predicted(preds, symbol.upper(), model_name)
            plot_probability_calibration(preds, symbol.upper(), model_name)

        # Diebold-Mariano test: does Ridge beat HAR?
        merged = har.merge(ridge[["trade_date", "y_hat"]], on="trade_date",
                           suffixes=("_har", "_ridge"))
        e_har   = (merged["y_true"] - merged["y_hat_har"])   ** 2
        e_ridge = (merged["y_true"] - merged["y_hat_ridge"]) ** 2
        dm_stat, dm_pval = diebold_mariano(e_har.values, e_ridge.values)
        print(f"\n  Diebold-Mariano (HAR vs Ridge): stat={dm_stat:.3f}, p={dm_pval:.4f}")
        print(f"  {'Ridge significantly better' if dm_pval < 0.05 and dm_stat > 0 else 'No significant difference'}")

        # Feature importance
        imp = compute_feature_importance(features, symbol.upper())
        imp.to_csv(f"output/feature_importance_{symbol}.csv", index=False)
        plot_feature_importance(imp, symbol.upper())

        print(f"\n  Top 10 features ({symbol.upper()}):")
        print(imp[["feature", "coef", "abs_coef"]].head(10).to_string(index=False))

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("output/metrics_summary.csv", index=False)
    print(f"\n\nSaved output/metrics_summary.csv")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
