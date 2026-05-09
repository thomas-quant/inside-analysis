import pandas as pd


def test_run_session_evaluation_writes_metrics_with_sessions(monkeypatch):
    import evaluate

    def preds(symbol, session, model):
        y_hat = [0.0, 0.05, -0.05, 0.1, -0.1, 0.0] if model == "har" else [0.01, 0.08, -0.08, 0.18, -0.16, 0.01]
        return pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=6),
            "y_true": [0.0, 0.1, -0.1, 0.2, -0.2, 0.0],
            "y_hat": y_hat,
            "pi_lower_90": [-1.0] * 6,
            "pi_upper_90": [1.0] * 6,
            "p_inside": [0.1] * 6,
            "p_outside": [0.2] * 6,
            "true_inside": [False, True, False, False, True, False],
            "true_outside": [False, False, True, False, False, True],
        })

    def fake_read_parquet(path):
        if path.startswith("output/predictions_"):
            stem = path.replace("output/predictions_", "").replace(".parquet", "")
            symbol, session, model = stem.split("_")
            return preds(symbol, session, model)
        return pd.DataFrame({"feature": [1, 2, 3]})

    metrics_written = {}
    importances_written = []

    def fake_metrics_to_csv(self, path, index=False):
        metrics_written[path] = self.copy()

    def fake_importance_to_csv(self, path, index=False):
        importances_written.append(path)

    def fake_feature_importance(df, symbol):
        return pd.DataFrame({"feature": ["x"], "coef": [1.0], "abs_coef": [1.0], "symbol": [symbol]})

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(evaluate, "plot_actual_vs_predicted", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate, "plot_probability_calibration", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate, "plot_feature_importance", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate, "compute_feature_importance", fake_feature_importance)
    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_metrics_to_csv)

    evaluate.run_session_evaluation()

    metrics = metrics_written["output/metrics_summary.csv"]
    assert set(metrics["session"]) == {"ETH", "RTH"}
    assert set(metrics["symbol"]) == {"ES", "NQ"}
    assert set(metrics["model"]) == {"HAR_OLS", "Full_Ridge"}
    assert len(metrics) == 8
