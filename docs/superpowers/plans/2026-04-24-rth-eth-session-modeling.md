# RTH + ETH Session Modeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class RTH inside/outside modeling beside existing ETH/Globex modeling, then compare predictability across sessions.

**Architecture:** Keep this repo. Generalize feature/target/model/eval paths by `session` = `eth` or `rth`. ETH remains current behavior; RTH gets its own daily bars, labels, target, predictions, metrics, plots. No volatility-regime dependency yet; later regime outputs can join on `trade_date`.

**Tech Stack:** Python 3, pandas, numpy, scikit-learn, pytest, parquet outputs.

---

## File Map

- Modify `feature_engineering.py`
  - Build feature sets for both ETH and RTH targets.
  - Save `output/features_{symbol}_{session}.parquet`.
  - Preserve old ETH column names/output semantics.
- Modify `model.py`
  - Loop over `symbol × session`.
  - Save `output/predictions_{symbol}_{session}_{model}.parquet`.
  - Add class-weighted logistic option only after tests.
- Modify `evaluate.py`
  - Loop over `symbol × session`.
  - Write session column to metrics.
  - Save session-specific plots/importances.
- Modify `tests/test_features.py`
  - Add RTH label/target tests.
- Modify `tests/test_model.py`
  - Add prediction-path/session loop tests.
- Optional modify `README.md`
  - Document new outputs + commands.

---

## Task 1: Add session-aware target builder

**Files:**
- Modify: `feature_engineering.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_features.py`:

```python
def test_add_target_uses_session_daily_high_low():
    import pandas as pd
    from feature_engineering import add_target

    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "Open": [100.0, 101.0, 102.0],
        "High": [110.0, 108.0, 112.0],
        "Low": [90.0, 92.0, 88.0],
        "Close": [105.0, 103.0, 100.0],
        "Volume": [1, 1, 1],
        "range_abs": [20.0, 16.0, 24.0],
    })

    out = add_target(df)

    assert bool(out.loc[1, "inside"]) is True
    assert bool(out.loc[1, "outside"]) is False
    assert bool(out.loc[2, "inside"]) is False
    assert bool(out.loc[2, "outside"]) is True
    assert out.loc[0, "y"] == pytest.approx(np.log(16.0 / 20.0))
```

- [ ] **Step 2: Run red**

Run:

```bash
python3 -m pytest tests/test_features.py::test_add_target_uses_session_daily_high_low -v
```

Expected: PASS if existing `add_target` already generic. If PASS, test guards behavior; no production change needed for this task.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features.py
git commit -m "test session target labels"
```

---

## Task 2: Generate RTH feature parquets

**Files:**
- Modify: `feature_engineering.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing test for RTH feature artifact builder**

Add to `tests/test_features.py`:

```python
def test_build_features_for_returns_eth_and_rth_feature_frames():
    import pandas as pd
    import feature_engineering as fe

    raw = pd.read_parquet("data/es_1m.parquet")
    vix = pd.read_parquet("data/vix_cboe.parquet")
    vix["date"] = pd.to_datetime(vix["date"])
    eco = pd.read_parquet("data/economic_events.parquet")

    eth, rth = fe._build_features_for("ES", raw, vix, eco)

    assert "range_abs" in eth.columns
    assert "range_abs" in rth.columns
    assert len(eth) > 1000
    assert len(rth) > 1000
    assert rth["trade_date"].is_monotonic_increasing
```

- [ ] **Step 2: Run red**

```bash
python3 -m pytest tests/test_features.py::test_build_features_for_returns_eth_and_rth_feature_frames -v
```

Expected: FAIL because returned `rth` lacks full feature columns/`range_abs`.

- [ ] **Step 3: Implement minimal session feature builder**

In `feature_engineering.py`, add helper near `_build_features_for`:

```python
def _add_common_features(
    daily: pd.DataFrame,
    raw: pd.DataFrame,
    vix: pd.DataFrame,
    eco: pd.DataFrame,
    rv_session: str,
) -> pd.DataFrame:
    """Add feature groups that can be computed for either ETH or RTH target bars."""
    out = compute_rv_features(daily, raw, session=rv_session)
    out = compute_range_features(out)
    out = compute_volume_features(out, raw)
    out = compute_session_features(out, raw)
    out = compute_calendar_features(out, eco)
    out = compute_vix_features(out, vix)
    return out
```

Modify `_build_features_for`:

```python
def _build_features_for(symbol: str, raw: pd.DataFrame,
                         vix: pd.DataFrame, eco: pd.DataFrame) -> tuple:
    """Returns (eth_daily_with_features, rth_daily_with_features) tuple."""
    eth_daily = build_eth_daily(raw)
    rth_daily = build_rth_daily(raw)

    eth = _add_common_features(eth_daily, raw, vix, eco, rv_session="rth")
    eth = compute_eth_rth_cross_features(eth, rth_daily)

    rth = _add_common_features(rth_daily, raw, vix, eco, rv_session="rth")
    rth = compute_eth_rth_cross_features(eth_daily, rth)

    return eth, rth
```

- [ ] **Step 4: Run green**

```bash
python3 -m pytest tests/test_features.py::test_build_features_for_returns_eth_and_rth_feature_frames -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add feature_engineering.py tests/test_features.py
git commit -m "build rth feature frame"
```

---

## Task 3: Save ETH and RTH features with targets/patterns/cross-instrument columns

**Files:**
- Modify: `feature_engineering.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing test for output paths**

Add to `tests/test_features.py`:

```python
def test_feature_engineering_main_writes_eth_and_rth_outputs(tmp_path):
    from pathlib import Path
    import subprocess

    subprocess.run(["python3", "feature_engineering.py"], check=True)

    for symbol in ["es", "nq"]:
        for session in ["eth", "rth"]:
            path = Path(f"output/features_{symbol}_{session}.parquet")
            assert path.exists(), f"missing {path}"
```

- [ ] **Step 2: Run red**

```bash
python3 -m pytest tests/test_features.py::test_feature_engineering_main_writes_eth_and_rth_outputs -v
```

Expected: FAIL: missing RTH outputs.

- [ ] **Step 3: Update `main()` save flow**

Replace main middle section after `_build_features_for` calls:

```python
print("Adding cross-instrument features...")
es, nq = compute_cross_instrument_features(es, nq)
es_rth, nq_rth = compute_cross_instrument_features(es_rth, nq_rth)

print("Adding targets...")
for frame in [es, nq, es_rth, nq_rth]:
    frame = add_target(frame)

es = add_target(es)
nq = add_target(nq)
es_rth = add_target(es_rth)
nq_rth = add_target(nq_rth)

print("Adding pattern features...")
es = compute_pattern_features(es)
nq = compute_pattern_features(nq)
es_rth = compute_pattern_features(es_rth)
nq_rth = compute_pattern_features(nq_rth)

outputs = {
    "output/features_es_eth.parquet": es,
    "output/features_nq_eth.parquet": nq,
    "output/features_es_rth.parquet": es_rth,
    "output/features_nq_rth.parquet": nq_rth,
}
for path, frame in outputs.items():
    frame.to_parquet(path, index=False)
    print(f"Saved {path}")
```

Remove accidental unused loop if present.

- [ ] **Step 4: Run green**

```bash
python3 -m pytest tests/test_features.py::test_feature_engineering_main_writes_eth_and_rth_outputs -v
```

Expected: PASS.

- [ ] **Step 5: Run feature tests**

```bash
python3 -m pytest tests/test_features.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add feature_engineering.py tests/test_features.py output/features_es_eth.parquet output/features_nq_eth.parquet output/features_es_rth.parquet output/features_nq_rth.parquet
git commit -m "write eth and rth feature outputs"
```

---

## Task 4: Update model outputs by session

**Files:**
- Modify: `model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_model.py`:

```python
def test_model_main_writes_session_prediction_outputs():
    from pathlib import Path
    import subprocess

    subprocess.run(["python3", "model.py"], check=True)

    for symbol in ["es", "nq"]:
        for session in ["eth", "rth"]:
            for model_name in ["har", "ridge"]:
                path = Path(f"output/predictions_{symbol}_{session}_{model_name}.parquet")
                assert path.exists(), f"missing {path}"
```

- [ ] **Step 2: Run red**

```bash
python3 -m pytest tests/test_model.py::test_model_main_writes_session_prediction_outputs -v
```

Expected: FAIL: missing session-specific outputs.

- [ ] **Step 3: Modify `model.py` main loop**

Replace file path logic in `main()` with:

```python
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
        har.to_parquet(f"output/predictions_{symbol}_{session}_har.parquet", index=False)

        print(f"  {symbol.upper()} {session.upper()} — Full Ridge model")
        ridge = walk_forward(df, FEATURE_COLS_ALL, TARGET_COL,
                             init_window=WALK_FORWARD_INIT, model_type="ridge")
        ridge["model"] = "Full_Ridge"
        ridge["session"] = session.upper()
        ridge.to_parquet(f"output/predictions_{symbol}_{session}_ridge.parquet", index=False)
```

- [ ] **Step 4: Run green**

```bash
python3 -m pytest tests/test_model.py::test_model_main_writes_session_prediction_outputs -v
```

Expected: PASS.

- [ ] **Step 5: Preserve old filenames if needed**

If downstream scripts still require old paths, add compatibility writes only for ETH:

```python
if session == "eth":
    har.to_parquet(f"output/predictions_{symbol}_har.parquet", index=False)
    ridge.to_parquet(f"output/predictions_{symbol}_ridge.parquet", index=False)
```

- [ ] **Step 6: Commit**

```bash
git add model.py tests/test_model.py output/predictions_*.parquet
git commit -m "write session prediction outputs"
```

---

## Task 5: Update evaluation by session

**Files:**
- Modify: `evaluate.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_model.py`:

```python
def test_evaluate_metrics_include_session_column():
    import subprocess
    import pandas as pd

    subprocess.run(["python3", "evaluate.py"], check=True)

    metrics = pd.read_csv("output/metrics_summary.csv")
    assert "session" in metrics.columns
    assert set(metrics["session"]) == {"ETH", "RTH"}
    assert set(metrics["symbol"]) == {"ES", "NQ"}
```

- [ ] **Step 2: Run red**

```bash
python3 -m pytest tests/test_model.py::test_evaluate_metrics_include_session_column -v
```

Expected: FAIL: no `session` column or missing RTH rows.

- [ ] **Step 3: Modify `evaluate.py` loops**

Use nested loop:

```python
for symbol in ["es", "nq"]:
    for session in ["eth", "rth"]:
        print(f"\n{'='*55}")
        print(f"  {symbol.upper()} {session.upper()}")

        har = pd.read_parquet(f"output/predictions_{symbol}_{session}_har.parquet")
        ridge = pd.read_parquet(f"output/predictions_{symbol}_{session}_ridge.parquet")
        har["trade_date"] = pd.to_datetime(har["trade_date"])
        ridge["trade_date"] = pd.to_datetime(ridge["trade_date"])

        features = pd.read_parquet(f"output/features_{symbol}_{session}.parquet")
```

In metric append dict, add:

```python
"session": session.upper(),
```

Rename output artifacts:

```python
plot_actual_vs_predicted(preds, f"{symbol.upper()}_{session.upper()}", model_name)
plot_probability_calibration(preds, f"{symbol.upper()}_{session.upper()}", model_name)
imp.to_csv(f"output/feature_importance_{symbol}_{session}.csv", index=False)
plot_feature_importance(imp, f"{symbol.upper()}_{session.upper()}")
```

- [ ] **Step 4: Run green**

```bash
python3 -m pytest tests/test_model.py::test_evaluate_metrics_include_session_column -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluate.py tests/test_model.py output/metrics_summary.csv output/feature_importance_* output/plots
git commit -m "evaluate eth and rth sessions"
```

---

## Task 6: Add non-regime contraction/expansion benchmark features

**Files:**
- Modify: `feature_engineering.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_features.py`:

```python
def test_pattern_features_include_nr_wr_and_streaks():
    import pandas as pd
    from feature_engineering import compute_range_features, add_target, compute_pattern_features

    df = pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=8, freq="D"),
        "Open": [10, 10, 10, 10, 10, 10, 10, 10],
        "High": [20, 19, 18, 17, 16, 30, 29, 28],
        "Low": [10, 11, 12, 13, 14, 5, 6, 7],
        "Close": [15, 15, 15, 15, 15, 20, 20, 20],
        "Volume": [1]*8,
    })
    df = compute_range_features(df)
    df = add_target(df)
    out = compute_pattern_features(df)

    for col in ["nr4_flag", "nr7_flag", "wr4_flag", "wr7_flag", "inside_streak", "outside_streak"]:
        assert col in out.columns

    assert out.loc[4, "nr4_flag"] == 1
    assert out.loc[5, "wr4_flag"] == 1
```

- [ ] **Step 2: Run red**

```bash
python3 -m pytest tests/test_features.py::test_pattern_features_include_nr_wr_and_streaks -v
```

Expected: FAIL missing columns.

- [ ] **Step 3: Implement minimal features**

Add inside `compute_pattern_features` after `range_percentile_22`:

```python
for n in [4, 7]:
    rolling_min = r.rolling(n, min_periods=n).min()
    rolling_max = r.rolling(n, min_periods=n).max()
    out[f"nr{n}_flag"] = (r == rolling_min).astype(int)
    out[f"wr{n}_flag"] = (r == rolling_max).astype(int)

def _streak(values: pd.Series) -> pd.Series:
    counts = []
    current = 0
    for value in values.fillna(False).astype(bool):
        current = current + 1 if value else 0
        counts.append(current)
    return pd.Series(counts, index=values.index, dtype=float)

out["inside_streak"] = _streak(out["inside"])
out["outside_streak"] = _streak(out["outside"])
```

Add new columns to `FEATURE_COLS_ALL` after `range_percentile_22`:

```python
"nr4_flag", "nr7_flag", "wr4_flag", "wr7_flag", "inside_streak", "outside_streak",
```

- [ ] **Step 4: Run green**

```bash
python3 -m pytest tests/test_features.py::test_pattern_features_include_nr_wr_and_streaks -v
```

Expected: PASS.

- [ ] **Step 5: Rebuild pipeline**

```bash
python3 feature_engineering.py
python3 model.py
python3 evaluate.py
python3 -m pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add feature_engineering.py tests/test_features.py output
git commit -m "add contraction expansion benchmark features"
```

---

## Task 7: Add low-risk classifier/model alternatives

**Files:**
- Modify: `model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write failing test for balanced classifier argument**

Add to `tests/test_model.py`:

```python
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
```

- [ ] **Step 2: Run red**

```bash
python3 -m pytest tests/test_model.py::test_compute_probabilities_accepts_balanced_class_weight -v
```

Expected: FAIL unexpected keyword.

- [ ] **Step 3: Implement parameter**

Change signature:

```python
def compute_probabilities(
    X_test: np.ndarray,
    X_train: np.ndarray,
    labels_inside_next: np.ndarray,
    labels_outside_next: np.ndarray,
    scaler: "StandardScaler | None" = None,
    class_weight: "str | dict | None" = None,
) -> tuple:
```

Change logistic:

```python
lr = LogisticRegression(C=0.1, max_iter=300, solver="lbfgs", class_weight=class_weight)
```

In `walk_forward`, call:

```python
p_in, p_out, p_nei = compute_probabilities(
    X_test_clf, X_train_clf, is_inside_next, is_outside_next,
    class_weight="balanced",
)
```

- [ ] **Step 4: Run green**

```bash
python3 -m pytest tests/test_model.py::test_compute_probabilities_accepts_balanced_class_weight -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "use balanced inside outside classifiers"
```

---

## Task 8: Optional sklearn tree classifier experiment, no new dependency

**Files:**
- Modify: `model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_model.py`:

```python
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
```

- [ ] **Step 2: Run red**

```bash
python3 -m pytest tests/test_model.py::test_compute_probabilities_supports_hist_gradient_boosting -v
```

Expected: FAIL unexpected keyword.

- [ ] **Step 3: Implement `clf_type`**

Add param:

```python
clf_type: str = "logistic",
```

Inside `_logistic_prob`, rename to `_clf_prob` and branch:

```python
if clf_type == "logistic":
    clf = LogisticRegression(C=0.1, max_iter=300, solver="lbfgs", class_weight=class_weight)
elif clf_type == "hgb":
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, random_state=42)
else:
    raise ValueError(f"Unknown clf_type={clf_type}")
clf.fit(Xs_tr, labels.astype(int))
return float(clf.predict_proba(Xs_te)[0, 1])
```

Do not make HGB default yet. Use it as experiment path only.

- [ ] **Step 4: Run green**

```bash
python3 -m pytest tests/test_model.py::test_compute_probabilities_supports_hist_gradient_boosting -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "add hgb classifier option"
```

---

## Task 9: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update overview**

Change “Primary session: ETH” to:

```markdown
**Sessions:** ETH/Globex and RTH are modeled as separate targets. ETH remains the original baseline; RTH has its own inside/outside labels and range-ratio target.
```

- [ ] **Step 2: Update outputs**

Add:

```markdown
output/features_{es,nq}_{eth,rth}.parquet
output/predictions_{es,nq}_{eth,rth}_{har,ridge}.parquet
```

- [ ] **Step 3: Run docs-safe tests**

```bash
python3 -m pytest tests/ -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "document eth rth modeling outputs"
```

---

## Self-review

- Spec coverage: ETH+RTH targets, model outputs, evaluation outputs, crude contraction features, no vol regime dependency, alternative model path covered.
- Placeholder scan: no TBD/TODO/later placeholders.
- Type consistency: `session` strings lower in paths, upper in metric column; existing `FEATURE_COLS_ALL` reused.
- Main risk: `compute_eth_rth_cross_features(eth_daily, rth)` for RTH features may include same-day RTH flags as features. After implementation, inspect no-lookahead. If RTH target uses `rth_inside_flag` same-day, remove those columns from RTH classifier feature list or convert to lagged forms before modeling.
