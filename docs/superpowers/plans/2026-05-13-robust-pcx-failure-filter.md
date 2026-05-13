# Robust PCX Failure Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a stronger, more robust PCX failure filter by training on PCX-alone rows, evaluating PCX+ICT/CISD slices, adding leak-safe PCX features, and selecting only robust ship candidates.

**Architecture:** Keep all work inside `research_setup_failures.py` and `tests/test_research_setup_failures.py`. Add small feature-engineering helpers, robust selection helpers, and wire selection into the existing PCX failure-mode CLI path without touching raw-data pipelines.

**Tech Stack:** Python 3, pandas/numpy, scikit-learn, optional XGBoost CUDA, pytest.

---

### Task 1: Add leak-safe PCX engineered features

**Files:**
- Modify: `tests/test_research_setup_failures.py`
- Modify: `research_setup_failures.py`

- [ ] **Step 1: Write failing test**

Append test:

```python
def test_add_pcx_failure_engineered_features_are_side_adjusted_and_numeric():
    from research_setup_failures import add_pcx_failure_engineered_features, failure_mode_feature_columns

    frame = pd.DataFrame({
        "direction": ["LONG", "SHORT"],
        "side": [1.0, -1.0],
        "signal_body_to_range": [0.4, 0.4],
        "prior_body_to_range": [0.2, 0.2],
        "signal_close_location": [0.8, 0.2],
        "prior_close_location": [0.7, 0.3],
        "signal_close_to_target_extreme": [0.1, 0.1],
        "prior_close_to_target_extreme": [0.2, 0.2],
        "signal_close_through": [0.3, -0.3],
        "prior_close_through": [0.1, -0.1],
        "signal_range_vs_prior_range": [1.5, 1.5],
        "range_percentile_22": [0.2, 0.2],
        "rv_regime_252": [0.5, 0.5],
        "hit": [True, False],
        "failure_any": [False, True],
    })

    out = add_pcx_failure_engineered_features(frame)

    assert list(out["signal_close_location_side_adj"].round(6)) == [0.8, 0.8]
    assert list(out["prior_close_location_side_adj"].round(6)) == [0.7, 0.7]
    assert list(out["signal_close_through_side_adj"].round(6)) == [0.3, 0.3]
    assert (out["pcx_quality"] > 0).all()
    assert "pcx_quality_x_compression" in failure_mode_feature_columns(out)
    assert "signal_close_location_side_adj" in failure_mode_feature_columns(out)
```

- [ ] **Step 2: Run red test**

Run:

```bash
/mnt/e/backup/code/Finance/research/markovian-ms/.venv/bin/python -m pytest tests/test_research_setup_failures.py::test_add_pcx_failure_engineered_features_are_side_adjusted_and_numeric -v
```

Expected: FAIL because `add_pcx_failure_engineered_features` is missing.

- [ ] **Step 3: Implement helper and include columns**

In `research_setup_failures.py`, add engineered feature names to `FAILURE_MODE_FEATURE_CANDIDATES`, implement `add_pcx_failure_engineered_features(frame)`, and call it in `build_setup_frame` before returning.

- [ ] **Step 4: Run green test**

Run same pytest command. Expected: PASS.

### Task 2: Add robust selection scoring

**Files:**
- Modify: `tests/test_research_setup_failures.py`
- Modify: `research_setup_failures.py`

- [ ] **Step 1: Write failing tests**

Append tests:

```python
def test_build_robust_selection_report_requires_positive_ict_and_cisd():
    from research_setup_failures import build_robust_selection_report

    slice_eval = pd.DataFrame({
        "target": ["failure_any", "failure_any"],
        "candidate_model": ["logistic", "logistic"],
        "filter": ["remove_top_20", "remove_top_20"],
        "eval_setup": ["pcx_ict", "pcx_ict_cisd"],
        "delta_kept_vs_base": [0.04, -0.01],
        "removed_n": [40, 30],
    })
    yearly = pd.DataFrame({
        "target": ["failure_any", "failure_any"],
        "candidate_model": ["logistic", "logistic"],
        "filter": ["remove_top_20", "remove_top_20"],
        "eval_setup": ["pcx_ict", "pcx_ict_cisd"],
        "delta_kept_vs_base": [0.02, 0.03],
    })

    out = build_robust_selection_report(slice_eval, yearly_by_slice=yearly, min_removed_n=20)

    assert bool(out.iloc[0]["ship_eligible"]) is False
    assert out.iloc[0]["eligibility_reason"] == "pcx_ict_cisd_delta<=0"


def test_build_robust_selection_report_penalizes_negative_yearly_stability():
    from research_setup_failures import build_robust_selection_report

    slice_eval = pd.DataFrame({
        "target": ["failure_any", "failure_any", "failure_any", "failure_any"],
        "candidate_model": ["stable", "stable", "unstable", "unstable"],
        "filter": ["remove_top_20"] * 4,
        "eval_setup": ["pcx_ict", "pcx_ict_cisd", "pcx_ict", "pcx_ict_cisd"],
        "delta_kept_vs_base": [0.04, 0.04, 0.05, 0.05],
        "removed_n": [40, 30, 40, 30],
    })
    yearly = pd.DataFrame({
        "target": ["failure_any", "failure_any", "failure_any", "failure_any"],
        "candidate_model": ["stable", "stable", "unstable", "unstable"],
        "filter": ["remove_top_20"] * 4,
        "eval_setup": ["pcx_ict", "pcx_ict_cisd", "pcx_ict", "pcx_ict_cisd"],
        "delta_kept_vs_base": [0.01, 0.02, -0.06, -0.04],
    })

    out = build_robust_selection_report(slice_eval, yearly_by_slice=yearly, min_removed_n=20)

    assert out.iloc[0]["candidate_model"] == "stable"
    unstable = out[out["candidate_model"].eq("unstable")].iloc[0]
    assert unstable["yearly_penalty"] > 0
    assert bool(unstable["ship_eligible"]) is False
```

- [ ] **Step 2: Run red tests**

Run both tests. Expected: FAIL because `build_robust_selection_report` is missing.

- [ ] **Step 3: Implement robust selection**

Add `build_robust_selection_report(slice_eval, yearly_by_slice=None, fixed_holdout=None, permutation=None, min_removed_n=20)` that returns robust metrics, penalties, `ship_eligible`, and `eligibility_reason`. Keep `build_selection_report` available as wrapper or legacy helper.

- [ ] **Step 4: Run green tests**

Run both tests. Expected: PASS.

### Task 3: Make ship config require eligibility

**Files:**
- Modify: `tests/test_research_setup_failures.py`
- Modify: `research_setup_failures.py`

- [ ] **Step 1: Write failing test**

Append:

```python
def test_select_ship_config_returns_empty_when_no_eligible_candidate():
    from research_setup_failures import select_ship_config

    selection = pd.DataFrame({
        "target": ["failure_any"],
        "candidate_model": ["logistic"],
        "filter": ["remove_top_20"],
        "selection_score": [0.20],
        "ship_eligible": [False],
        "eligibility_reason": ["pcx_ict_delta<=0"],
    })

    out = select_ship_config(selection)

    assert out.empty
```

- [ ] **Step 2: Run red test**

Expected: FAIL because current `select_ship_config` returns top row regardless.

- [ ] **Step 3: Update `select_ship_config`**

If `ship_eligible` column exists, filter to eligible rows before selecting. Return empty DataFrame when none.

- [ ] **Step 4: Run green test**

Expected: PASS.

### Task 4: Wire robust selection into PCX failure mode runner

**Files:**
- Modify: `tests/test_research_setup_failures.py`
- Modify: `research_setup_failures.py`

- [ ] **Step 1: Write failing integration test**

Append:

```python
def test_pcx_failure_mode_selection_contains_robust_columns(tmp_path):
    from research_setup_failures import run_pcx_failure_mode_research, build_slice_membership, build_yearly_by_slice, build_robust_selection_report

    daily = _daily_features(140)
    signals = _signals(130)
    feature_path = tmp_path / "features.parquet"
    signal_path = tmp_path / "signals.csv"
    daily.to_parquet(feature_path, index=False)
    signals.to_csv(signal_path, index=False)

    summary, scores, slice_eval = run_pcx_failure_mode_research(
        feature_path=feature_path,
        signal_path=signal_path,
        markovian_root=None,
        train_setup="pcx_wick",
        eval_setups=["pcx_wick", "pcx_ict", "pcx_ict_cisd"],
        targets=["failure_any"],
        model_names=["logistic", "ensemble_rank_mean"],
        init_window=40,
        n_splits=3,
        threshold_fractions=[0.20, 0.30],
    )
    membership = build_slice_membership(daily, signals, setups=["pcx_wick", "pcx_ict", "pcx_ict_cisd"], markovian_root=None)
    yearly = build_yearly_by_slice(scores, membership, filter_col="remove_top_20")
    selection = build_robust_selection_report(slice_eval, yearly_by_slice=yearly, min_removed_n=1)

    assert not selection.empty
    assert {"ship_eligible", "eligibility_reason", "yearly_penalty", "pcx_ict_delta", "pcx_ict_cisd_delta"}.issubset(selection.columns)
    assert {"logistic", "ensemble_rank_mean"}.issubset(set(selection["candidate_model"]))
```

- [ ] **Step 2: Run red/green as appropriate**

If previous tasks already make it pass, note that this is integration coverage. If it fails, fix runner/helper metadata propagation.

- [ ] **Step 3: Update CLI defaults**

Change `--models` default for PCX failure mode runtime path to include `ensemble_rank_mean` when `--pcx-failure-mode` is used and no explicit models are supplied. Keep normal mode default unchanged. Update selection path in `main()` to call `build_robust_selection_report` after yearly/fixed metrics are available.

- [ ] **Step 4: Run targeted integration test**

Expected: PASS.

### Task 5: Verification and research run

**Files:**
- No code changes unless verification finds a bug; add failing regression test before any fix.

- [ ] **Step 1: Run full setup failure tests**

```bash
/mnt/e/backup/code/Finance/research/markovian-ms/.venv/bin/python -m pytest tests/test_research_setup_failures.py -v
```

Expected: PASS.

- [ ] **Step 2: Run safe research command**

```bash
/mnt/e/backup/code/Finance/research/markovian-ms/.venv/bin/python research_setup_failures.py --pcx-failure-mode --train-setup pcx_wick --eval-setups pcx_wick,pcx_ict,pcx_ict_cisd --targets failure_any,inside_failure --models logistic,hgb,ensemble_rank_mean --thresholds 0.10,0.15,0.20,0.25,0.30 --splits 3
```

Expected: writes PCX failure outputs under `output/`, no raw 1-minute data reads.

- [ ] **Step 3: Inspect ship config**

```bash
cat output/pcx_failure_mode_ship_config.csv
```

Expected: either one eligible candidate or header-only/empty if no robust candidate.

- [ ] **Step 4: Commit implementation**

```bash
git add research_setup_failures.py tests/test_research_setup_failures.py docs/superpowers/plans/2026-05-13-robust-pcx-failure-filter.md
git commit -m "improve robust pcx failure filter selection"
```
