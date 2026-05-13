# PCX Failure-Mode Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve PCX failure-mode filtering with ensembles, threshold grid, side slices, conservative GPU XGBoost variants, and selection report.

**Architecture:** Extend `research_setup_failures.py` model scoring and reporting helpers. Preserve blocked CV and safe inputs. Add tests in `tests/test_research_setup_failures.py`.

**Tech Stack:** Python, pandas, numpy, scikit-learn, xgboost, pytest.

---

### Task 1: Add conservative XGB variants
- [ ] Test `xgb_gpu_depth1`, `xgb_gpu_depth2_l1`, `xgb_gpu_depth2_subsample` params.
- [ ] Implement variants in `xgb_gpu_params()`.
- [ ] Run targeted test.

### Task 2: Add ensemble models
- [ ] Test `_fit_scores("ensemble_mean", ...)` returns valid scores.
- [ ] Test `_fit_scores("ensemble_rank_mean", ...)` returns valid scores.
- [ ] Implement ensemble fitting using logistic, HGB, and `xgb_gpu_depth2`; skip unavailable XGB inside ensemble by averaging available models.
- [ ] Run targeted tests.

### Task 3: Add threshold grid support
- [ ] Test `parse_thresholds("0.05,0.15,0.30")`.
- [ ] Add `threshold_fractions` arg to `run_setup_failure_research()` and `run_pcx_failure_mode_research()`.
- [ ] Add CLI `--thresholds`; default PCX failure-mode grid `0.05,0.10,0.15,0.20,0.25,0.30`.
- [ ] Run parser/runner tests.

### Task 4: Add side-specific slice evaluation
- [ ] Test `build_side_slice_eval()` returns LONG/SHORT rows.
- [ ] Implement using existing scored rows and slice membership.
- [ ] Add CLI output `output/pcx_failure_mode_side_slice_eval.csv`.

### Task 5: Add selection report
- [ ] Test `build_selection_report()` ranks high-transfer rows above weak rows and penalizes low removed count.
- [ ] Implement selection report from slice eval.
- [ ] Add CLI output `output/pcx_failure_mode_selection.csv`.

### Task 6: Verify
- [ ] Run `python -m pytest tests/test_research_setup_failures.py tests/test_research_inside.py -v`.
- [ ] Run PCX failure-mode smoke with GPU and ensembles.
- [ ] Report best rows.
