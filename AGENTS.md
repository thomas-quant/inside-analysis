# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python research pipeline for ES/NQ inside-outside day prediction.

- `feature_engineering.py`: builds ETH/RTH daily data and 40+ model features from input parquets.
- `model.py`: runs walk-forward HAR/OLS and Ridge models plus logistic classification heads.
- `evaluate.py`: computes out-of-sample metrics and writes plots/reports.
- `baseline_stats.py`: standalone inside/outside base-rate analysis.
- `inside.py`: older exploratory CSV script; not part of the main pipeline.
- `tests/`: pytest regression and model-structure tests.
- `data/`: required local input parquets; treat as non-source data.
- `output/`: generated features, predictions, metrics, and plots.
- `docs/plans/`: design and implementation notes.

## Build, Test, and Development Commands

Run from the repository root.

```bash
python3 feature_engineering.py   # generate output/features_{es,nq}_eth.parquet
python3 model.py                 # generate output/predictions_{es,nq}_{har,ridge}.parquet
python3 evaluate.py              # generate metrics_summary.csv and output/plots/
python3 baseline_stats.py        # print base-rate summaries
python3 -m pytest tests/ -v      # run full test suite
```

Use targeted tests while editing:

```bash
python3 -m pytest tests/test_features.py::test_rv_1d_known_date -v
```

## Coding Style & Naming Conventions

Use Python 3, four-space indentation, and clear `snake_case` names for functions, variables, and columns. Keep pipeline functions small and explicit: `compute_*_features`, `build_*_daily`, and `test_*` naming is already established. Prefer pandas/numpy vectorized operations where practical. Preserve existing column names because tests and downstream parquet consumers rely on them.

## Testing Guidelines

Tests use `pytest`. `tests/test_features.py` depends on `data/*.parquet`; `tests/test_model.py` depends on generated `output/*.parquet`. Regenerate the pipeline before model tests when feature logic changes. Add regression tests for known dates or invariant bounds when adding features, e.g. nonnegative volatility, probabilities in `[0, 1]`, or no-lookahead checks.

## Commit & Pull Request Guidelines

Git history shows short descriptive commit messages without prefixes, e.g. `created model for prediction of inside days`. Keep commits focused and use concise lowercase summaries. Pull requests should include: purpose, changed pipeline stage, commands run, affected outputs, and any metric changes. Include plots or metric-table snippets when evaluation output changes.

## Security & Configuration Tips

Do not commit raw market data, large generated outputs, credentials, or environment-specific paths. Keep `data/` local. Document any new data files, required columns, and date/session conventions in `README.md` or `docs/plans/`.
