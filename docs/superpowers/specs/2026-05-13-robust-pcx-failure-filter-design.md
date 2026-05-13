# Robust PCX Failure Filter Design

## Goal
Improve the PCX failure filter so it is stronger and more robust for PCX+ICT and PCX-alone usage. The filter should identify trades to skip before the target session using only signal-date daily features and PCX signal metadata.

## Scope
- Use safe inputs only: `output/features_nq_eth.parquet` and `markovian-ms` `signal_features.csv`.
- Do not read raw 1-minute data or regenerate feature pipelines.
- Primary train universe: `pcx_wick` (PCX alone) for larger sample size.
- Primary deployment eval slices: `pcx_ict` and `pcx_ict_cisd`.
- Primary target: `failure_any`; secondary target: `inside_failure`.

## Approach
1. Train blocked time-series failure models on the PCX-alone universe.
2. Score future blocks only; train thresholds are learned from train scores only.
3. Evaluate each candidate on PCX-alone, PCX+ICT, and PCX+ICT+CISD slices.
4. Select only configurations that improve kept hit rate on the important slices and remove enough trades to be useful.
5. Add robustness checks: yearly slice stability, fixed holdout, and permutation comparison.

## Feature Additions
Keep features leak-safe by deriving them from signal-day/prior-day PCX rows or signal-date daily features only.

Add compact engineered PCX features:
- side-adjusted signal close location
- side-adjusted prior close location
- target-side wick pressure
- adverse-side wick pressure
- close-through alignment with trade side
- PCX quality composite
- compression × PCX quality interactions
- volatility regime × PCX quality interactions

These features should be numeric and included through `failure_mode_feature_columns` only when present.

## Candidate Models
Default candidates:
- `logistic`
- `hgb`
- `ensemble_rank_mean`

Optional GPU candidates remain available:
- `xgb_gpu_depth1`
- `xgb_gpu_depth2`
- `xgb_gpu_depth2_l1`
- `xgb_gpu_depth2_subsample`

Unavailable CUDA/XGBoost candidates should be skipped without failing the full run.

## Thresholds
Evaluate removal thresholds:
- 10%, 15%, 20%, 25%, 30%

Selection should prefer enough removal frequency over tiny high-noise removals.

## Robust Selection
Create a robust selection report with one row per `(target, candidate_model, filter)`.

Metrics:
- PCX+ICT delta kept hit rate vs base
- PCX+ICT+CISD delta kept hit rate vs base
- removed count on both slices
- yearly minimum delta for both slices
- fixed holdout delta for both slices
- permutation p-value when requested

Selection score:
- reward mean delta across PCX+ICT and PCX+ICT+CISD
- penalize low removed count
- penalize negative yearly minimum delta
- penalize fixed holdout fail
- penalize weak permutation evidence when available

A candidate is ship-eligible only if:
- PCX+ICT delta > 0
- PCX+ICT+CISD delta > 0
- removed count meets minimum on PCX+ICT
- fixed holdout passes when run

## Outputs
Keep existing outputs and add/strengthen:
- `output/pcx_failure_mode_selection.csv`: robust ranking and eligibility columns
- `output/pcx_failure_mode_ship_config.csv`: single best eligible config, empty if none
- `output/pcx_failure_mode_skip_list.csv`: skip rows from selected config
- `output/pcx_failure_mode_fixed_holdout.csv`: fixed holdout slice metrics
- `output/pcx_failure_mode_yearly_by_slice.csv`: yearly robustness by slice

## Testing
Add regression tests for:
- engineered PCX features are side-adjusted and leak-safe
- robust selection penalizes negative yearly slices
- robust selection rejects candidates with no PCX+ICT/CISD improvement
- ship config is empty when no candidate is eligible
- runner emits selection/ship outputs with ensemble candidate

Run:
```bash
/mnt/e/backup/code/Finance/research/markovian-ms/.venv/bin/python -m pytest tests/test_research_setup_failures.py -v
```

## Acceptance
The implementation is acceptable when tests pass and the generated selection report clearly says either:
- a robust ship candidate exists, with positive PCX+ICT and PCX+ICT+CISD deltas; or
- no robust candidate exists, with reasons visible in eligibility columns.
