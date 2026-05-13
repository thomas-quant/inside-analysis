# PCX Failure-Mode Improvements Design

## Goal

Improve PCX failure-mode filtering by adding ensemble models, broader threshold grids, side-specific slice evaluation, conservative GPU XGBoost variants, and a selection report ranked on transfer quality rather than broad-universe score only.

## Approach

Keep the existing safe pipeline: train on `pcx_wick`, score with blocked time-series CV, and evaluate transfer into `pcx_ict` and `pcx_ict_cisd`. Do not add raw data reads. Add model and reporting layers around existing scored outputs.

## Changes

1. Add GPU XGBoost conservative variants:
   - `xgb_gpu_depth1`
   - `xgb_gpu_depth2_l1`
   - `xgb_gpu_depth2_subsample`

2. Add ensemble models inside blocked CV:
   - `ensemble_mean`: mean of logistic, HGB, XGB depth2 scores.
   - `ensemble_rank_mean`: mean of within-block train/test percentile ranks for logistic, HGB, XGB depth2.
   These derive thresholds from train ensemble scores only.

3. Add threshold grid support:
   - default PCX failure-mode thresholds: 5%, 10%, 15%, 20%, 25%, 30%.
   - CLI flag `--thresholds 0.05,0.10,...`.

4. Add side-specific slice evaluation:
   - evaluate `LONG` and `SHORT` sub-slices for each setup.
   - output `output/pcx_failure_mode_side_slice_eval.csv`.

5. Add selection report:
   - rank rows by `pcx_ict` and `pcx_ict_cisd` transfer deltas.
   - penalize low removed counts.
   - output `output/pcx_failure_mode_selection.csv`.

## Success Criteria

- Existing tests pass.
- GPU smoke still runs.
- New models appear in summary/slice outputs.
- Threshold outputs include `remove_top_15`, `remove_top_25`, `remove_top_30`.
- Side slice output includes `LONG` and `SHORT`.
- Selection report ranks model/target/filter combinations using narrow-slice transfer.
