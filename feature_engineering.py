"""
feature_engineering.py
Compute all predictive features for inside/outside day modelling.
Primary session: ETH (Globex). RTH used as feature dimension.
Input:  data/*.parquet
Output: output/features_{es,nq}_eth.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
RTH_START    = pd.Timestamp("09:30").time()
RTH_END      = pd.Timestamp("16:14").time()
MIN_RTH_BARS = 30
MIN_ETH_BARS = 100   # full ETH day ≈ 1,380 bars

_FOMC_TITLES = {"US FOMC Statement", "US Federal Funds Rate",
                "US FOMC Economic Projections"}
_NFP_TITLE   = "US Non-Farm Employment Change"


def _globex_trade_date(dt_series: pd.Series) -> pd.Series:
    """Bars at 18:00–23:59 ET belong to the next calendar day (CME convention)."""
    dates = dt_series.dt.normalize()
    return dates + pd.to_timedelta((dt_series.dt.hour >= 18).astype(int), unit="D")


def build_eth_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-min bars to ETH (Globex) daily OHLCV.
    Excludes 17:00–17:59 ET maintenance break.
    Trade date assigned by globex convention (18:00+ rolls to next calendar day).
    """
    t = df["DateTime_ET"].dt.time
    in_break = (t >= pd.Timestamp("17:00").time()) & (t < pd.Timestamp("18:00").time())
    eth = df[~in_break].copy()
    eth["trade_date"] = _globex_trade_date(eth["DateTime_ET"])

    bar_counts = eth.groupby("trade_date").size()
    valid = bar_counts[bar_counts >= MIN_ETH_BARS].index

    daily = (
        eth.groupby("trade_date")
        .agg(Open=("Open", "first"), High=("High", "max"),
             Low=("Low", "min"), Close=("Close", "last"),
             Volume=("Volume", "sum"))
        .reset_index()
    )
    return daily[daily["trade_date"].isin(valid)].sort_values("trade_date").reset_index(drop=True)


def build_rth_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-min bars to RTH (09:30–16:14 ET) daily OHLCV.
    Trade date = calendar date of the bar.
    """
    t = df["DateTime_ET"].dt.time
    rth = df[(t >= RTH_START) & (t <= RTH_END)].copy()
    rth["trade_date"] = rth["DateTime_ET"].dt.normalize()

    bar_counts = rth.groupby("trade_date").size()
    valid = bar_counts[bar_counts >= MIN_RTH_BARS].index

    daily = (
        rth.groupby("trade_date")
        .agg(Open=("Open", "first"), High=("High", "max"),
             Low=("Low", "min"), Close=("Close", "last"),
             Volume=("Volume", "sum"))
        .reset_index()
    )
    return daily[daily["trade_date"].isin(valid)].sort_values("trade_date").reset_index(drop=True)


def compute_rv_features(daily: pd.DataFrame, raw_1min: pd.DataFrame,
                         session: str = "rth") -> pd.DataFrame:
    """
    Group 1: Realized Volatility features.
    session='rth' → compute RV from RTH bars (merged onto ETH trade dates).
    session='eth' → compute RV from full ETH bars.

    Features: rv_1d, rv_5d, rv_22d, rv_ratio_1_5, rv_percentile_252, parkinson_vol
    """
    if session == "rth":
        t = raw_1min["DateTime_ET"].dt.time
        bars = raw_1min[(t >= RTH_START) & (t <= RTH_END)].copy()
        bars["trade_date"] = bars["DateTime_ET"].dt.normalize()
    else:
        t = raw_1min["DateTime_ET"].dt.time
        in_break = (t >= pd.Timestamp("17:00").time()) & (t < pd.Timestamp("18:00").time())
        bars = raw_1min[~in_break].copy()
        bars["trade_date"] = _globex_trade_date(bars["DateTime_ET"])

    bars = bars.sort_values("DateTime_ET")
    bars["log_ret"] = np.log(bars["Close"] / bars["Close"].shift(1))
    # Zero out cross-session returns (first bar of each trade_date)
    bars.loc[bars["trade_date"] != bars["trade_date"].shift(1), "log_ret"] = np.nan

    rv_series = (
        bars.groupby("trade_date")["log_ret"]
        .apply(lambda r: (r ** 2).sum())
        .rename("rv_1d")
        .reset_index()
    )

    # For Parkinson: use session H and L
    hl = (
        bars.groupby("trade_date")
        .agg(ses_H=("High", "max"), ses_L=("Low", "min"))
        .reset_index()
    )

    out = daily.merge(rv_series, on="trade_date", how="left")
    out = out.merge(hl, on="trade_date", how="left")

    out["rv_5d"]  = out["rv_1d"].rolling(5,  min_periods=3).mean()
    out["rv_22d"] = out["rv_1d"].rolling(22, min_periods=10).mean()
    out["rv_ratio_1_5"] = out["rv_1d"] / out["rv_5d"]
    out["rv_percentile_252"] = (
        out["rv_1d"]
        .rolling(252, min_periods=30)
        .apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    )
    out["parkinson_vol"] = (np.log(out["ses_H"] / out["ses_L"]) ** 2) / (4 * np.log(2))
    return out.drop(columns=["ses_H", "ses_L"])


def compute_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 2: Range Structure features.
    overnight_gap: (Open_t - Close_{t-1}) / range_{t-1}
    """
    out = df.copy()
    out["range_abs"]       = out["High"] - out["Low"]
    prev_range             = out["range_abs"].shift(1)
    out["range_pct_of_prev"] = out["range_abs"] / prev_range

    # ATR(14): simple mean of range over 14 days
    atr14 = out["range_abs"].rolling(14, min_periods=5).mean()
    out["atr_ratio"]   = out["range_abs"] / atr14
    out["range_ma_5"]  = out["range_abs"].rolling(5,  min_periods=3).mean()
    out["range_ma_22"] = out["range_abs"].rolling(22, min_periods=10).mean()

    # Where did the close land within today's range? 0 = at low, 1 = at high
    out["close_location"] = (out["Close"] - out["Low"]) / out["range_abs"]

    # Overnight gap at open of day t (relative to t-1 range)
    out["overnight_gap"] = (out["Open"] - out["Close"].shift(1)) / prev_range

    return out


def compute_volume_features(df: pd.DataFrame, raw_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Group 3: Volume features.
    - volume_prev          : RTH volume on day t (from daily bars)
    - volume_zscore_22     : z-score of volume vs trailing 22-day window
    - volume_rth_vs_globex : RTH vol / total globex vol for same session-date
    - volume_first_hour_pct: NYAM (09:30–10:29) vol as % of total RTH vol
    """
    out = df.copy()
    out["volume_prev"] = out["Volume"]

    vol_mean = out["Volume"].rolling(22, min_periods=10).mean()
    vol_std  = out["Volume"].rolling(22, min_periods=10).std()
    out["volume_zscore_22"] = (out["Volume"] - vol_mean) / vol_std

    # Globex total volume per trade date (exclude maintenance break)
    raw = raw_1min.copy()
    t = raw["DateTime_ET"].dt.time
    raw = raw[~((t >= pd.Timestamp("17:00").time()) & (t < pd.Timestamp("18:00").time()))].copy()
    raw["trade_date"] = _globex_trade_date(raw["DateTime_ET"])
    globex_vol = raw.groupby("trade_date")["Volume"].sum().rename("globex_vol")
    out = out.merge(globex_vol, on="trade_date", how="left")
    out["volume_rth_vs_globex"] = out["Volume"] / out["globex_vol"]
    out = out.drop(columns=["globex_vol"])

    # First-hour (NYAM: 09:30–10:29) volume
    t2 = raw_1min["DateTime_ET"].dt.time
    nyam_mask = (t2 >= pd.Timestamp("09:30").time()) & (t2 < pd.Timestamp("10:30").time())
    nyam = raw_1min[nyam_mask].copy()
    nyam["trade_date"] = nyam["DateTime_ET"].dt.normalize()
    nyam_vol = nyam.groupby("trade_date")["Volume"].sum().rename("nyam_vol")
    out = out.merge(nyam_vol, on="trade_date", how="left")
    out["volume_first_hour_pct"] = out["nyam_vol"] / out["Volume"]
    out = out.drop(columns=["nyam_vol"])

    return out


def _session_range(raw_1min: pd.DataFrame, session_name: str) -> pd.Series:
    """Daily H-L range for a named session, indexed by Globex trade_date."""
    s = raw_1min[raw_1min["session"] == session_name].copy()
    s["trade_date"] = _globex_trade_date(s["DateTime_ET"])
    return (
        s.groupby("trade_date")
         .apply(lambda g: g["High"].max() - g["Low"].min())
         .rename(f"{session_name.lower()}_range")
    )


def compute_session_features(daily: pd.DataFrame, raw_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Group 4: Intraday Session Structure features.
    - nyam_range_pct    : NYAM H-L / daily H-L
    - london_range_pct  : LONDON session H-L / daily H-L
    - asia_range_pct    : ASIA session H-L / daily H-L
    - overnight_range_pct: bars before 09:30 H-L / daily H-L
    - session_vol_entropy : Shannon entropy of per-session RV shares
    """
    out = daily.copy()
    daily_range = out["High"] - out["Low"]

    for ses in ["NYAM", "LONDON", "ASIA"]:
        sr = _session_range(raw_1min, ses)
        col = f"{ses.lower()}_range_pct"
        sr_aligned = sr.reindex(out["trade_date"]).values
        out[col] = np.clip(sr_aligned / daily_range.values, 0.0, 1.0)

    # Overnight range: full Globex overnight before RTH open for each trade_date
    t = raw_1min["DateTime_ET"].dt.time
    overnight_mask = (t < RTH_START) | (t >= pd.Timestamp("18:00").time())
    overnight = raw_1min[overnight_mask].copy()
    overnight["trade_date"] = _globex_trade_date(overnight["DateTime_ET"])
    ovn_range = (
        overnight.groupby("trade_date")
                 .apply(lambda g: g["High"].max() - g["Low"].min())
                 .rename("ovn_range")
    )
    ovn_df = ovn_range.reset_index()
    ovn_df.columns = ["trade_date", "ovn_range"]
    out = out.merge(ovn_df, on="trade_date", how="left")
    out["overnight_range_pct"] = out["ovn_range"] / daily_range
    out = out.drop(columns=["ovn_range"])

    # Session-level realized variance for entropy computation
    session_rv = {}
    for ses in ["ASIA", "LONDON", "NYAM", "LUNCH", "PM", "OTHER"]:
        mask = raw_1min["session"] == ses
        s = raw_1min[mask].copy()
        s["trade_date"] = _globex_trade_date(s["DateTime_ET"])
        s = s.sort_values("DateTime_ET")
        s["log_ret"] = np.log(s["Close"] / s["Close"].shift(1))
        s.loc[s["trade_date"] != s["trade_date"].shift(1), "log_ret"] = np.nan
        rv = s.groupby("trade_date")["log_ret"].apply(lambda r: (r**2).sum())
        session_rv[ses] = rv

    rv_df = pd.DataFrame(session_rv).fillna(0)
    rv_df["total"] = rv_df.sum(axis=1)

    def _entropy(row):
        total = row["total"]
        if total == 0:
            return np.nan
        shares = row.drop("total") / total
        shares = shares[shares > 0]
        return -(shares * np.log(shares)).sum()

    entropy = rv_df.apply(_entropy, axis=1).rename("session_vol_entropy")
    entropy_df = entropy.reset_index()
    entropy_df.columns = ["trade_date", "session_vol_entropy"]
    out = out.merge(entropy_df, on="trade_date", how="left")

    return out


def compute_eth_rth_cross_features(eth_daily: pd.DataFrame,
                                    rth_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Group 4b: ETH / RTH cross-session features.
    - rth_pct_of_eth      : RTH H-L / ETH H-L  (mean ~81%)
    - overnight_pct_of_eth: 1 - rth_pct_of_eth
    - overnight_gap_eth   : (RTH_Open - prev_ETH_Close) / prev_ETH_range
    - rth_inside_flag     : 1 if RTH session was an inside day (within RTH context)
    - rth_outside_flag    : 1 if RTH session was an outside day (within RTH context)
    - eth_rth_divergence  : 1 if ETH inside/outside classification differs from RTH
    """
    rth = rth_daily.copy()
    rth["rth_range"] = rth["High"] - rth["Low"]
    rth["prev_RTH_H"] = rth["High"].shift(1)
    rth["prev_RTH_L"] = rth["Low"].shift(1)
    rth["rth_inside_flag"]  = (
        (rth["High"] <= rth["prev_RTH_H"]) & (rth["Low"] >= rth["prev_RTH_L"])
    ).astype(int)
    rth["rth_outside_flag"] = (
        (rth["High"] >  rth["prev_RTH_H"]) & (rth["Low"] <  rth["prev_RTH_L"])
    ).astype(int)
    rth_cols = rth[["trade_date", "rth_range", "Open", "rth_inside_flag", "rth_outside_flag"]]

    out = eth_daily.copy()
    out["eth_range"] = out["High"] - out["Low"]

    out["prev_ETH_H"]     = out["High"].shift(1)
    out["prev_ETH_L"]     = out["Low"].shift(1)
    out["prev_ETH_range"] = out["eth_range"].shift(1)
    out["prev_ETH_Close"] = out["Close"].shift(1)
    out["eth_inside"]  = (out["High"] <= out["prev_ETH_H"]) & (out["Low"] >= out["prev_ETH_L"])
    out["eth_outside"] = (out["High"] >  out["prev_ETH_H"]) & (out["Low"] <  out["prev_ETH_L"])

    out = out.merge(rth_cols.rename(columns={"Open": "rth_open"}),
                    on="trade_date", how="left")

    out["rth_pct_of_eth"]       = out["rth_range"] / out["eth_range"]
    out["overnight_pct_of_eth"] = 1.0 - out["rth_pct_of_eth"]

    out["overnight_gap_eth"] = (
        (out["rth_open"] - out["prev_ETH_Close"]) / out["prev_ETH_range"]
    )

    out["eth_rth_divergence"] = (
        (out["eth_inside"].astype(int)  != out["rth_inside_flag"]) |
        (out["eth_outside"].astype(int) != out["rth_outside_flag"])
    ).astype(int)

    drop_cols = ["eth_range", "prev_ETH_H", "prev_ETH_L", "prev_ETH_range",
                 "prev_ETH_Close", "eth_inside", "eth_outside", "rth_range", "rth_open"]
    return out.drop(columns=drop_cols, errors="ignore")


def compute_calendar_features(
    daily: pd.DataFrame,
    eco: pd.DataFrame,
    max_event_date: "pd.Timestamp | None" = None,
) -> pd.DataFrame:
    """
    Group 5: Calendar & Macro Event features.
    Economic events are in UTC; convert to ET date by subtracting 4 hours.
    Handles timezone-aware datetime_utc by converting to tz-naive after offset.
    Optionally filters scheduled events after max_event_date to avoid using
    future calendar revisions beyond the available market-data horizon.
    """
    out = daily.copy()
    out["day_of_week"] = pd.to_datetime(out["trade_date"]).dt.dayofweek  # 0=Mon

    if max_event_date is None and "trade_date" in out.columns and len(out) > 0:
        max_event_date = pd.to_datetime(out["trade_date"]).max()

    eco = eco.copy()
    # Subtract 4h to convert UTC to ET, then normalize to date (tz-naive)
    et_times = eco["datetime_utc"] - pd.Timedelta(hours=4)
    if hasattr(et_times.dt, "tz") and et_times.dt.tz is not None:
        et_times = et_times.dt.tz_localize(None)
    eco["et_date"] = et_times.dt.normalize()

    if max_event_date is not None:
        max_event_date = pd.to_datetime(max_event_date).normalize()
        eco = eco[eco["et_date"] <= max_event_date].copy()

    # Drop non-high-impact events
    eco_high = eco[eco["impact"] == "high"].copy()

    # FOMC days
    fomc = eco_high[eco_high["title"].isin(_FOMC_TITLES)][["et_date"]].drop_duplicates()
    fomc["is_fomc_day"] = 1
    out = out.merge(fomc, left_on="trade_date", right_on="et_date", how="left")
    out["is_fomc_day"] = out["is_fomc_day"].fillna(0).astype(int)
    out = out.drop(columns=["et_date"], errors="ignore")

    # NFP days
    nfp = eco_high[eco_high["title"] == _NFP_TITLE][["et_date"]].drop_duplicates()
    nfp["is_nfp_day"] = 1
    out = out.merge(nfp, left_on="trade_date", right_on="et_date", how="left")
    out["is_nfp_day"] = out["is_nfp_day"].fillna(0).astype(int)
    out = out.drop(columns=["et_date"], errors="ignore")

    # All high-impact events per date
    event_counts = eco_high.groupby("et_date").size().rename("n_events").reset_index()
    out = out.merge(event_counts, left_on="trade_date", right_on="et_date", how="left")
    out["high_impact_today"] = (out["n_events"].fillna(0) > 0).astype(int)
    out = out.drop(columns=["et_date", "n_events"], errors="ignore")

    # Tomorrow's events — shift event dates back by 1 day
    event_counts_shifted = event_counts.copy()
    event_counts_shifted["et_date"] = event_counts_shifted["et_date"] - pd.Timedelta(days=1)
    out = out.merge(event_counts_shifted, left_on="trade_date", right_on="et_date", how="left")
    out["high_impact_tomorrow"] = (out["n_events"].fillna(0) > 0).astype(int)
    out = out.drop(columns=["et_date", "n_events"], errors="ignore")

    # Count of events in next 2 calendar days
    event_set = set(eco_high["et_date"].dt.date)

    def _events_next_2d(date):
        return sum(
            (date + pd.Timedelta(days=d)).date() in event_set
            for d in [1, 2]
        )

    out["n_events_next_2d"] = out["trade_date"].apply(_events_next_2d)

    # Days since last FOMC
    fomc_dates = sorted(fomc["et_date"].dropna().tolist())

    def _days_since_fomc(date):
        past = [d for d in fomc_dates if d <= date]
        return (date - past[-1]).days if past else np.nan

    out["days_since_fomc"] = out["trade_date"].apply(_days_since_fomc)

    return out


def compute_vix_features(daily: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """
    Group 6: VIX / Implied Volatility features.
    - vix_close         : VIX index close on day t
    - vix_change_1d     : day-over-day VIX change
    - vix_rv_spread     : VIX_t - sqrt(rv_22d * 252) — implied minus realised vol
    - vix_percentile_252: percentile of VIX within trailing 252-day window
    """
    out = daily.copy()
    vix = vix.copy()
    vix["trade_date"] = pd.to_datetime(vix["date"])
    vix = vix.sort_values("trade_date")
    vix["vix_change_1d"] = vix["c"].diff()
    vix["vix_percentile_252"] = (
        vix["c"]
        .rolling(252, min_periods=30)
        .apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    )

    out = out.merge(
        vix[["trade_date", "c", "vix_change_1d", "vix_percentile_252"]].rename(
            columns={"c": "vix_close"}),
        on="trade_date", how="left"
    )

    if "rv_22d" in out.columns:
        out["vix_rv_spread"] = out["vix_close"] - np.sqrt(out["rv_22d"] * 252) * 100
    else:
        out["vix_rv_spread"] = np.nan

    return out


def compute_cross_instrument_features(
    daily_es: pd.DataFrame, daily_nq: pd.DataFrame
) -> tuple:
    """
    Group 7: Cross-instrument features (ES ↔ NQ).
    - es_nq_rv_ratio    : rv_1d_ES / rv_1d_NQ
    - es_nq_range_ratio : range_ES / range_NQ
    """
    def _break_state(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        out = df[["trade_date", "High", "Low"]].copy()
        prev_h = out["High"].shift(1)
        prev_l = out["Low"].shift(1)
        valid = prev_h.notna() & prev_l.notna()
        break_high = (out["High"] > prev_h) & valid
        break_low = (out["Low"] < prev_l) & valid
        out[f"{prefix}_break_high"] = break_high.astype(float).where(valid)
        out[f"{prefix}_break_low"] = break_low.astype(float).where(valid)
        out[f"{prefix}_outside"] = (break_high & break_low).astype(float).where(valid)
        out[f"{prefix}_one_side"] = (break_high ^ break_low).astype(float).where(valid)
        out[f"{prefix}_high_only"] = (break_high & ~break_low).astype(float).where(valid)
        out[f"{prefix}_low_only"] = (~break_high & break_low).astype(float).where(valid)
        return out.drop(columns=["High", "Low"])

    cross = daily_es[["trade_date", "rv_1d", "range_abs"]].merge(
        daily_nq[["trade_date", "rv_1d", "range_abs"]],
        on="trade_date", suffixes=("_es", "_nq")
    )
    cross = cross.merge(_break_state(daily_es, "es"), on="trade_date", how="left")
    cross = cross.merge(_break_state(daily_nq, "nq"), on="trade_date", how="left")

    cross["es_nq_rv_ratio"]    = cross["rv_1d_es"]    / cross["rv_1d_nq"]
    cross["es_nq_range_ratio"] = cross["range_abs_es"] / cross["range_abs_nq"]

    cross["both_outside"] = ((cross["es_outside"] == 1) & (cross["nq_outside"] == 1)).astype(float)
    cross["both_one_side"] = ((cross["es_one_side"] == 1) & (cross["nq_one_side"] == 1)).astype(float)
    cross["es_nq_outside_divergence"] = (cross["es_outside"] != cross["nq_outside"]).astype(float)
    cross.loc[cross[["es_outside", "nq_outside"]].isna().any(axis=1), "es_nq_outside_divergence"] = np.nan
    cross["nq_outside_es_one_side"] = ((cross["nq_outside"] == 1) & (cross["es_one_side"] == 1)).astype(float)
    cross["es_outside_nq_one_side"] = ((cross["es_outside"] == 1) & (cross["nq_one_side"] == 1)).astype(float)
    cross["break_direction_agreement"] = (
        (cross["es_break_high"] == cross["nq_break_high"]) &
        (cross["es_break_low"] == cross["nq_break_low"])
    ).astype(float)
    cross.loc[cross[["es_break_high", "nq_break_high", "es_break_low", "nq_break_low"]].isna().any(axis=1),
              "break_direction_agreement"] = np.nan

    for col in ["es_nq_outside_divergence", "nq_outside_es_one_side", "es_outside_nq_one_side"]:
        cross[f"{col}_rate_5"] = cross[col].rolling(5, min_periods=1).mean()
        cross[f"{col}_rate_22"] = cross[col].rolling(22, min_periods=5).mean()

    cross = cross.rename(columns={
        "es_nq_outside_divergence_rate_5": "cross_outside_divergence_rate_5",
        "es_nq_outside_divergence_rate_22": "cross_outside_divergence_rate_22",
    })

    cross_cols = cross[[
        "trade_date", "es_nq_rv_ratio", "es_nq_range_ratio",
        "both_outside", "both_one_side", "es_nq_outside_divergence",
        "nq_outside_es_one_side", "es_outside_nq_one_side",
        "break_direction_agreement", "cross_outside_divergence_rate_5",
        "cross_outside_divergence_rate_22", "nq_outside_es_one_side_rate_5",
        "nq_outside_es_one_side_rate_22", "es_outside_nq_one_side_rate_5",
        "es_outside_nq_one_side_rate_22",
    ]]

    es_out = daily_es.merge(cross_cols, on="trade_date", how="left")
    nq_out = daily_nq.merge(cross_cols, on="trade_date", how="left")
    return es_out, nq_out


def compute_pattern_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Group 8: Inside/Outside Pattern features.
    Directly predictive of whether the next day will be inside.

    - h_proximity        : (prev_H - H) / prev_range  — how far H is below prev H
                           0 = H is exactly at prev H (on the inside boundary)
    - l_proximity        : (L - prev_L) / prev_range  — how far L is above prev L
    - hl_containment     : h_proximity + l_proximity  — both close = market trapped
    - range_vs_max_3d    : range_t / max(range_{t-3..t-1}) — compression vs recent peak
    - range_vs_max_5d    : range_t / max(range_{t-5..t-1})
    - range_vs_max_10d   : range_t / max(range_{t-10..t-1})
    - contraction_streak : count of last 3 days where range shrank (0–3)
    - close_vs_midpoint  : |close_location - 0.5| — 0 = doji, 0.5 = close at extreme
    - inside_lag1        : was yesterday an inside day (ETH basis)
    - outside_lag1       : was yesterday an outside day
    - range_percentile_22: percentile of today's range in trailing 22-day window
    """
    out = daily.copy()
    r    = out["range_abs"]
    pr   = r.shift(1)
    pH   = out["High"].shift(1)
    pL   = out["Low"].shift(1)

    out["h_proximity"]    = (pH - out["High"]) / pr
    out["l_proximity"]    = (out["Low"] - pL)  / pr
    out["hl_containment"] = out["h_proximity"] + out["l_proximity"]

    out["break_high"] = (out["High"] > pH)
    out["break_low"] = (out["Low"] < pL)
    out["high_only_break"] = out["break_high"] & ~out["break_low"]
    out["low_only_break"] = ~out["break_high"] & out["break_low"]
    out["one_side_break"] = out["break_high"] ^ out["break_low"]
    out["dist_to_prev_high"] = ((pH - out["High"]).clip(lower=0)) / pr
    out["dist_to_prev_low"] = ((out["Low"] - pL).clip(lower=0)) / pr
    overnight_gap = out["overnight_gap"] if "overnight_gap" in out.columns else (out["Open"] - out["Close"].shift(1)) / pr
    out["gap_direction"] = np.sign(overnight_gap).fillna(0.0)
    out["abs_overnight_gap"] = overnight_gap.abs()

    for n in [3, 5, 10]:
        roll_max = r.rolling(n, min_periods=n).max().shift(1)
        out[f"range_vs_max_{n}d"] = r / roll_max

    contraction = (r < pr).astype(float)
    out["contraction_streak"] = contraction.rolling(3, min_periods=1).sum()

    out["close_vs_midpoint"] = (out["close_location"] - 0.5).abs()

    out["inside_lag1"]  = out["inside"].shift(1).astype(float)
    out["outside_lag1"] = out["outside"].shift(1).astype(float)

    out["range_percentile_22"] = (
        r.rolling(22, min_periods=10)
         .apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    )

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

    return out


def add_target(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add target variable and inside/outside/neither labels.
    y = log(range_{t+1} / range_t) — predicts NEXT day's log range ratio.
    """
    out = daily.copy()
    out["prev_High"] = out["High"].shift(1)
    out["prev_Low"]  = out["Low"].shift(1)
    out["inside"]  = (out["High"] <= out["prev_High"]) & (out["Low"] >= out["prev_Low"])
    out["outside"] = (out["High"] >  out["prev_High"]) & (out["Low"] <  out["prev_Low"])
    out["neither"] = ~out["inside"] & ~out["outside"]
    out = out.drop(columns=["prev_High", "prev_Low"])

    # Target: log ratio of NEXT day's range to today's range
    out["y"] = np.log(out["range_abs"].shift(-1) / out["range_abs"])

    return out


# ── Feature column groups (for model.py to reference) ────────────────────────
FEATURE_COLS_HAR = ["rv_1d", "rv_5d", "rv_22d"]

FEATURE_COLS_ALL = [
    # Group 1 — Realized Volatility (computed on RTH session bars)
    "rv_1d", "rv_5d", "rv_22d", "rv_ratio_1_5", "rv_percentile_252", "parkinson_vol",
    # Group 2 — Range Structure (ETH daily bars)
    "range_pct_of_prev", "atr_ratio", "range_ma_5", "range_ma_22",
    "close_location", "overnight_gap",
    # Group 3 — Volume
    "volume_prev", "volume_zscore_22", "volume_rth_vs_globex", "volume_first_hour_pct",
    # Group 4 — Intraday Session (% of daily range)
    "nyam_range_pct", "london_range_pct", "asia_range_pct", "session_vol_entropy",
    # Group 4b — ETH/RTH Cross-session
    "rth_pct_of_eth", "overnight_pct_of_eth", "overnight_gap_eth",
    "rth_inside_flag", "rth_outside_flag", "eth_rth_divergence",
    # Group 5 — Calendar & Macro
    "day_of_week", "high_impact_today", "high_impact_tomorrow",
    "n_events_next_2d", "is_fomc_day", "is_nfp_day", "days_since_fomc",
    # Group 6 — VIX
    "vix_close", "vix_change_1d", "vix_rv_spread", "vix_percentile_252",
    # Group 7 — Cross-instrument (ES ↔ NQ)
    "es_nq_rv_ratio", "es_nq_range_ratio",
    "both_outside", "both_one_side", "es_nq_outside_divergence",
    "nq_outside_es_one_side", "es_outside_nq_one_side",
    "break_direction_agreement", "cross_outside_divergence_rate_5",
    "cross_outside_divergence_rate_22", "nq_outside_es_one_side_rate_5",
    "nq_outside_es_one_side_rate_22", "es_outside_nq_one_side_rate_5",
    "es_outside_nq_one_side_rate_22",
    # Group 8 — Inside/Outside Pattern (lagged structure + HL proximity + compression)
    "h_proximity", "l_proximity", "hl_containment",
    "break_high", "break_low", "high_only_break", "low_only_break",
    "one_side_break", "dist_to_prev_high", "dist_to_prev_low",
    "gap_direction", "abs_overnight_gap",
    "range_vs_max_3d", "range_vs_max_5d", "range_vs_max_10d",
    "contraction_streak", "close_vs_midpoint",
    "inside_lag1", "outside_lag1", "range_percentile_22",
    "nr4_flag", "nr7_flag", "wr4_flag", "wr7_flag",
    "inside_streak", "outside_streak",
]

TARGET_COL  = "y"
LABEL_COLS  = ["inside", "outside", "neither"]
ID_COLS     = ["trade_date", "Open", "High", "Low", "Close", "Volume", "range_abs"]


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


def _build_features_for(symbol: str, raw: pd.DataFrame,
                         vix: pd.DataFrame, eco: pd.DataFrame) -> tuple:
    """Returns (eth_daily_with_features, rth_daily_with_features) tuple."""
    eth_daily = build_eth_daily(raw)
    rth_daily = build_rth_daily(raw)

    eth = _add_common_features(eth_daily, raw, vix, eco, rv_session="rth")
    eth = compute_eth_rth_cross_features(eth, rth_daily)

    rth = _add_common_features(rth_daily, raw, vix, eco, rv_session="rth")
    rth_cross = compute_eth_rth_cross_features(eth_daily, rth)
    cross_cols = [
        "trade_date",
        "rth_inside_flag",
        "rth_outside_flag",
        "rth_pct_of_eth",
        "overnight_pct_of_eth",
        "overnight_gap_eth",
        "eth_rth_divergence",
    ]
    rth = rth.merge(rth_cross[cross_cols], on="trade_date", how="left")

    return eth, rth


def finalize_feature_frames(
    es: pd.DataFrame,
    nq: pd.DataFrame,
    es_rth: pd.DataFrame,
    nq_rth: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Add cross-instrument features, targets, and pattern features for ETH/RTH outputs."""
    es, nq = compute_cross_instrument_features(es, nq)
    es_rth, nq_rth = compute_cross_instrument_features(es_rth, nq_rth)

    outputs = {
        "output/features_es_eth.parquet": es,
        "output/features_nq_eth.parquet": nq,
        "output/features_es_rth.parquet": es_rth,
        "output/features_nq_rth.parquet": nq_rth,
    }

    finalized = {}
    for path, frame in outputs.items():
        finalized[path] = compute_pattern_features(add_target(frame))
    return finalized


def main():
    Path("output").mkdir(exist_ok=True)

    print("Loading 1-min data...")
    es_raw = pd.read_parquet("data/es_1m.parquet")
    nq_raw = pd.read_parquet("data/nq_1m.parquet")

    vix = pd.read_parquet("data/vix_cboe.parquet")
    vix["date"] = pd.to_datetime(vix["date"])
    eco = pd.read_parquet("data/economic_events.parquet")

    print("Engineering ES features...")
    es, es_rth = _build_features_for("ES", es_raw, vix, eco)

    print("Engineering NQ features...")
    nq, nq_rth = _build_features_for("NQ", nq_raw, vix, eco)

    print("Finalizing ETH/RTH feature outputs...")
    outputs = finalize_feature_frames(es, nq, es_rth, nq_rth)

    for path, frame in outputs.items():
        frame.to_parquet(path, index=False)
        print(f"Saved {path}")

    for label, frame in [
        ("ES ETH", outputs["output/features_es_eth.parquet"]),
        ("NQ ETH", outputs["output/features_nq_eth.parquet"]),
        ("ES RTH", outputs["output/features_es_rth.parquet"]),
        ("NQ RTH", outputs["output/features_nq_rth.parquet"]),
    ]:
        complete = frame[FEATURE_COLS_ALL].notna().all(axis=1).sum()
        print(f"{label}: {len(frame)} days, {complete} fully complete rows")


if __name__ == "__main__":
    main()
