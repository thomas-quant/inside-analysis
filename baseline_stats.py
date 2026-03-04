"""
Baseline inside/outside day frequency stats for ES and NQ futures.
Resamples 1-minute data to daily bars using two session definitions:
  - RTH  : 09:30–16:14 ET  (Regular Trading Hours)
  - Globex: 18:00 ET (prev day) – 17:00 ET (full overnight session)
"""

import pandas as pd
import numpy as np

# ── CONFIG ─────────────────────────────────────────────────────────────────────
INSTRUMENTS = {
    "ES": "data/es_1m.parquet",
    "NQ": "data/nq_1m.parquet",
}
SESSION_MODES = ["RTH", "Globex"]
OUTPUT_DIR = "output"
# ──────────────────────────────────────────────────────────────────────────────


def assign_trade_date_globex(dt_series: pd.Series) -> pd.Series:
    """
    Globex convention: session starts at 18:00 ET.
    Bars from 18:00–23:59 belong to the *next* calendar day.
    """
    dates = dt_series.dt.normalize()  # midnight of calendar date
    after_close = dt_series.dt.hour >= 18
    return dates + pd.to_timedelta(after_close.astype(int), unit="D")


def build_daily_bars(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Resample 1-min OHLCV to daily bars for the given session mode."""
    d = df.copy()

    if mode == "RTH":
        t = d["DateTime_ET"].dt.time
        mask = (t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:15").time())
        d = d[mask].copy()
        d["trade_date"] = d["DateTime_ET"].dt.normalize()
    else:  # Globex
        # exclude the 17:00–17:59 maintenance break
        t = d["DateTime_ET"].dt.time
        break_mask = (t >= pd.Timestamp("17:00").time()) & (t < pd.Timestamp("18:00").time())
        d = d[~break_mask].copy()
        d["trade_date"] = assign_trade_date_globex(d["DateTime_ET"])

    daily = (
        d.groupby("trade_date")
        .agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum"),
        )
        .reset_index()
    )

    # drop days with very few bars (holidays, early closes, data gaps)
    bar_counts = d.groupby("trade_date").size()
    min_bars = 30 if mode == "RTH" else 100
    valid_dates = bar_counts[bar_counts >= min_bars].index
    daily = daily[daily["trade_date"].isin(valid_dates)].copy()

    daily = daily.sort_values("trade_date").reset_index(drop=True)
    return daily


def flag_inside_outside(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["prev_High"] = daily["High"].shift(1)
    daily["prev_Low"] = daily["Low"].shift(1)
    daily["prev_range"] = daily["prev_High"] - daily["prev_Low"]
    daily["curr_range"] = daily["High"] - daily["Low"]

    # Inside: current range completely within previous range
    daily["inside"] = (daily["High"] <= daily["prev_High"]) & (daily["Low"] >= daily["prev_Low"])
    # Outside: current range completely engulfs previous range
    daily["outside"] = (daily["High"] > daily["prev_High"]) & (daily["Low"] < daily["prev_Low"])
    # Neither (directional trend day)
    daily["neither"] = ~daily["inside"] & ~daily["outside"]

    # Range as % of previous day's range
    daily["range_pct_of_prev"] = (daily["curr_range"] / daily["prev_range"] * 100).round(2)

    return daily.dropna(subset=["prev_High", "prev_Low"]).reset_index(drop=True)


def run_length_stats(flag: pd.Series, max_len: int = 5) -> pd.DataFrame:
    """
    For each run length N (2..max_len), compute:
      - count of N-day windows where all days match flag
      - as % of all windows of that length
    """
    s = flag.values
    n_total = len(s)
    rows = []
    for N in range(2, max_len + 1):
        windows = n_total - N + 1
        if windows <= 0:
            continue
        count = sum(s[i : i + N].all() for i in range(windows))
        rows.append({"run_length": N, "count": count, "windows": windows,
                     "pct": round(count / windows * 100, 2)})
    return pd.DataFrame(rows)


def baseline_stats(daily: pd.DataFrame, label: str) -> dict:
    n = len(daily)
    n_inside = daily["inside"].sum()
    n_outside = daily["outside"].sum()
    n_neither = daily["neither"].sum()

    stats = {
        "label": label,
        "total_days": n,
        "inside_count": int(n_inside),
        "outside_count": int(n_outside),
        "neither_count": int(n_neither),
        "inside_pct": round(n_inside / n * 100, 2),
        "outside_pct": round(n_outside / n * 100, 2),
        "neither_pct": round(n_neither / n * 100, 2),
    }
    return stats


def weekday_breakdown(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["weekday"] = daily["trade_date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    totals = daily["weekday"].value_counts().reindex(order, fill_value=0)
    in_cnt = daily[daily["inside"]]["weekday"].value_counts().reindex(order, fill_value=0)
    out_cnt = daily[daily["outside"]]["weekday"].value_counts().reindex(order, fill_value=0)

    return pd.DataFrame({
        "total": totals,
        "inside": in_cnt,
        "inside_pct": (in_cnt / totals * 100).round(2),
        "outside": out_cnt,
        "outside_pct": (out_cnt / totals * 100).round(2),
    })


def monthly_breakdown(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["month"] = daily["trade_date"].dt.month_name()
    order = ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]

    totals = daily["month"].value_counts().reindex(order, fill_value=0)
    in_cnt = daily[daily["inside"]]["month"].value_counts().reindex(order, fill_value=0)
    out_cnt = daily[daily["outside"]]["month"].value_counts().reindex(order, fill_value=0)

    return pd.DataFrame({
        "total": totals,
        "inside": in_cnt,
        "inside_pct": (in_cnt / totals * 100).round(2),
        "outside": out_cnt,
        "outside_pct": (out_cnt / totals * 100).round(2),
    })


def print_section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def analyze(symbol: str, path: str) -> None:
    print(f"\n{'#' * 70}")
    print(f"#  {symbol}")
    print(f"{'#' * 70}")

    raw = pd.read_parquet(path)

    for mode in SESSION_MODES:
        daily = build_daily_bars(raw, mode)
        daily = flag_inside_outside(daily)

        tag = f"{symbol} [{mode}]"

        # ── 1. Overall baseline ────────────────────────────────────────────
        print_section(f"{tag}  |  Overall baseline")
        stats = baseline_stats(daily, tag)
        date_range = f"{daily['trade_date'].min().date()} → {daily['trade_date'].max().date()}"
        print(f"  Period     : {date_range}")
        print(f"  Total days : {stats['total_days']}")
        print(f"  Inside     : {stats['inside_count']:>4d}  ({stats['inside_pct']:>5.1f}%)")
        print(f"  Outside    : {stats['outside_count']:>4d}  ({stats['outside_pct']:>5.1f}%)")
        print(f"  Neither    : {stats['neither_count']:>4d}  ({stats['neither_pct']:>5.1f}%)")

        # range stats
        print(f"\n  Range stats (current day vs previous day range):")
        print(f"    Inside  – median range/prev : {daily[daily['inside']]['range_pct_of_prev'].median():.1f}%"
              f"  mean: {daily[daily['inside']]['range_pct_of_prev'].mean():.1f}%")
        print(f"    Outside – median range/prev : {daily[daily['outside']]['range_pct_of_prev'].median():.1f}%"
              f"  mean: {daily[daily['outside']]['range_pct_of_prev'].mean():.1f}%")
        print(f"    Neither – median range/prev : {daily[daily['neither']]['range_pct_of_prev'].median():.1f}%"
              f"  mean: {daily[daily['neither']]['range_pct_of_prev'].mean():.1f}%")

        # ── 2. Weekday breakdown ───────────────────────────────────────────
        print_section(f"{tag}  |  Weekday breakdown")
        print(weekday_breakdown(daily).to_string())

        # ── 3. Monthly breakdown ───────────────────────────────────────────
        print_section(f"{tag}  |  Monthly breakdown")
        print(monthly_breakdown(daily).to_string())

        # ── 4. Consecutive run-length probabilities ────────────────────────
        print_section(f"{tag}  |  Consecutive run-length probabilities")
        in_runs = run_length_stats(daily["inside"])
        out_runs = run_length_stats(daily["outside"])

        print(f"\n  Inside streaks:")
        print(in_runs.to_string(index=False))
        print(f"\n  Outside streaks:")
        print(out_runs.to_string(index=False))

        # ── 5. Save daily bars to CSV ──────────────────────────────────────
        out_path = f"{OUTPUT_DIR}/{symbol.lower()}_{mode.lower()}_daily.csv"
        cols = ["trade_date", "Open", "High", "Low", "Close", "Volume",
                "inside", "outside", "neither", "range_pct_of_prev"]
        daily[cols].to_csv(out_path, index=False)
        print(f"\n  → Saved daily bars: {out_path}")


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for symbol, path in INSTRUMENTS.items():
        analyze(symbol, path)
