"""
optimize_weights.py
====================
Grid search по весам ансамбля + порогу уверенности.
Работает на уже собранных данных rolling_backtest CSV
(нужны колонки: lgbm_prob, xgb_prob, lstm_daily, lstm_hourly, actual_delta).

Usage:
    python optimize_weights.py data/rolling_backtest_2026-05-01_2026-05-22.csv
    python optimize_weights.py data/rolling_backtest_*.csv --step 0.05
"""
import argparse
import sys
from itertools import product

import numpy as np
import pandas as pd


def load_data(paths: list[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Require raw model probabilities
    required = ["lgbm_prob", "xgb_prob", "lstm_daily", "lstm_hourly", "actual_delta"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: CSV is missing columns: {missing}")
        print("Re-run rolling_backtest.py with the updated code to save raw model outputs.")
        sys.exit(1)

    # Convert to numeric
    for col in ["lgbm_prob", "xgb_prob", "lstm_daily", "lstm_hourly", "actual_delta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without actual data
    df = df.dropna(subset=["actual_delta", "lgbm_prob", "lstm_daily"])

    # Filter out holidays (days where >80% tickers have actual_delta == 0)
    day_zero = df.groupby("as_of")["actual_delta"].apply(
        lambda x: (x == 0).sum() / len(x), include_groups=False
    )
    holiday_dates = day_zero[day_zero > 0.8].index.tolist()
    if holiday_dates:
        print(f"Excluding {len(holiday_dates)} non-trading days: {holiday_dates}")
        df = df[~df["as_of"].isin(holiday_dates)]

    print(f"Data: {len(df)} predictions, {df['as_of'].nunique()} real days, "
          f"{df['ticker'].nunique()} tickers\n")
    return df


def evaluate(df: pd.DataFrame, w_lgb, w_xgb, w_lstm_d, w_lstm_h,
             conf_threshold, delta_min, news_weight=0.05, market_weight=0.03):
    """Simulate ensemble with given weights and threshold."""
    # Convert LSTM regression output to probability
    lstm_d_prob = 1 / (1 + np.exp(-df["lstm_daily"].values * 3))
    lstm_h_prob = 1 / (1 + np.exp(-df["lstm_hourly"].fillna(0).values * 3))

    p = w_lgb * df["lgbm_prob"].values + w_xgb * df["xgb_prob"].fillna(0.5).values + \
        w_lstm_d * lstm_d_prob + w_lstm_h * lstm_h_prob

    p = np.clip(p, 0.01, 0.99)
    delta = (p - 0.5) * 6

    # Signals
    signals = np.where(
        (p > conf_threshold) & (delta > delta_min), "BUY",
        np.where((p < (1 - conf_threshold)) & (delta < -delta_min), "SELL", "HOLD")
    )

    actual = df["actual_delta"].values

    # Correctness
    correct = np.zeros(len(df), dtype=bool)
    correct[signals == "BUY"] = actual[signals == "BUY"] > 0
    correct[signals == "SELL"] = actual[signals == "SELL"] < 0
    correct[signals == "HOLD"] = np.abs(actual[signals == "HOLD"]) < 0.4

    n_buy = (signals == "BUY").sum()
    n_sell = (signals == "SELL").sum()
    n_hold = (signals == "HOLD").sum()
    n_active = n_buy + n_sell

    acc_all = correct.mean() * 100
    acc_active = correct[signals != "HOLD"].mean() * 100 if n_active > 0 else 0
    acc_buy = correct[signals == "BUY"].mean() * 100 if n_buy > 0 else 0
    acc_sell = correct[signals == "SELL"].mean() * 100 if n_sell > 0 else 0

    # MAE
    mae = np.abs(delta - actual).mean()

    return {
        "acc_all": acc_all, "acc_active": acc_active,
        "acc_buy": acc_buy, "acc_sell": acc_sell,
        "n_buy": n_buy, "n_sell": n_sell, "n_hold": n_hold,
        "mae": mae,
    }


def grid_search(df, step=0.1):
    """Try all weight combinations (step intervals) that sum to 1.0."""
    levels = np.arange(0.0, 1.0 + step/2, step)
    levels = np.round(levels, 2)

    # Confidence thresholds to try
    thresholds = [0.54, 0.56, 0.58, 0.60, 0.62, 0.65]
    delta_mins = [0.2, 0.4, 0.6]

    results = []
    n_combos = 0

    for w1 in levels:
        for w2 in levels:
            for w3 in levels:
                w4 = round(1.0 - w1 - w2 - w3, 2)
                if w4 < -0.001 or w4 > 1.001:
                    continue
                w4 = max(0, w4)
                # Skip degenerate (one model = 100%)
                if max(w1, w2, w3, w4) > 0.85:
                    continue

                for ct in thresholds:
                    for dm in delta_mins:
                        n_combos += 1
                        r = evaluate(df, w1, w2, w3, w4, ct, dm)
                        r.update({
                            "w_lgb": w1, "w_xgb": w2,
                            "w_lstm_d": w3, "w_lstm_h": w4,
                            "conf_threshold": ct, "delta_min": dm,
                        })
                        results.append(r)

    print(f"Tested {n_combos} combinations\n")
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Grid search for ensemble weights")
    parser.add_argument("csv", nargs="+", help="Path(s) to rolling_backtest CSV")
    parser.add_argument("--step", type=float, default=0.1,
                        help="Weight step size (default 0.1 = 10%%)")
    parser.add_argument("--top", type=int, default=15,
                        help="Show top N results")
    args = parser.parse_args()

    df = load_data(args.csv)
    results = grid_search(df, step=args.step)

    # Filter: need enough active signals for statistical significance
    results = results[results["n_buy"] + results["n_sell"] >= 30]

    print("=" * 90)
    print("TOP BY OVERALL ACCURACY (all signals)")
    print("=" * 90)
    top_all = results.nlargest(args.top, "acc_all")
    for _, r in top_all.iterrows():
        print(f"  acc={r.acc_all:5.1f}%  active={r.acc_active:5.1f}%  "
              f"BUY={r.acc_buy:4.0f}%({int(r.n_buy):3d})  SELL={r.acc_sell:4.0f}%({int(r.n_sell):3d})  "
              f"HOLD={int(r.n_hold):3d}  MAE={r.mae:.2f}  "
              f"w=[{r.w_lgb:.1f} {r.w_xgb:.1f} {r.w_lstm_d:.1f} {r.w_lstm_h:.1f}]  "
              f"ct={r.conf_threshold:.2f} dm={r.delta_min:.1f}")

    print()
    print("=" * 90)
    print("TOP BY ACTIVE SIGNALS ACCURACY (BUY + SELL only)")
    print("=" * 90)
    top_active = results.nlargest(args.top, "acc_active")
    for _, r in top_active.iterrows():
        print(f"  active={r.acc_active:5.1f}%  acc_all={r.acc_all:5.1f}%  "
              f"BUY={r.acc_buy:4.0f}%({int(r.n_buy):3d})  SELL={r.acc_sell:4.0f}%({int(r.n_sell):3d})  "
              f"HOLD={int(r.n_hold):3d}  MAE={r.mae:.2f}  "
              f"w=[{r.w_lgb:.1f} {r.w_xgb:.1f} {r.w_lstm_d:.1f} {r.w_lstm_h:.1f}]  "
              f"ct={r.conf_threshold:.2f} dm={r.delta_min:.1f}")

    print()
    print("=" * 90)
    print("TOP BY BALANCED SCORE (0.4*acc_all + 0.6*acc_active)")
    print("=" * 90)
    results["balanced"] = 0.4 * results["acc_all"] + 0.6 * results["acc_active"]
    top_bal = results.nlargest(args.top, "balanced")
    for _, r in top_bal.iterrows():
        print(f"  balanced={r.balanced:5.1f}  acc={r.acc_all:5.1f}%  active={r.acc_active:5.1f}%  "
              f"BUY={r.acc_buy:4.0f}%({int(r.n_buy):3d})  SELL={r.acc_sell:4.0f}%({int(r.n_sell):3d})  "
              f"HOLD={int(r.n_hold):3d}  "
              f"w=[{r.w_lgb:.1f} {r.w_xgb:.1f} {r.w_lstm_d:.1f} {r.w_lstm_h:.1f}]  "
              f"ct={r.conf_threshold:.2f} dm={r.delta_min:.1f}")

    # Save full results
    out = "data/weight_optimization_results.csv"
    results.to_csv(out, index=False)
    print(f"\nFull results saved to {out} ({len(results)} valid combos)")

    # Best recommendation
    best = top_bal.iloc[0]
    print(f"\n{'='*90}")
    print(f"RECOMMENDATION:")
    print(f"  Weights: LGB={best.w_lgb:.1f}  XGB={best.w_xgb:.1f}  "
          f"LSTM_d={best.w_lstm_d:.1f}  LSTM_h={best.w_lstm_h:.1f}")
    print(f"  Threshold: conf={best.conf_threshold:.2f}  delta_min={best.delta_min:.1f}")
    print(f"  Expected: accuracy={best.acc_all:.1f}%  active={best.acc_active:.1f}%")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
