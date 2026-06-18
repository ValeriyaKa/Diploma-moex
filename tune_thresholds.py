"""
Tune ensemble weights and signal thresholds on existing rolling backtest CSV.

Matches the exact formula from generate_predictions.py:
  lstm_d_prob    = sigmoid(lstm_daily  * 3)
  lstm_h_prob    = sigmoid(lstm_hourly * 3)
  ensemble_prob  = w_lgbm*lgbm_prob + w_xgb*xgb_prob + w_lstm_d*lstm_d_prob + w_lstm_h*lstm_h_prob
  ensemble_delta = (ensemble_prob - 0.5) * 6
  BUY  if ensemble_prob > conf_thresh  and ensemble_delta > delta_min
  SELL if ensemble_prob < 1-conf_thresh and ensemble_delta < -delta_min
  HOLD otherwise

mark_correct (matches backtest_prediction.py):
  BUY  → actual_delta > 0
  SELL → actual_delta < 0
  HOLD → abs(actual_delta) <= hold_band (0.4)

Usage:
    python tune_thresholds.py --csv data/rolling_backtest_2026-04-01_2026-05-26.csv
    python tune_thresholds.py --csv data/rolling_backtest_2026-04-01_2026-05-26.csv --top 20 --min-cov 0.2
"""
from __future__ import annotations

import argparse
import itertools
import numpy as np
import pandas as pd


def sigmoid(x: float, k: float = 3.0) -> float:
    return 1.0 / (1.0 + np.exp(-x * k))


def mark_correct(signal: str, actual_delta: float, hold_band: float = 0.4) -> bool:
    if signal == "BUY":
        return actual_delta > 0
    if signal == "SELL":
        return actual_delta < 0
    return abs(actual_delta) <= hold_band  # HOLD correct only if small move


def compute_signal(row, w_lgbm: float, w_xgb: float,
                   w_lstm_d: float, w_lstm_h: float,
                   conf_thresh: float, delta_min: float) -> tuple[str, float, float]:
    lgbm_prob  = row["lgbm_prob"]   if not np.isnan(row["lgbm_prob"])   else 0.5
    xgb_prob   = row["xgb_prob"]    if not np.isnan(row["xgb_prob"])    else 0.5
    lstm_d     = row["lstm_daily"]  if not np.isnan(row["lstm_daily"])  else 0.0
    lstm_h     = row["lstm_hourly"] if not np.isnan(row["lstm_hourly"]) else 0.0

    lstm_d_prob = sigmoid(lstm_d)
    lstm_h_prob = sigmoid(lstm_h)

    # Normalize weights
    w_total = w_lgbm + w_xgb + w_lstm_d + w_lstm_h
    if w_total == 0:
        w_total = 1.0

    ens_prob = (w_lgbm   * lgbm_prob
              + w_xgb    * xgb_prob
              + w_lstm_d * lstm_d_prob
              + w_lstm_h * lstm_h_prob) / w_total

    ens_prob  = max(0.01, min(0.99, ens_prob))
    ens_delta = (ens_prob - 0.5) * 6

    if ens_prob > conf_thresh and ens_delta > delta_min:
        signal = "BUY"
    elif ens_prob < (1 - conf_thresh) and ens_delta < -delta_min:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, ens_prob, ens_delta


def evaluate(df: pd.DataFrame,
             w_lgbm: float, w_xgb: float, w_lstm_d: float, w_lstm_h: float,
             conf_thresh: float, delta_min: float) -> dict | None:
    scored = df[df["actual_delta"].notna()].copy()
    if len(scored) == 0:
        return None

    results = [
        compute_signal(row, w_lgbm, w_xgb, w_lstm_d, w_lstm_h, conf_thresh, delta_min)
        for _, row in scored.iterrows()
    ]
    signals, probs, deltas = zip(*results)

    scored["new_signal"] = signals
    scored["new_correct"] = [
        mark_correct(s, a) for s, a in zip(signals, scored["actual_delta"])
    ]

    n_total   = len(scored)
    n_correct = scored["new_correct"].sum()
    overall_acc = n_correct / n_total

    active   = scored[scored["new_signal"] != "HOLD"]
    n_active = len(active)
    coverage = n_active / n_total

    dir_acc = active["new_correct"].mean() if n_active > 0 else 0.0
    mae     = (active["new_signal"].map({"BUY": 1, "SELL": -1}) * 0  # placeholder
               ).mean()  # not useful; replace with delta error
    if n_active > 0:
        act_active = active["actual_delta"]
        pred_dir   = active["new_signal"].map({"BUY": 1, "SELL": -1})
        mae = act_active.abs().mean()  # mean absolute actual move on active signals

    return {
        "w_lgbm":      round(w_lgbm,   2),
        "w_xgb":       round(w_xgb,    2),
        "w_lstm_d":    round(w_lstm_d,  2),
        "w_lstm_h":    round(w_lstm_h,  2),
        "conf_thresh": conf_thresh,
        "delta_min":   delta_min,
        "overall_acc": round(overall_acc, 4),
        "dir_acc":     round(dir_acc,     4),
        "coverage":    round(coverage,    4),
        "n_signals":   n_active,
        "n_buy":       int((scored["new_signal"] == "BUY").sum()),
        "n_sell":      int((scored["new_signal"] == "SELL").sum()),
        "n_hold":      int((scored["new_signal"] == "HOLD").sum()),
        # score: directional precision weighted by sqrt(coverage) — rewards balance
        "score":       round(dir_acc * np.sqrt(coverage + 0.01), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",      required=True, help="Path to rolling_backtest CSV")
    parser.add_argument("--top",      default=15,    type=int,   help="Show top N configs")
    parser.add_argument("--min-cov",  default=0.10,  type=float, help="Min signal coverage fraction")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows, {df['ticker'].nunique()} tickers")
    print(f"Scored rows (with actual): {df['actual_delta'].notna().sum()}")

    # Verify lstm_hourly has real values
    h_nonzero = (df["lstm_hourly"].notna() & (df["lstm_hourly"].abs() > 0.001)).sum()
    print(f"Rows with non-zero lstm_hourly: {h_nonzero}/{len(df)}")
    print()

    # --- Search grid ---
    # Weights: (lgbm, xgb, lstm_daily, lstm_hourly) — will be normalized
    weight_combos = [
        # current baseline
        (0.5, 0.1, 0.4, 0.0),
        # vary lgbm vs lstm_d
        (0.6, 0.1, 0.3, 0.0),
        (0.7, 0.1, 0.2, 0.0),
        (0.4, 0.1, 0.5, 0.0),
        # add xgb weight
        (0.5, 0.2, 0.3, 0.0),
        (0.4, 0.2, 0.4, 0.0),
        (0.6, 0.2, 0.2, 0.0),
        (0.5, 0.3, 0.2, 0.0),
        # include lstm_hourly
        (0.5, 0.1, 0.3, 0.1),
        (0.4, 0.1, 0.4, 0.1),
        (0.5, 0.1, 0.2, 0.2),
        (0.4, 0.1, 0.3, 0.2),
        # lgbm only
        (1.0, 0.0, 0.0, 0.0),
        # lgbm + xgb only
        (0.7, 0.3, 0.0, 0.0),
        (0.6, 0.4, 0.0, 0.0),
        (0.5, 0.5, 0.0, 0.0),
    ]
    conf_thresholds = [0.55, 0.57, 0.58, 0.60, 0.62, 0.63, 0.65, 0.67]
    delta_mins      = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]

    total = len(weight_combos) * len(conf_thresholds) * len(delta_mins)
    print(f"Evaluating {total} configurations...")

    results = []
    for (wl, wx, wld, wlh), ct, dm in itertools.product(
            weight_combos, conf_thresholds, delta_mins):
        r = evaluate(df, wl, wx, wld, wlh, ct, dm)
        if r and r["coverage"] >= args.min_cov:
            results.append(r)

    if not results:
        print("No configurations met the minimum coverage requirement.")
        return

    results_df = pd.DataFrame(results).sort_values("score", ascending=False)

    print(f"\nTop {args.top} configurations (score = dir_acc × √coverage):\n")
    cols = ["w_lgbm","w_xgb","w_lstm_d","w_lstm_h",
            "conf_thresh","delta_min","overall_acc","dir_acc","coverage","n_buy","n_sell","score"]
    print(results_df[cols].head(args.top).to_string(index=False))

    print("\n--- Best by dir_acc (min coverage 20%) ---")
    hc = results_df[results_df["coverage"] >= 0.20]
    if not hc.empty:
        best = hc.sort_values("dir_acc", ascending=False).iloc[0]
        print(best[cols].to_string())

    print("\n--- Best score overall ---")
    best = results_df.iloc[0]
    print(best[cols].to_string())

    # Normalize weights for display
    b = results_df.iloc[0]
    wt = b["w_lgbm"] + b["w_xgb"] + b["w_lstm_d"] + b["w_lstm_h"]
    if wt == 0: wt = 1
    print("\n--- Recommended settings ---")
    print(f"CONF_THRESHOLD = {b['conf_thresh']}")
    print(f"DELTA_MIN      = {b['delta_min']}")
    print(f"# Ensemble weights (in generate_predictions.py):")
    print(f"w_lgbm   = {round(b['w_lgbm']/wt,  3)}   # lgbm_prob")
    print(f"w_xgb    = {round(b['w_xgb']/wt,   3)}   # xgb_prob")
    print(f"w_lstm_d = {round(b['w_lstm_d']/wt, 3)}   # sigmoid(lstm_daily*3)")
    print(f"w_lstm_h = {round(b['w_lstm_h']/wt, 3)}   # sigmoid(lstm_hourly*3)")

    out = args.csv.replace(".csv", "_tune_results.csv")
    results_df.to_csv(out, index=False)
    print(f"\nFull results → {out}")


if __name__ == "__main__":
    main()
