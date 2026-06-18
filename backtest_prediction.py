"""
Backtest one-day MOEX predictions.

Example:
    python backtest_prediction.py --as-of 2026-05-11 --target 2026-05-12
    python backtest_prediction.py --ticker SBER --as-of 2026-05-11 --target 2026-05-12 --save
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

from generate_predictions import (
    TICKERS,
    build_daily_features,
    build_hourly_features,
    load_macro,
    load_news,
    predict_full,
)

load_dotenv()
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def parse_date(value: str) -> date:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    try:
        return datetime.strptime(f"{value}.{date.today().year}", "%d.%m.%Y").date()
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        f"Invalid date '{value}'. Use YYYY-MM-DD, DD.MM.YYYY or DD.MM."
    )


def day_start(day: date) -> str:
    return day.isoformat()


def next_day(day: date) -> str:
    return (day + timedelta(days=1)).isoformat()


def _trade_date(value) -> date | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.tz_convert("Europe/Moscow").date()


def _filter_by_trade_date(rows: list[dict], max_day: date | None = None,
                          exact_day: date | None = None,
                          limit: int | None = None) -> list[dict]:
    out = []
    for row in rows:
        td = _trade_date(row.get("time"))
        if td is None:
            continue
        if exact_day is not None and td != exact_day:
            continue
        if max_day is not None and td > max_day:
            continue
        item = dict(row)
        item["_trade_date"] = td
        out.append(item)
    out.sort(key=lambda r: pd.to_datetime(r["time"], utc=True, errors="coerce"))
    if limit is not None:
        out = out[-limit:]
    for row in out:
        row.pop("_trade_date", None)
    return out


def fetch_daily_as_of(ticker: str, as_of: date, limit: int = 100):
    candles = supabase.table("candles")\
        .select("time,open,high,low,close,volume")\
        .eq("ticker", ticker).eq("interval", "1d")\
        .lt("time", next_day(as_of + timedelta(days=2)))\
        .order("time", desc=True).limit(limit + 10).execute().data
    indicators = supabase.table("indicators")\
        .select("time,rsi_14,macd_hist,bb_upper,bb_lower,atr_14,vol_ratio")\
        .eq("ticker", ticker)\
        .lt("time", next_day(as_of + timedelta(days=2)))\
        .order("time", desc=True).limit(limit + 10).execute().data
    candles = _filter_by_trade_date(candles, max_day=as_of, limit=limit)
    indicators = _filter_by_trade_date(indicators, max_day=as_of, limit=limit)
    return candles, indicators


def fetch_hourly_as_of(ticker: str, as_of: date, limit: int = 150):
    rows = supabase.table("candles")\
        .select("time,close,volume")\
        .eq("ticker", ticker).eq("interval", "1h")\
        .lt("time", next_day(as_of + timedelta(days=2)))\
        .order("time", desc=True).limit(limit + 50).execute().data
    return _filter_by_trade_date(rows, max_day=as_of, limit=limit)


def fetch_actual_daily(ticker: str, target: date):
    rows = supabase.table("candles")\
        .select("time,close")\
        .eq("ticker", ticker).eq("interval", "1d")\
        .gte("time", day_start(target - timedelta(days=1)))\
        .lt("time", next_day(target + timedelta(days=1)))\
        .order("time", desc=False).limit(5).execute().data
    rows = _filter_by_trade_date(rows, exact_day=target, limit=1)
    return rows[0] if rows else None


def load_macro_as_of(as_of: date):
    try:
        rows = supabase.table("macro")\
            .select("*")\
            .lte("time", next_day(as_of))\
            .order("time", desc=True).limit(1).execute().data
        if rows:
            return {
                "imoex": float(rows[0].get("imoex") or 3000),
                "usd_rub": float(rows[0].get("usd_rub") or 88),
            }
    except Exception:
        pass
    return load_macro()


def mark_correct(signal: str, actual_delta: float, hold_band: float = 0.4) -> bool:
    if signal == "BUY":
        return actual_delta > 0
    if signal == "SELL":
        return actual_delta < 0
    return abs(actual_delta) <= hold_band


def run_backtest(tickers: list[str], as_of: date, target: date, save: bool):
    macro = load_macro_as_of(as_of)
    ticker_sentiment, market_sentiment = load_news()
    results = []

    print(f"Backtest as_of={as_of.isoformat()} target={target.isoformat()}")
    print(f"Macro: IMOEX={macro['imoex']:.0f}, USD/RUB={macro['usd_rub']:.2f}")

    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"[{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
            candles, indicators = fetch_daily_as_of(ticker, as_of)
            df_daily = build_daily_features(
                candles,
                indicators,
                macro,
                ticker_sentiment.get(ticker, 0.0),
                market_sentiment,
            )
            if df_daily is None:
                print("skip: not enough daily data")
                continue

            hourly = fetch_hourly_as_of(ticker, as_of)
            df_hourly = build_hourly_features(hourly)
            pred = predict_full(
                ticker,
                df_daily,
                df_hourly,
                ticker_sentiment.get(ticker, 0.0),
                market_sentiment,
            )
            pred["target_date"] = target

            current_close = float(df_daily.iloc[-1]["close"])
            actual = fetch_actual_daily(ticker, target)
            if actual:
                actual_close = float(actual["close"])
                actual_delta = (actual_close / current_close - 1) * 100
                error_pct = pred["predicted_delta"] - actual_delta
                is_correct = mark_correct(pred["signal"], actual_delta)
            else:
                actual_close = None
                actual_delta = None
                error_pct = None
                is_correct = None

            row = {
                "ticker": ticker,
                "as_of": as_of.isoformat(),
                "target_date": target.isoformat(),
                "current_close": round(current_close, 4),
                "predicted_close": pred.get("predicted_close"),
                "predicted_delta": pred["predicted_delta"],
                "confidence": pred["confidence"],
                "signal": pred["signal"],
                "actual_close": round(actual_close, 4) if actual_close is not None else None,
                "actual_delta": round(actual_delta, 4) if actual_delta is not None else None,
                "error_pct": round(error_pct, 4) if error_pct is not None else None,
                "is_correct": is_correct,
            }
            results.append(row)
            print(
                f"{row['signal']} pred={row['predicted_delta']:+.2f}% "
                f"actual={row['actual_delta'] if row['actual_delta'] is not None else 'N/A'}"
            )

            if save:
                supabase.table("predictions").upsert({
                    "target_date": row["target_date"],
                    "ticker": ticker,
                    "model_version": f"backtest_asof_{as_of.isoformat()}",
                    "predicted_close": row["predicted_close"],
                    "predicted_delta": row["predicted_delta"],
                    "confidence": row["confidence"],
                    "signal": row["signal"],
                    "actual_delta": row["actual_delta"],
                    "is_correct": row["is_correct"],
                    "top_features": pred.get("top_features", {}),
                }, on_conflict="ticker,target_date,model_version").execute()
            time.sleep(0.1)
        except Exception as exc:
            print(f"error: {exc}")

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"backtest_{as_of.isoformat()}_{target.isoformat()}.csv"
    if results:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        scored = [r for r in results if r["is_correct"] is not None]
        if scored:
            acc = sum(1 for r in scored if r["is_correct"]) / len(scored) * 100
            mae = sum(abs(r["error_pct"]) for r in scored if r["error_pct"] is not None) / len(scored)
            print(f"\nAccuracy: {acc:.1f}% ({len(scored)} scored), MAE={mae:.2f} pp")
        print(f"Saved: {out_path}")
    else:
        print("No results.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--as-of", required=True, type=parse_date)
    parser.add_argument("--target", required=True, type=parse_date)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else TICKERS
    if args.target <= args.as_of:
        raise SystemExit("--target must be after --as-of")
    run_backtest(tickers, args.as_of, args.target, args.save)


if __name__ == "__main__":
    main()
