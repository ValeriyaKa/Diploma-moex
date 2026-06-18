"""
Run rolling one-day backtests over a date range.

Example:
    python rolling_backtest.py --from 11.05 --to today
    python rolling_backtest.py --from 11.05.2026 --to 14.05.2026 --save
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

from backtest_prediction import (
    fetch_actual_daily,
    fetch_daily_as_of,
    fetch_hourly_as_of,
    load_macro_as_of,
    mark_correct,
    parse_date,
)
from generate_predictions import (
    TICKERS,
    build_daily_features,
    build_hourly_features,
    load_news,
    predict_full,
)

load_dotenv()
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# ─── Fallback: actual_close из features.csv ───────────────────────────────
_features_cache: pd.DataFrame | None = None


def _load_features_cache() -> pd.DataFrame:
    """Загружает features.csv один раз и кэширует в памяти."""
    global _features_cache
    if _features_cache is not None:
        return _features_cache
    for features_path in [Path("data") / "features.csv", Path("features.csv")]:
        if features_path.exists():
            df = pd.read_csv(features_path)
            df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True)
            df["date"] = df["time"].dt.date
            _features_cache = df[df["timeframe"] == "1d"][["ticker", "date", "close"]]
            print(f"  [features cache] {len(_features_cache):,} дневных строк из {features_path}")
            return _features_cache
    _features_cache = pd.DataFrame(columns=["ticker", "date", "close"])
    print("  [features cache] features.csv не найден — fallback недоступен")
    return _features_cache


def fetch_actual_from_features(ticker: str, target: date) -> dict | None:
    """
    Fallback: берёт actual_close из локального features.csv.
    Используется когда fetch_actual_daily (Supabase) не вернул данные.
    """
    df = _load_features_cache()
    if df.empty:
        return None
    rows = df[(df["ticker"] == ticker) & (df["date"] == target)]
    if not rows.empty:
        return {"close": float(rows.iloc[0]["close"])}
    return None


# ─── Торговый календарь ───────────────────────────────────────────────────
def _next_weekday(day: date) -> date:
    target = day + timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


def _iter_weekdays(start: date, end: date):
    day = start
    while day <= end:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


try:
    import exchange_calendars as xcals
    _moex_cal = xcals.get_calendar("XMOS")
    def next_trading_day(day: date) -> date:
        try:
            return _moex_cal.next_session(day).date()
        except Exception as exc:
            print(f"WARNING: MOEX calendar next_session failed ({type(exc).__name__}); using weekday fallback")
            return _next_weekday(day)
    def iter_trading_days(start: date, end: date):
        try:
            sessions = _moex_cal.sessions_in_range(str(start), str(end))
            for s in sessions:
                yield s.date()
        except Exception as exc:
            print(f"WARNING: MOEX calendar range failed ({type(exc).__name__}); using weekday fallback")
            yield from _iter_weekdays(start, end)
except Exception:
    def next_trading_day(day: date) -> date:
        return _next_weekday(day)
    def iter_trading_days(start: date, end: date):
        yield from _iter_weekdays(start, end)


def summarize(rows: list[dict]) -> dict:
    scored = [row for row in rows if row["is_correct"] is not None]
    if not scored:
        return {"total": 0, "accuracy": None, "mae": None}
    correct = sum(1 for row in scored if row["is_correct"])
    mae = sum(abs(row["error_pct"]) for row in scored) / len(scored)
    return {
        "total": len(scored),
        "correct": correct,
        "accuracy": correct / len(scored) * 100,
        "mae": mae,
    }


def run(args):
    tickers = [args.ticker.upper()] if args.ticker else TICKERS
    ticker_sentiment, market_sentiment = load_news()
    rows = []

    # Прогреваем кэш features.csv один раз до цикла
    _load_features_cache()

    for as_of in iter_trading_days(args.date_from, args.date_to):
        target = next_trading_day(as_of)
        if target > args.date_to:
            continue

        macro = load_macro_as_of(as_of)
        print(f"\nBacktest {as_of.isoformat()} -> {target.isoformat()}")

        for i, ticker in enumerate(tickers, 1):
            try:
                print(f"  [{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
                candles, indicators = fetch_daily_as_of(ticker, as_of)
                df_daily = build_daily_features(
                    candles,
                    indicators,
                    macro,
                    ticker_sentiment.get(ticker, 0.0),
                    market_sentiment,
                    ticker=ticker,
                )
                if df_daily is None:
                    print("skip")
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

                current_close = float(df_daily.iloc[-1]["close"])

                # ── Получаем actual_close: сначала Supabase, потом features.csv ──
                actual = fetch_actual_daily(ticker, target)
                if actual is None:
                    actual = fetch_actual_from_features(ticker, target)
                    if actual is not None:
                        print("[features.csv] ", end="")

                if actual:
                    actual_close = float(actual["close"])
                    actual_delta = (actual_close / current_close - 1) * 100
                    error_pct = pred["predicted_delta"] - actual_delta
                    is_correct = mark_correct(pred["signal"], actual_delta)
                    print(
                        f"{pred['signal']} pred={pred['predicted_delta']:+.2f}% "
                        f"actual={actual_delta:+.2f}%"
                    )
                else:
                    actual_close = None
                    actual_delta = None
                    error_pct = None
                    is_correct = None
                    print(f"{pred['signal']} actual=N/A")

                row = {
                    "as_of": as_of.isoformat(),
                    "target_date": target.isoformat(),
                    "ticker": ticker,
                    "signal": pred["signal"],
                    "confidence": pred["confidence"],
                    "current_close": round(current_close, 4),
                    "predicted_close": pred.get("predicted_close"),
                    "predicted_delta": pred["predicted_delta"],
                    "actual_close": round(actual_close, 4) if actual_close is not None else None,
                    "actual_delta": round(actual_delta, 4) if actual_delta is not None else None,
                    "error_pct": round(error_pct, 4) if error_pct is not None else None,
                    "is_correct": is_correct,
                    "lgbm_prob": pred.get("lgbm_prob"),
                    "xgb_prob": pred.get("xgb_prob"),
                    "lstm_daily": pred.get("lstm_daily"),
                    "lstm_hourly": pred.get("lstm_hourly"),
                }
                rows.append(row)

                if args.save:
                    supabase.table("predictions").upsert({
                        "target_date": row["target_date"],
                        "ticker": ticker,
                        "model_version": f"rolling_backtest_asof_{as_of.isoformat()}",
                        "predicted_close": row["predicted_close"],
                        "predicted_delta": row["predicted_delta"],
                        "confidence": row["confidence"],
                        "signal": row["signal"],
                        "actual_delta": row["actual_delta"],
                        "is_correct": row["is_correct"],
                        "top_features": pred.get("top_features", {}),
                    }, on_conflict="ticker,target_date,model_version").execute()

                time.sleep(0.05)
            except Exception as exc:
                print(f"error: {exc}")

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"rolling_backtest_{args.date_from.isoformat()}_{args.date_to.isoformat()}.csv"
    if rows:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = summarize(rows)
    print("\nRolling backtest summary")
    print("=" * 60)
    print(f"Rows: {len(rows)}")
    print(f"Scored: {summary['total']}")
    if summary["accuracy"] is not None:
        print(f"Accuracy: {summary['accuracy']:.2f}%")
        print(f"MAE: {summary['mae']:.2f} percentage points")
    print(f"Report: {out_path}")

    if args.fail_under is not None and summary["accuracy"] is not None:
        if summary["accuracy"] < args.fail_under:
            raise SystemExit(
                f"Accuracy {summary['accuracy']:.2f}% is below threshold {args.fail_under:.2f}%"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="date_from", required=True, type=parse_date)
    parser.add_argument("--to", dest="date_to", default="today", type=parse_date)
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--fail-under", default=None, type=float)
    args = parser.parse_args()

    if args.date_to <= args.date_from:
        raise SystemExit("--to must be after --from")
    run(args)


if __name__ == "__main__":
    main()
