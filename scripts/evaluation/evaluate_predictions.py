"""
Evaluate saved MOEX predictions against real daily candles in Supabase.

Examples:
    python evaluate_predictions.py --from 11.05 --to today --save
    python evaluate_predictions.py --from 2026-05-11 --to 2026-05-14 --fail-under 45
    python evaluate_predictions.py --ticker SBER --from 11.05 --to tomorrow --save
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def parse_date(value: str) -> date:
    value = value.strip().lower()
    if value in {"today", "сегодня"}:
        return date.today()
    if value in {"tomorrow", "завтра"}:
        return date.today() + timedelta(days=1)

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
        f"Invalid date '{value}'. Use YYYY-MM-DD, DD.MM.YYYY, DD.MM, today or tomorrow."
    )


def next_day(day: date) -> str:
    return (day + timedelta(days=1)).isoformat()


def fetch_predictions(date_from: date, date_to: date, ticker: str | None, model_version: str | None):
    query = supabase.table("predictions")\
        .select("id,target_date,ticker,model_version,predicted_close,predicted_delta,confidence,signal")\
        .gte("target_date", date_from.isoformat())\
        .lte("target_date", date_to.isoformat())\
        .order("target_date", desc=False)\
        .order("ticker", desc=False)
    if ticker:
        query = query.eq("ticker", ticker)
    if model_version:
        query = query.eq("model_version", model_version)
    return query.execute().data or []


def fetch_actual_close(ticker: str, target: date):
    rows = supabase.table("candles")\
        .select("time,close")\
        .eq("ticker", ticker).eq("interval", "1d")\
        .gte("time", target.isoformat())\
        .lt("time", next_day(target))\
        .order("time", desc=False).limit(1).execute().data
    return float(rows[0]["close"]) if rows else None


def fetch_prev_close(ticker: str, target: date):
    rows = supabase.table("candles")\
        .select("time,close")\
        .eq("ticker", ticker).eq("interval", "1d")\
        .lt("time", target.isoformat())\
        .order("time", desc=True).limit(1).execute().data
    return float(rows[0]["close"]) if rows else None


def mark_correct(signal: str, actual_delta: float, hold_band: float) -> bool:
    signal = (signal or "HOLD").upper()
    if signal == "BUY":
        return actual_delta > 0
    if signal == "SELL":
        return actual_delta < 0
    return abs(actual_delta) <= hold_band


def summarize(scored: list[dict]) -> dict:
    if not scored:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": None,
            "mae": None,
            "by_signal": {},
        }

    correct = sum(1 for row in scored if row["is_correct"])
    mae = sum(abs(row["error_pct"]) for row in scored) / len(scored)
    by_signal = {}
    for row in scored:
        sig = row["signal"]
        bucket = by_signal.setdefault(sig, {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(bool(row["is_correct"]))

    for bucket in by_signal.values():
        bucket["accuracy"] = round(bucket["correct"] / bucket["total"] * 100, 2)

    return {
        "total": len(scored),
        "correct": correct,
        "accuracy": correct / len(scored) * 100,
        "mae": mae,
        "by_signal": by_signal,
    }


def evaluate(args) -> tuple[list[dict], list[dict], dict]:
    predictions = fetch_predictions(args.date_from, args.date_to, args.ticker, args.model_version)
    scored = []
    skipped = []

    for pred in predictions:
        ticker = pred["ticker"]
        target = parse_date(str(pred["target_date"])[:10])
        actual_close = fetch_actual_close(ticker, target)
        prev_close = fetch_prev_close(ticker, target)

        if actual_close is None or prev_close is None:
            skipped.append({
                **pred,
                "reason": "missing_actual_candle" if actual_close is None else "missing_previous_candle",
            })
            continue

        actual_delta = (actual_close / prev_close - 1) * 100
        predicted_delta = float(pred.get("predicted_delta") or 0)
        error_pct = predicted_delta - actual_delta
        is_correct = mark_correct(pred.get("signal", "HOLD"), actual_delta, args.hold_band)

        row = {
            "id": pred.get("id"),
            "target_date": target.isoformat(),
            "ticker": ticker,
            "model_version": pred.get("model_version"),
            "signal": pred.get("signal"),
            "confidence": pred.get("confidence"),
            "predicted_close": pred.get("predicted_close"),
            "predicted_delta": round(predicted_delta, 4),
            "prev_close": round(prev_close, 4),
            "actual_close": round(actual_close, 4),
            "actual_delta": round(actual_delta, 4),
            "error_pct": round(error_pct, 4),
            "is_correct": is_correct,
        }
        scored.append(row)

        if args.save and pred.get("id") is not None:
            supabase.table("predictions").update({
                "actual_delta": row["actual_delta"],
                "is_correct": row["is_correct"],
            }).eq("id", pred["id"]).execute()

    return scored, skipped, summarize(scored)


def write_report(scored: list[dict], skipped: list[dict], summary: dict, date_from: date, date_to: date) -> Path:
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"prediction_quality_{date_from.isoformat()}_{date_to.isoformat()}.csv"

    fieldnames = [
        "target_date", "ticker", "model_version", "signal", "confidence",
        "predicted_close", "predicted_delta", "prev_close", "actual_close",
        "actual_delta", "error_pct", "is_correct",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scored:
            writer.writerow({k: row.get(k) for k in fieldnames})

    skipped_path = out_dir / f"prediction_quality_skipped_{date_from.isoformat()}_{date_to.isoformat()}.csv"
    if skipped:
        with skipped_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(skipped[0].keys()))
            writer.writeheader()
            writer.writerows(skipped)

    return out_path


def print_summary(scored: list[dict], skipped: list[dict], summary: dict, report_path: Path):
    print("\nPrediction quality")
    print("=" * 60)
    print(f"Scored:  {summary['total']}")
    print(f"Skipped: {len(skipped)}")
    if summary["total"]:
        print(f"Correct: {summary['correct']}/{summary['total']}")
        print(f"Accuracy: {summary['accuracy']:.2f}%")
        print(f"MAE: {summary['mae']:.2f} percentage points")
        print("\nBy signal:")
        for signal, data in sorted(summary["by_signal"].items()):
            print(
                f"  {signal:4s}: {data['correct']}/{data['total']} "
                f"({data['accuracy']:.2f}%)"
            )
    else:
        print("No scored predictions. Check that predictions and daily candles exist for this range.")

    print(f"\nReport: {report_path}")
    if skipped:
        reasons = {}
        for row in skipped:
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
        print(f"Skipped reasons: {reasons}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="date_from", required=True, type=parse_date)
    parser.add_argument("--to", dest="date_to", default="today", type=parse_date)
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--hold-band", default=0.4, type=float)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--fail-under", default=None, type=float)
    args = parser.parse_args()

    if args.date_to < args.date_from:
        raise SystemExit("--to must be after or equal to --from")
    if args.ticker:
        args.ticker = args.ticker.upper()

    scored, skipped, summary = evaluate(args)
    report_path = write_report(scored, skipped, summary, args.date_from, args.date_to)
    print_summary(scored, skipped, summary, report_path)

    if args.fail_under is not None and summary["accuracy"] is not None:
        if summary["accuracy"] < args.fail_under:
            raise SystemExit(
                f"Accuracy {summary['accuracy']:.2f}% is below threshold {args.fail_under:.2f}%"
            )


if __name__ == "__main__":
    main()
