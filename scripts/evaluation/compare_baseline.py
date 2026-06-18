"""
Сравнение модели со статистическим baseline и динамикой индекса IMOEX.

Что считается:
  1. Точность направления (accuracy): модель vs baseline "всегда BUY" vs baseline "momentum"
  2. Симулированная доходность: стратегия модели vs Buy&Hold IMOEX за тот же период

Примеры запуска:
    python compare_baseline.py --from 2026-01-01 --to today
    python compare_baseline.py --from 01.01.2026 --to 14.05.2026 --save
    python compare_baseline.py --from 01.01.2026 --to today --ticker SBER
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

HOLD_BAND = 0.4  # ±0.4% считается «нейтральным» движением


# ═══════════════════════════════════════════════════════════════
# ПАРСИНГ ДАТЫ
# ═══════════════════════════════════════════════════════════════

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
        f"Неверный формат даты '{value}'. Используй YYYY-MM-DD, DD.MM.YYYY, DD.MM, today."
    )


def next_day(d: date) -> str:
    return (d + timedelta(days=1)).isoformat()


# ═══════════════════════════════════════════════════════════════
# ЗАГРУЗКА ДАННЫХ
# ═══════════════════════════════════════════════════════════════

def fetch_scored_predictions(date_from: date, date_to: date, ticker: str | None) -> list[dict]:
    """Загружаем предсказания у которых уже есть actual_delta (оценённые)."""
    query = (
        supabase.table("predictions")
        .select("target_date,ticker,signal,confidence,predicted_delta,actual_delta,is_correct")
        .gte("target_date", date_from.isoformat())
        .lte("target_date", date_to.isoformat())
        .not_.is_("actual_delta", "null")
        .order("target_date", desc=False)
        .order("ticker", desc=False)
    )
    if ticker:
        query = query.eq("ticker", ticker)
    return query.execute().data or []


def fetch_imoex(date_from: date, date_to: date) -> list[dict]:
    """Загружаем дневные свечи по индексу IMOEX из таблицы macro."""
    rows = (
        supabase.table("macro")
        .select("time,imoex")
        .gte("time", date_from.isoformat())
        .lte("time", date_to.isoformat())
        .not_.is_("imoex", "null")
        .order("time", desc=False)
        .execute()
        .data
    ) or []
    return rows


def fetch_candles_for_ticker(ticker: str, date_from: date, date_to: date) -> list[dict]:
    """Загружаем дневные свечи по конкретному тикеру для симуляции доходности."""
    rows = (
        supabase.table("candles")
        .select("time,close")
        .eq("ticker", ticker)
        .eq("interval", "1d")
        .gte("time", date_from.isoformat())
        .lte("time", date_to.isoformat())
        .order("time", desc=False)
        .execute()
        .data
    ) or []
    return rows


# ═══════════════════════════════════════════════════════════════
# BASELINE-ПРЕДСКАЗАНИЯ
# ═══════════════════════════════════════════════════════════════

def baseline_always_buy_correct(actual_delta: float) -> bool:
    """Baseline 1: всегда предсказываем рост (BUY)."""
    return actual_delta > 0


def baseline_momentum_correct(prev_delta: float | None, actual_delta: float) -> bool:
    """Baseline 2: предсказываем то же направление, что и вчера (momentum)."""
    if prev_delta is None:
        return actual_delta > 0  # если нет данных — считаем BUY
    if prev_delta > HOLD_BAND:
        return actual_delta > 0
    elif prev_delta < -HOLD_BAND:
        return actual_delta < 0
    else:
        return abs(actual_delta) <= HOLD_BAND


# ═══════════════════════════════════════════════════════════════
# СИМУЛЯЦИЯ ДОХОДНОСТИ
# ═══════════════════════════════════════════════════════════════

def simulate_model_return(predictions: list[dict]) -> float:
    """
    Симулируем торговлю по сигналам модели.
    BUY  → держим позицию на следующий день (+actual_delta)
    SELL → шортим                           (-actual_delta)
    HOLD → не торгуем                       (0)
    Возвращаем суммарную доходность в % (не накопленную, а суммарную по сделкам).
    """
    # Группируем по дате, чтобы считать как портфель
    by_date: dict[str, list] = defaultdict(list)
    for p in predictions:
        by_date[p["target_date"]].append(p)

    total_return = 0.0
    trading_days = 0

    for day_preds in by_date.values():
        day_return = 0.0
        trades = 0
        for p in day_preds:
            sig = (p.get("signal") or "HOLD").upper()
            delta = float(p.get("actual_delta") or 0)
            if sig == "BUY":
                day_return += delta
                trades += 1
            elif sig == "SELL":
                day_return -= delta
                trades += 1
        if trades > 0:
            total_return += day_return / trades  # средняя доходность по портфелю за день
            trading_days += 1

    return round(total_return, 4), trading_days


def simulate_imoex_return(imoex_rows: list[dict]) -> float:
    """
    Считаем суммарную доходность Buy&Hold IMOEX за период.
    (close[-1] / close[0] - 1) * 100
    """
    if len(imoex_rows) < 2:
        return None
    first = float(imoex_rows[0]["imoex"])
    last  = float(imoex_rows[-1]["imoex"])
    if first == 0:
        return None
    return round((last / first - 1) * 100, 4)


# ═══════════════════════════════════════════════════════════════
# ОСНОВНАЯ ОЦЕНКА
# ═══════════════════════════════════════════════════════════════

def evaluate(predictions: list[dict]) -> dict:
    """Считаем точность модели и обоих baseline."""
    # Строим словарь предыдущего actual_delta по тикеру для momentum
    prev_delta_by_ticker: dict[str, float | None] = {}

    model_correct   = 0
    always_buy_corr = 0
    momentum_corr   = 0
    total           = 0

    for p in predictions:
        ticker       = p["ticker"]
        actual_delta = float(p.get("actual_delta") or 0)
        is_correct   = p.get("is_correct")

        if is_correct is None:
            continue

        # Модель
        model_correct += int(bool(is_correct))

        # Baseline 1: всегда BUY
        always_buy_corr += int(baseline_always_buy_correct(actual_delta))

        # Baseline 2: momentum
        prev = prev_delta_by_ticker.get(ticker)
        momentum_corr += int(baseline_momentum_correct(prev, actual_delta))

        prev_delta_by_ticker[ticker] = actual_delta
        total += 1

    if total == 0:
        return {"total": 0}

    return {
        "total": total,
        "model_accuracy":      round(model_correct   / total * 100, 2),
        "always_buy_accuracy": round(always_buy_corr / total * 100, 2),
        "momentum_accuracy":   round(momentum_corr   / total * 100, 2),
        "model_correct":       model_correct,
        "always_buy_correct":  always_buy_corr,
        "momentum_correct":    momentum_corr,
    }


# ═══════════════════════════════════════════════════════════════
# ВЫВОД
# ═══════════════════════════════════════════════════════════════

def print_report(stats: dict, model_return: float, trading_days: int, imoex_return: float | None):
    print("\n" + "=" * 62)
    print("  СРАВНЕНИЕ МОДЕЛИ С BASELINE И IMOEX")
    print("=" * 62)

    if stats.get("total", 0) == 0:
        print("Нет оценённых предсказаний за указанный период.")
        print("Запусти сначала: python evaluate_predictions.py --from ... --to ... --save")
        return

    print(f"\nВсего оценённых предсказаний: {stats['total']}")
    print()
    print(f"{'Метод':<30} {'Верных':>8} {'Accuracy':>10}")
    print("-" * 52)
    print(f"{'Модель (LSTM + LightGBM)':<30} {stats['model_correct']:>8} {stats['model_accuracy']:>9.2f}%")
    print(f"{'Baseline: всегда BUY':<30} {stats['always_buy_correct']:>8} {stats['always_buy_accuracy']:>9.2f}%")
    print(f"{'Baseline: momentum':<30} {stats['momentum_correct']:>8} {stats['momentum_accuracy']:>9.2f}%")

    print()
    print("Прирост модели над baseline:")
    delta_buy      = stats["model_accuracy"] - stats["always_buy_accuracy"]
    delta_momentum = stats["model_accuracy"] - stats["momentum_accuracy"]
    print(f"  vs всегда BUY : {delta_buy:+.2f} п.п.")
    print(f"  vs momentum   : {delta_momentum:+.2f} п.п.")

    print()
    print("─" * 62)
    print("СИМУЛЯЦИЯ ДОХОДНОСТИ")
    print("─" * 62)
    print(f"  Стратегия модели : {model_return:+.2f}% ({trading_days} торговых дней)")
    if imoex_return is not None:
        print(f"  IMOEX Buy&Hold   : {imoex_return:+.2f}%")
        delta_imoex = model_return - imoex_return
        print(f"  Модель vs IMOEX  : {delta_imoex:+.2f} п.п.")
    else:
        print("  IMOEX: нет данных в таблице macro за этот период")

    print("=" * 62)


def save_report(
    stats: dict,
    model_return: float,
    trading_days: int,
    imoex_return: float | None,
    date_from: date,
    date_to: date,
) -> Path:
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"baseline_comparison_{date_from.isoformat()}_{date_to.isoformat()}.csv"

    rows = [
        {"metric": "period_from",            "value": date_from.isoformat()},
        {"metric": "period_to",              "value": date_to.isoformat()},
        {"metric": "total_predictions",      "value": stats.get("total", 0)},
        {"metric": "model_accuracy_pct",     "value": stats.get("model_accuracy")},
        {"metric": "always_buy_accuracy_pct","value": stats.get("always_buy_accuracy")},
        {"metric": "momentum_accuracy_pct",  "value": stats.get("momentum_accuracy")},
        {"metric": "model_return_pct",       "value": model_return},
        {"metric": "imoex_return_pct",       "value": imoex_return},
        {"metric": "trading_days",           "value": trading_days},
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nОтчёт сохранён: {path}")
    return path


# ═══════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Сравнение модели с baseline и IMOEX")
    parser.add_argument("--from", dest="date_from", required=True, type=parse_date,
                        help="Начало периода (YYYY-MM-DD / DD.MM / today)")
    parser.add_argument("--to",   dest="date_to",   default="today", type=parse_date,
                        help="Конец периода (YYYY-MM-DD / DD.MM / today)")
    parser.add_argument("--ticker", default=None, help="Фильтр по тикеру (например SBER)")
    parser.add_argument("--save", action="store_true", help="Сохранить CSV-отчёт в папку data/")
    args = parser.parse_args()

    if args.date_to < args.date_from:
        raise SystemExit("--to должна быть позже --from")
    if args.ticker:
        args.ticker = args.ticker.upper()

    print(f"Загружаю предсказания за {args.date_from} – {args.date_to}...")
    predictions = fetch_scored_predictions(args.date_from, args.date_to, args.ticker)

    if not predictions:
        print("Нет оценённых предсказаний. Сначала запусти evaluate_predictions.py --save")
        return

    print(f"Загружаю данные IMOEX...")
    imoex_rows = fetch_imoex(args.date_from, args.date_to)

    stats         = evaluate(predictions)
    model_return, trading_days = simulate_model_return(predictions)
    imoex_return  = simulate_imoex_return(imoex_rows)

    print_report(stats, model_return, trading_days, imoex_return)

    if args.save:
        save_report(stats, model_return, trading_days, imoex_return, args.date_from, args.date_to)


if __name__ == "__main__":
    main()
