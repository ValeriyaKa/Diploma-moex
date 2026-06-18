"""
Анализ систематических ошибок модели.

Находит паттерны где модель стабильно ошибается:
  - по тикеру
  - по месяцу
  - по дню недели
  - по направлению сигнала
  - по диапазону уверенности
  - по диапазону lstm_daily

Сохраняет data/error_patterns.json — используется train_models.py
для повышения весов проблемных паттернов при обучении.

Usage:
    python analyze_errors.py --csv data/rolling_backtest_2026-04-01_2026-05-25.csv
    python analyze_errors.py --csv data/rolling_backtest_2026-04-01_2026-05-25.csv --min-n 5
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def error_rate(df: pd.DataFrame) -> float:
    """Доля неверных активных сигналов (BUY/SELL)."""
    active = df[df["signal"] != "HOLD"]
    if len(active) == 0:
        return 0.5
    wrong = (~active["is_correct"].astype(bool)).sum()
    return wrong / len(active)


def analyze(df: pd.DataFrame, min_n: int = 5) -> dict:
    """Возвращает словарь паттернов с error_rate и boost_weight."""
    df = df.copy()
    df = df[df["actual_delta"].notna()].copy()
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["month"]       = df["target_date"].dt.month
    df["dow"]         = df["target_date"].dt.dayofweek
    df["is_correct"]  = df["is_correct"].astype(bool)

    patterns = {}

    # ── 1. По тикеру ──────────────────────────────────────────────
    ticker_stats = {}
    for ticker, g in df.groupby("ticker"):
        active = g[g["signal"] != "HOLD"]
        if len(active) < min_n:
            continue
        er = error_rate(g)
        ticker_stats[ticker] = {
            "error_rate": round(er, 3),
            "n_active":   int(len(active)),
            "n_total":    int(len(g)),
        }
    # Boost для тикеров с error_rate > 0.55
    patterns["bad_tickers"] = {
        t: v for t, v in ticker_stats.items() if v["error_rate"] > 0.55
    }

    # ── 2. По месяцу ──────────────────────────────────────────────
    month_stats = {}
    for month, g in df.groupby("month"):
        active = g[g["signal"] != "HOLD"]
        if len(active) < min_n:
            continue
        er = error_rate(g)
        month_stats[int(month)] = {
            "error_rate": round(er, 3),
            "n_active":   int(len(active)),
        }
    patterns["month_error_rates"] = month_stats

    # ── 3. По дню недели ──────────────────────────────────────────
    dow_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"}
    dow_stats = {}
    for dow, g in df.groupby("dow"):
        active = g[g["signal"] != "HOLD"]
        if len(active) < min_n:
            continue
        er = error_rate(g)
        dow_stats[dow_names.get(int(dow), str(dow))] = {
            "error_rate": round(er, 3),
            "n_active":   int(len(active)),
        }
    patterns["dow_error_rates"] = dow_stats

    # ── 4. По типу сигнала ────────────────────────────────────────
    sig_stats = {}
    for sig, g in df.groupby("signal"):
        if sig == "HOLD":
            continue
        er = (~g["is_correct"]).sum() / max(len(g), 1)
        sig_stats[sig] = {
            "error_rate": round(float(er), 3),
            "n":          int(len(g)),
        }
    patterns["signal_error_rates"] = sig_stats

    # ── 5. По диапазону уверенности ───────────────────────────────
    df["conf_bin"] = pd.cut(
        df["confidence"],
        bins=[0, 0.55, 0.60, 0.65, 0.70, 1.0],
        labels=["<0.55","0.55-0.60","0.60-0.65","0.65-0.70",">0.70"],
    )
    conf_stats = {}
    for bin_label, g in df.groupby("conf_bin", observed=True):
        active = g[g["signal"] != "HOLD"]
        if len(active) < min_n:
            continue
        er = error_rate(g)
        conf_stats[str(bin_label)] = {
            "error_rate": round(er, 3),
            "n_active":   int(len(active)),
        }
    patterns["confidence_error_rates"] = conf_stats

    # ── 6. По диапазону lstm_daily ────────────────────────────────
    if "lstm_daily" in df.columns:
        df["lstm_bin"] = pd.cut(
            df["lstm_daily"].fillna(0),
            bins=[-10, -0.5, -0.1, 0.1, 0.5, 10],
            labels=["strong_neg","neg","neutral","pos","strong_pos"],
        )
        lstm_stats = {}
        for bin_label, g in df.groupby("lstm_bin", observed=True):
            active = g[g["signal"] != "HOLD"]
            if len(active) < min_n:
                continue
            er = error_rate(g)
            lstm_stats[str(bin_label)] = {
                "error_rate": round(er, 3),
                "n_active":   int(len(active)),
            }
        patterns["lstm_daily_error_rates"] = lstm_stats

    # ── Сводные boost_weights для train_models.py ─────────────────
    # weight = 1 + 2 * max(0, error_rate - 0.50)
    # Если модель ошибается в 70% случаев → weight = 1 + 2*0.2 = 1.4
    # Если ошибается в 50% → weight = 1.0 (норма)

    boost = {}

    # По месяцам
    boost["months"] = {
        str(m): round(1 + 2 * max(0, v["error_rate"] - 0.50), 2)
        for m, v in month_stats.items()
    }

    # По дням недели
    boost["dow"] = {
        str(k): round(1 + 2 * max(0, v["error_rate"] - 0.50), 2)
        for k, v in dow_stats.items()
    }

    # По тикерам
    boost["tickers"] = {
        t: round(1 + 2 * max(0, v["error_rate"] - 0.50), 2)
        for t, v in ticker_stats.items()
    }

    patterns["boost_weights"] = boost

    return patterns


def print_report(patterns: dict):
    print("\n" + "="*60)
    print("АНАЛИЗ СИСТЕМАТИЧЕСКИХ ОШИБОК МОДЕЛИ")
    print("="*60)

    print("\n📅 Ошибки по месяцам:")
    month_names = {1:"Янв",2:"Фев",3:"Мар",4:"Апр",5:"Май",6:"Июн",
                   7:"Июл",8:"Авг",9:"Сен",10:"Окт",11:"Ноя",12:"Дек"}
    for m, v in sorted(patterns.get("month_error_rates",{}).items()):
        bar = "█" * int(v["error_rate"] * 20)
        flag = " ⚠️  BOOST" if v["error_rate"] > 0.55 else ""
        print(f"  {month_names.get(m, m):3s}  {bar:<20s} {v['error_rate']*100:.1f}%  (n={v['n_active']}){flag}")

    print("\n📆 Ошибки по дням недели:")
    for d, v in patterns.get("dow_error_rates",{}).items():
        bar = "█" * int(v["error_rate"] * 20)
        flag = " ⚠️  BOOST" if v["error_rate"] > 0.55 else ""
        print(f"  {d:3s}  {bar:<20s} {v['error_rate']*100:.1f}%  (n={v['n_active']}){flag}")

    print("\n📊 Ошибки по сигналу:")
    for s, v in patterns.get("signal_error_rates",{}).items():
        print(f"  {s:5s}  ошибка {v['error_rate']*100:.1f}%  (n={v['n']})")

    print("\n🎯 Ошибки по уверенности модели:")
    for b, v in patterns.get("confidence_error_rates",{}).items():
        print(f"  {b:12s}  ошибка {v['error_rate']*100:.1f}%  (n={v['n_active']})")

    print("\n🔁 LSTM daily диапазоны:")
    for b, v in patterns.get("lstm_daily_error_rates",{}).items():
        print(f"  {b:12s}  ошибка {v['error_rate']*100:.1f}%  (n={v['n_active']})")

    print("\n🚨 Проблемные тикеры (error_rate > 55%):")
    bad = patterns.get("bad_tickers", {})
    if bad:
        for t, v in sorted(bad.items(), key=lambda x: -x[1]["error_rate"]):
            print(f"  {t:6s}  {v['error_rate']*100:.1f}%  (активных сигналов: {v['n_active']})")
    else:
        print("  Нет тикеров с систематически высокой ошибкой")

    print("\n📦 boost_weights сохранены в data/error_patterns.json")
    print("   → используй --boost-errors при следующем обучении")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   required=True, help="rolling_backtest CSV")
    parser.add_argument("--min-n", default=5, type=int,
                        help="Минимум активных сигналов для включения группы")
    parser.add_argument("--out",   default="data/error_patterns.json")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Загружено: {len(df)} строк, {df['ticker'].nunique()} тикеров")
    print(f"Активных сигналов (BUY/SELL): {(df['signal'] != 'HOLD').sum()}")
    print(f"Засчитанных (с actual): {df['actual_delta'].notna().sum()}")

    patterns = analyze(df, min_n=args.min_n)
    print_report(patterns)

    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Сохранено: {args.out}")


if __name__ == "__main__":
    main()
