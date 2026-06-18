"""
wfv_multifold.py — Многофолдовая Walk-Forward Validation
с детальным анализом ошибок по каждому фолду.

КРИТИЧЕСКИ ВАЖНО: используется только для АНАЛИЗА и ДИАГНОСТИКИ.
Точность оценивается ТОЛЬКО на строках, где actual_close != current_close
(т.е. где данные о следующем дне действительно были собраны).

Использование:
    python wfv_multifold.py --csv data/rolling_backtest_*.csv
    python wfv_multifold.py --csv data/rolling_backtest_2025-09-01_2026-04-01.csv \\
                             data/rolling_backtest_2026-04-01_2026-05-26.csv

Выводит:
- Точность модели по фолдам
- Лучшие/худшие тикеры
- Анализ ошибок по дням недели, месяцам, уровням уверенности
- Рекомендации по фильтрации сигналов
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


# ─── CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation Analysis")
    parser.add_argument("--csv", nargs="+", required=True,
                        help="Один или несколько CSV файлов бэктеста")
    parser.add_argument("--folds", type=int, default=4,
                        help="Количество временны́х фолдов (default: 4)")
    parser.add_argument("--conf-min", type=float, default=0.55,
                        help="Минимальный порог confidence (default: 0.55)")
    parser.add_argument("--out", type=str, default="data/wfv_analysis.json",
                        help="Путь для JSON-отчёта")
    return parser.parse_args()


# ─── Загрузка и очистка данных ──────────────────────────────────────────────
def load_backtest(csv_paths: list[str]) -> pd.DataFrame:
    """Загружает и соединяет CSV-файлы бэктеста."""
    dfs = []
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"⚠️  Файл не найден: {path}")
            continue
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"  Loaded {path}: {len(df):,} rows")

    if not dfs:
        raise FileNotFoundError("Ни один CSV файл не загружен")

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["as_of", "ticker"])

    # Парсим даты
    df["as_of"] = pd.to_datetime(df["as_of"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["dow"] = df["as_of"].dt.day_name()
    df["month"] = df["as_of"].dt.month
    df["year"] = df["as_of"].dt.year

    # КРИТИЧЕСКИ ВАЖНО: помечаем строки с ВАЛИДНЫМИ данными о фактической цене
    # actual_delta == 0 И actual_close == current_close → данные не были собраны
    if "actual_delta" in df.columns and "actual_close" in df.columns:
        df["has_valid_actual"] = (
            (df["actual_close"].notna()) &
            (df["actual_close"] != df["current_close"])
        )
    else:
        df["has_valid_actual"] = False

    total = len(df)
    valid = df["has_valid_actual"].sum()
    print(f"\n📊 Всего строк: {total:,}")
    print(f"   Строк с ВАЛИДНЫМ actual_close: {valid:,} ({valid/total*100:.1f}%)")
    print(f"   ⚠️  Остальные {total-valid:,} строк оценить невозможно (actual == current)")

    return df


# ─── Функции анализа ────────────────────────────────────────────────────────
def accuracy_on_valid(df: pd.DataFrame, label: str = "") -> dict:
    """
    Возвращает точность ТОЛЬКО на строках с валидным actual_close.
    Это единственная честная метрика модели.
    """
    valid = df[df["has_valid_actual"]]
    active_valid = valid[valid["signal"] != "HOLD"]

    if len(active_valid) == 0:
        return {"label": label, "n_valid": 0, "accuracy": None}

    acc = active_valid["is_correct"].mean()
    dir_correct = (
        ((active_valid["signal"] == "BUY") & (active_valid["actual_delta"] > 0)) |
        ((active_valid["signal"] == "SELL") & (active_valid["actual_delta"] < 0))
    ).mean()

    buy_acc = active_valid[active_valid["signal"] == "BUY"]["is_correct"].mean() if (active_valid["signal"] == "BUY").any() else None
    sell_acc = active_valid[active_valid["signal"] == "SELL"]["is_correct"].mean() if (active_valid["signal"] == "SELL").any() else None

    return {
        "label": label,
        "n_valid_rows": len(valid),
        "n_active_valid": len(active_valid),
        "accuracy_vs_delta_min": round(float(acc), 4),
        "directional_accuracy": round(float(dir_correct), 4),
        "buy_accuracy": round(float(buy_acc), 4) if buy_acc is not None else None,
        "sell_accuracy": round(float(sell_acc), 4) if sell_acc is not None else None,
    }


def fold_analysis(df: pd.DataFrame, n_folds: int = 4) -> list[dict]:
    """
    Разбивает данные на n_folds последовательных временны́х фолдов.
    Каждый фолд оценивается независимо.
    """
    print(f"\n🔀 Walk-Forward Validation: {n_folds} фолдов")
    df_sorted = df.sort_values("as_of")
    dates = df_sorted["as_of"].unique()

    fold_size = len(dates) // n_folds
    results = []

    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(dates)
        fold_dates = dates[start_idx:end_idx]

        fold_df = df_sorted[df_sorted["as_of"].isin(fold_dates)]
        fold_start = pd.Timestamp(fold_dates[0]).strftime("%Y-%m-%d")
        fold_end = pd.Timestamp(fold_dates[-1]).strftime("%Y-%m-%d")
        label = f"Fold {i+1}: {fold_start} → {fold_end}"

        stats = accuracy_on_valid(fold_df, label)
        stats["n_total_rows"] = len(fold_df)
        stats["n_active_signals"] = (fold_df["signal"] != "HOLD").sum()
        stats["coverage"] = round((fold_df["signal"] != "HOLD").mean(), 4)

        results.append(stats)
        print(f"  {label}")
        print(f"    Строк: {stats['n_total_rows']:,} | Активных: {stats['n_active_signals']}")
        if stats["n_valid_rows"] > 0:
            print(f"    Точность (valid): {stats['accuracy_vs_delta_min']:.1%} | Направление: {stats['directional_accuracy']:.1%}")
        else:
            print(f"    ⚠️  Нет валидных строк для оценки")

    return results


def ticker_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Анализ точности по каждому тикеру (на валидных строках)."""
    valid = df[df["has_valid_actual"] & (df["signal"] != "HOLD")]
    if valid.empty:
        return pd.DataFrame()

    stats = valid.groupby("ticker").agg(
        n_signals=("is_correct", "count"),
        accuracy=("is_correct", "mean"),
        directional=("actual_delta", lambda x: (
            ((valid.loc[x.index, "signal"] == "BUY") & (x > 0)) |
            ((valid.loc[x.index, "signal"] == "SELL") & (x < 0))
        ).mean()),
        avg_confidence=("confidence", "mean"),
    ).round(4).sort_values("accuracy", ascending=False)

    return stats


def confidence_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Анализ точности по уровням уверенности модели."""
    valid = df[df["has_valid_actual"] & (df["signal"] != "HOLD")].copy()
    if valid.empty:
        return pd.DataFrame()

    bins = [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 1.0]
    valid["conf_bucket"] = pd.cut(valid["confidence"], bins=bins)
    return valid.groupby("conf_bucket", observed=False)["is_correct"].agg(["mean", "count"]).round(4)


def dow_analysis(df: pd.DataFrame) -> dict:
    """Анализ точности по дням недели."""
    valid = df[df["has_valid_actual"] & (df["signal"] != "HOLD")]
    if valid.empty:
        return {}
    return valid.groupby("dow")["is_correct"].agg(["mean", "count"]).round(4).to_dict()


def signal_filter_recommendations(df: pd.DataFrame, conf_threshold: float = 0.55) -> dict:
    """
    Вырабатывает рекомендации по фильтрации сигналов на основе анализа.
    """
    valid = df[df["has_valid_actual"] & (df["signal"] != "HOLD")]
    recs = {}

    if valid.empty:
        return {"note": "Нет валидных данных для рекомендаций"}

    # Тикеры с accuracy < 0.35 → исключить
    ticker_acc = valid.groupby("ticker")["is_correct"].agg(["mean", "count"])
    bad_tickers = ticker_acc[
        (ticker_acc["mean"] < 0.35) & (ticker_acc["count"] >= 5)
    ].index.tolist()
    good_tickers = ticker_acc[
        (ticker_acc["mean"] >= 0.45) & (ticker_acc["count"] >= 5)
    ].index.tolist()

    recs["exclude_tickers"] = bad_tickers
    recs["prefer_tickers"] = good_tickers

    # Лучший порог confidence
    bins = [0.50, 0.55, 0.58, 0.60, 0.63, 0.65, 1.0]
    valid_copy = valid.copy()
    valid_copy["conf_bin"] = pd.cut(valid_copy["confidence"], bins=bins)
    conf_acc = valid_copy.groupby("conf_bin", observed=False)["is_correct"].agg(["mean", "count"])
    best_conf_rows = conf_acc[conf_acc["count"] >= 5].sort_values("mean", ascending=False)
    if not best_conf_rows.empty:
        best_bin = str(best_conf_rows.index[0])
        recs["best_confidence_range"] = best_bin
        recs["best_confidence_accuracy"] = round(float(best_conf_rows.iloc[0]["mean"]), 4)

    # Лучшие дни недели
    dow_acc = valid.groupby("dow")["is_correct"].agg(["mean", "count"])
    best_days = dow_acc[
        (dow_acc["mean"] >= 0.45) & (dow_acc["count"] >= 5)
    ].index.tolist()
    recs["best_days_of_week"] = best_days

    # Потенциальный прирост от фильтрации
    filtered = valid[
        (~valid["ticker"].isin(bad_tickers)) &
        (valid["confidence"] >= conf_threshold)
    ]
    if len(filtered) > 0:
        recs["filtered_accuracy"] = round(float(filtered["is_correct"].mean()), 4)
        recs["filtered_n_signals"] = len(filtered)
        recs["baseline_accuracy"] = round(float(valid["is_correct"].mean()), 4)
        recs["accuracy_gain"] = round(
            recs["filtered_accuracy"] - recs["baseline_accuracy"], 4
        )

    return recs


# ─── Главная функция ─────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 65)
    print("  Multi-Fold Walk-Forward Validation Analysis")
    print("=" * 65)

    # Загружаем данные
    print("\n📂 Загрузка данных...")
    df = load_backtest(args.csv)

    # Общая точность
    print("\n📈 Общая точность на валидных строках:")
    overall = accuracy_on_valid(df, "Всего (оба периода)")
    for k, v in overall.items():
        print(f"   {k}: {v}")

    # Фолды
    folds = fold_analysis(df, args.folds)

    # По тикерам
    print("\n🏆 Точность по тикерам (TOP-10):")
    ticker_stats = ticker_analysis(df)
    if not ticker_stats.empty:
        print(ticker_stats.head(10).to_string())
        print("\n⚠️  Худшие тикеры:")
        print(ticker_stats.tail(10).to_string())

    # Рекомендации
    print("\n💡 Рекомендации по фильтрации:")
    recs = signal_filter_recommendations(df, args.conf_min)
    for k, v in recs.items():
        print(f"   {k}: {v}")

    # Confidence analysis
    print("\n📊 Точность по confidence:")
    conf_stats = confidence_analysis(df)
    if not conf_stats.empty:
        print(conf_stats.to_string())

    # Сохраняем отчёт
    report = {
        "generated_at": datetime.now().isoformat(),
        "csv_files": args.csv,
        "overall_accuracy": overall,
        "folds": folds,
        "ticker_accuracy": ticker_stats.reset_index().to_dict(orient="records") if not ticker_stats.empty else [],
        "confidence_accuracy": conf_stats.reset_index().to_dict(orient="records") if not conf_stats.empty else [],
        "recommendations": recs,
    }

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✅ Отчёт сохранён: {args.out}")


if __name__ == "__main__":
    main()
