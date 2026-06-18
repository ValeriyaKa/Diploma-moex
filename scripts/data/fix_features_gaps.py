"""
fix_features_gaps.py — Исправление пропусков и переименований тикеров в features.csv

Что делает:
1. YDEX (Яндекс): заполняет пропуск июль 2024 данными YNDX с MOEX API
   (YNDX → YDEX: редомициляция завершилась, старый тикер YNDX делистингован)
2. FIXP (Fix Price): forward-fill для пропуска июнь–август 2025
   (торги были приостановлены; заполняем последней известной ценой)
3. UWGN: оставляем как есть (IPO декабрь 2024, исторических данных нет)
4. X5: оставляем как есть (ре-листинг после реструктуризации)

Использование:
    pip install aiomoex aiohttp
    python fix_features_gaps.py
    python fix_features_gaps.py --dry-run   # только показать что изменится
"""

import argparse
import asyncio
from datetime import timedelta
from pathlib import Path

import aiohttp
import aiomoex
import numpy as np
import pandas as pd

FEATURES_CSV = Path("data") / "features.csv"


# ─── Расчёт фичей ──────────────────────────────────────────────────────────
def calc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет технические индикаторы к DataFrame."""
    df = df.sort_values("time").reset_index(drop=True)

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    df["atr_14"] = df["close"].rolling(14).max() - df["close"].rolling(14).min()
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    return df


# ─── Загрузка исторических данных с MOEX ───────────────────────────────────
async def fetch_candles_moex(ticker: str, start: str, end: str,
                              interval: int = 24) -> pd.DataFrame:
    """Скачивает свечи с MOEX ISS API."""
    async with aiohttp.ClientSession() as session:
        try:
            data = await aiomoex.get_market_candles(
                session, ticker, interval=interval,
                start=start, end=end,
                market="shares", engine="stock",
            )
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df = df.rename(columns={"begin": "time"})
            df["time"] = pd.to_datetime(df["time"], utc=True)
            return df[["time", "close", "volume"]]
        except Exception as e:
            print(f"  ⚠️  {ticker}: {e}")
            return pd.DataFrame()


# ─── 1. YDEX: заполнить пропуск из YNDX ────────────────────────────────────
async def fix_ydex(features: pd.DataFrame, dry_run: bool = False) -> pd.DataFrame:
    """
    Пропуск YDEX ~41 день (2024-07-08 по 2024-08-16).
    Заполняем данными старого тикера YNDX за тот же период.
    """
    print("\n🔧 YDEX: анализ пропуска...")
    ydex_daily = features[
        (features["ticker"] == "YDEX") & (features["timeframe"] == "1d")
    ].sort_values("time")

    # Находим пропуски больше 10 рабочих дней
    ydex_daily = ydex_daily.copy()
    ydex_daily["gap"] = ydex_daily["time"].diff().dt.days
    gaps = ydex_daily[ydex_daily["gap"] > 10]

    if gaps.empty:
        print("  ✅ YDEX: пропусков нет, всё в порядке")
        return features

    for _, row in gaps.iterrows():
        gap_end = row["time"]
        gap_start = gap_end - timedelta(days=int(row["gap"]))
        print(f"  Пропуск: {gap_start.date()} → {gap_end.date()} ({int(row['gap'])} дней)")

        if dry_run:
            print(f"  [dry-run] Скачаем YNDX с {gap_start.date()} по {gap_end.date()}")
            continue

        # Скачиваем YNDX за период пропуска
        start_str = (gap_start - timedelta(days=30)).strftime("%Y-%m-%d")
        end_str = gap_end.strftime("%Y-%m-%d")
        print(f"  → Скачиваем YNDX ({start_str} → {end_str})...")

        yndx_df = await fetch_candles_moex("YNDX", start_str, end_str)
        if yndx_df.empty:
            print("  ⚠️  YNDX: данные не получены, пропуск остаётся")
            continue

        # Оставляем только строки в диапазоне пропуска
        mask = (yndx_df["time"] >= gap_start) & (yndx_df["time"] < gap_end)
        fill_rows = yndx_df[mask].copy()
        if fill_rows.empty:
            print("  ⚠️  YNDX: нет данных в период пропуска")
            continue

        print(f"  ✅ Получено {len(fill_rows)} строк из YNDX для заполнения пропуска")

        # Нужны imoex и usd_rub — берём из соседних строк YDEX
        ydex_macro = features[
            (features["ticker"] == "YDEX") & (features["timeframe"] == "1d")
        ][["time", "imoex", "usd_rub"]].sort_values("time")

        fill_rows["ticker"] = "YDEX"
        fill_rows["timeframe"] = "1d"

        # Заполняем макро данными из ближайшего дня
        fill_rows = fill_rows.merge(
            ydex_macro.rename(columns={"imoex": "imoex_src", "usd_rub": "usd_rub_src"}),
            on="time", how="left"
        )
        fill_rows["imoex"] = fill_rows["imoex_src"].combine_first(
            ydex_macro["imoex"].iloc[0]  # крайнее значение
        )
        fill_rows["usd_rub"] = fill_rows["usd_rub_src"].combine_first(
            ydex_macro["usd_rub"].iloc[0]
        )

        # Пересчитываем технические индикаторы на полной истории
        all_ydex = pd.concat([
            features[(features["ticker"] == "YDEX") & (features["timeframe"] == "1d")],
            fill_rows[["time", "close", "volume", "ticker", "timeframe", "imoex", "usd_rub"]]
        ]).sort_values("time").drop_duplicates("time")

        all_ydex = calc_features(all_ydex)

        # Убираем старые YDEX 1d из features и добавляем пересчитанные
        features = features[
            ~((features["ticker"] == "YDEX") & (features["timeframe"] == "1d"))
        ]
        features = pd.concat([features, all_ydex], ignore_index=True)
        print(f"  ✅ YDEX дополнен: +{len(fill_rows)} строк из YNDX")

    return features


# ─── 2. FIXP: forward-fill пропуска ────────────────────────────────────────
def fix_fixp(features: pd.DataFrame, dry_run: bool = False) -> pd.DataFrame:
    """
    Пропуск FIXP ~61 день (2025-06-20 → 2025-08-20).
    Fix Price была приостановлена к торгам — исторических данных нет.
    Заполняем последней известной ценой (forward-fill), объём = 0.
    Это позволяет модели "видеть" тикер, но с нейтральным сигналом.
    """
    print("\n🔧 FIXP: анализ пропуска...")
    fixp_daily = features[
        (features["ticker"] == "FIXP") & (features["timeframe"] == "1d")
    ].sort_values("time").copy()

    fixp_daily["gap"] = fixp_daily["time"].diff().dt.days
    gaps = fixp_daily[fixp_daily["gap"] > 10]

    if gaps.empty:
        print("  ✅ FIXP: пропусков нет")
        return features

    for _, row in gaps.iterrows():
        gap_end = row["time"]
        gap_start = gap_end - timedelta(days=int(row["gap"]))
        print(f"  Пропуск: {gap_start.date()} → {gap_end.date()} ({int(row['gap'])} дней)")

        # Последняя известная цена до пропуска
        before_gap = fixp_daily[fixp_daily["time"] < gap_start]
        if before_gap.empty:
            continue
        last_row = before_gap.iloc[-1]
        last_close = float(last_row["close"])
        print(f"  Последняя цена перед пропуском: {last_close}")

        if dry_run:
            print(f"  [dry-run] Forward-fill {last_close} для {int(row['gap'])-1} дней")
            continue

        # Генерируем даты-заполнители (рабочие дни)
        fill_dates = pd.date_range(
            start=gap_start + timedelta(days=1),
            end=gap_end - timedelta(days=1),
            freq="B",  # business days
        ).tz_localize("UTC")

        if fill_dates.empty:
            continue

        # Макро: берём из других тикеров за те же даты
        macro_ref = features[
            (features["ticker"] == "SBER") & (features["timeframe"] == "1d")
        ][["time", "imoex", "usd_rub"]].sort_values("time")
        macro_ref = macro_ref.set_index("time")

        fill_rows = []
        for dt in fill_dates:
            # Найти ближайший макро
            imoex_val = np.nan
            usd_rub_val = np.nan
            if not macro_ref.empty:
                idx = macro_ref.index.searchsorted(dt)
                if idx < len(macro_ref):
                    imoex_val = macro_ref.iloc[min(idx, len(macro_ref)-1)]["imoex"]
                    usd_rub_val = macro_ref.iloc[min(idx, len(macro_ref)-1)]["usd_rub"]

            fill_rows.append({
                "time": dt,
                "close": last_close,
                "volume": 0.0,          # нет торгов
                "ticker": "FIXP",
                "timeframe": "1d",
                "imoex": imoex_val,
                "usd_rub": usd_rub_val,
            })

        fill_df = pd.DataFrame(fill_rows)

        # Пересчитываем индикаторы на полной истории FIXP
        all_fixp = pd.concat([
            features[(features["ticker"] == "FIXP") & (features["timeframe"] == "1d")],
            fill_df
        ]).sort_values("time").drop_duplicates("time")
        all_fixp = calc_features(all_fixp)

        features = features[
            ~((features["ticker"] == "FIXP") & (features["timeframe"] == "1d"))
        ]
        features = pd.concat([features, all_fixp], ignore_index=True)
        print(f"  ✅ FIXP: добавлено {len(fill_rows)} строк forward-fill")

    return features


# ─── 3. UWGN: информационное сообщение ─────────────────────────────────────
def report_uwgn(features: pd.DataFrame):
    uwgn = features[(features["ticker"] == "UWGN") & (features["timeframe"] == "1d")]
    print(f"\nℹ️  UWGN: {len(uwgn)} строк с {uwgn['time'].min().date()} — это IPO декабря 2024.")
    print("   Исторических данных нет. UWGN будет обучаться только на коротком периоде.")
    print("   Рекомендация: добавить в TICKERS только если > 250 торговых дней.")


# ─── Главная функция ────────────────────────────────────────────────────────
async def main(dry_run: bool):
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"features.csv не найден: {FEATURES_CSV}")

    print(f"📂 Загружаем {FEATURES_CSV}...")
    features = pd.read_csv(FEATURES_CSV)
    features["time"] = pd.to_datetime(features["time"], format="ISO8601", utc=True)

    rows_before = len(features)
    print(f"   Строк до обработки: {rows_before:,}")

    # 1. YDEX
    features = await fix_ydex(features, dry_run)

    # 2. FIXP
    features = fix_fixp(features, dry_run)

    # 3. UWGN (только информация)
    report_uwgn(features)

    if not dry_run:
        # Сохраняем
        features = features.sort_values(["ticker", "timeframe", "time"])
        features.to_csv(FEATURES_CSV, index=False)
        rows_after = len(features)
        print(f"\n✅ features.csv сохранён: {rows_before:,} → {rows_after:,} строк (+{rows_after-rows_before})")
    else:
        print("\n[dry-run] features.csv не изменён")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Только показать что изменится, не сохранять")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run))
