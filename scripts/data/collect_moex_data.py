"""
collect_moex_data.py — сбор данных MOEX ISS до сегодняшней даты
и загрузка в Supabase.

Использование:
    pip install aiomoex aiohttp pandas python-dotenv supabase
    python collect_moex_data.py

Что делает:
1. Скачивает дневные свечи для всех 35 тикеров с последней даты в features.csv до сегодня
2. Скачивает часовые свечи (для LSTM hourly)
3. Обновляет features.csv (добавляет новые строки)
4. Загружает в Supabase таблицу moex_candles
"""

import asyncio
import os
import json
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

import aiohttp
import aiomoex
import pandas as pd
import numpy as np

# ─── Конфиг ────────────────────────────────────────────────────────────────
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://ynxqawesgqsrgyhogvgs.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_OtDPuZsNI9ZkfczbELN29g_VW1NqruN")

TICKERS = [
    "AFLT", "ALRS", "BANEP", "CBOM", "CHMF", "FESH", "FIXP", "GAZP",
    "GMKN", "LENT", "LKOH", "MAGN", "MGNT", "MOEX", "MTSS", "NLMK",
    "NVTK", "OZON", "PIKK", "PLZL", "POSI", "ROSN", "RTKM", "SBER",
    "SFIN", "SIBN", "SMLT", "SNGS", "T", "TATN", "UWGN", "VKCO",
    "VTBR", "X5", "YDEX"
]

FEATURES_CSV = os.path.join(os.path.dirname(__file__), "data", "features.csv")
END_DATE = date.today().isoformat()

# ─── Индексы и макро ────────────────────────────────────────────────────────
IMOEX_TICKER = "IMOEX"
USD_RUB_TICKER = "USD000UTSTOM"  # futures или CBRF


# ─── Помощники ──────────────────────────────────────────────────────────────
def calc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет технические индикаторы к DataFrame с колонками [time, close, volume]."""
    df = df.sort_values("time").reset_index(drop=True)

    # RSI-14
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # MACD histogram (12,26,9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - signal

    # Bollinger Bands 20
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    # ATR-14
    df["atr_14"] = (df["close"].rolling(14).max() - df["close"].rolling(14).min())

    # Volume ratio (vs 20-day avg)
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    return df


async def fetch_candles(session: aiohttp.ClientSession,
                        ticker: str,
                        start: str,
                        end: str,
                        interval: int = 24) -> pd.DataFrame:
    """
    Загружает свечи с MOEX ISS API через aiomoex.
    interval=24 → дневные, interval=60 → часовые
    """
    try:
        data = await aiomoex.get_market_candles(
            session, ticker,
            interval=interval,
            start=start,
            end=end,
            market="shares",
            engine="stock",
        )
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df = df.rename(columns={"begin": "time", "close": "close", "volume": "volume"})
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df["ticker"] = ticker
        df["timeframe"] = "1d" if interval == 24 else "1h"
        return df[["time", "close", "volume", "ticker", "timeframe"]]
    except Exception as e:
        print(f"  ⚠️  {ticker} interval={interval}: {e}")
        return pd.DataFrame()


async def fetch_index(session: aiohttp.ClientSession,
                      start: str, end: str) -> pd.DataFrame:
    """Загружает IMOEX дневные значения."""
    try:
        data = await aiomoex.get_market_candles(
            session, IMOEX_TICKER,
            interval=24, start=start, end=end,
            market="index", engine="stock",
        )
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["begin"], utc=True)
        df = df[["time", "close"]].rename(columns={"close": "imoex"})
        return df
    except Exception as e:
        print(f"  ⚠️  IMOEX: {e}")
        return pd.DataFrame()


async def fetch_usdrub(session: aiohttp.ClientSession,
                       start: str, end: str) -> pd.DataFrame:
    """Загружает USD/RUB из MOEX (инструмент USD000UTSTOM, рынок фьючерсов)."""
    try:
        data = await aiomoex.get_market_candles(
            session, USD_RUB_TICKER,
            interval=24, start=start, end=end,
            market="selt", engine="currency",
        )
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["begin"], utc=True)
        df = df[["time", "close"]].rename(columns={"close": "usd_rub"})
        return df
    except Exception as e:
        print(f"  ⚠️  USD/RUB: {e}")
        return pd.DataFrame()


async def collect_all(start: str, end: str) -> pd.DataFrame:
    """Собирает данные для всех тикеров за период [start, end]."""
    print(f"📡 Сбор данных MOEX с {start} по {end}...")
    print(f"   Тикеров: {len(TICKERS)}")

    async with aiohttp.ClientSession() as session:
        # Индексы
        print("  → IMOEX и USD/RUB...")
        imoex_df = await fetch_index(session, start, end)
        usdrub_df = await fetch_usdrub(session, start, end)

        # Дневные свечи по всем тикерам
        print("  → Дневные свечи...")
        tasks_d = [fetch_candles(session, t, start, end, 24) for t in TICKERS]
        results_d = await asyncio.gather(*tasks_d)

        # Часовые свечи (для LSTM hourly, если используется)
        print("  → Часовые свечи...")
        tasks_h = [fetch_candles(session, t, start, end, 60) for t in TICKERS]
        results_h = await asyncio.gather(*tasks_h)

    # Склеиваем дневные
    daily_dfs = [r for r in results_d if not r.empty]
    hourly_dfs = [r for r in results_h if not r.empty]

    if not daily_dfs:
        print("❌ Не получены дневные данные!")
        return pd.DataFrame()

    df_all = pd.concat(daily_dfs + hourly_dfs, ignore_index=True)
    df_all = df_all.sort_values(["ticker", "time"])

    # Добавляем макро-данные
    if not imoex_df.empty:
        df_all = df_all.merge(imoex_df, on="time", how="left")
        df_all["imoex"] = df_all["imoex"].ffill()
    else:
        df_all["imoex"] = np.nan

    if not usdrub_df.empty:
        df_all = df_all.merge(usdrub_df, on="time", how="left")
        df_all["usd_rub"] = df_all["usd_rub"].ffill()
    else:
        df_all["usd_rub"] = np.nan

    # Технические индикаторы (по тикеру + timeframe)
    print("  → Расчёт технических индикаторов...")
    groups = []
    for (ticker, tf), grp in df_all.groupby(["ticker", "timeframe"]):
        grp_feat = calc_features(grp.copy())
        groups.append(grp_feat)
    df_all = pd.concat(groups, ignore_index=True)

    print(f"✅ Собрано: {len(df_all):,} строк для {df_all['ticker'].nunique()} тикеров")
    return df_all


def update_features_csv(new_data: pd.DataFrame) -> pd.DataFrame:
    """Добавляет новые строки в features.csv (не дублирует существующие)."""
    if os.path.exists(FEATURES_CSV):
        existing = pd.read_csv(FEATURES_CSV)
        existing["time"] = pd.to_datetime(existing["time"], format="ISO8601", utc=True)
        new_data["time"] = pd.to_datetime(new_data["time"], format="ISO8601", utc=True)

        # Убираем пересечения
        existing_keys = set(zip(existing["time"].astype(str), existing["ticker"],
                                existing["timeframe"]))
        mask = ~new_data.apply(
            lambda r: (str(r["time"]), r["ticker"], r["timeframe"]) in existing_keys,
            axis=1
        )
        truly_new = new_data[mask]
        print(f"📝 Новых строк для добавления: {len(truly_new):,}")

        if len(truly_new) > 0:
            combined = pd.concat([existing, truly_new], ignore_index=True)
            combined = combined.sort_values(["ticker", "timeframe", "time"])
            combined.to_csv(FEATURES_CSV, index=False)
            print(f"✅ features.csv обновлён: {len(combined):,} строк")
        else:
            combined = existing
            print("✅ Данные уже актуальны, новых строк нет")
        return combined
    else:
        os.makedirs(os.path.dirname(FEATURES_CSV), exist_ok=True)
        new_data.to_csv(FEATURES_CSV, index=False)
        print(f"✅ features.csv создан: {len(new_data):,} строк")
        return new_data


def upload_to_supabase(df: pd.DataFrame, table: str = "moex_candles"):
    """Загружает данные в Supabase через REST API (upsert)."""
    try:
        from supabase import create_client
    except ImportError:
        print("⚠️  supabase-py не установлен: pip install supabase")
        print("   Данные сохранены только в features.csv")
        return

    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)

        # Конвертируем для JSON
        upload_df = df.copy()
        upload_df["time"] = upload_df["time"].astype(str)
        upload_df = upload_df.where(pd.notna(upload_df), None)

        records = upload_df.to_dict(orient="records")
        batch_size = 500
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            client.table(table).upsert(
                batch,
                on_conflict="time,ticker,timeframe"
            ).execute()
            total += len(batch)
            print(f"  ↑ Supabase: загружено {total}/{len(records)}")

        print(f"✅ Supabase: всего загружено {total:,} записей в таблицу '{table}'")
    except Exception as e:
        print(f"⚠️  Ошибка загрузки в Supabase: {e}")
        print("   Данные сохранены только в features.csv")


def get_last_date_in_features() -> str:
    """Возвращает последнюю дату в features.csv для определения start периода."""
    if not os.path.exists(FEATURES_CSV):
        return "2022-01-01"
    df = pd.read_csv(FEATURES_CSV, usecols=["time", "timeframe"])
    daily = df[df["timeframe"] == "1d"]["time"]
    if daily.empty:
        return "2022-01-01"
    last = pd.to_datetime(daily).max()
    # Берём за 30 дней до последней даты для пересчёта индикаторов
    start = (last - timedelta(days=30)).strftime("%Y-%m-%d")
    print(f"📅 Последняя дата в features.csv: {last.date()}, сбор с {start}")
    return start


async def main():
    print("=" * 60)
    print("  MOEX Data Collector")
    print(f"  End date: {END_DATE}")
    print("=" * 60)

    start = get_last_date_in_features()
    new_data = await collect_all(start, END_DATE)

    if new_data.empty:
        print("❌ Данные не получены. Проверь соединение с интернетом.")
        return

    # Обновляем CSV
    update_features_csv(new_data)

    # Загружаем в Supabase
    print("\n📤 Загрузка в Supabase...")
    upload_to_supabase(new_data)

    print("\n🎉 Готово!")
    print(f"   features.csv: актуальные данные до {END_DATE}")
    print(f"   Supabase таблица: moex_candles")


if __name__ == "__main__":
    asyncio.run(main())
