"""
backend/collector/moex.py
═══════════════════════════════════════════════════════════════
Сборщик данных с MOEX ISS API + расчёт индикаторов.
"""
from __future__ import annotations
import asyncio, logging, os
from datetime import date, timedelta
from typing import Optional

import aiohttp, aiomoex
import numpy as np
import pandas as pd
import pandas_ta as ta

from backend.db.database import upsert_candles, upsert_indicators

log = logging.getLogger(__name__)
REQUEST_DELAY = float(os.environ.get("MOEX_REQUEST_DELAY", 0.3))
MAX_FETCH_RETRIES = int(os.environ.get("MOEX_FETCH_RETRIES", 3))
FETCH_RETRY_DELAY = float(os.environ.get("MOEX_FETCH_RETRY_DELAY", 2.0))

TICKERS = [
    "GAZP","LKOH","NVTK","ROSN","TATN","SNGS","SIBN","BANEP",
    "SBER","VTBR","T","MOEX","SFIN","CBOM",
    "GMKN","PLZL","ALRS","CHMF","NLMK","MAGN",
    "YDEX","OZON","VKCO","POSI",
    "MGNT","X5","LENT","FIXP",
    "MTSS","RTKM",
    "PIKK","SMLT",
    "AFLT","FESH","UWGN",
]

INTERVAL_CODES = {"1d": 24, "1h": 60}


async def _fetch_candles(session, ticker, interval, date_from, date_till):
    import traceback
    for attempt in range(1, MAX_FETCH_RETRIES + 1):
        try:
            data = await aiomoex.get_board_candles(
                session, security=ticker, interval=interval,
                start=date_from, end=date_till,
                board="TQBR", market="shares",
            )
        except Exception as e:
            log.warning(f"[{ticker}] ISS ошибка (attempt {attempt}/{MAX_FETCH_RETRIES}): {e}")
            if attempt < MAX_FETCH_RETRIES:
                await asyncio.sleep(FETCH_RETRY_DELAY)
                continue
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        try:
            df = pd.DataFrame(data)
            log.debug(f"[{ticker}] columns={list(df.columns)}, rows={len(df)}")
            if "begin" not in df.columns:
                log.error(f"[{ticker}] No 'begin' column. Columns: {list(df.columns)}")
                return pd.DataFrame()
            df["begin"] = pd.to_datetime(df["begin"]).dt.tz_localize("Europe/Moscow")
            df.rename(columns={"begin": "time"}, inplace=True)
            df["ticker"] = ticker
            missing = [c for c in ["time","ticker","open","high","low","close","volume","value"] if c not in df.columns]
            if missing:
                log.warning(f"[{ticker}] Missing columns {missing}, available: {list(df.columns)}")
                for c in missing:
                    df[c] = None
            return df[["time","ticker","open","high","low","close","volume","value"]]
        except Exception:
            log.error(f"[{ticker}] DataFrame parse error:\n{traceback.format_exc()}")
            return pd.DataFrame()

    return pd.DataFrame()


def _calc_indicators(df: pd.DataFrame, ticker: str) -> list[tuple]:
    d = df.copy().astype({
        "open": float, "high": float, "low": float,
        "close": float, "volume": float,
    })
    d.sort_values("time", inplace=True)

    # Нужно минимум 50 строк для SMA-50
    if len(d) < 50:
        return []

    d["sma_10"] = ta.sma(d["close"], 10)
    d["sma_20"] = ta.sma(d["close"], 20)
    d["sma_50"] = ta.sma(d["close"], 50)
    d["ema_12"] = ta.ema(d["close"], 12)
    d["ema_26"] = ta.ema(d["close"], 26)
    d["rsi_14"] = ta.rsi(d["close"], 14)

    # MACD — проверяем что результат не None
    macd = ta.macd(d["close"])
    if macd is None or macd.empty:
        return []
    macd_cols = macd.columns.tolist()
    d["macd"]        = macd[[c for c in macd_cols if c.startswith("MACD_")][0]]
    d["macd_signal"] = macd[[c for c in macd_cols if c.startswith("MACDs_")][0]]
    d["macd_hist"]   = macd[[c for c in macd_cols if c.startswith("MACDh_")][0]]

    # Bollinger Bands — проверяем что результат не None
    bb = ta.bbands(d["close"], 20)
    if bb is None or bb.empty:
        return []
    bb_cols = bb.columns.tolist()
    d["bb_upper"]  = bb[[c for c in bb_cols if c.startswith("BBU")][0]]
    d["bb_middle"] = bb[[c for c in bb_cols if c.startswith("BBM")][0]]
    d["bb_lower"]  = bb[[c for c in bb_cols if c.startswith("BBL")][0]]

    d["atr_14"]    = ta.atr(d["high"], d["low"], d["close"], 14)
    d["obv"]       = ta.obv(d["close"], d["volume"])
    d["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean()

    d.dropna(subset=["rsi_14"], inplace=True)

    def _safe(v):
        try:
            x = float(v)
            return None if np.isnan(x) else round(x, 6)
        except Exception:
            return None

    records = []
    for _, row in d.iterrows():
        records.append((
            row["time"], ticker,
            _safe(row["sma_10"]),   _safe(row["sma_20"]),
            _safe(row["sma_50"]),   _safe(row["ema_12"]),
            _safe(row["ema_26"]),   _safe(row["rsi_14"]),
            _safe(row["macd"]),     _safe(row["macd_signal"]),
            _safe(row["macd_hist"]),
            _safe(row["bb_upper"]), _safe(row["bb_middle"]),
            _safe(row["bb_lower"]), _safe(row["atr_14"]),
            int(row["obv"]) if not np.isnan(row["obv"]) else None,
            _safe(row["vol_ratio"]),
        ))
    return records


async def load_history(ticker, date_from="2022-01-01", date_till=None):
    if date_till is None:
        date_till = date.today().isoformat()

    log.info(f"[{ticker}] Загрузка {date_from} → {date_till}")

    async with aiohttp.ClientSession() as session:
        daily_df = pd.DataFrame()

        for iname, icode in INTERVAL_CODES.items():
            df = await _fetch_candles(session, ticker, icode, date_from, date_till)
            if df.empty:
                log.warning(f"[{ticker}] Нет данных для {iname}")
                continue

            df["interval"] = iname
            records = [
                (row["time"], row["ticker"], row["interval"],
                 float(row["open"]),  float(row["high"]),
                 float(row["low"]),   float(row["close"]),
                 int(row["volume"]),
                 float(row["value"]) if not pd.isna(row["value"]) else None)
                for _, row in df.iterrows()
            ]
            n = await upsert_candles(records)
            log.info(f"[{ticker}] {iname}: {n} свечей")

            if iname == "1d":
                daily_df = df
            await asyncio.sleep(REQUEST_DELAY)

        if not daily_df.empty:
            ind_records = _calc_indicators(daily_df, ticker)
            await upsert_indicators(ind_records)
            log.info(f"[{ticker}] Индикаторы: {len(ind_records)} записей")


async def load_history_all(date_from="2022-01-01"):
    for i, ticker in enumerate(TICKERS):
        log.info(f"[{i+1}/{len(TICKERS)}] {ticker}")
        await load_history(ticker, date_from)
        await asyncio.sleep(REQUEST_DELAY * 2)


async def daily_update():
    date_from = (date.today() - timedelta(days=1)).isoformat()
    date_till = date.today().isoformat()
    log.info(f"=== Обновление {date_from} → {date_till} ===")

    for ticker in TICKERS:
        try:
            total_1d = 0
            total_1h = 0
            connector = aiohttp.TCPConnector(limit_per_host=1, keepalive_timeout=30)
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                for iname, icode in INTERVAL_CODES.items():
                    df = await _fetch_candles(session, ticker, icode, date_from, date_till)
                    if df.empty:
                        log.info(f"[{ticker}] Нет новых данных для {iname}")
                        continue

                    df["interval"] = iname
                    records = [
                        (row["time"], row["ticker"], iname,
                         float(row["open"]),  float(row["high"]),
                         float(row["low"]),   float(row["close"]),
                         int(row["volume"]),
                         float(row["value"]) if not pd.isna(row["value"]) else None)
                        for _, row in df.iterrows()
                    ]
                    await upsert_candles(records)

                    if iname == "1d":
                        ind = _calc_indicators(df, ticker)
                        await upsert_indicators(ind)
                        total_1d += len(records)
                        log.info(f"[{ticker}] ✓ {len(records)} дневных свечей, {len(ind)} индикаторов")
                    else:
                        total_1h += len(records)
                        log.info(f"[{ticker}] ✓ {len(records)} часовых свечей")

                    await asyncio.sleep(REQUEST_DELAY)

                if total_1d == 0 and total_1h == 0:
                    log.info(f"[{ticker}] Нет новых данных")
        except Exception as e:
            log.error(f"[{ticker}] ✗ {e}")

    log.info("=== Обновление завершено ===")