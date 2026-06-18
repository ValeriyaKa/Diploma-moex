"""
backend/collector/download_csv.py
Скачивает свечи 35 акций MOEX и сохраняет в CSV.
Запуск: python -m backend.collector.download_csv
"""
import asyncio, aiohttp, aiomoex
import pandas as pd, pandas_ta as ta, numpy as np
import logging, os
from datetime import date
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

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

DATE_FROM = "2022-01-01"
DATE_TILL = date.today().isoformat()
os.makedirs("data", exist_ok=True)


async def fetch(session, ticker, interval_code, date_from, date_till):
    try:
        data = await aiomoex.get_board_candles(
            session, security=ticker, interval=interval_code,
            start=date_from, end=date_till, board="TQBR", market="shares",
        )
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["begin"] = pd.to_datetime(df["begin"]).dt.tz_localize("Europe/Moscow")
        df.rename(columns={"begin": "time"}, inplace=True)
        df["ticker"] = ticker
        return df
    except Exception as e:
        log.warning(f"[{ticker}] Ошибка: {e}")
        return pd.DataFrame()


def calc_indicators(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Рассчитывает индикаторы. Названия колонок зависят от версии pandas-ta,
    поэтому ищем по паттерну а не по точному имени.
    """
    d = df.copy().astype({
        "open": float, "high": float, "low": float,
        "close": float, "volume": float,
    })
    d.sort_values("time", inplace=True)

    d["sma_10"] = ta.sma(d["close"], 10)
    d["sma_20"] = ta.sma(d["close"], 20)
    d["sma_50"] = ta.sma(d["close"], 50)
    d["ema_12"] = ta.ema(d["close"], 12)
    d["ema_26"] = ta.ema(d["close"], 26)
    d["rsi_14"] = ta.rsi(d["close"], 14)

    # MACD — ищем колонки по содержимому имени
    macd = ta.macd(d["close"])
    macd_cols = macd.columns.tolist()
    # Колонки выглядят как MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    macd_line   = [c for c in macd_cols if c.startswith("MACD_")][0]
    macd_signal = [c for c in macd_cols if c.startswith("MACDs_")][0]
    macd_hist   = [c for c in macd_cols if c.startswith("MACDh_")][0]
    d["macd"]        = macd[macd_line]
    d["macd_signal"] = macd[macd_signal]
    d["macd_hist"]   = macd[macd_hist]

    # Bollinger Bands — ищем колонки по содержимому
    bb = ta.bbands(d["close"], 20)
    bb_cols = bb.columns.tolist()
    # Колонки: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0 или похожие
    bb_upper  = [c for c in bb_cols if c.startswith("BBU")][0]
    bb_middle = [c for c in bb_cols if c.startswith("BBM")][0]
    bb_lower  = [c for c in bb_cols if c.startswith("BBL")][0]
    d["bb_upper"]  = bb[bb_upper]
    d["bb_middle"] = bb[bb_middle]
    d["bb_lower"]  = bb[bb_lower]

    d["atr_14"]    = ta.atr(d["high"], d["low"], d["close"], 14)
    d["obv"]       = ta.obv(d["close"], d["volume"])
    d["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean()

    d.dropna(subset=["rsi_14"], inplace=True)
    d["ticker"] = ticker
    return d


async def download_all():
    all_candles    = []
    all_indicators = []
    failed = []

    log.info(f"Скачиваю {len(TICKERS)} акций: {DATE_FROM} → {DATE_TILL}\n")

    async with aiohttp.ClientSession() as session:
        for i, ticker in enumerate(TICKERS):
            log.info(f"[{i+1}/{len(TICKERS)}] {ticker}")

            # Дневные свечи
            df_1d = await fetch(session, ticker, 24, DATE_FROM, DATE_TILL)
            if not df_1d.empty:
                df_1d["interval"] = "1d"
                all_candles.append(
                    df_1d[["time","ticker","interval",
                           "open","high","low","close","volume","value"]]
                )
                log.info(f"  1d: {len(df_1d)} свечей")

                # Индикаторы только по дневным
                try:
                    df_ind = calc_indicators(df_1d.copy(), ticker)
                    all_indicators.append(df_ind[[
                        "time","ticker",
                        "sma_10","sma_20","sma_50","ema_12","ema_26",
                        "rsi_14","macd","macd_signal","macd_hist",
                        "bb_upper","bb_middle","bb_lower",
                        "atr_14","obv","vol_ratio",
                    ]])
                except Exception as e:
                    log.warning(f"  Индикаторы ошибка: {e}")
            else:
                log.warning(f"  1d: нет данных")
                failed.append(ticker)

            await asyncio.sleep(0.3)

            # Часовые свечи
            df_1h = await fetch(session, ticker, 60, DATE_FROM, DATE_TILL)
            if not df_1h.empty:
                df_1h["interval"] = "1h"
                all_candles.append(
                    df_1h[["time","ticker","interval",
                           "open","high","low","close","volume","value"]]
                )
                log.info(f"  1h: {len(df_1h)} свечей")

            await asyncio.sleep(0.3)

    # Сохраняем
    log.info("\nСохраняю файлы...")

    if all_candles:
        df_out = pd.concat(all_candles, ignore_index=True)
        df_out["time"] = df_out["time"].astype(str)
        df_out.sort_values(["ticker","interval","time"], inplace=True)
        df_out.to_csv("data/candles_all.csv", index=False)
        n1d = len(df_out[df_out["interval"]=="1d"])
        n1h = len(df_out[df_out["interval"]=="1h"])
        log.info(f"  data/candles_all.csv — {len(df_out):,} строк (1d: {n1d:,} | 1h: {n1h:,})")

    if all_indicators:
        df_ind = pd.concat(all_indicators, ignore_index=True)
        df_ind["time"] = df_ind["time"].astype(str)
        df_ind.sort_values(["ticker","time"], inplace=True)
        df_ind.to_csv("data/indicators.csv", index=False)
        log.info(f"  data/indicators.csv  — {len(df_ind):,} строк")

    log.info(f"\nГотово! Успешно: {len(TICKERS)-len(failed)}/{len(TICKERS)}")
    if failed:
        log.warning(f"Не загрузились: {', '.join(failed)}")

    log.info("""
Следующий шаг — залей CSV в Supabase:
  1. Table Editor → candles   → Insert → Import CSV → candles_all.csv
  2. Table Editor → indicators → Insert → Import CSV → indicators.csv
""")


if __name__ == "__main__":
    asyncio.run(download_all())