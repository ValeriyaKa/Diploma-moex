"""
load_candles.py — простая синхронная загрузка свечей с MOEX ISS API.
Обходит проблемный async-коллектор. Запускать перед generate_predictions.py.

Использование:
    python load_candles.py              # все тикеры, с 2023-01-01
    python load_candles.py --ticker SBER
    python load_candles.py --from 2022-01-01
"""
import os, sys, time, json, argparse, traceback
import requests
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
load_dotenv()

TICKERS = [
    "GAZP","LKOH","NVTK","ROSN","TATN","SNGS","SIBN","BANEP",
    "SBER","VTBR","T","MOEX","SFIN","CBOM",
    "GMKN","PLZL","ALRS","CHMF","NLMK","MAGN",
    "YDEX","OZON","VKCO","POSI",
    "MGNT","X5","LENT","FIXP",
    "MTSS","RTKM","PIKK","SMLT","AFLT","FESH","UWGN",
]


def parse_cli_date(value: str) -> str:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
        try:
            return pd.to_datetime(value, format=fmt).date().isoformat()
        except ValueError:
            pass
    try:
        return pd.to_datetime(f"{value}.{date.today().year}", format="%d.%m.%Y").date().isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Неверная дата '{value}'. Используй YYYY-MM-DD, DD.MM.YYYY или DD.MM."
        )

supabase_client = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])

CANDLE_COLS = ["time", "ticker", "interval", "open", "high", "low", "close", "volume", "value"]
IND_COLS = [
    "time", "ticker", "sma_10", "sma_20", "sma_50", "ema_12", "ema_26", "rsi_14",
    "macd", "macd_signal", "macd_hist", "bb_upper", "bb_middle", "bb_lower",
    "atr_14", "obv", "vol_ratio",
]



def fetch_candles_moex(
    ticker: str,
    date_from: str,
    date_till: str,
    interval_code: int = 24,
) -> pd.DataFrame:
    """Загружает свечи с MOEX ISS API (постранично). interval_code: 24=1d, 60=1h."""
    url = (
        f"https://iss.moex.com/iss/engines/stock/markets/shares"
        f"/boards/TQBR/securities/{ticker}/candles.json"
    )
    all_rows = []
    start = 0
    page = 1
    while True:
        print(f"    стр.{page} (записей загружено: {len(all_rows)})...", end=" ", flush=True)
        try:
            r = requests.get(url, params={
                "from": date_from, "till": date_till,
                "interval": interval_code, "start": start,
                "iss.meta": "off",
            }, timeout=15)
            r.raise_for_status()
            if not r.content:
                print("пустой ответ")
                break
            body = r.json()
            candles = body.get("candles", {})
            cols = candles.get("columns", [])
            data = candles.get("data", [])
            if not data:
                print("конец данных")
                break
            for row in data:
                all_rows.append(dict(zip(cols, row)))
            print(f"+{len(data)}")
            if len(data) < 500:
                break
            start += len(data)
            page += 1
            time.sleep(0.2)
        except requests.exceptions.Timeout:
            print(f"TIMEOUT (15с) — пропускаем")
            break
        except Exception as e:
            print(f"ОШИБКА: {e}")
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # Колонка времени называется 'begin' в ответе MOEX
    time_col = "begin" if "begin" in df.columns else df.columns[0]
    df = df.rename(columns={time_col: "time"})
    df["time"] = pd.to_datetime(df["time"])
    df["ticker"] = ticker
    for col in ["open","high","low","close","volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "value" not in df.columns:
        df["value"] = None
    return df


def calc_indicators(df: pd.DataFrame, ticker: str) -> list:
    if len(df) < 50:
        return []
    d = df.sort_values("time").copy()
    for col in ["open","high","low","close","volume"]:
        d[col] = d[col].astype(float)

    d["sma_10"] = ta.sma(d["close"], 10)
    d["sma_20"] = ta.sma(d["close"], 20)
    d["sma_50"] = ta.sma(d["close"], 50)
    d["ema_12"] = ta.ema(d["close"], 12)
    d["ema_26"] = ta.ema(d["close"], 26)
    d["rsi_14"] = ta.rsi(d["close"], 14)

    macd = ta.macd(d["close"])
    if macd is None or macd.empty:
        return []
    mcols = macd.columns.tolist()
    macd_col    = next((c for c in mcols if c.startswith("MACD_")),    None)
    macds_col   = next((c for c in mcols if c.startswith("MACDs_")),   None)
    macdh_col   = next((c for c in mcols if c.startswith("MACDh_")),   None)
    if not all([macd_col, macds_col, macdh_col]):
        return []
    d["macd"]        = macd[macd_col]
    d["macd_signal"] = macd[macds_col]
    d["macd_hist"]   = macd[macdh_col]

    bb = ta.bbands(d["close"], 20)
    if bb is None or bb.empty:
        return []
    bcols = bb.columns.tolist()
    bbu = next((c for c in bcols if c.startswith("BBU")), None)
    bbm = next((c for c in bcols if c.startswith("BBM")), None)
    bbl = next((c for c in bcols if c.startswith("BBL")), None)
    if not all([bbu, bbm, bbl]):
        return []
    d["bb_upper"]  = bb[bbu]
    d["bb_middle"] = bb[bbm]
    d["bb_lower"]  = bb[bbl]

    d["atr_14"]    = ta.atr(d["high"], d["low"], d["close"], 14)
    d["obv"]       = ta.obv(d["close"], d["volume"])
    d["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean()

    d.dropna(subset=["rsi_14"], inplace=True)

    def _safe(v):
        try:
            x = float(v)
            return None if (np.isnan(x) or np.isinf(x)) else round(x, 6)
        except Exception:
            return None

    records = []
    for _, row in d.iterrows():
        obv_val = row.get("obv")
        try:
            obv_int = int(float(obv_val)) if obv_val is not None and not np.isnan(float(obv_val)) else None
        except Exception:
            obv_int = None
        records.append((
            row["time"], ticker,
            _safe(row.get("sma_10")),   _safe(row.get("sma_20")),
            _safe(row.get("sma_50")),   _safe(row.get("ema_12")),
            _safe(row.get("ema_26")),   _safe(row.get("rsi_14")),
            _safe(row.get("macd")),     _safe(row.get("macd_signal")),
            _safe(row.get("macd_hist")),
            _safe(row.get("bb_upper")), _safe(row.get("bb_middle")),
            _safe(row.get("bb_lower")), _safe(row.get("atr_14")),
            obv_int,
            _safe(row.get("vol_ratio")),
        ))
    return records


def _make_candle_records(df: pd.DataFrame, ticker: str, interval: str) -> list:
    records = []
    for _, row in df.iterrows():
        try:
            vol = int(float(row["volume"])) if pd.notna(row.get("volume")) else 0
            val = float(row["value"]) if pd.notna(row.get("value")) else None
            records.append((
                row["time"], ticker, interval,
                float(row["open"]), float(row["high"]),
                float(row["low"]),  float(row["close"]),
                vol, val,
            ))
        except Exception:
            continue
    return records


def load_ticker(ticker: str, date_from: str, date_till: str):
    print(f"  Загрузка {ticker} {date_from}→{date_till}...", end=" ", flush=True)

    df_daily = fetch_candles_moex(ticker, date_from, date_till, interval_code=24)
    df_hourly = fetch_candles_moex(ticker, date_from, date_till, interval_code=60)
    if df_daily.empty and df_hourly.empty:
        print("нет данных")
        return

    candle_records = []
    if not df_daily.empty:
        candle_records.extend(_make_candle_records(df_daily, ticker, "1d"))
    if not df_hourly.empty:
        candle_records.extend(_make_candle_records(df_hourly, ticker, "1h"))
    ind_records = calc_indicators(df_daily, ticker) if not df_daily.empty else []

    if not candle_records and not ind_records:
        print("нет записей для сохранения")
        return

    print(
        f"сохраняю {len(candle_records)} свечей "
        f"(1d={len(df_daily)}, 1h={len(df_hourly)}) + {len(ind_records)} индикаторов...",
        end=" ",
        flush=True,
    )

    candle_dicts = [
        {k: (str(v) if k == "time" else v) for k, v in zip(CANDLE_COLS, r)}
        for r in candle_records
    ]
    ind_dicts = [
        {k: (str(v) if k == "time" else v) for k, v in zip(IND_COLS, r)}
        for r in ind_records
    ]

    for i in range(0, len(candle_dicts), 25):
        supabase_client.table("candles").upsert(
            candle_dicts[i:i+25], on_conflict="time,ticker,interval"
        ).execute()

    for i in range(0, len(ind_dicts), 25):
        supabase_client.table("indicators").upsert(
            ind_dicts[i:i+25], on_conflict="time,ticker"
        ).execute()

    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--from", dest="date_from", default="2023-01-01", type=parse_cli_date)
    parser.add_argument("--till", dest="date_till", default=date.today().isoformat(), type=parse_cli_date)
    args = parser.parse_args()

    date_till = args.date_till
    tickers = [args.ticker.upper()] if args.ticker else TICKERS

    print(f"Загрузка {len(tickers)} тикеров с {args.date_from} по {date_till}")
    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}]", end=" ")
        try:
            load_ticker(t, args.date_from, date_till)
        except Exception:
            print(f"ОШИБКА:\n{traceback.format_exc()}")
        time.sleep(0.5)

    print("Готово! Теперь запусти generate_predictions.py")
