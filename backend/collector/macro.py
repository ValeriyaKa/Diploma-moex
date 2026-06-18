"""
backend/collector/macro.py - FIXED with pagination
Collects macroeconomic data from MOEX ISS API + Yahoo Finance.
MOEX ISS returns max 100 rows per request, so we paginate.

Usage:
    python -m backend.collector.macro
"""
import os, logging, time
from datetime import date
import requests
import pandas as pd

log = logging.getLogger(__name__)


def _moex_paginated(url_template, date_from, date_till):
    """Fetch all pages from MOEX ISS API (100 rows per page)."""
    all_data = []
    start = 0
    while True:
        url = url_template.format(
            date_from=date_from, date_till=date_till, start=start
        )
        try:
            r = requests.get(url, timeout=15, proxies={"http": None, "https": None})
            data = r.json()["history"]["data"]
            if not data:
                break
            all_data.extend(data)
            if len(data) < 100:
                break
            start += 100
            time.sleep(0.2)
        except Exception as e:
            log.warning(f"  MOEX page error at start={start}: {e}")
            break
    return all_data


def fetch_moex_macro(date_from, date_till):
    """Fetch USD/RUB, EUR/RUB, IMOEX, RTSI from MOEX with pagination."""

    # USD/RUB
    log.info("Fetching USD/RUB...")
    usd_data = _moex_paginated(
        "https://iss.moex.com/iss/history/engines/currency/markets/selt"
        "/boards/CETS/securities/USD000UTSTOM.json"
        "?from={date_from}&till={date_till}&start={start}"
        "&iss.meta=off&history.columns=TRADEDATE,CLOSE",
        date_from, date_till
    )
    usd_map = {r[0]: r[1] for r in usd_data}
    log.info(f"  USD/RUB: {len(usd_data)} rows")
    time.sleep(0.3)

    # EUR/RUB
    log.info("Fetching EUR/RUB...")
    eur_data = _moex_paginated(
        "https://iss.moex.com/iss/history/engines/currency/markets/selt"
        "/boards/CETS/securities/EUR_RUB__TOM.json"
        "?from={date_from}&till={date_till}&start={start}"
        "&iss.meta=off&history.columns=TRADEDATE,CLOSE",
        date_from, date_till
    )
    eur_map = {r[0]: r[1] for r in eur_data}
    log.info(f"  EUR/RUB: {len(eur_data)} rows")
    time.sleep(0.3)

    # IMOEX
    log.info("Fetching IMOEX...")
    imoex_data = _moex_paginated(
        "https://iss.moex.com/iss/history/engines/stock/markets/index"
        "/securities/IMOEX.json"
        "?from={date_from}&till={date_till}&start={start}"
        "&iss.meta=off&history.columns=TRADEDATE,CLOSE",
        date_from, date_till
    )
    imoex_map = {r[0]: r[1] for r in imoex_data}
    log.info(f"  IMOEX: {len(imoex_data)} rows")
    time.sleep(0.3)

    # RTSI
    log.info("Fetching RTSI...")
    rtsi_data = _moex_paginated(
        "https://iss.moex.com/iss/history/engines/stock/markets/index"
        "/securities/RTSI.json"
        "?from={date_from}&till={date_till}&start={start}"
        "&iss.meta=off&history.columns=TRADEDATE,CLOSE",
        date_from, date_till
    )
    rtsi_map = {r[0]: r[1] for r in rtsi_data}
    log.info(f"  RTSI: {len(rtsi_data)} rows")
    time.sleep(0.3)

    # Brent (MOEX futures)
    log.info("Fetching Brent...")
    brent_data = _moex_paginated(
        "https://iss.moex.com/iss/history/engines/futures/markets/forts"
        "/boards/RFUD/securities/BRN5.json"
        "?from={date_from}&till={date_till}&start={start}"
        "&iss.meta=off&history.columns=TRADEDATE,CLOSE",
        date_from, date_till
    )
    brent_map = {r[0]: r[1] for r in brent_data}
    log.info(f"  Brent: {len(brent_data)} rows")

    # Merge all dates
    all_dates = sorted(set(
        list(usd_map.keys()) + list(imoex_map.keys()) + list(rtsi_map.keys())
    ))

    rows = []
    for d in all_dates:
        rows.append({
            "time": d,
            "usd_rub": usd_map.get(d),
            "eur_rub": eur_map.get(d),
            "imoex": imoex_map.get(d),
            "rtsi": rtsi_map.get(d),
            "brent_usd": brent_map.get(d),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_yahoo_macro(date_from, date_till):
    """Fetch S&P500, Gold, Natural Gas, CSI300 from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed")
        return pd.DataFrame()

    tickers = {
        "^GSPC":    "sp500",
        "GC=F":     "gold_usd",
        "NG=F":     "gas_usd",
        "000300.SS": "csi300",
    }

    all_data = {}
    for yf_ticker, col_name in tickers.items():
        log.info(f"Fetching {col_name} from Yahoo Finance...")
        try:
            data = yf.download(yf_ticker, start=date_from, end=date_till,
                               progress=False, auto_adjust=True)
            if not data.empty:
                close = data["Close"]
                # Handle MultiIndex columns
                if hasattr(close, "columns"):
                    close = close.iloc[:, 0]
                all_data[col_name] = close
                log.info(f"  {col_name}: {len(close)} rows")
        except Exception as e:
            log.warning(f"  {col_name} error: {e}")
        time.sleep(0.5)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    df.index.name = "time"
    return df.reset_index()


def collect_all(date_from="2022-01-01", date_till=None):
    if date_till is None:
        date_till = date.today().isoformat()

    log.info(f"Collecting macro: {date_from} -> {date_till}\n")

    df_moex = fetch_moex_macro(date_from, date_till)
    log.info(f"\nMOEX total: {len(df_moex)} rows")

    df_yahoo = fetch_yahoo_macro(date_from, date_till)
    log.info(f"Yahoo total: {len(df_yahoo)} rows")

    # Merge
    if not df_moex.empty and not df_yahoo.empty:
        df = df_moex.merge(df_yahoo, on="time", how="outer")
    elif not df_moex.empty:
        df = df_moex
    elif not df_yahoo.empty:
        df = df_yahoo
    else:
        log.error("No data collected!")
        return

    df.sort_values("time", inplace=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/macro.csv", index=False)
    log.info(f"\nSaved data/macro.csv: {len(df)} rows")

    # Save to Supabase
    _save_macro_to_db(df)
    log.info(f"Date range: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

    # Verify
    last_imoex = df.dropna(subset=["imoex"]).tail(1)
    last_usd = df.dropna(subset=["usd_rub"]).tail(1)
    if not last_imoex.empty:
        log.info(f"Latest IMOEX: {last_imoex.iloc[0]['imoex']} ({last_imoex.iloc[0]['time']})")
    if not last_usd.empty:
        log.info(f"Latest USD/RUB: {last_usd.iloc[0]['usd_rub']} ({last_usd.iloc[0]['time']})")


def _save_macro_to_db(df: pd.DataFrame):
    """Upsert macro DataFrame into Supabase."""
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv()

    sb = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])

    rows = df.to_dict("records")
    saved, errors = 0, 0
    for i in range(0, len(rows), 25):
        batch = rows[i:i+25]
        clean = []
        for r in batch:
            rec = {}
            for k, v in r.items():
                if pd.isna(v):
                    rec[k] = None
                else:
                    rec[k] = v
            clean.append(rec)
        try:
            sb.table("macro").upsert(clean, on_conflict="time").execute()
            saved += len(clean)
        except Exception as e:
            errors += len(clean)
            if errors <= 25:  # print first error for debugging
                log.warning(f"Macro upsert error: {e}")
    log.info(f"Supabase: saved {saved}, errors {errors}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    from dotenv import load_dotenv
    load_dotenv()
    collect_all()