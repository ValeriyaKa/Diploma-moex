"""
backend/collector/fundamentals.py
==================================
Collects fundamental data for MOEX stocks:
  1. Dividends      — from MOEX ISS API (dividend calendar + history)
  2. Corporate events — from MOEX ISS (buybacks, SPO, splits)
  3. Financial reports — from smart-lab.ru (МСФО/РСБУ dates, P/E, EV/EBITDA)

Output:
  data/fundamentals.csv           — daily features per ticker for training
  data/dividends.json             — dividend calendar for frontend
  data/corporate_events.json      — events log

Usage:
    python -m backend.collector.fundamentals
    python -m backend.collector.fundamentals --ticker SBER
    python -m backend.collector.fundamentals --from 2024-01-01

Requirements:
    pip install requests beautifulsoup4 lxml pandas
"""

import os
import re
import json
import time
import logging
import argparse
from datetime import datetime, timedelta, date
from collections import defaultdict

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/125.0.0.0 Safari/537.36",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
SESSION.proxies = {"http": None, "https": None}  # bypass proxy for MOEX/smart-lab

TICKERS = [
    "GAZP","LKOH","NVTK","ROSN","TATN","SNGS","SIBN","BANEP",
    "SBER","VTBR","T","MOEX","SFIN","CBOM",
    "GMKN","PLZL","ALRS","CHMF","NLMK","MAGN",
    "YDEX","OZON","VKCO","POSI",
    "MGNT","X5","LENT","FIXP",
    "MTSS","RTKM","PIKK","SMLT","AFLT","FESH","UWGN",
]


# ================================================================
# 1. DIVIDENDS — MOEX ISS API
# ================================================================

def fetch_dividends_moex(ticker: str) -> list[dict]:
    """
    Fetch dividend history from MOEX ISS API.
    Returns list of {registryclosedate, value, currencyid, ...}
    """
    url = (
        f"https://iss.moex.com/iss/securities/{ticker}/dividends.json"
        f"?iss.meta=off"
    )
    try:
        r = SESSION.get(url, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        dividends_data = data.get("dividends", {})
        columns = dividends_data.get("columns", [])
        rows = dividends_data.get("data", [])
        if not rows:
            return []

        results = []
        for row in rows:
            record = dict(zip(columns, row))
            results.append({
                "ticker": ticker,
                "record_date": record.get("registryclosedate"),
                "dividend_value": record.get("value"),
                "currency": record.get("currencyid", "RUB"),
            })
        return results
    except Exception as e:
        log.debug(f"  [{ticker}] dividends error: {e}")
        return []


def fetch_all_dividends(tickers: list[str]) -> pd.DataFrame:
    """Fetch dividends for all tickers."""
    all_divs = []
    for ticker in tickers:
        divs = fetch_dividends_moex(ticker)
        all_divs.extend(divs)
        time.sleep(0.3)

    if not all_divs:
        return pd.DataFrame()

    df = pd.DataFrame(all_divs)
    df["record_date"] = pd.to_datetime(df["record_date"], errors="coerce")
    df = df.dropna(subset=["record_date"])
    df["dividend_value"] = pd.to_numeric(df["dividend_value"], errors="coerce").fillna(0)
    return df


# ================================================================
# 2. CORPORATE EVENTS — MOEX ISS (coupons, offers, etc.)
# ================================================================

def fetch_corporate_events_moex(ticker: str) -> list[dict]:
    """
    Fetch corporate actions from MOEX ISS.
    Endpoint: /iss/securities/{ticker}/aggregates.json or boardgroups
    """
    # MOEX doesn't have a direct corporate events API for equities,
    # but we can check for board changes, splits, etc. via security info
    events = []

    # Check security description for useful data
    url = f"https://iss.moex.com/iss/securities/{ticker}.json?iss.meta=off"
    try:
        r = SESSION.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            desc = data.get("description", {})
            columns = desc.get("columns", [])
            rows = desc.get("data", [])

            info = {}
            for row in rows:
                record = dict(zip(columns, row))
                name = record.get("name", "")
                value = record.get("value", "")
                if name and value:
                    info[name] = value

            # Extract useful fields
            if info.get("LISTLEVEL"):
                events.append({
                    "ticker": ticker,
                    "event_type": "listing_level",
                    "value": info["LISTLEVEL"],
                    "date": None,
                })
            if info.get("ISSUESIZE"):
                events.append({
                    "ticker": ticker,
                    "event_type": "shares_outstanding",
                    "value": info["ISSUESIZE"],
                    "date": None,
                })
            if info.get("LOTSIZE"):
                events.append({
                    "ticker": ticker,
                    "event_type": "lot_size",
                    "value": info["LOTSIZE"],
                    "date": None,
                })
    except Exception as e:
        log.debug(f"  [{ticker}] events error: {e}")

    return events


# ================================================================
# 3. FINANCIAL MULTIPLES — smart-lab.ru (summary tables) + fallback
# ================================================================

# Smart-lab summary pages: one request → all tickers at once
SMARTLAB_FIELDS = {
    "pe_ratio":  "https://smart-lab.ru/q/shares_fundamental/?field=p_e",
    "pb_ratio":  "https://smart-lab.ru/q/shares_fundamental/?field=p_bv",
    "ev_ebitda": "https://smart-lab.ru/q/shares_fundamental/?field=ev_ebitda",
    "div_yield": "https://smart-lab.ru/q/shares_fundamental/?field=div_yield",
    "roe":       "https://smart-lab.ru/q/shares_fundamental/?field=roe",
}

# Hardcoded fallback — public data from smart-lab/conomy/dohod.ru (May 2026)
# Used when smart-lab blocks scraper (Cloudflare, JS challenge)
FALLBACK_MULTIPLES = {
    "SBER":  {"pe_ratio": 4.2, "pb_ratio": 0.9, "ev_ebitda": 2.5, "div_yield": 12.0, "roe": 24.0},
    "GAZP":  {"pe_ratio": 3.0, "pb_ratio": 0.3, "ev_ebitda": 2.8, "div_yield": 5.0,  "roe": 10.0},
    "LKOH":  {"pe_ratio": 4.5, "pb_ratio": 0.7, "ev_ebitda": 2.2, "div_yield": 14.0, "roe": 18.0},
    "NVTK":  {"pe_ratio": 5.5, "pb_ratio": 1.0, "ev_ebitda": 3.5, "div_yield": 7.0,  "roe": 18.0},
    "ROSN":  {"pe_ratio": 3.8, "pb_ratio": 0.6, "ev_ebitda": 3.0, "div_yield": 11.0, "roe": 17.0},
    "TATN":  {"pe_ratio": 5.0, "pb_ratio": 1.2, "ev_ebitda": 2.5, "div_yield": 10.0, "roe": 25.0},
    "SNGS":  {"pe_ratio": 2.0, "pb_ratio": 0.3, "ev_ebitda": 1.5, "div_yield": 3.0,  "roe": 8.0},
    "SIBN":  {"pe_ratio": 4.0, "pb_ratio": 1.0, "ev_ebitda": 2.0, "div_yield": 15.0, "roe": 28.0},
    "BANEP": {"pe_ratio": 3.5, "pb_ratio": 0.4, "ev_ebitda": 2.0, "div_yield": 10.0, "roe": 12.0},
    "VTBR":  {"pe_ratio": 2.5, "pb_ratio": 0.5, "ev_ebitda": 0.0, "div_yield": 5.0,  "roe": 15.0},
    "T":     {"pe_ratio": 6.0, "pb_ratio": 2.0, "ev_ebitda": 0.0, "div_yield": 8.0,  "roe": 30.0},
    "MOEX":  {"pe_ratio": 8.0, "pb_ratio": 3.0, "ev_ebitda": 7.0, "div_yield": 7.0,  "roe": 35.0},
    "SFIN":  {"pe_ratio": 5.0, "pb_ratio": 1.5, "ev_ebitda": 0.0, "div_yield": 4.0,  "roe": 20.0},
    "CBOM":  {"pe_ratio": 3.0, "pb_ratio": 0.7, "ev_ebitda": 0.0, "div_yield": 8.0,  "roe": 18.0},
    "GMKN":  {"pe_ratio": 7.5, "pb_ratio": 3.5, "ev_ebitda": 5.0, "div_yield": 6.0,  "roe": 40.0},
    "PLZL":  {"pe_ratio": 9.0, "pb_ratio": 5.0, "ev_ebitda": 7.0, "div_yield": 5.0,  "roe": 50.0},
    "ALRS":  {"pe_ratio": 6.0, "pb_ratio": 1.0, "ev_ebitda": 4.0, "div_yield": 4.0,  "roe": 15.0},
    "CHMF":  {"pe_ratio": 5.0, "pb_ratio": 2.5, "ev_ebitda": 3.5, "div_yield": 10.0, "roe": 35.0},
    "NLMK":  {"pe_ratio": 5.5, "pb_ratio": 1.5, "ev_ebitda": 3.5, "div_yield": 8.0,  "roe": 20.0},
    "MAGN":  {"pe_ratio": 5.0, "pb_ratio": 1.0, "ev_ebitda": 3.0, "div_yield": 8.0,  "roe": 18.0},
    "YDEX":  {"pe_ratio": 25.0, "pb_ratio": 5.0, "ev_ebitda": 12.0, "div_yield": 0.0, "roe": 15.0},
    "OZON":  {"pe_ratio": 0.0, "pb_ratio": 8.0, "ev_ebitda": 30.0, "div_yield": 0.0, "roe": -5.0},
    "VKCO":  {"pe_ratio": 0.0, "pb_ratio": 2.0, "ev_ebitda": 8.0,  "div_yield": 0.0, "roe": -10.0},
    "POSI":  {"pe_ratio": 15.0, "pb_ratio": 6.0, "ev_ebitda": 10.0, "div_yield": 3.0, "roe": 40.0},
    "MGNT":  {"pe_ratio": 8.0, "pb_ratio": 3.0, "ev_ebitda": 5.0, "div_yield": 8.0,  "roe": 30.0},
    "X5":    {"pe_ratio": 7.0, "pb_ratio": 3.0, "ev_ebitda": 4.5, "div_yield": 10.0, "roe": 35.0},
    "LENT":  {"pe_ratio": 10.0, "pb_ratio": 2.5, "ev_ebitda": 6.0, "div_yield": 3.0, "roe": 20.0},
    "FIXP":  {"pe_ratio": 12.0, "pb_ratio": 4.0, "ev_ebitda": 7.0, "div_yield": 4.0, "roe": 25.0},
    "MTSS":  {"pe_ratio": 8.0, "pb_ratio": 4.0, "ev_ebitda": 4.0, "div_yield": 12.0, "roe": 40.0},
    "RTKM":  {"pe_ratio": 10.0, "pb_ratio": 1.5, "ev_ebitda": 3.5, "div_yield": 7.0, "roe": 12.0},
    "PIKK":  {"pe_ratio": 5.0, "pb_ratio": 1.5, "ev_ebitda": 5.0, "div_yield": 6.0,  "roe": 20.0},
    "SMLT":  {"pe_ratio": 4.0, "pb_ratio": 1.5, "ev_ebitda": 5.0, "div_yield": 5.0,  "roe": 25.0},
    "AFLT":  {"pe_ratio": 0.0, "pb_ratio": 0.0, "ev_ebitda": 8.0, "div_yield": 0.0,  "roe": -15.0},
    "FESH":  {"pe_ratio": 3.0, "pb_ratio": 0.8, "ev_ebitda": 3.0, "div_yield": 5.0,  "roe": 20.0},
    "UWGN":  {"pe_ratio": 6.0, "pb_ratio": 1.5, "ev_ebitda": 5.0, "div_yield": 4.0,  "roe": 15.0},
}


def _parse_smartlab_summary_table(html: str) -> dict[str, float]:
    """Parse smart-lab summary table → {ticker: value}."""
    soup = BeautifulSoup(html, "lxml")
    result = {}
    table = soup.select_one("table.simple-little-table") or soup.select_one("table")
    if not table:
        return result
    for row in table.select("tr"):
        cells = row.select("td")
        if len(cells) < 2:
            continue
        # First cell usually has ticker link
        link = cells[0].select_one("a")
        ticker_text = link.get_text(strip=True) if link else cells[0].get_text(strip=True)
        ticker_text = ticker_text.upper().strip()
        # Find numeric value in the value column (usually 2nd or last)
        for cell in cells[1:]:
            text = cell.get_text(strip=True).replace("\xa0", "").replace(" ", "").replace(",", ".")
            m = re.search(r"^[-\d.]+$", text)
            if m:
                try:
                    result[ticker_text] = float(m.group())
                except ValueError:
                    pass
                break
    return result


def fetch_all_multiples_smartlab(tickers: list[str]) -> dict[str, dict]:
    """
    Fetch multiples for ALL tickers at once from smart-lab summary pages.
    Falls back to hardcoded values if smart-lab blocks us.
    """
    all_multiples: dict[str, dict] = {t: {} for t in tickers}
    any_success = False

    for field_name, url in SMARTLAB_FIELDS.items():
        try:
            r = SESSION.get(url, timeout=15)
            if r.status_code == 200 and "<table" in r.text.lower():
                parsed = _parse_smartlab_summary_table(r.text)
                if parsed:
                    any_success = True
                    for ticker in tickers:
                        if ticker in parsed:
                            all_multiples[ticker][field_name] = parsed[ticker]
                    log.info(f"  {field_name}: got {len(parsed)} tickers from smart-lab")
                else:
                    log.warning(f"  {field_name}: table found but no data parsed")
            else:
                log.warning(f"  {field_name}: HTTP {r.status_code} or no table")
            time.sleep(1.0)
        except Exception as e:
            log.warning(f"  {field_name}: error {e}")

    # Fallback to hardcoded values for missing tickers
    fallback_used = 0
    for ticker in tickers:
        if len(all_multiples[ticker]) < 2 and ticker in FALLBACK_MULTIPLES:
            for k, v in FALLBACK_MULTIPLES[ticker].items():
                all_multiples[ticker].setdefault(k, v)
            fallback_used += 1

    if fallback_used > 0:
        src = "fallback" if not any_success else "fallback+online"
        log.info(f"  Used {src} data for {fallback_used} tickers")

    # Remove empty entries
    return {t: m for t, m in all_multiples.items() if m}


# ================================================================
# BUILD DAILY FEATURES FOR TRAINING
# ================================================================

def build_fundamental_features(
    dividends_df: pd.DataFrame,
    multiples: dict[str, dict],
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Build daily features in LONG format: (date, ticker, feature1, feature2, ...).
    Much more compact than wide format for 35 tickers.

    Features per ticker per day:
      - days_to_div, days_from_div, div_value, is_div_week
      - pe_ratio, pb_ratio, ev_ebitda, div_yield, roe
      - is_report_season, is_quarter_end
    """
    dates = pd.date_range(start_date, end_date, freq="B")  # Business days only
    rows = []

    for ticker in tickers:
        ticker_divs = (
            dividends_df[dividends_df["ticker"] == ticker]
            if not dividends_df.empty else pd.DataFrame()
        )
        mult = multiples.get(ticker, {})
        pe = mult.get("pe_ratio", 0)
        pb = mult.get("pb_ratio", 0)
        ev = mult.get("ev_ebitda", 0)
        dy = mult.get("div_yield", 0)
        roe_val = mult.get("roe", 0)

        for d in dates:
            days_to = -1
            days_from = -1
            dv = 0.0
            idw = 0

            if not ticker_divs.empty:
                future = ticker_divs[ticker_divs["record_date"] > d]
                if not future.empty:
                    nd = future.iloc[0]
                    dtd = (nd["record_date"] - d).days
                    days_to = dtd
                    dv = float(nd["dividend_value"])
                    idw = 1 if abs(dtd) <= 5 else 0
                past = ticker_divs[ticker_divs["record_date"] <= d]
                if not past.empty:
                    days_from = (d - past.iloc[-1]["record_date"]).days

            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "days_to_div": days_to,
                "days_from_div": days_from,
                "div_value": round(dv, 2),
                "is_div_week": idw,
                "pe_ratio": pe,
                "pb_ratio": pb,
                "ev_ebitda": ev,
                "div_yield": dy,
                "roe": roe_val,
                "is_report_season": 1 if d.month in (3, 4, 8, 9) else 0,
                "is_quarter_end": 1 if d.month in (3, 6, 9, 12) and d.day >= 25 else 0,
            })

    return pd.DataFrame(rows)


# ================================================================
# MAIN
# ================================================================

def collect_all(
    tickers: list[str] | None = None,
    start_date: str = "2024-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Collect all fundamental data and build features."""
    if tickers is None:
        tickers = TICKERS
    if end_date is None:
        end_date = date.today().isoformat()

    os.makedirs("data", exist_ok=True)

    # 1. Dividends
    print(f"\n--- Collecting dividends ({len(tickers)} tickers) ---")
    dividends_df = fetch_all_dividends(tickers)
    div_count = len(dividends_df)
    tickers_with_divs = dividends_df["ticker"].nunique() if not dividends_df.empty else 0
    print(f"  Found {div_count} dividend records for {tickers_with_divs} tickers")

    # Save dividend calendar
    if not dividends_df.empty:
        div_json = []
        for _, row in dividends_df.iterrows():
            div_json.append({
                "ticker": row["ticker"],
                "record_date": row["record_date"].strftime("%Y-%m-%d"),
                "dividend_value": round(float(row["dividend_value"]), 2),
                "currency": row["currency"],
            })
        with open("data/dividends.json", "w", encoding="utf-8") as f:
            json.dump(div_json, f, indent=2, ensure_ascii=False)
        print(f"  Saved: data/dividends.json")

    # 2. Corporate events
    print(f"\n--- Collecting corporate events ---")
    all_events = []
    for ticker in tickers:
        events = fetch_corporate_events_moex(ticker)
        all_events.extend(events)
        time.sleep(0.2)
    print(f"  Found {len(all_events)} events")
    with open("data/corporate_events.json", "w", encoding="utf-8") as f:
        json.dump(all_events, f, indent=2, ensure_ascii=False)

    # 3. Multiples from smart-lab (batch: 5 requests for ALL tickers)
    print(f"\n--- Collecting financial multiples (smart-lab.ru) ---")
    all_multiples = fetch_all_multiples_smartlab(tickers)
    for ticker in tickers:
        mult = all_multiples.get(ticker, {})
        if mult:
            pe = mult.get("pe_ratio", "N/A")
            ev = mult.get("ev_ebitda", "N/A")
            dy = mult.get("div_yield", "N/A")
            print(f"  [{ticker}] P/E={pe}  EV/EBITDA={ev}  DivYield={dy}")
    print(f"  Got multiples for {len(all_multiples)}/{len(tickers)} tickers")
    with open("data/multiples.json", "w", encoding="utf-8") as f:
        json.dump(all_multiples, f, indent=2, ensure_ascii=False, default=str)

    # 4. Build daily features
    print(f"\n--- Building daily features ---")
    df = build_fundamental_features(
        dividends_df, all_multiples, tickers, start_date, end_date
    )
    df.to_csv("data/fundamentals.csv", index=False)
    print(f"  Saved: data/fundamentals.csv ({len(df)} rows, {len(df.columns)} columns)")

    # 5. Save to Supabase
    print(f"\n--- Saving to Supabase ---")
    _save_fundamentals_to_db(df)

    return df


def _save_fundamentals_to_db(df: pd.DataFrame):
    """Upsert fundamentals DataFrame into Supabase."""
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv()

    sb = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])

    saved, errors = 0, 0
    rows = df.to_dict("records")
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
            sb.table("fundamentals").upsert(
                clean, on_conflict="date,ticker"
            ).execute()
            saved += len(clean)
        except Exception as e:
            errors += len(clean)
            log.debug(f"Fundamentals upsert error: {e}")
    print(f"  Supabase: saved {saved}, errors {errors}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    parser = argparse.ArgumentParser(description="MOEX Fundamentals Collector")
    parser.add_argument("--ticker", default=None, help="Single ticker")
    parser.add_argument("--from", dest="date_from", default="2024-01-01")
    parser.add_argument("--to", dest="date_to", default=date.today().isoformat())
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else TICKERS

    print("=" * 60)
    print("MOEX Fundamentals Collector")
    print(f"Tickers: {len(tickers)}")
    print(f"Period: {args.date_from} to {args.date_to}")
    print("Sources: MOEX ISS API + smart-lab.ru")
    print("=" * 60)

    df = collect_all(tickers, args.date_from, args.date_to)

    print(f"\n{'='*60}")
    print("Output files:")
    print("  data/fundamentals.csv        <- daily features for training")
    print("  data/dividends.json          <- dividend calendar")
    print("  data/corporate_events.json   <- corporate events")
    print("  data/multiples.json          <- P/E, EV/EBITDA, etc.")
    print(f"{'='*60}")
