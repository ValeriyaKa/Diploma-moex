"""
backend/collector/news_finnhub_v2.py
====================================
Collects historical news sentiment using Finnhub API.
Uses HTTPS proxy for Finnhub (blocked in RU without it).
Saves to news_sentiment_historical.csv + news_sentiment.json.

Free tier: 60 calls/minute. For 35 tickers we need ~35 calls
per week chunk → can cover ~4 years in ~100 minutes.

Usage:
    python -m backend.collector.news_finnhub_v2
    python -m backend.collector.news_finnhub_v2 --from 2023-01-01
"""
import os, time, json, logging, argparse
from datetime import datetime, timedelta, date
import requests
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"

# Finnhub needs company-news with a symbol. For Russian stocks
# there are no direct symbols, so we use general news search.
# But company-news works for some tickers on international exchanges.

# We'll use general news + keyword filtering
TICKER_KEYWORDS = {
    "GAZP": ["gazprom"],
    "LKOH": ["lukoil"],
    "NVTK": ["novatek"],
    "ROSN": ["rosneft"],
    "TATN": ["tatneft"],
    "SNGS": ["surgutneftegas"],
    "SIBN": ["gazprom neft"],
    "SBER": ["sberbank"],
    "VTBR": ["vtb bank", "vtb "],
    "T": ["tinkoff", "t-bank", "t-technologies"],
    "MOEX": ["moscow exchange", "moex"],
    "GMKN": ["nornickel", "norilsk"],
    "PLZL": ["polyus"],
    "ALRS": ["alrosa"],
    "CHMF": ["severstal"],
    "NLMK": ["nlmk"],
    "MAGN": ["magnitogorsk", "mmk steel"],
    "YDEX": ["yandex"],
    "OZON": ["ozon"],
    "VKCO": ["vkontakte", "vk company"],
    "POSI": ["positive tech"],
    "MGNT": ["magnit"],
    "X5": ["x5 retail", "x5 group"],
    "MTSS": ["mts russia", "mts telecom"],
    "AFLT": ["aeroflot"],
    "PIKK": ["pik group"],
}

POSITIVE = ["growth", "rise", "gain", "surge", "rally", "profit", "record",
            "boost", "upgrade", "beat", "strong", "recovery", "deal",
            "dividend", "buyback", "optimism", "increase"]
NEGATIVE = ["fall", "decline", "loss", "drop", "crash", "crisis", "risk",
            "cut", "weak", "default", "sanction", "war", "conflict",
            "recession", "bankrupt", "layoff", "downgrade", "miss",
            "fear", "concern", "threat", "slump", "collapse"]


def _sentiment(text):
    t = text.lower()
    p = sum(1 for w in POSITIVE if w in t)
    n = sum(1 for w in NEGATIVE if w in t)
    if p + n == 0:
        return 0.0
    return round((p - n) / (p + n), 3)


def _get_proxy():
    """Get proxy from env if available."""
    proxy = os.environ.get("FINNHUB_PROXY", "")
    if not proxy:
        # Try common proxy env vars
        proxy = os.environ.get("HTTPS_PROXY_BACKUP", "")
    return proxy


def fetch_general_news(api_key, category="general", proxy=None):
    """Fetch latest general news from Finnhub."""
    proxies = {"https": f"http://{proxy}"} if proxy else None
    try:
        r = requests.get(
            f"{FINNHUB_BASE}/news",
            params={"category": category, "token": api_key},
            timeout=30,
            proxies=proxies,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.warning(f"  General news error: {e}")
    return []


def fetch_company_news_by_date(api_key, symbol, date_from, date_to, proxy=None):
    """
    Fetch company news for a US-listed symbol.
    Works for: YNDX (Yandex), OZON, etc.
    """
    proxies = {"https": f"http://{proxy}"} if proxy else None
    try:
        r = requests.get(
            f"{FINNHUB_BASE}/company-news",
            params={
                "symbol": symbol,
                "from": date_from,
                "to": date_to,
                "token": api_key,
            },
            timeout=30,
            proxies=proxies,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.debug(f"  Company news error: {e}")
    return []


def search_news(api_key, keyword, date_from, date_to, proxy=None):
    """
    Search for news mentioning a keyword using Finnhub.
    Uses market news endpoint and filters by keyword.
    For historical data, chunks by month.
    """
    proxies = {"https": f"http://{proxy}"} if proxy else None

    # Finnhub /news only returns recent news, not historical.
    # For historical, we use /company-news with related US tickers
    # or fall back to general news filtering.

    # Try general news (only recent ~1 week)
    try:
        r = requests.get(
            f"{FINNHUB_BASE}/news",
            params={"category": "general", "token": api_key},
            timeout=30,
            proxies=proxies,
        )
        if r.status_code != 200:
            return []
        articles = r.json()
    except Exception as e:
        log.warning(f"  Search error: {e}")
        return []

    # Filter by keyword
    results = []
    kw = keyword.lower()
    for a in articles:
        text = (a.get("headline", "") + " " + a.get("summary", "")).lower()
        if kw in text:
            ts = a.get("datetime", 0)
            if ts:
                dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                if date_from <= dt <= date_to:
                    results.append({
                        "date": dt,
                        "sentiment": _sentiment(text),
                        "headline": a.get("headline", "")[:200],
                    })
    return results


# US-listed Russian company tickers for company-news endpoint
US_TICKERS = {
    "OZON": "OZON",
    "YDEX": "YNDX",  # may be delisted but worth trying
}


def collect_historical(api_key, date_from="2022-01-01", date_to=None, proxy=None):
    """Main collection routine."""
    if date_to is None:
        date_to = date.today().isoformat()

    os.makedirs("data", exist_ok=True)

    # Generate monthly chunks
    chunks = []
    s = datetime.strptime(date_from, "%Y-%m-%d")
    e = datetime.strptime(date_to, "%Y-%m-%d")
    while s < e:
        chunk_end = min(s + timedelta(days=29), e)
        chunks.append((s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        s = chunk_end + timedelta(days=1)

    print(f"Period: {date_from} -> {date_to} ({len(chunks)} months)")
    print(f"Proxy: {proxy or 'none'}")

    all_rows = []
    api_calls = 0

    # 1. Try company-news for US-listed Russian stocks (historical)
    for moex_ticker, us_symbol in US_TICKERS.items():
        print(f"\n[{moex_ticker}] company-news ({us_symbol})")
        for cs, ce in chunks:
            data = fetch_company_news_by_date(api_key, us_symbol, cs, ce, proxy)
            api_calls += 1

            if data and isinstance(data, list):
                for a in data:
                    ts = a.get("datetime", 0)
                    if ts:
                        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                        text = (a.get("headline", "") + " " + a.get("summary", "")).lower()
                        all_rows.append({
                            "date": dt,
                            "ticker": moex_ticker,
                            "sentiment": _sentiment(text),
                        })
                print(f"  {cs}: {len(data)} articles", end="")

            if api_calls % 55 == 0:
                print(" (cooling 62s)", end="")
                time.sleep(62)
            else:
                time.sleep(1.1)
        print()

    # 2. General news search for all tickers (recent only)
    print("\n--- General news search (recent) ---")
    general = fetch_general_news(api_key, proxy=proxy)
    api_calls += 1
    print(f"Got {len(general)} general articles")

    for moex_ticker, keywords in TICKER_KEYWORDS.items():
        for a in general:
            text = (a.get("headline", "") + " " + a.get("summary", "")).lower()
            for kw in keywords:
                if kw in text:
                    ts = a.get("datetime", 0)
                    if ts:
                        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                        all_rows.append({
                            "date": dt,
                            "ticker": moex_ticker,
                            "sentiment": _sentiment(text),
                        })
                    break

    # 3. Market-wide sentiment from general news
    print("\n--- Market-wide sentiment ---")
    market_keywords = ["russia", "moscow", "ruble", "oil", "sanctions",
                       "emerging market", "opec", "central bank"]
    for a in general:
        text = (a.get("headline", "") + " " + a.get("summary", "")).lower()
        if any(kw in text for kw in market_keywords):
            ts = a.get("datetime", 0)
            if ts:
                dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                all_rows.append({
                    "date": dt,
                    "ticker": "_MARKET",
                    "sentiment": _sentiment(text),
                })

    print(f"\nTotal Finnhub records: {len(all_rows)}")
    print(f"API calls used: {api_calls}")

    # ---- Also build price-derived sentiment for full history ----
    print("\n--- Price-derived sentiment (full history from Supabase) ---")
    df_price = _build_price_sentiment(date_from, date_to)

    # ---- Merge ----
    if all_rows:
        df_news = pd.DataFrame(all_rows)
        df_news = df_news[df_news["ticker"] != "_MARKET"]
        df_news_daily = df_news.groupby(["date", "ticker"])["sentiment"].mean().reset_index()
        print(f"Finnhub ticker-day records: {len(df_news_daily)}")

        if not df_price.empty:
            # Combine: news where available, price-derived as fallback
            df_combined = df_price.merge(
                df_news_daily, on=["date", "ticker"], how="left",
                suffixes=("_price", "_news")
            )
            # Prefer real news sentiment, fallback to price-derived
            df_combined["sentiment"] = df_combined["sentiment_news"].fillna(
                df_combined["sentiment_price"]
            )
            df_combined.drop(columns=["sentiment_price", "sentiment_news"],
                             inplace=True, errors="ignore")
        else:
            df_combined = df_news_daily
            df_combined["news_volume"] = 1.0
    elif not df_price.empty:
        print("No Finnhub data, using price-derived only")
        df_combined = df_price
    else:
        print("ERROR: No data at all!")
        return pd.DataFrame()

    # ---- Pivot & save ----
    _save_results(df_combined, date_from, date_to)

    return df_combined


def _build_price_sentiment(date_from, date_to):
    """Build sentiment proxy from Supabase candles."""
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv()

    try:
        sb = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
    except Exception as e:
        log.warning(f"Supabase error: {e}")
        return pd.DataFrame()

    tickers = list(TICKER_KEYWORDS.keys())
    all_rows = []

    for ticker in tickers:
        rows = []
        offset = 0
        while True:
            q = sb.table("candles").select("time,close,volume")\
                .eq("ticker", ticker).eq("interval", "1d")\
                .gte("time", date_from).lte("time", date_to)\
                .order("time").range(offset, offset + 999)
            batch = q.execute().data
            rows.extend(batch)
            if len(batch) < 1000:
                break
            offset += 1000

        if len(rows) < 20:
            continue

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"])
        df.sort_values("time", inplace=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        df["ret"] = df["close"].pct_change()
        df["vol_ma"] = df["volume"].rolling(20, min_periods=5).mean()
        df["vol_z"] = ((df["volume"] - df["vol_ma"]) /
                       df["vol_ma"].replace(0, 1)).clip(-3, 5)

        df["sentiment"] = (df["ret"] * 10).clip(-1, 1)
        df["sentiment"] = (df["sentiment"] *
                           (1 + df["vol_z"].clip(0, 3) * 0.3)).clip(-1, 1).round(3)
        df["news_volume"] = df["vol_z"].clip(0, 5).round(1)

        for _, row in df.dropna(subset=["sentiment"]).iterrows():
            all_rows.append({
                "date": row["time"].strftime("%Y-%m-%d"),
                "ticker": ticker,
                "sentiment": row["sentiment"],
                "news_volume": row["news_volume"],
            })

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    print(f"  Price-derived: {len(df)} records, {df['ticker'].nunique()} tickers")
    return df


def _save_results(df_combined, date_from, date_to):
    """Pivot and save results."""
    tickers = list(TICKER_KEYWORDS.keys())

    df_sent = df_combined.pivot_table(
        index="date", columns="ticker", values="sentiment", aggfunc="mean"
    ).reset_index()
    df_sent.columns = ["date"] + [f"sent_{c}" for c in df_sent.columns[1:]]

    if "news_volume" in df_combined.columns:
        df_vol = df_combined.pivot_table(
            index="date", columns="ticker", values="news_volume", aggfunc="mean"
        ).reset_index()
        df_vol.columns = ["date"] + [f"news_count_{c}" for c in df_vol.columns[1:]]
        df_sent = df_sent.merge(df_vol, on="date", how="outer")

    # Market sentiment
    sent_cols = [c for c in df_sent.columns if c.startswith("sent_")]
    df_sent["market_sentiment"] = df_sent[sent_cols].mean(axis=1).round(3)

    # Rolling
    df_sent.sort_values("date", inplace=True)
    for col in sent_cols:
        df_sent[f"{col}_3d"] = df_sent[col].rolling(3, min_periods=1).mean().round(3)
        df_sent[f"{col}_7d"] = df_sent[col].rolling(7, min_periods=1).mean().round(3)
    df_sent["market_sentiment_3d"] = df_sent["market_sentiment"].rolling(3, min_periods=1).mean().round(3)
    df_sent["market_sentiment_7d"] = df_sent["market_sentiment"].rolling(7, min_periods=1).mean().round(3)

    df_sent.to_csv("data/news_sentiment_historical.csv", index=False)
    print(f"\nSaved: data/news_sentiment_historical.csv")
    print(f"  Rows: {len(df_sent)}, Columns: {len(df_sent.columns)}")
    print(f"  Date range: {df_sent['date'].min()} -> {df_sent['date'].max()}")

    # JSON
    ticker_sentiment = {}
    for ticker in tickers:
        col = f"sent_{ticker}"
        if col in df_sent.columns:
            last = df_sent[col].dropna().tail(5)
            if not last.empty:
                ticker_sentiment[ticker] = round(float(last.mean()), 3)

    mkt = df_sent["market_sentiment"].dropna().tail(5)
    market_sent = round(float(mkt.mean()), 3) if not mkt.empty else 0.0

    with open("data/news_sentiment.json", "w") as f:
        json.dump({
            "ticker_sentiment": ticker_sentiment,
            "market_sentiment": market_sent,
            "updated": date.today().isoformat(),
            "source": "finnhub+price_derived",
        }, f, indent=2)

    print(f"Saved: data/news_sentiment.json ({len(ticker_sentiment)} tickers)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="date_from", default="2022-01-01")
    parser.add_argument("--to", dest="date_to", default=date.today().isoformat())
    parser.add_argument("--proxy", default=None,
                        help="Proxy for Finnhub (e.g. 127.0.0.1:10085)")
    args = parser.parse_args()

    api_key = os.environ.get("FINNHUB_API_KEY", "")
    proxy = args.proxy or os.environ.get("FINNHUB_PROXY", "")

    if not api_key:
        print("ERROR: Set FINNHUB_API_KEY in .env")
        exit(1)

    print("=" * 60)
    print("Finnhub + Price-Derived Sentiment Collector")
    print(f"Period: {args.date_from} to {args.date_to}")
    print(f"Finnhub API: {'yes' if api_key else 'no'}")
    print(f"Proxy: {proxy or 'direct'}")
    print("=" * 60)

    collect_historical(api_key, args.date_from, args.date_to, proxy or None)
