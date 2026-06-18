"""
backend/collector/news_finnhub.py
=================================
Collects historical news sentiment from Finnhub API.
Free tier: 60 calls/minute.

Finnhub doesn't have Russian tickers directly, so we search
by company name in English and aggregate daily sentiment.

Usage:
    python -m backend.collector.news_finnhub
    python -m backend.collector.news_finnhub --from 2022-01-01
"""
import os, time, json, logging, argparse
from datetime import datetime, timedelta, date
import requests
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"

# Map MOEX tickers to Finnhub-searchable keywords
TICKER_KEYWORDS = {
    "GAZP": ["Gazprom"],
    "LKOH": ["Lukoil"],
    "NVTK": ["Novatek"],
    "ROSN": ["Rosneft"],
    "TATN": ["Tatneft"],
    "SNGS": ["Surgutneftegas"],
    "SIBN": ["Gazprom Neft"],
    "BANEP": ["Bashneft"],
    "SBER": ["Sberbank"],
    "VTBR": ["VTB Bank"],
    "T": ["Tinkoff", "T-Bank"],
    "MOEX": ["Moscow Exchange"],
    "GMKN": ["Nornickel", "Norilsk Nickel"],
    "PLZL": ["Polyus Gold"],
    "ALRS": ["Alrosa"],
    "CHMF": ["Severstal"],
    "NLMK": ["NLMK"],
    "MAGN": ["Magnitogorsk"],
    "YDEX": ["Yandex"],
    "OZON": ["Ozon"],
    "VKCO": ["VK Company", "VKontakte"],
    "POSI": ["Positive Technologies"],
    "MGNT": ["Magnit"],
    "X5": ["X5 Retail"],
    "LENT": ["Lenta retail"],
    "FIXP": ["Fix Price"],
    "MTSS": ["MTS Russia"],
    "RTKM": ["Rostelecom"],
    "PIKK": ["PIK Group"],
    "SMLT": ["Samolet Group"],
    "AFLT": ["Aeroflot"],
    "FESH": ["FESCO transport"],
    "UWGN": ["United Wagon"],
    "SFIN": ["SFI Group Russia"],
    "CBOM": ["Credit Bank Moscow"],
}

# Global market keywords
GLOBAL_KEYWORDS = [
    "Russia economy",
    "Russia sanctions",
    "oil OPEC crude",
    "Federal Reserve rate",
    "China trade",
    "emerging markets",
    "Russia Ukraine",
]


def _api_get(endpoint, params, api_key):
    """Make Finnhub API request with rate limiting."""
    params["token"] = api_key
    for attempt in range(3):
        try:
            r = requests.get(
                f"{FINNHUB_BASE}/{endpoint}",
                params=params,
                timeout=15,
                proxies={"http": None, "https": None},
            )
            if r.status_code == 429:
                log.warning("  Rate limited, waiting 60s...")
                time.sleep(60)
                continue
            if r.status_code != 200:
                log.debug(f"  API {r.status_code}: {r.text[:100]}")
                return None
            return r.json()
        except Exception as e:
            log.warning(f"  Request error: {e}")
            time.sleep(5)
    return None


def fetch_news_sentiment(keyword, date_from, date_to, api_key):
    """
    Fetch general news for keyword and extract sentiment.
    Uses Finnhub /news endpoint for market news
    and /company-news for company-specific news.
    """
    # Use general news search
    data = _api_get("news", {
        "category": "general",
        "minId": 0,
    }, api_key)

    if not data:
        return []

    results = []
    keyword_lower = keyword.lower()
    for article in data:
        headline = (article.get("headline", "") + " " + article.get("summary", "")).lower()
        if keyword_lower in headline:
            ts = article.get("datetime", 0)
            if ts:
                dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                # Simple sentiment from headline
                sentiment = _simple_sentiment(headline)
                results.append({"date": dt, "sentiment": sentiment})

    return results


def fetch_market_news_chunk(date_from, date_to, api_key):
    """
    Fetch market news for a date range using Finnhub general news.
    Returns daily aggregated sentiment.
    """
    data = _api_get("news", {
        "category": "general",
    }, api_key)

    if not data:
        return pd.DataFrame()

    rows = []
    for article in data:
        ts = article.get("datetime", 0)
        if ts:
            dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            headline = article.get("headline", "")
            sentiment = _simple_sentiment(headline.lower())
            rows.append({"date": dt, "sentiment": sentiment})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.groupby("date")["sentiment"].mean().reset_index()


def _simple_sentiment(text):
    """Quick sentiment scoring from headline text."""
    pos = ["growth", "rise", "gain", "surge", "rally", "profit", "record",
           "boost", "upgrade", "beat", "strong", "recovery", "deal",
           "dividend", "buyback", "optimism", "increase", "up "]
    neg = ["fall", "decline", "loss", "drop", "crash", "crisis", "risk",
           "cut", "weak", "default", "sanction", "war", "conflict",
           "recession", "bankrupt", "layoff", "downgrade", "miss",
           "fear", "concern", "threat", "slump", "collapse"]

    p = sum(1 for w in pos if w in text)
    n = sum(1 for w in neg if w in text)
    total = p + n
    if total == 0:
        return 0.0
    return round((p - n) / total, 3)


# ================================================================
# ALTERNATIVE: Use company-news endpoint (better for specific tickers)
# ================================================================

def fetch_company_news(symbol, date_from, date_to, api_key):
    """
    Fetch company news from Finnhub. Works with US tickers.
    For Russian companies we'll use general news search.
    """
    data = _api_get("company-news", {
        "symbol": symbol,
        "from": date_from,
        "to": date_to,
    }, api_key)

    if not data or not isinstance(data, list):
        return pd.DataFrame()

    rows = []
    for article in data:
        ts = article.get("datetime", 0)
        if ts:
            dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            headline = article.get("headline", "")
            sentiment = _simple_sentiment(headline.lower())
            rows.append({"date": dt, "sentiment": sentiment})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    daily = df.groupby("date").agg(
        sentiment=("sentiment", "mean"),
        count=("sentiment", "count"),
    ).reset_index()
    return daily


# ================================================================
# MAIN COLLECTOR: chunks by week to get full history
# ================================================================

def collect_historical(api_key, date_from="2022-01-01", date_to=None):
    """
    Collect historical sentiment by querying Finnhub in weekly chunks.
    Free tier = 60 calls/min, so we can do ~50 tickers * 200 weeks.
    """
    if date_to is None:
        date_to = date.today().isoformat()

    os.makedirs("data", exist_ok=True)

    # Generate weekly chunks
    chunks = []
    s = datetime.strptime(date_from, "%Y-%m-%d")
    e = datetime.strptime(date_to, "%Y-%m-%d")
    while s < e:
        chunk_end = min(s + timedelta(days=6), e)
        chunks.append((s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        s = chunk_end + timedelta(days=1)

    log.info(f"Period: {date_from} -> {date_to} ({len(chunks)} weeks)")
    log.info(f"Tickers: {len(TICKER_KEYWORDS)}")

    # ---- Ticker sentiment via general news search ----
    # Finnhub general news only returns recent articles,
    # so for historical data we use a different approach:
    # Generate proxy sentiment from price momentum + news volume

    # Try to get what we can from Finnhub for recent period
    all_rows = []
    api_calls = 0

    # For each ticker, search recent news
    for ticker, keywords in TICKER_KEYWORDS.items():
        log.info(f"[{ticker}] {keywords[0]}")

        for cs, ce in chunks[-4:]:  # last 4 weeks from Finnhub
            for kw in keywords:
                data = _api_get("news", {"category": "general"}, api_key)
                api_calls += 1

                if data and isinstance(data, list):
                    kw_lower = kw.lower()
                    for article in data:
                        headline = (article.get("headline", "") + " " +
                                    article.get("summary", "")).lower()
                        if kw_lower in headline:
                            ts = article.get("datetime", 0)
                            if ts:
                                dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                                sent = _simple_sentiment(headline)
                                all_rows.append({
                                    "date": dt, "ticker": ticker,
                                    "sentiment": sent,
                                })

                # Rate limit: 60/min
                if api_calls % 55 == 0:
                    log.info(f"  ({api_calls} calls, cooling down 60s...)")
                    time.sleep(61)
                else:
                    time.sleep(1.1)

    # ---- Build price-derived sentiment for full history ----
    log.info("\nBuilding price-derived sentiment for full history...")
    df_price_sent = _build_price_sentiment(date_from, date_to)

    # ---- Merge Finnhub + price-derived ----
    if all_rows:
        df_finnhub = pd.DataFrame(all_rows)
        df_finnhub_daily = df_finnhub.groupby(["date", "ticker"])["sentiment"].mean().reset_index()
        log.info(f"Finnhub news: {len(df_finnhub_daily)} ticker-day records")

        # Merge: prefer Finnhub where available, fallback to price-derived
        if not df_price_sent.empty:
            df_merged = df_price_sent.merge(
                df_finnhub_daily, on=["date", "ticker"], how="left", suffixes=("_price", "_news")
            )
            df_merged["sentiment"] = df_merged["sentiment_news"].fillna(df_merged["sentiment_price"])
            df_merged.drop(columns=["sentiment_price", "sentiment_news"], inplace=True, errors="ignore")
        else:
            df_merged = df_finnhub_daily
    else:
        log.info("No Finnhub news found, using price-derived sentiment only")
        df_merged = df_price_sent

    if df_merged.empty:
        log.error("No sentiment data collected!")
        return pd.DataFrame()

    # ---- Pivot to wide format for training ----
    df_pivot = df_merged.pivot_table(
        index="date", columns="ticker", values="sentiment", aggfunc="mean"
    ).reset_index()
    df_pivot.columns = ["date"] + [f"sent_{c}" for c in df_pivot.columns[1:]]

    # Add volume-based news count proxy
    if "news_volume" in df_merged.columns:
        df_vol = df_merged.pivot_table(
            index="date", columns="ticker", values="news_volume", aggfunc="mean"
        ).reset_index()
        df_vol.columns = ["date"] + [f"news_count_{c}" for c in df_vol.columns[1:]]
        df_pivot = df_pivot.merge(df_vol, on="date", how="outer")

    # Market sentiment (average across all tickers)
    sent_cols = [c for c in df_pivot.columns if c.startswith("sent_")]
    df_pivot["market_sentiment"] = df_pivot[sent_cols].mean(axis=1)

    # Rolling features
    df_pivot.sort_values("date", inplace=True)
    for col in sent_cols:
        df_pivot[f"{col}_7d"] = df_pivot[col].rolling(7, min_periods=1).mean()
    df_pivot["market_sentiment_7d"] = df_pivot["market_sentiment"].rolling(7, min_periods=1).mean()

    # Save
    df_pivot.to_csv("data/news_sentiment_historical.csv", index=False)
    log.info(f"\nSaved: data/news_sentiment_historical.csv")
    log.info(f"  Rows: {len(df_pivot)}, Columns: {len(df_pivot.columns)}")
    log.info(f"  Date range: {df_pivot['date'].min()} -> {df_pivot['date'].max()}")

    # Also save latest sentiment JSON
    ticker_sentiment = {}
    for ticker in TICKER_KEYWORDS:
        col = f"sent_{ticker}"
        if col in df_pivot.columns:
            last = df_pivot[col].dropna().tail(5)
            if not last.empty:
                ticker_sentiment[ticker] = round(float(last.mean()), 3)

    mkt = df_pivot["market_sentiment"].dropna().tail(5)
    market_sent = round(float(mkt.mean()), 3) if not mkt.empty else 0.0

    with open("data/news_sentiment.json", "w") as f:
        json.dump({
            "ticker_sentiment": ticker_sentiment,
            "market_sentiment": market_sent,
            "updated": datetime.now().isoformat(),
            "source": "finnhub+price_proxy",
        }, f, indent=2)

    log.info(f"Saved: data/news_sentiment.json ({len(ticker_sentiment)} tickers)")

    return df_pivot


def _build_price_sentiment(date_from, date_to):
    """
    Build sentiment proxy from price data in Supabase.
    Uses daily returns + volume anomalies as sentiment indicator.
    """
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv()

    try:
        sb = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
    except Exception as e:
        log.warning(f"Can't connect to Supabase: {e}")
        return pd.DataFrame()

    tickers = list(TICKER_KEYWORDS.keys())
    all_rows = []

    for ticker in tickers:
        # Fetch daily candles
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

        if len(rows) < 10:
            continue

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"])
        df.sort_values("time", inplace=True)

        # Daily return
        df["ret"] = df["close"].pct_change()
        # Volume z-score (relative to 20-day average)
        df["vol_ma"] = df["volume"].rolling(20, min_periods=5).mean()
        df["vol_z"] = (df["volume"] - df["vol_ma"]) / df["vol_ma"].replace(0, 1)

        # Sentiment = return normalized + volume signal
        # Big positive return + high volume = strong positive sentiment
        # Big negative return + high volume = strong negative sentiment
        df["sentiment"] = (df["ret"] * 10).clip(-1, 1)  # normalize returns
        # Amplify by volume anomaly
        df["sentiment"] = df["sentiment"] * (1 + df["vol_z"].clip(0, 2) * 0.3)
        df["sentiment"] = df["sentiment"].clip(-1, 1).round(3)

        # News volume proxy from actual volume anomalies
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

    df_out = pd.DataFrame(all_rows)
    log.info(f"  Price-derived sentiment: {len(df_out)} records for {df_out['ticker'].nunique()} tickers")
    return df_out


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="date_from", default="2022-01-01")
    parser.add_argument("--to", dest="date_to", default=date.today().isoformat())
    args = parser.parse_args()

    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("FINNHUB_API_KEY", "")

    if not api_key:
        print("ERROR: Set FINNHUB_API_KEY in .env or environment")
        exit(1)

    print("=" * 60)
    print("Finnhub + Price-Derived Sentiment Collector")
    print(f"Period: {args.date_from} to {args.date_to}")
    print(f"Tickers: {len(TICKER_KEYWORDS)}")
    print("=" * 60)

    df = collect_historical(api_key, args.date_from, args.date_to)
    print(f"\nDone! {len(df)} rows")
