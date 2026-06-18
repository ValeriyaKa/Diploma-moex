"""
backend/collector/news_historical.py
====================================
Collects historical news sentiment from GDELT for 2022-2026.
Uses GDELT Doc API (timelinetone) for daily sentiment per keyword.

For each ticker we search for company name keywords and get
daily average tone from world news. Tone ranges from -100 to +100,
we normalize to -1..+1.

Also collects global market sentiment:
  - "Russia economy sanctions"
  - "oil prices OPEC"
  - "Federal Reserve interest rate"
  - "China trade war"

Usage:
    python -m backend.collector.news_historical

Output:
    data/news_sentiment_historical.csv  (daily sentiment per ticker, 2022-2026)
    data/news_sentiment.json            (latest sentiment for inference)
"""
import os, time, json, logging, argparse
from datetime import datetime, timedelta, date
import requests
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ================================================================
# GDELT DOC API
# ================================================================

GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Company keywords for GDELT search (English names work best)
TICKER_KEYWORDS = {
    "GAZP": "Gazprom",
    "LKOH": "Lukoil",
    "NVTK": "Novatek gas Russia",
    "ROSN": "Rosneft",
    "TATN": "Tatneft",
    "SNGS": "Surgutneftegas",
    "SIBN": "Gazprom Neft",
    "BANEP": "Bashneft",
    "SBER": "Sberbank",
    "VTBR": "VTB Bank Russia",
    "T": "T-Technologies Tinkoff Russia",
    "MOEX": "Moscow Exchange MOEX",
    "SFIN": "SFI Russia finance",
    "CBOM": "MKB Moscow Credit Bank",
    "GMKN": "Nornickel",
    "PLZL": "Polyus Gold Russia",
    "ALRS": "Alrosa diamonds",
    "CHMF": "Severstal steel",
    "NLMK": "NLMK steel Russia",
    "MAGN": "Magnitogorsk steel",
    "YDEX": "Yandex Russia YDEX",
    "OZON": "Ozon Russia ecommerce",
    "VKCO": "VK company Russia",
    "POSI": "Positive Technologies Russia",
    "MGNT": "Magnit Russia retail",
    "X5": "X5 Retail Group Russia",
    "LENT": "Lenta Russia retail",
    "FIXP": "Fix Price Russia",
    "MTSS": "MTS Russia telecom",
    "RTKM": "Rostelecom Russia",
    "PIKK": "PIK Group Russia construction",
    "SMLT": "Samolet Group Russia",
    "AFLT": "Aeroflot Russia airline",
    "FESH": "FESCO transport Russia",
    "UWGN": "United Wagon Company Russia",
}

# Global/political sentiment keywords
GLOBAL_KEYWORDS = {
    "russia_economy": "Russia economy sanctions",
    "oil_opec": "oil prices OPEC crude",
    "fed_rate": "Federal Reserve interest rate",
    "china_trade": "China trade tariffs",
    "geopolitics": "Russia Ukraine conflict geopolitics",
    "inflation": "inflation central bank monetary policy",
    "emerging_markets": "emerging markets Russia investment",
}


_PROXY = None  # set via --proxy flag

def fetch_gdelt_tone(keyword, start_date, end_date, mode="timelinetone"):
    params = {
        "query": keyword,
        "mode": mode,
        "startdatetime": start_date.replace("-", "") + "000000",
        "enddatetime": end_date.replace("-", "") + "235959",
        "maxrecords": 250,
        "format": "json",
    }
    proxies = {"http": f"http://{_PROXY}", "https": f"http://{_PROXY}"} if _PROXY else None

    for attempt in range(5):
        try:
            r = requests.get(GDELT_API, params=params, timeout=30, proxies=proxies)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                log.warning(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                log.warning(f"  GDELT API error {r.status_code}")
                return pd.DataFrame()

            data = r.json()
            if not data or "timeline" not in data:
                return pd.DataFrame()

            timeline = data["timeline"]
            if not timeline or "data" not in timeline[0]:
                return pd.DataFrame()

            rows = []
            for entry in timeline[0]["data"]:
                dt = entry.get("date", "")
                val = entry.get("value", 0)
                if dt:
                    try:
                        d = datetime.strptime(dt[:8], "%Y%m%d").strftime("%Y-%m-%d")
                        rows.append({"date": d, "tone": float(val)})
                    except:
                        pass

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            df["tone"] = (df["tone"] / 10.0).clip(-1, 1)
            return df

        except Exception as e:
            log.warning(f"  GDELT error: {e}")
            time.sleep(10)

    return pd.DataFrame()


def fetch_gdelt_volume(keyword, start_date, end_date):
    """Fetch daily article volume for a keyword."""
    params = {
        "query": keyword,
        "mode": "timelinevolraw",
        "startdatetime": start_date.replace("-", "") + "000000",
        "enddatetime": end_date.replace("-", "") + "235959",
        "maxrecords": 250,
        "format": "json",
    }

    proxies = {"http": f"http://{_PROXY}", "https": f"http://{_PROXY}"} if _PROXY else None
    try:
        r = requests.get(GDELT_API, params=params, timeout=30, proxies=proxies)
        if r.status_code != 200:
            return pd.DataFrame()

        data = r.json()
        if not data or "timeline" not in data:
            return pd.DataFrame()

        timeline = data["timeline"]
        if not timeline or "data" not in timeline[0]:
            return pd.DataFrame()

        rows = []
        for entry in timeline[0]["data"]:
            dt = entry.get("date", "")
            val = entry.get("value", 0)
            if dt:
                try:
                    d = datetime.strptime(dt[:8], "%Y%m%d").strftime("%Y-%m-%d")
                    rows.append({"date": d, "volume": int(val)})
                except:
                    pass

        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except:
        return pd.DataFrame()


# ================================================================
# COLLECT HISTORICAL DATA IN CHUNKS
# ================================================================

def add_training_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """Add rolling news features used by DataSphere training."""
    if df_all.empty:
        return df_all

    df_all = df_all.copy()
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all.sort_values("date", inplace=True)

    sent_cols = [c for c in df_all.columns if c.startswith("sent_")]
    count_cols = [c for c in df_all.columns if c.startswith("news_count_")]

    for col in sent_cols:
        df_all[f"{col}_3d"] = df_all[col].rolling(3, min_periods=1).mean()
        df_all[f"{col}_7d"] = df_all[col].rolling(7, min_periods=1).mean()

    for col in count_cols:
        df_all[f"{col}_3d"] = df_all[col].rolling(3, min_periods=1).sum()
        df_all[f"{col}_7d"] = df_all[col].rolling(7, min_periods=1).sum()

    if "market_sentiment" in df_all.columns:
        df_all["market_sentiment_3d"] = df_all["market_sentiment"].rolling(3, min_periods=1).mean()
        df_all["market_sentiment_7d"] = df_all["market_sentiment"].rolling(7, min_periods=1).mean()
    if "market_news_count" in df_all.columns:
        df_all["market_news_count_3d"] = df_all["market_news_count"].rolling(3, min_periods=1).sum()
        df_all["market_news_count_7d"] = df_all["market_news_count"].rolling(7, min_periods=1).sum()

    df_all["date"] = df_all["date"].dt.strftime("%Y-%m-%d")
    return df_all


def collect_historical(start_date="2024-01-01", end_date=None, sleep_seconds=5.0):
    """
    Collect historical sentiment in 3-month chunks.
    GDELT API works best with shorter time ranges.
    """
    if end_date is None:
        end_date = date.today().isoformat()

    os.makedirs("data", exist_ok=True)

    # Break into 90-day chunks
    chunks = []
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    while s < e:
        chunk_end = min(s + timedelta(days=89), e)
        chunks.append((s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        s = chunk_end + timedelta(days=1)

    log.info(f"Collecting {len(chunks)} chunks from {start_date} to {end_date}")

    # --- Ticker sentiment ---
    all_ticker_data = []

    for ticker, keyword in TICKER_KEYWORDS.items():
        log.info(f"[{ticker}] Searching: '{keyword}'")
        ticker_dfs = []

        for i, (cs, ce) in enumerate(chunks):
            df_tone = fetch_gdelt_tone(keyword, cs, ce)
            df_vol = fetch_gdelt_volume(keyword, cs, ce)
            if not df_tone.empty:
                df = df_tone
                if not df_vol.empty:
                    df = df.merge(df_vol, on="date", how="left")
                else:
                    df["volume"] = 0
                df["ticker"] = ticker
                ticker_dfs.append(df)

            # Rate limiting - be nice to GDELT
            time.sleep(sleep_seconds)

        if ticker_dfs:
            df_t = pd.concat(ticker_dfs)
            all_ticker_data.append(df_t)
            log.info(f"  Got {len(df_t)} days of sentiment")
        else:
            log.info(f"  No data found")

        time.sleep(2.0)

    # --- Global/political sentiment ---
    all_global_data = []

    for key, keyword in GLOBAL_KEYWORDS.items():
        log.info(f"[GLOBAL] Searching: '{keyword}'")
        global_dfs = []

        for cs, ce in chunks:
            df_tone = fetch_gdelt_tone(keyword, cs, ce)
            df_vol = fetch_gdelt_volume(keyword, cs, ce)
            if not df_tone.empty:
                df = df_tone
                if not df_vol.empty:
                    df = df.merge(df_vol, on="date", how="left")
                else:
                    df["volume"] = 0
                df["category"] = key
                global_dfs.append(df)
            time.sleep(sleep_seconds)

        if global_dfs:
            df_g = pd.concat(global_dfs)
            all_global_data.append(df_g)
            log.info(f"  Got {len(df_g)} days")

        time.sleep(2.0)

    # ================================================================
    # SAVE HISTORICAL CSV
    # ================================================================

    # Ticker sentiment: pivot to one column per ticker
    if all_ticker_data:
        df_tickers = pd.concat(all_ticker_data)
        df_pivot = df_tickers.pivot_table(
            index="date", columns="ticker", values="tone", aggfunc="mean"
        ).reset_index()
        df_pivot.columns = ["date"] + [f"sent_{c}" for c in df_pivot.columns[1:]]

        df_vol_pivot = df_tickers.pivot_table(
            index="date", columns="ticker", values="volume", aggfunc="sum"
        ).reset_index()
        df_vol_pivot.columns = ["date"] + [f"news_count_{c}" for c in df_vol_pivot.columns[1:]]
        df_pivot = df_pivot.merge(df_vol_pivot, on="date", how="outer")
    else:
        df_pivot = pd.DataFrame({"date": []})

    # Global sentiment: average across categories
    if all_global_data:
        df_global = pd.concat(all_global_data)
        df_global_daily = df_global.groupby("date")["tone"].mean().reset_index()
        df_global_daily.columns = ["date", "market_sentiment"]

        df_global_volume = df_global.groupby("date")["volume"].sum().reset_index()
        df_global_volume.columns = ["date", "market_news_count"]
        df_global_daily = df_global_daily.merge(df_global_volume, on="date", how="outer")

        # Also get specific categories
        df_cats = df_global.pivot_table(
            index="date", columns="category", values="tone", aggfunc="mean"
        ).reset_index()
        df_cats.columns = ["date"] + [f"global_{c}" for c in df_cats.columns[1:]]
    else:
        df_global_daily = pd.DataFrame({"date": [], "market_sentiment": []})
        df_cats = pd.DataFrame({"date": []})

    # Merge all
    df_all = df_pivot
    if not df_global_daily.empty:
        df_all = df_all.merge(df_global_daily, on="date", how="outer")
    if not df_cats.empty and len(df_cats.columns) > 1:
        df_all = df_all.merge(df_cats, on="date", how="outer")

    df_all.sort_values("date", inplace=True)
    df_all = add_training_features(df_all)
    df_all.to_csv("data/news_sentiment_historical.csv", index=False)

    log.info(f"\nSaved: data/news_sentiment_historical.csv")
    log.info(f"  Rows: {len(df_all)}")
    log.info(f"  Date range: {df_all['date'].min()} -> {df_all['date'].max()}")
    log.info(f"  Columns: {df_all.columns.tolist()}")

    # ================================================================
    # ALSO UPDATE news_sentiment.json FOR LATEST PREDICTIONS
    # ================================================================

    # Get latest sentiment per ticker
    ticker_sentiment = {}
    for ticker in TICKER_KEYWORDS:
        col = f"sent_{ticker}"
        if col in df_all.columns:
            last_vals = df_all[col].dropna().tail(5)
            if not last_vals.empty:
                ticker_sentiment[ticker] = round(float(last_vals.mean()), 3)

    # Get latest market sentiment
    if "market_sentiment" in df_all.columns:
        last_market = df_all["market_sentiment"].dropna().tail(5)
        market_sent = round(float(last_market.mean()), 3) if not last_market.empty else 0.0
    else:
        market_sent = 0.0

    sentiment_json = {
        "ticker_sentiment": ticker_sentiment,
        "market_sentiment": market_sent,
        "updated": datetime.now().isoformat(),
        "source": "GDELT",
    }

    with open("data/news_sentiment.json", "w") as f:
        json.dump(sentiment_json, f, indent=2)

    log.info(f"\nSaved: data/news_sentiment.json")
    log.info(f"  Tickers with sentiment: {len(ticker_sentiment)}")
    log.info(f"  Market sentiment: {market_sent:+.3f}")

    # Show top positive/negative
    if ticker_sentiment:
        sorted_t = sorted(ticker_sentiment.items(), key=lambda x: x[1])
        log.info(f"\n  Most negative: {sorted_t[0][0]} ({sorted_t[0][1]:+.3f})")
        log.info(f"  Most positive: {sorted_t[-1][0]} ({sorted_t[-1][1]:+.3f})")

    return df_all


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="date_from", default=None)
    parser.add_argument("--to", dest="date_to", default=date.today().isoformat())
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--proxy", default=None,
                        help="HTTP proxy for GDELT (e.g. 127.0.0.1:10085)")
    args = parser.parse_args()

    if args.proxy:
        import backend.collector.news_historical as _mod
        _mod._PROXY = args.proxy

    if args.date_from is None:
        args.date_from = (date.today() - timedelta(days=365 * args.years)).isoformat()

    print("="*60)
    print("GDELT Historical News Sentiment Collector")
    print("Fetching daily sentiment for 35 tickers + global events")
    print(f"Period: {args.date_from} to {args.date_to}")
    print(f"Proxy: {args.proxy or 'direct'}")
    print("="*60)

    df = collect_historical(args.date_from, args.date_to, sleep_seconds=args.sleep)

    print(f"\nDone! {len(df)} rows collected")
    print("Files:")
    print("  data/news_sentiment_historical.csv  <- for training")
    print("  data/news_sentiment.json            <- for predictions")
