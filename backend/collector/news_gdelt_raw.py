"""
backend/collector/news_gdelt_raw.py
====================================
Collects historical news sentiment from GDELT raw GKG files.
GDELT API is blocked in RU, but raw data files at
data.gdeltproject.org are accessible.

GKG (Global Knowledge Graph) files contain:
  - Tone (sentiment) per article
  - Themes/organizations mentioned
  - Source URLs

We download daily GKG files, filter for Russian company mentions,
and aggregate daily sentiment per ticker.

Usage:
    python -m backend.collector.news_gdelt_raw
    python -m backend.collector.news_gdelt_raw --from 2023-01-01
    python -m backend.collector.news_gdelt_raw --from 2022-01-01 --sample-per-day 4
"""
import os, io, time, json, logging, argparse, zipfile
from datetime import datetime, timedelta, date
import requests
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

GDELT_BASE = "http://data.gdeltproject.org/gdeltv2"
NO_PROXY = {"http": None, "https": None}

# GKG column indices (tab-separated, no header)
# See: http://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf
GKG_DATE = 0        # YYYYMMDDHHMMSS
GKG_SRCID = 2       # SourceCollectionIdentifier (numeric)
GKG_DOMAIN = 3      # SourceCommonName — domain name (e.g. rbc.ru)
GKG_URL = 4         # DocumentIdentifier — FULL article URL
GKG_THEMES = 7      # V2Themes
GKG_PERSONS = 11    # V2Persons
GKG_ORGS = 13       # V2Organizations
GKG_TONE = 15       # V2Tone: avg_tone,pos_score,neg_score,polarity,...

# Company keywords to search in organizations/themes/sources
TICKER_KEYWORDS = {
    "GAZP": ["gazprom"],
    "LKOH": ["lukoil"],
    "NVTK": ["novatek"],
    "ROSN": ["rosneft"],
    "TATN": ["tatneft"],
    "SNGS": ["surgutneftegas"],
    "SIBN": ["gazprom neft", "gazpromneft"],
    "BANEP": ["bashneft"],
    "SBER": ["sberbank"],
    "VTBR": ["vtb bank", "vtb "],
    "T": ["tinkoff", "t-bank", "tcs group"],
    "MOEX": ["moscow exchange", "moex"],
    "GMKN": ["nornickel", "norilsk nickel", "nornikel"],
    "PLZL": ["polyus", "polus gold"],
    "ALRS": ["alrosa"],
    "CHMF": ["severstal"],
    "NLMK": ["nlmk"],
    "MAGN": ["magnitogorsk", "mmk "],
    "YDEX": ["yandex"],
    "OZON": ["ozon"],
    "VKCO": ["vkontakte", "vk company", "mail.ru"],
    "POSI": ["positive technolog"],
    "MGNT": ["magnit"],
    "X5": ["x5 retail", "x5 group", "pyaterochka"],
    "LENT": ["lenta retail", "lenta "],
    "FIXP": ["fix price"],
    "MTSS": ["mts russia", "mts telecom", "mobile telesystems"],
    "RTKM": ["rostelecom"],
    "PIKK": ["pik group"],
    "SMLT": ["samolet"],
    "AFLT": ["aeroflot"],
    "FESH": ["fesco"],
    "UWGN": ["united wagon", "uralvagonzavod"],
    "SFIN": ["sfi group"],
    "CBOM": ["credit bank of moscow", "mkb bank"],
}

# Political/market keywords
MARKET_KEYWORDS = ["russia", "russian", "moscow", "kremlin", "ruble",
                   "sanctions russia", "opec", "oil price"]


def _gkg_url(dt_str):
    """Build GKG file URL for a given datetime string YYYYMMDDHHMMSS."""
    return f"{GDELT_BASE}/{dt_str}.gkg.csv.zip"


def _get_timestamps_for_day(day: date, samples_per_day=4):
    """
    Generate GKG file timestamps for a given day.
    GKG files are published every 15 minutes.
    samples_per_day=4 means we take files at 00:00, 06:00, 12:00, 18:00.
    """
    interval = 24 // samples_per_day
    timestamps = []
    for h in range(0, 24, interval):
        ts = f"{day.strftime('%Y%m%d')}{h:02d}0000"
        timestamps.append(ts)
    return timestamps


def download_and_parse_gkg(timestamp):
    """
    Download one GKG file and extract relevant records.
    Returns list of dicts with date, ticker, tone, source.
    """
    url = _gkg_url(timestamp)
    try:
        r = requests.get(url, timeout=30, proxies=NO_PROXY)
        if r.status_code != 200:
            return []
    except Exception as e:
        log.debug(f"  Download error {timestamp}: {e}")
        return []

    # Unzip
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            fname = zf.namelist()[0]
            raw = zf.read(fname).decode("utf-8", errors="ignore")
    except Exception as e:
        log.debug(f"  Unzip error {timestamp}: {e}")
        return []

    results = []
    day_str = timestamp[:8]
    day_fmt = f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}"

    for line in raw.split("\n"):
        fields = line.split("\t")
        if len(fields) < 16:
            continue

        # Get tone
        tone_field = fields[GKG_TONE]
        if not tone_field:
            continue
        try:
            avg_tone = float(tone_field.split(",")[0])
        except (ValueError, IndexError):
            continue

        # Get searchable text (organizations + themes + domain + URL)
        orgs = fields[GKG_ORGS].lower() if len(fields) > GKG_ORGS else ""
        themes = fields[GKG_THEMES].lower() if len(fields) > GKG_THEMES else ""
        domain = fields[GKG_DOMAIN].lower() if len(fields) > GKG_DOMAIN else ""
        article_url = fields[GKG_URL].strip() if len(fields) > GKG_URL else ""
        search_text = f"{orgs} {themes} {domain} {article_url.lower()}"

        # Full article URL
        source_url = article_url if article_url.startswith("http") else ""

        # Match tickers
        for ticker, keywords in TICKER_KEYWORDS.items():
            for kw in keywords:
                if kw in search_text:
                    # Normalize tone from [-10,+10] to [-1,+1]
                    norm_tone = max(-1.0, min(1.0, avg_tone / 10.0))
                    results.append({
                        "date": day_fmt,
                        "ticker": ticker,
                        "tone": round(norm_tone, 3),
                        "url": source_url,
                    })
                    break

        # Market sentiment
        if any(kw in search_text for kw in MARKET_KEYWORDS):
            norm_tone = max(-1.0, min(1.0, avg_tone / 10.0))
            results.append({
                "date": day_fmt,
                "ticker": "_MARKET",
                "tone": round(norm_tone, 3),
                "url": source_url,
            })

    return results


def collect_historical(date_from="2022-01-01", date_to=None, samples_per_day=4):
    """
    Collect historical sentiment from GDELT raw GKG files.
    samples_per_day: how many 15-min files to sample per day (4=every 6h).
    """
    if date_to is None:
        date_to = date.today().isoformat()

    os.makedirs("data", exist_ok=True)

    start = datetime.strptime(date_from, "%Y-%m-%d").date()
    end = datetime.strptime(date_to, "%Y-%m-%d").date()
    total_days = (end - start).days + 1

    print(f"Collecting {total_days} days, {samples_per_day} samples/day")
    print(f"Total files to download: ~{total_days * samples_per_day}")

    all_results = []
    current = start
    day_num = 0

    while current <= end:
        day_num += 1
        timestamps = _get_timestamps_for_day(current, samples_per_day)
        day_records = []

        for ts in timestamps:
            records = download_and_parse_gkg(ts)
            day_records.extend(records)
            time.sleep(0.3)  # be nice to GDELT servers

        ticker_count = len(set(r["ticker"] for r in day_records if r["ticker"] != "_MARKET"))

        if day_num % 30 == 0 or day_num <= 3:
            print(f"[{day_num}/{total_days}] {current} -> "
                  f"{len(day_records)} records, {ticker_count} tickers")

        all_results.extend(day_records)
        current += timedelta(days=1)

    if not all_results:
        print("No data collected!")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    print(f"\nTotal raw records: {len(df)}")
    print(f"Tickers found: {df[df['ticker'] != '_MARKET']['ticker'].nunique()}")

    # Save raw records with URLs for FinBERT enrichment later
    if "url" not in df.columns:
        print("WARNING: 'url' column missing! Check GKG_DOCID index.")
    else:
        url_counts = df["url"].str.len().describe()
        print(f"\nURL stats: {(df['url'].str.len() > 0).sum()} non-empty "
              f"out of {len(df)} total")
        # Show sample URLs
        sample_urls = df[df["url"].str.len() > 0]["url"].head(5).tolist()
        if sample_urls:
            print("Sample URLs:")
            for u in sample_urls:
                print(f"  {u[:100]}")

    df_with_urls = df[df.get("url", pd.Series(dtype=str)).str.len() > 0].copy() \
        if "url" in df.columns else pd.DataFrame()
    if not df_with_urls.empty:
        raw_path = "data/gdelt_articles_urls.csv"
        df_with_urls.to_csv(raw_path, index=False)
        print(f"Saved article URLs: {raw_path} ({len(df_with_urls)} records)")
        # Show top domains
        domains = df_with_urls["url"].str.extract(r"https?://([^/]+)")[0].value_counts().head(10)
        print("Top domains:")
        for domain, cnt in domains.items():
            print(f"  {domain}: {cnt}")

    # ---- Aggregate daily per ticker ----
    df_tickers = df[df["ticker"] != "_MARKET"]
    df_market = df[df["ticker"] == "_MARKET"]

    # Pivot: daily sentiment per ticker
    if not df_tickers.empty:
        df_sent = df_tickers.groupby(["date", "ticker"]).agg(
            tone=("tone", "mean"),
            count=("tone", "count"),
        ).reset_index()

        df_tone_pivot = df_sent.pivot_table(
            index="date", columns="ticker", values="tone", aggfunc="mean"
        ).reset_index()
        df_tone_pivot.columns = ["date"] + [f"sent_{c}" for c in df_tone_pivot.columns[1:]]

        df_count_pivot = df_sent.pivot_table(
            index="date", columns="ticker", values="count", aggfunc="sum"
        ).reset_index()
        df_count_pivot.columns = ["date"] + [f"news_count_{c}" for c in df_count_pivot.columns[1:]]

        df_pivot = df_tone_pivot.merge(df_count_pivot, on="date", how="outer")
    else:
        df_pivot = pd.DataFrame({"date": []})

    # Market sentiment
    if not df_market.empty:
        df_mkt = df_market.groupby("date").agg(
            market_sentiment=("tone", "mean"),
            market_news_count=("tone", "count"),
        ).reset_index()
        df_pivot = df_pivot.merge(df_mkt, on="date", how="outer")
    else:
        df_pivot["market_sentiment"] = 0.0
        df_pivot["market_news_count"] = 0

    # Rolling features
    df_pivot.sort_values("date", inplace=True)
    sent_cols = [c for c in df_pivot.columns if c.startswith("sent_")]

    for col in sent_cols:
        df_pivot[f"{col}_3d"] = df_pivot[col].rolling(3, min_periods=1).mean().round(3)
        df_pivot[f"{col}_7d"] = df_pivot[col].rolling(7, min_periods=1).mean().round(3)

    if "market_sentiment" in df_pivot.columns:
        df_pivot["market_sentiment_3d"] = df_pivot["market_sentiment"].rolling(3, min_periods=1).mean().round(3)
        df_pivot["market_sentiment_7d"] = df_pivot["market_sentiment"].rolling(7, min_periods=1).mean().round(3)

    # ---- Save ----
    df_pivot.to_csv("data/news_sentiment_historical.csv", index=False)
    print(f"\nSaved: data/news_sentiment_historical.csv")
    print(f"  Rows: {len(df_pivot)}")
    print(f"  Columns: {len(df_pivot.columns)}")
    if len(df_pivot) > 0:
        print(f"  Date range: {df_pivot['date'].min()} -> {df_pivot['date'].max()}")
        print(f"  Tickers with sentiment: {len(sent_cols)}")

    # Save latest sentiment JSON
    ticker_sentiment = {}
    for ticker in TICKER_KEYWORDS:
        col = f"sent_{ticker}"
        if col in df_pivot.columns:
            last = df_pivot[col].dropna().tail(5)
            if not last.empty:
                ticker_sentiment[ticker] = round(float(last.mean()), 3)

    mkt = df_pivot.get("market_sentiment", pd.Series(dtype=float)).dropna().tail(5)
    market_sent = round(float(mkt.mean()), 3) if not mkt.empty else 0.0

    with open("data/news_sentiment.json", "w") as f:
        json.dump({
            "ticker_sentiment": ticker_sentiment,
            "market_sentiment": market_sent,
            "updated": date.today().isoformat(),
            "source": "GDELT_raw_GKG",
        }, f, indent=2)

    print(f"Saved: data/news_sentiment.json ({len(ticker_sentiment)} tickers)")

    if ticker_sentiment:
        s = sorted(ticker_sentiment.items(), key=lambda x: x[1])
        print(f"  Most negative: {s[0][0]} ({s[0][1]:+.3f})")
        print(f"  Most positive: {s[-1][0]} ({s[-1][1]:+.3f})")

    return df_pivot


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="date_from", default="2022-01-01")
    parser.add_argument("--to", dest="date_to", default=date.today().isoformat())
    parser.add_argument("--sample-per-day", type=int, default=4,
                        help="GKG files to sample per day (1-96, default 4)")
    args = parser.parse_args()

    print("=" * 60)
    print("GDELT Raw GKG Sentiment Collector")
    print(f"Period: {args.date_from} to {args.date_to}")
    print(f"Samples per day: {args.sample_per_day}")
    print(f"Source: data.gdeltproject.org (raw files)")
    print("=" * 60)

    collect_historical(args.date_from, args.date_to, args.sample_per_day)
