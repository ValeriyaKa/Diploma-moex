"""
backend/collector/news_from_prices.py
=====================================
Builds market-implied sentiment from price data already in Supabase.

Approach (used in academic research):
  - Daily return → normalized to [-1, +1]
  - Volume anomaly (z-score vs 20-day MA) amplifies signal
  - Rolling averages (3d, 7d) smooth noise
  - Cross-ticker correlation captures market-wide sentiment

No external APIs needed — uses candles already in DB.

Usage:
    python -m backend.collector.news_from_prices
    python -m backend.collector.news_from_prices --from 2022-01-01
"""
import os, logging, argparse
from datetime import date
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

TICKERS = [
    "GAZP","LKOH","NVTK","ROSN","TATN","SNGS","SIBN","BANEP",
    "SBER","VTBR","T","MOEX","SFIN","CBOM",
    "GMKN","PLZL","ALRS","CHMF","NLMK","MAGN",
    "YDEX","OZON","VKCO","POSI",
    "MGNT","X5","LENT","FIXP",
    "MTSS","RTKM","PIKK","SMLT","AFLT","FESH","UWGN",
]


def _fetch_candles(sb, ticker, date_from, date_to):
    """Fetch daily candles from Supabase with pagination."""
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
    return rows


def build_sentiment(date_from="2022-01-01", date_to=None):
    """Build price-derived sentiment for all tickers."""
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv()

    if date_to is None:
        date_to = date.today().isoformat()

    sb = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
    os.makedirs("data", exist_ok=True)

    all_ticker_dfs = []

    for i, ticker in enumerate(TICKERS):
        print(f"[{i+1}/{len(TICKERS)}] {ticker}", end=" ", flush=True)

        rows = _fetch_candles(sb, ticker, date_from, date_to)
        if len(rows) < 20:
            print("skip (too few rows)")
            continue

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"])
        df.sort_values("time", inplace=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        # 1. Daily return → sentiment base
        df["ret"] = df["close"].pct_change()

        # 2. Volume z-score (vs 20-day rolling mean)
        df["vol_ma20"] = df["volume"].rolling(20, min_periods=5).mean()
        df["vol_z"] = ((df["volume"] - df["vol_ma20"]) /
                       df["vol_ma20"].replace(0, 1)).clip(-3, 5)

        # 3. Sentiment = return * (1 + volume_boost)
        #    Big move + high volume = strong signal
        #    Small move or low volume = weak signal
        df["sentiment"] = (df["ret"] * 10).clip(-1, 1)
        df["sentiment"] = (df["sentiment"] *
                           (1 + df["vol_z"].clip(0, 3) * 0.3)).clip(-1, 1)
        df["sentiment"] = df["sentiment"].round(3)

        # 4. News volume proxy from volume anomalies
        df["news_volume"] = df["vol_z"].clip(0, 5).round(1)

        df["date"] = df["time"].dt.strftime("%Y-%m-%d")
        df["ticker"] = ticker

        valid = df.dropna(subset=["sentiment"])[["date", "ticker", "sentiment", "news_volume"]]
        all_ticker_dfs.append(valid)
        print(f"{len(valid)} days")

    if not all_ticker_dfs:
        print("No data!")
        return pd.DataFrame()

    df_all = pd.concat(all_ticker_dfs, ignore_index=True)
    print(f"\nTotal: {len(df_all)} records, {df_all['ticker'].nunique()} tickers")

    # ---- Pivot to wide format ----
    df_sent = df_all.pivot_table(
        index="date", columns="ticker", values="sentiment", aggfunc="mean"
    ).reset_index()
    df_sent.columns = ["date"] + [f"sent_{c}" for c in df_sent.columns[1:]]

    df_vol = df_all.pivot_table(
        index="date", columns="ticker", values="news_volume", aggfunc="mean"
    ).reset_index()
    df_vol.columns = ["date"] + [f"news_count_{c}" for c in df_vol.columns[1:]]

    df_pivot = df_sent.merge(df_vol, on="date", how="outer")

    # ---- Market sentiment (average across all tickers) ----
    sent_cols = [c for c in df_pivot.columns if c.startswith("sent_")]
    df_pivot["market_sentiment"] = df_pivot[sent_cols].mean(axis=1).round(3)

    count_cols = [c for c in df_pivot.columns if c.startswith("news_count_")]
    df_pivot["market_news_count"] = df_pivot[count_cols].mean(axis=1).round(1)

    # ---- Rolling features ----
    df_pivot.sort_values("date", inplace=True)
    for col in sent_cols:
        df_pivot[f"{col}_3d"] = df_pivot[col].rolling(3, min_periods=1).mean().round(3)
        df_pivot[f"{col}_7d"] = df_pivot[col].rolling(7, min_periods=1).mean().round(3)

    df_pivot["market_sentiment_3d"] = df_pivot["market_sentiment"].rolling(3, min_periods=1).mean().round(3)
    df_pivot["market_sentiment_7d"] = df_pivot["market_sentiment"].rolling(7, min_periods=1).mean().round(3)

    # ---- Save CSV ----
    df_pivot.to_csv("data/news_sentiment_historical.csv", index=False)
    print(f"\nSaved: data/news_sentiment_historical.csv")
    print(f"  Rows: {len(df_pivot)}")
    print(f"  Columns: {len(df_pivot.columns)}")
    print(f"  Date range: {df_pivot['date'].min()} -> {df_pivot['date'].max()}")

    # ---- Save latest sentiment JSON ----
    import json

    ticker_sentiment = {}
    for ticker in TICKERS:
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
            "updated": date.today().isoformat(),
            "source": "price_derived",
        }, f, indent=2)

    print(f"\nSaved: data/news_sentiment.json")
    print(f"  Tickers: {len(ticker_sentiment)}")
    print(f"  Market sentiment: {market_sent:+.3f}")

    # Top/bottom
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
    args = parser.parse_args()

    print("=" * 60)
    print("Price-Derived Sentiment Builder")
    print(f"Period: {args.date_from} to {args.date_to}")
    print(f"Source: Supabase candles (no external APIs)")
    print("=" * 60)

    build_sentiment(args.date_from, args.date_to)
