"""
datasphere/export.py
====================
Exports ALL training data from Supabase → CSV files → S3.

Tables exported:
  candles + indicators → features.csv
  macro               → macro.csv
  news_sentiment      → news_sentiment.csv
  fundamentals        → fundamentals.csv

Usage:
    python datasphere/export.py                # full export + upload
    python datasphere/export.py --upload-only  # re-upload existing files
"""
import os, sys, time
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

_sb = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])

TICKERS = [
    "GAZP","LKOH","NVTK","ROSN","TATN","SNGS","SIBN","BANEP",
    "SBER","VTBR","T","MOEX","SFIN","CBOM",
    "GMKN","PLZL","ALRS","CHMF","NLMK","MAGN",
    "YDEX","OZON","VKCO","POSI",
    "MGNT","X5","LENT","FIXP",
    "MTSS","RTKM","PIKK","SMLT","AFLT","FESH","UWGN",
]


# ================================================================
# PAGINATION HELPER
# ================================================================

def _fetch_all(table, select, ticker=None, interval=None,
               order_col="time", page_size=1000):
    """Fetch all rows via Supabase API with pagination."""
    rows = []
    offset = 0
    while True:
        q = _sb.table(table).select(select)
        if ticker:
            q = q.eq("ticker", ticker)
        if interval:
            q = q.eq("interval", interval)
        q = q.order(order_col).range(offset, offset + page_size - 1)
        batch = q.execute().data
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def _fetch_table(table, select="*", order_col="time", page_size=1000):
    """Fetch entire table (no ticker filter)."""
    rows = []
    offset = 0
    while True:
        q = _sb.table(table).select(select).order(order_col)
        q = q.range(offset, offset + page_size - 1)
        batch = q.execute().data
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


# ================================================================
# EXPORT: FEATURES (candles + indicators + macro)
# ================================================================

def export_features():
    """Export candles + indicators from Supabase → features.csv"""
    print("\n" + "="*60)
    print("EXPORT: features.csv (candles + indicators)")
    print("="*60)

    # Load macro for enrichment
    macro_dict = {}
    macro_rows = _fetch_table("macro", "time,imoex,usd_rub")
    if macro_rows:
        df_m = pd.DataFrame(macro_rows)
        df_m["time"] = pd.to_datetime(df_m["time"]).dt.date
        macro_dict = df_m.set_index("time")[["imoex","usd_rub"]].to_dict("index")
        print(f"  Macro: {len(macro_dict)} records from Supabase")

    all_daily = []
    all_hourly = []

    for i, ticker in enumerate(TICKERS):
        print(f"[{i+1}/{len(TICKERS)}] {ticker}", end=" ", flush=True)
        t0 = time.time()

        # Daily candles
        candles = _fetch_all("candles", "time,close,volume", ticker, interval="1d")

        # Indicators
        indicators = _fetch_all(
            "indicators",
            "time,rsi_14,macd_hist,bb_upper,bb_lower,atr_14,vol_ratio",
            ticker,
        )

        if candles and indicators:
            df_c = pd.DataFrame(candles)
            df_i = pd.DataFrame(indicators)
            df_c["time"] = pd.to_datetime(df_c["time"])
            df_i["time"] = pd.to_datetime(df_i["time"])
            df = df_c.merge(df_i, on="time", how="inner")
            df["ticker"] = ticker
            df["timeframe"] = "1d"
            dt = df["time"].dt.date
            df["imoex"] = dt.map(lambda d: macro_dict.get(d, {}).get("imoex", 3000) or 3000)
            df["usd_rub"] = dt.map(lambda d: macro_dict.get(d, {}).get("usd_rub", 88) or 88)
            df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
            all_daily.append(df)
            print(f"1d:{len(df)}", end=" ")

        # Hourly
        hourly = _fetch_all("candles", "time,close,volume", ticker, interval="1h")
        if hourly:
            df_h = pd.DataFrame(hourly)
            df_h["ticker"] = ticker
            df_h["timeframe"] = "1h"
            for col in ["rsi_14","macd_hist","bb_upper","bb_lower",
                        "atr_14","vol_ratio","imoex","usd_rub"]:
                df_h[col] = None
            all_hourly.append(df_h)
            print(f"1h:{len(df_h)}", end=" ")

        print(f"({time.time()-t0:.1f}s)")

    dfs = []
    if all_daily: dfs.append(pd.concat(all_daily, ignore_index=True))
    if all_hourly: dfs.append(pd.concat(all_hourly, ignore_index=True))

    if not dfs:
        print("No candle data!")
        return

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("data/features.csv", index=False)
    n1d = len(df[df["timeframe"]=="1d"])
    n1h = len(df[df["timeframe"]=="1h"])
    print(f"\nSaved data/features.csv: {len(df):,} rows (1d:{n1d:,} 1h:{n1h:,}) {df['ticker'].nunique()} tickers")


# ================================================================
# EXPORT: MACRO
# ================================================================

def export_macro():
    """Export macro table from Supabase → macro.csv"""
    print("\n" + "="*60)
    print("EXPORT: macro.csv")
    print("="*60)

    rows = _fetch_table("macro", "time,imoex,usd_rub")
    if not rows:
        print("  No macro data in Supabase!")
        return

    df = pd.DataFrame(rows)
    df.to_csv("data/macro.csv", index=False)
    print(f"  Saved data/macro.csv: {len(df)} rows")


# ================================================================
# EXPORT: NEWS SENTIMENT
# ================================================================

def export_news():
    """Export news_sentiment table from Supabase → news_sentiment.csv"""
    print("\n" + "="*60)
    print("EXPORT: news_sentiment.csv")
    print("="*60)

    rows = _fetch_table(
        "news_sentiment",
        "published,source,title,sentiment,tickers,is_political",
        order_col="published",
    )
    if not rows:
        print("  No news data in Supabase!")
        return

    df = pd.DataFrame(rows)
    df.to_csv("data/news_sentiment.csv", index=False)
    print(f"  Saved data/news_sentiment.csv: {len(df)} rows")


# ================================================================
# EXPORT: FUNDAMENTALS
# ================================================================

def export_fundamentals():
    """Export fundamentals table from Supabase → fundamentals.csv"""
    print("\n" + "="*60)
    print("EXPORT: fundamentals.csv")
    print("="*60)

    # Try different column names for ordering (table schema varies)
    for order_col in ["date", "time", "report_date", "created_at"]:
        try:
            rows = _fetch_table(
                "fundamentals",
                "*",
                order_col=order_col,
            )
            break
        except Exception:
            rows = []
            continue

    if not rows:
        print("  No fundamentals data in Supabase (or table missing)!")
        print("  Using existing data/fundamentals.csv if available")
        return

    df = pd.DataFrame(rows)
    # Ensure 'date' column exists for downstream compatibility
    if "date" not in df.columns and "time" in df.columns:
        df.rename(columns={"time": "date"}, inplace=True)
    elif "date" not in df.columns and "report_date" in df.columns:
        df.rename(columns={"report_date": "date"}, inplace=True)
    df.to_csv("data/fundamentals.csv", index=False)
    print(f"  Saved data/fundamentals.csv: {len(df)} rows, {df['ticker'].nunique()} tickers")


# ================================================================
# S3 UPLOAD (via yc CLI)
# ================================================================

DATA_FILES = [
    "data/features.csv",
    "data/macro.csv",
    "data/news_sentiment_historical.csv",
    "data/news_sentiment_historical_merged.csv",
    "data/news_finbert_historical.csv",
    "data/news_sentiment.json",
    "data/fundamentals.csv",
]


def _upload_all_data():
    """Upload all data files to S3 via yc CLI."""
    import subprocess, shutil

    bucket = os.environ.get("YC_BUCKET_NAME", "moex-models-diploma")

    if not shutil.which("yc"):
        print("\nNo 'yc' CLI found, saved locally only.")
        return

    print(f"\nUploading data to s3://{bucket}/ ...")
    uploaded, skipped = 0, 0
    for path in DATA_FILES:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            try:
                subprocess.run(
                    ["yc", "storage", "s3api", "put-object",
                     "--bucket", bucket,
                     "--key", path.replace("\\", "/"),
                     "--body", path],
                    check=True, capture_output=True, text=True, timeout=300,
                )
                print(f"  {path} ({size_mb:.1f} MB)")
                uploaded += 1
            except subprocess.CalledProcessError as e:
                print(f"  {path} ERROR: {e.stderr.strip()}")
            except subprocess.TimeoutExpired:
                print(f"  {path} ERROR: timeout (>5min)")
        else:
            print(f"  {path} — not found, skipping")
            skipped += 1
    print(f"Uploaded: {uploaded}, skipped: {skipped}")


def upload_only():
    """Re-upload all existing data files without recomputing."""
    found = [f for f in DATA_FILES if os.path.exists(f)]
    if not found:
        print("No data files found. Run collectors first.")
        return
    print(f"Found {len(found)} data files to upload:")
    for f in found:
        print(f"  {f} ({os.path.getsize(f) / 1024:.0f} KB)")
    _upload_all_data()


# ================================================================
# MAIN
# ================================================================

def main():
    os.makedirs("data", exist_ok=True)

    export_features()
    export_macro()
    export_news()
    export_fundamentals()

    # Upload all to S3
    _upload_all_data()

    print("\n" + "="*60)
    print("DONE! Files ready for DataSphere training:")
    for f in DATA_FILES:
        if os.path.exists(f):
            size = os.path.getsize(f) / (1024*1024)
            print(f"  {f} ({size:.1f} MB)")
    print("="*60)


if __name__ == "__main__":
    if "--upload-only" in sys.argv:
        upload_only()
    else:
        main()
