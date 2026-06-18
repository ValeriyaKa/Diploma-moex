from __future__ import annotations
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
_sb = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])


# ── Compat stubs для FastAPI lifespan и health-check ─────────

async def get_pool():
    return _PoolStub()

async def close_pool():
    pass


class _PoolStub:
    async def fetchval(self, sql, *args):
        return 1


@asynccontextmanager
async def db_conn():
    yield _PoolStub()


# ── Securities ───────────────────────────────────────────────

async def get_securities():
    return _sb.table("securities")\
        .select("ticker,short_name,sector,is_active")\
        .order("ticker").execute().data


async def get_accuracy_stats(days: int = 30):
    from datetime import date, timedelta
    cutoff = str(date.today() - timedelta(days=days))
    rows = _sb.table("predictions")\
        .select("signal,is_correct")\
        .gte("target_date", cutoff)\
        .execute().data
    stats: dict[str, dict] = {}
    for row in rows:
        sig = row["signal"]
        if sig not in stats:
            stats[sig] = {"correct": 0, "total": 0}
        if row["is_correct"] is not None:
            stats[sig]["total"] += 1
            if row["is_correct"]:
                stats[sig]["correct"] += 1
    return [{"signal": k, **v} for k, v in stats.items()]


# ── Candles ───────────────────────────────────────────────────

async def get_candles(
    ticker: str,
    interval: str = "1d",
    limit: int = 252,
    date_from: str | None = None,
    date_to: str | None = None,
):
    query = _sb.table("candles")\
        .select("time,open,high,low,close,volume")\
        .eq("ticker", ticker).eq("interval", interval)
    if date_from:
        query = query.gte("time", date_from)
    if date_to:
        query = query.lt("time", date_to)
    data = query.order("time", desc=True).limit(limit).execute().data
    return list(reversed(data))


async def upsert_candles(records: list[tuple]) -> int:
    if not records: return 0
    cols = ["time", "ticker", "interval", "open", "high", "low", "close", "volume", "value"]
    dicts = [{k: (str(v) if k == "time" else v) for k, v in zip(cols, r)} for r in records]
    for i in range(0, len(dicts), 25):
        _sb.table("candles").upsert(
            dicts[i:i+25], on_conflict="time,ticker,interval"
        ).execute()
    return len(records)


# ── Indicators ────────────────────────────────────────────────

async def get_indicators(ticker: str, limit: int = 252):
    data = _sb.table("indicators")\
        .select("time,rsi_14,macd_hist,bb_upper,bb_lower,atr_14,vol_ratio,sma_20,ema_12")\
        .eq("ticker", ticker)\
        .order("time", desc=True).limit(limit).execute().data
    return list(reversed(data))


async def upsert_indicators(records: list[tuple]) -> int:
    if not records: return 0
    cols = [
        "time", "ticker", "sma_10", "sma_20", "sma_50", "ema_12", "ema_26", "rsi_14",
        "macd", "macd_signal", "macd_hist", "bb_upper", "bb_middle", "bb_lower",
        "atr_14", "obv", "vol_ratio",
    ]
    dicts = [{k: (str(v) if k == "time" else v) for k, v in zip(cols, r)} for r in records]
    for i in range(0, len(dicts), 25):
        _sb.table("indicators").upsert(
            dicts[i:i+25], on_conflict="time,ticker"
        ).execute()
    return len(records)


# ── Macro ─────────────────────────────────────────────────────

async def get_macro(limit: int = 60):
    data = _sb.table("macro")\
        .select("*")\
        .order("time", desc=True).limit(limit).execute().data
    return list(reversed(data))


# ── Predictions ───────────────────────────────────────────────

async def save_prediction(pred: dict):
    result = _sb.table("predictions").upsert({
        "target_date":     str(pred["target_date"]),
        "ticker":          pred["ticker"],
        "model_version":   pred.get("model_version", "v1.0.0"),
        "predicted_close": pred.get("predicted_close"),
        "predicted_delta": pred["predicted_delta"],
        "confidence":      pred["confidence"],
        "signal":          pred["signal"],
        "top_features":    pred.get("top_features", {}),
    }, on_conflict="ticker,target_date,model_version").execute()
    return result.data[0]["id"] if result.data else 0


async def get_predictions(ticker: str, days: int = 30):
    return _sb.table("predictions")\
        .select("target_date,predicted_delta,confidence,signal,"
                "top_features,actual_delta,is_correct,created_at")\
        .eq("ticker", ticker)\
        .order("target_date", desc=True).limit(days).execute().data


async def get_latest_predictions():
    """Use supabase client instead of psycopg2 for reliability."""
    try:
        from supabase import create_client
        sb = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
        
        # Get latest prediction per ticker
        rows = sb.table("predictions")\
            .select("ticker,target_date,predicted_delta,confidence,signal,top_features")\
            .order("created_at", desc=True)\
            .limit(100)\
            .execute().data
        
        # Deduplicate - keep latest per ticker
        seen = {}
        for r in rows:
            if r["ticker"] not in seen:
                seen[r["ticker"]] = r
        
        # Add security info
        secs = sb.table("securities")\
            .select("ticker,short_name,sector")\
            .execute().data
        sec_map = {s["ticker"]: s for s in secs}
        
        result = []
        for ticker, pred in seen.items():
            sec = sec_map.get(ticker, {})
            pred["short_name"] = sec.get("short_name", ticker)
            pred["sector"] = sec.get("sector", "")
            pred["current_close"] = None
            pred["volume"] = None
            result.append(pred)
        
        return result
    except Exception as e:
        print(f"get_latest_predictions error: {e}")
        return []
