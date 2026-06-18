"""
backend/api/main.py
FastAPI server - all endpoints use psycopg2 query() for Supabase stability.
"""
from __future__ import annotations
import time, os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from dotenv import load_dotenv
load_dotenv()

from backend.db.database import (
    get_pool, close_pool,
    get_securities, get_accuracy_stats,
    get_candles, get_indicators, get_macro,
    get_predictions, get_latest_predictions, save_prediction,
)
from backend.ml.inference import predict_one, run_daily_predictions, TICKERS
from backend.ml.hourly_forecast import forecast_next_day_hourly


# ═══════════════════════════════════════════════════════════════
# SIMPLE MEMORY CACHE
# ═══════════════════════════════════════════════════════════════

_cache: dict[str, tuple[float, object]] = {}


def cache_get(key: str, ttl: int = 120):
    if key in _cache:
        ts, val = _cache[key]
        if time.time() - ts < ttl:
            return val
        del _cache[key]
    return None


def cache_set(key: str, value, ttl: int = 120):
    _cache[key] = (time.time(), value)


def cache_del(key: str):
    _cache.pop(key, None)


# ═══════════════════════════════════════════════════════════════
# LIFESPAN
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


# ═══════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="MOEX Predictor API",
    description="Stock prediction system for Moscow Exchange",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════
# SECURITIES
# ═══════════════════════════════════════════════════════════════

@app.get("/api/securities", summary="List of tracked stocks")
async def list_securities():
    return await get_securities()


# ═══════════════════════════════════════════════════════════════
# CANDLES
# ═══════════════════════════════════════════════════════════════

@app.get("/api/candles/{ticker}", summary="OHLCV candles")
async def candles(
    ticker: str,
    interval: str = Query("1d", description="Interval: 1d or 1h"),
    limit: int = Query(252, ge=1, le=2000),
    date_from: str | None = Query(None, description="Inclusive ISO start date/time"),
    date_to: str | None = Query(None, description="Exclusive ISO end date/time"),
):
    key = f"c:{ticker}:{interval}:{limit}:{date_from}:{date_to}"
    if cached := cache_get(key, ttl=60):
        return cached

    rows = await get_candles(ticker.upper(), interval, limit, date_from, date_to)
    if not rows:
        raise HTTPException(404, f"No data for {ticker}")

    cache_set(key, rows, ttl=60)
    return rows


# ═══════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════

@app.get("/api/indicators/{ticker}", summary="Technical indicators")
async def indicators(
    ticker: str,
    limit: int = Query(60, ge=5, le=500),
):
    rows = await get_indicators(ticker.upper(), limit)
    if not rows:
        raise HTTPException(404, f"No indicators for {ticker}")
    return rows


# ═══════════════════════════════════════════════════════════════
# MACRO
# ═══════════════════════════════════════════════════════════════

@app.get("/api/macro", summary="Macro data: IMOEX, USD/RUB")
async def macro(limit: int = Query(30, ge=1, le=252)):
    return await get_macro(limit)


# ═══════════════════════════════════════════════════════════════
# PREDICTIONS
# ═══════════════════════════════════════════════════════════════

@app.get("/api/predictions", summary="Latest predictions for all stocks")
async def latest_predictions():
    key = "pred:latest"
    if cached := cache_get(key, ttl=120):
        return cached

    rows = await get_latest_predictions()
    cache_set(key, rows, ttl=120)
    return rows


@app.get("/api/predictions/{ticker}", summary="Prediction history for stock")
async def ticker_predictions(
    ticker: str,
    days: int = Query(30, ge=1, le=365),
):
    rows = await get_predictions(ticker.upper(), days)
    if not rows:
        raise HTTPException(404, f"No predictions for {ticker}")
    return rows


@app.get("/api/hourly-forecast/{ticker}", summary="Hourly forecast for next trading day")
async def hourly_forecast(ticker: str):
    key = f"hourly-forecast:{ticker.upper()}"
    if cached := cache_get(key, ttl=120):
        return cached

    forecast = await forecast_next_day_hourly(ticker.upper())
    if not forecast:
        raise HTTPException(404, f"No hourly forecast for {ticker}")

    cache_set(key, forecast, ttl=120)
    return forecast


@app.post("/api/predictions/{ticker}/generate", summary="Generate prediction")
async def generate_one(ticker: str):
    pred = await predict_one(ticker.upper())
    if not pred:
        raise HTTPException(500, "Failed to generate prediction")
    pid = await save_prediction(pred)
    cache_del("pred:latest")
    return {**pred, "id": pid}


@app.post("/api/predictions/generate-all", summary="Generate all predictions")
async def generate_all():
    results = await run_daily_predictions()
    for p in results:
        await save_prediction(p)
    cache_del("pred:latest")
    return {
        "generated": len(results),
        "tickers": [r["ticker"] for r in results]
    }


# ═══════════════════════════════════════════════════════════════
# ACCURACY
# ═══════════════════════════════════════════════════════════════

@app.get("/api/accuracy", summary="Prediction accuracy stats")
async def accuracy_stats():
    rows = await get_accuracy_stats(days=30)
    return {
        row["signal"]: {
            "correct": row["correct"],
            "total": row["total"],
            "pct": round(row["correct"] / row["total"] * 100, 1),
        }
        for row in rows if row["total"] > 0
    }


# ═══════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════

@app.get("/health", summary="Service health check")
async def health():
    db_ok = False
    try:
        rows = await get_securities()
        db_ok = True
    except Exception:
        pass
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "ok" if db_ok else "error",
        "cache_keys": len(_cache),
        "tickers": len(TICKERS),
    }
