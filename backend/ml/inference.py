from __future__ import annotations
"""
backend/ml/inference.py - Final working version
"""
import asyncio, logging, os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch

from backend.db.database import get_candles, get_indicators, get_macro, save_prediction
from backend.ml.model import StockLSTM, LSTM_FEATURES, LOOKBACK

log = logging.getLogger(__name__)

MODELS_DIR    = Path(os.environ.get("MODELS_DIR", "./models"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0.0")

CONF_THRESHOLD = 0.63
DELTA_MIN      = 1.2

LGBM_FEATURES = [
    "rsi_14", "macd_hist", "bb_position", "atr_14", "vol_ratio",
    "close_pct_1", "close_pct_3", "close_pct_5",
    "rsi_lag_1", "rsi_lag_2", "close_lag_1", "close_lag_3",
    "day_of_week", "month", "is_monday", "is_friday",
    "imoex", "usd_rub",
    "news_sentiment", "news_sentiment_3d", "news_sentiment_7d",
    "news_count", "news_count_3d", "news_count_7d",
    "market_sentiment", "market_sentiment_3d", "market_sentiment_7d",
    "market_news_count", "market_news_count_3d", "market_news_count_7d",
]

TICKERS = [
    "GAZP","LKOH","NVTK","ROSN","TATN","SNGS","SIBN","BANEP",
    "SBER","VTBR","T","MOEX","SFIN","CBOM",
    "GMKN","PLZL","ALRS","CHMF","NLMK","MAGN",
    "YDEX","OZON","VKCO","POSI",
    "MGNT","X5","LENT","FIXP",
    "MTSS","RTKM",
    "PIKK","SMLT",
    "AFLT","FESH","UWGN",
]

_cache: dict = {}


def _load_lgbm(ticker: str):
    key = f"lgbm_{ticker}"
    if key not in _cache:
        p = MODELS_DIR / f"lgbm_{ticker}.pkl"
        if not p.exists():
            raise FileNotFoundError(f"No model: {p}")
        _cache[key] = joblib.load(p)
    return _cache[key]


def _load_lgbm_features(ticker: str):
    key = f"lgbm_features_{ticker}"
    if key not in _cache:
        p = MODELS_DIR / f"lgbm_{ticker}_features.pkl"
        _cache[key] = joblib.load(p) if p.exists() else LGBM_FEATURES
    return _cache[key]


def _load_explainer(ticker: str):
    key = f"exp_{ticker}"
    if key not in _cache:
        p = MODELS_DIR / f"lgbm_{ticker}_explainer.pkl"
        _cache[key] = joblib.load(p) if p.exists() else None
    return _cache[key]


def _load_lstm(ticker: str):
    key = f"lstm_{ticker}"
    if key not in _cache:
        mp = MODELS_DIR / f"lstm_{ticker}.pt"
        sp = MODELS_DIR / f"lstm_scaler_{ticker}.pkl"
        if not mp.exists():
            _cache[key] = (None, None)
        else:
            model = StockLSTM(input_size=len(LSTM_FEATURES))
            model.load_state_dict(torch.load(mp, map_location="cpu", weights_only=True))
            model.eval()
            scaler = joblib.load(sp) if sp.exists() else None
            _cache[key] = (model, scaler)
    return _cache[key]


async def _build_feature_row(ticker: str) -> Optional[pd.Series]:
    """Builds a feature vector for tomorrow prediction."""
    candles    = await get_candles(ticker, "1d", 70)
    indicators = await get_indicators(ticker, 70)
    macro      = await get_macro(5)

    if len(candles) < 15 or len(indicators) < 15:
        log.warning(f"[{ticker}] Not enough data")
        return None

    df = pd.DataFrame(candles).merge(
        pd.DataFrame(indicators)[[
            "time","rsi_14","macd_hist","bb_upper","bb_lower","atr_14","vol_ratio"
        ]],
        on="time", how="left"
    )

    df["close"]  = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df["close_pct_1"] = df["close"].pct_change(1) * 100
    df["close_pct_3"] = df["close"].pct_change(3) * 100
    df["close_pct_5"] = df["close"].pct_change(5) * 100
    df["rsi_lag_1"]   = df["rsi_14"].shift(1)
    df["rsi_lag_2"]   = df["rsi_14"].shift(2)
    df["close_lag_1"] = df["close"].shift(1)
    df["close_lag_3"] = df["close"].shift(3)

    bb_upper = df["bb_upper"].astype(float)
    bb_lower = df["bb_lower"].astype(float)
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    df["day_of_week"] = pd.to_datetime(df["time"]).dt.dayofweek
    df["month"]       = pd.to_datetime(df["time"]).dt.month
    df["is_monday"]   = (df["day_of_week"] == 0).astype(int)
    df["is_friday"]   = (df["day_of_week"] == 4).astype(int)

    m = pd.DataFrame(macro).iloc[-1] if macro else {}
    df["imoex"]   = float(m.get("imoex",   3000) or 3000)
    df["usd_rub"] = float(m.get("usd_rub", 88)   or 88)
    df["news_sentiment"] = 0.0
    df["news_sentiment_3d"] = 0.0
    df["news_sentiment_7d"] = 0.0
    df["news_count"] = 0.0
    df["news_count_3d"] = 0.0
    df["news_count_7d"] = 0.0
    df["market_sentiment"] = 0.0
    df["market_sentiment_3d"] = 0.0
    df["market_sentiment_7d"] = 0.0
    df["market_news_count"] = 0.0
    df["market_news_count_3d"] = 0.0
    df["market_news_count_7d"] = 0.0

    df.dropna(subset=["rsi_14", "close_pct_1"], inplace=True)
    return df.iloc[-1] if not df.empty else None


def _get_shap_top(explainer, X: pd.DataFrame, n: int = 3) -> dict:
    if explainer is None:
        return {}
    try:
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        importance = dict(zip(LGBM_FEATURES, np.abs(sv[0])))
        top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:n]
        return {k: round(float(v), 4) for k, v in top}
    except Exception:
        return {}


def _make_explanation(top_features: dict, signal: str, delta: float) -> str:
    names = {
        "rsi_14": "RSI indicator",
        "macd_hist": "MACD histogram",
        "vol_ratio": "Volume ratio",
        "bb_position": "Bollinger Bands position",
        "close_pct_1": "Yesterday price change",
        "atr_14": "Volatility (ATR)",
    }
    parts  = [names.get(k, k) for k in list(top_features.keys())[:2]]
    reason = ", ".join(parts) if parts else "Technical factors"
    action = {
        "BUY":  f"Consider buying. Forecast: +{delta:.1f}%",
        "SELL": f"Possible decline of {abs(delta):.1f}%",
        "HOLD": "Neutral signal. Wait for confirmation.",
    }
    return f"{reason}. {action.get(signal, '')}"


async def predict_one(ticker: str) -> Optional[dict]:
    """Generate prediction for one ticker."""
    row = await _build_feature_row(ticker)
    if row is None:
        return None

    lgbm_features = _load_lgbm_features(ticker)
    for feature in lgbm_features:
        if feature not in row.index:
            row[feature] = 0
    X = pd.DataFrame([row[lgbm_features].fillna(0).to_dict()])

    # LightGBM
    try:
        lgbm_prob    = float(_load_lgbm(ticker).predict_proba(X)[0, 1])
        top_features = _get_shap_top(_load_explainer(ticker), X)
    except FileNotFoundError:
        lgbm_prob, top_features = 0.5, {}

    # LSTM
    lstm_delta = 0.0
    try:
        lstm_model, scaler = _load_lstm(ticker)
        if lstm_model is not None:
            candles = await get_candles(ticker, "1d", 70)
            closes  = pd.DataFrame(candles)["close"].astype(float).values
            if len(closes) >= LOOKBACK + 1:
                log_returns = np.diff(
                    np.log(closes[-LOOKBACK-1:] + 1e-9)
                ).astype(np.float32)
                seq = torch.tensor(log_returns.reshape(1, LOOKBACK, 1))
                with torch.no_grad():
                    lstm_delta = float(lstm_model(seq).item())
    except Exception as e:
        log.debug(f"[{ticker}] LSTM: {e}")

    # Ensemble
    lgbm_delta = (lgbm_prob - 0.5) * 6
    comb_delta = 0.6 * lgbm_delta + 0.4 * lstm_delta
    comb_prob  = 0.6 * lgbm_prob  + 0.4 * (1 / (1 + np.exp(-lstm_delta / 2)))

    # Signal
    if comb_prob > CONF_THRESHOLD and comb_delta > DELTA_MIN:
        signal = "BUY"
    elif comb_prob < (1 - CONF_THRESHOLD) and comb_delta < -DELTA_MIN:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Следующий торговый день MOEX (учитывает праздники, не только выходные)
    try:
        import exchange_calendars as xcals
        _moex = xcals.get_calendar("XMOS")
        target_date = _moex.next_session(date.today()).date()
    except Exception:
        target_date = date.today() + timedelta(days=1)
        while target_date.weekday() >= 5:
            target_date += timedelta(days=1)

    current_close = float(row.get("close", 0))

    return {
        "ticker":          ticker,
        "target_date":     target_date,
        "model_version":   MODEL_VERSION,
        "predicted_close": round(current_close * (1 + comb_delta / 100), 2),
        "predicted_delta": round(comb_delta, 2),
        "confidence":      round(comb_prob, 4),
        "signal":          signal,
        "top_features":    top_features,
        "current_close":   current_close,
        "explanation":     _make_explanation(top_features, signal, comb_delta),
    }


async def run_daily_predictions() -> list[dict]:
    """Generate predictions for all tickers with reconnection."""
    import asyncio
    from backend.db.database import get_pool, close_pool
    
    results = []
    for ticker in TICKERS:
        try:
            # Reconnect pool every 5 tickers to avoid timeout
            if len(results) % 5 == 0 and len(results) > 0:
                await close_pool()
                await asyncio.sleep(1)
                await get_pool()
            
            pred = await predict_one(ticker)
            if pred:
                results.append(pred)
                print(f"[{ticker}] {pred['signal']:4s} delta={pred['predicted_delta']:+.1f}% conf={pred['confidence']:.2f}")
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"[{ticker}] Error: {e}")
            # Reconnect on error
            try:
                await close_pool()
                await asyncio.sleep(2)
                await get_pool()
            except Exception:
                pass
    
    print(f"Predictions: {len(results)}/{len(TICKERS)}")
    return results
