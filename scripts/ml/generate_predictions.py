"""
generate_predictions.py - V3: Full 3-model ensemble
  1. LightGBM  (daily, 20 features)      -> probability of up/down
  2. LSTM daily (60 days, 11 features)    -> daily % change prediction
  3. LSTM hourly (48 hours, 8 features)   -> intraday momentum signal

Ensemble: 50% LightGBM + 30% LSTM daily + 20% LSTM hourly
News sentiment and macro adjust the final probability.
"""
import os, json, joblib, time
import numpy as np, pandas as pd, torch
from datetime import date, timedelta
from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv
load_dotenv()

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "./models"))
CONF_THRESHOLD = 0.50
DELTA_MIN = 0.3

LGBM_FEATURES_DEFAULT = [
    "rsi_14","macd_hist","bb_position","atr_14","vol_ratio",
    "close_pct_1","close_pct_3","close_pct_5",
    "rsi_lag_1","rsi_lag_2","close_lag_1","close_lag_3",
    "day_of_week","month","is_monday","is_friday",
    "imoex","usd_rub","news_sentiment","market_sentiment",
    "news_sentiment_3d","news_sentiment_7d","news_count","news_count_3d","news_count_7d",
    "market_sentiment_3d","market_sentiment_7d","market_news_count","market_news_count_3d","market_news_count_7d",
]

LSTM_D_FEATURES_DEFAULT = [
    "close_pct_1","close_pct_3","rsi_14","macd_hist",
    "bb_position","vol_ratio","atr_14",
    "imoex","usd_rub","news_sentiment","market_sentiment",
    "news_sentiment_3d","news_sentiment_7d","market_sentiment_3d","market_sentiment_7d",
]

LSTM_H_FEATURES_DEFAULT = [
    "close_pct_1h","close_pct_3h","vol_ratio_h",
    "hour_sin","hour_cos","is_open_hour","is_close_hour","day_pct",
]

LOOKBACK_D = 60
LOOKBACK_H = 48

TICKERS = [
    "GAZP","LKOH","NVTK","ROSN","TATN","SNGS","SIBN","BANEP",
    "SBER","VTBR","T","MOEX","SFIN","CBOM",
    "GMKN","PLZL","ALRS","CHMF","NLMK","MAGN",
    "YDEX","OZON","VKCO","POSI",
    "MGNT","X5","LENT","FIXP",
    "MTSS","RTKM","PIKK","SMLT","AFLT","FESH","UWGN",
]

supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])


class StockLSTM(torch.nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers,
                                   batch_first=True,
                                   dropout=dropout if num_layers > 1 else 0.0)
        self.att = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 1), torch.nn.Softmax(dim=1))
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 128), torch.nn.GELU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64), torch.nn.GELU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32), torch.nn.GELU(),
            torch.nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        ctx = (out * self.att(out)).sum(1)
        return self.head(ctx).squeeze(-1)
    
# ================================================================
# DATA LOADING
# ================================================================

def get_daily_data(ticker):
    candles = supabase.table("candles")\
        .select("time,open,high,low,close,volume")\
        .eq("ticker", ticker).eq("interval", "1d")\
        .order("time", desc=True).limit(80).execute().data
    indicators = supabase.table("indicators")\
        .select("time,rsi_14,macd_hist,bb_upper,bb_lower,atr_14,vol_ratio")\
        .eq("ticker", ticker)\
        .order("time", desc=True).limit(80).execute().data
    return list(reversed(candles)), list(reversed(indicators))


def get_hourly_data(ticker):
    candles = supabase.table("candles")\
        .select("time,close,volume")\
        .eq("ticker", ticker).eq("interval", "1h")\
        .order("time", desc=True).limit(100).execute().data
    return list(reversed(candles))


def load_macro():
    if os.path.exists("data/macro.csv"):
        df = pd.read_csv("data/macro.csv")
        imoex_rows = df.dropna(subset=["imoex"])
        usd_rows = df.dropna(subset=["usd_rub"])
        return {
            "imoex": float(imoex_rows.iloc[-1]["imoex"]) if not imoex_rows.empty else 3000,
            "usd_rub": float(usd_rows.iloc[-1]["usd_rub"]) if not usd_rows.empty else 88,
        }
    return {"imoex": 3000.0, "usd_rub": 88.0}


def load_news():
    if os.path.exists("data/news_sentiment.json"):
        with open("data/news_sentiment.json") as f:
            data = json.load(f)
        return data.get("ticker_sentiment", {}), data.get("market_sentiment", 0.0)
    return {}, 0.0


# ================================================================
# FEATURE BUILDING
# ================================================================

def build_daily_features(candles, indicators, macro, news_sent, market_sent, ticker=None):
    if len(candles) < 25 or len(indicators) < 15:
        return None
    df = pd.DataFrame(candles).merge(pd.DataFrame(indicators), on="time", how="left")
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["open"] = pd.to_numeric(df.get("open", df["close"]), errors="coerce")
    df["high"] = pd.to_numeric(df.get("high", df["close"]), errors="coerce")
    df["low"] = pd.to_numeric(df.get("low", df["close"]), errors="coerce")

    # Momentum
    for d in [1, 3, 5, 10, 20]:
        df[f"close_pct_{d}"] = df["close"].pct_change(d) * 100
    df["gap_open"] = ((df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + 1e-9)) * 100
    df["high_low_range"] = ((df["high"] - df["low"]) / (df["close"] + 1e-9)) * 100

    # Volume spike
    vol_avg = df["volume"].rolling(20).mean()
    df["volume_spike"] = (df["volume"] > vol_avg * 2).astype(float)

    # RSI divergence
    rsi = pd.to_numeric(df["rsi_14"], errors="coerce")
    price_5 = df["close"].rolling(5).max()
    rsi_5 = rsi.rolling(5).max()
    df["rsi_divergence"] = (
        (df["close"] >= price_5 * 0.99).astype(float) -
        (rsi_5 < rsi_5.shift(5)).astype(float)
    )

    # Lagged
    df["rsi_lag_1"] = rsi.shift(1)
    df["rsi_lag_2"] = rsi.shift(2)
    df["close_lag_1"] = df["close"].shift(1)
    df["close_lag_3"] = df["close"].shift(3)

    # BB position
    bb_u = pd.to_numeric(df.get("bb_upper", 0), errors="coerce")
    bb_l = pd.to_numeric(df.get("bb_lower", 0), errors="coerce")
    df["bb_position"] = (df["close"] - bb_l) / (bb_u - bb_l + 1e-9)

    # Calendar
    dt = pd.to_datetime(df["time"])
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)

    # Macro
    # Macro
    df["imoex"] = macro["imoex"]
    df["usd_rub"] = macro["usd_rub"]
    # Read macro history for pct change
    if os.path.exists("data/macro.csv"):
        df_m = pd.read_csv("data/macro.csv")
        imoex_vals = pd.to_numeric(df_m["imoex"], errors="coerce").dropna()
        usd_vals = pd.to_numeric(df_m["usd_rub"], errors="coerce").dropna()
        df["imoex_pct"] = float(imoex_vals.pct_change().iloc[-1] * 100) if len(imoex_vals) > 1 else 0.0
        df["usd_rub_pct"] = float(usd_vals.pct_change().iloc[-1] * 100) if len(usd_vals) > 1 else 0.0
    else:
        df["imoex_pct"] = 0.0
        df["usd_rub_pct"] = 0.0

    # Sentiment
    df["news_sentiment"] = news_sent
    df["market_sentiment"] = market_sent
    df["news_sentiment_3d"] = news_sent
    df["news_sentiment_7d"] = news_sent
    df["news_count"] = 0
    df["news_count_3d"] = 0
    df["news_count_7d"] = 0
    df["market_sentiment_3d"] = market_sent
    df["market_sentiment_7d"] = market_sent
    df["market_news_count"] = 0
    df["market_news_count_3d"] = 0
    df["market_news_count_7d"] = 0

    # Fundamental features — load from fundamentals.csv
    _fund_feats = ["days_to_div", "days_from_div", "div_value", "is_div_week",
                   "pe_ratio", "pb_ratio", "ev_ebitda", "div_yield", "roe",
                   "is_report_season", "is_quarter_end"]
    fund_loaded = False
    if ticker and os.path.exists("data/fundamentals.csv"):
        try:
            df_fund = pd.read_csv("data/fundamentals.csv")
            tf = df_fund[df_fund["ticker"] == ticker].copy()
            if not tf.empty:
                tf["date"] = pd.to_datetime(tf["date"]).dt.strftime("%Y-%m-%d")
                tf = tf.set_index("date")
                df["_date_str"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
                for feat in _fund_feats:
                    if feat in tf.columns:
                        df[feat] = df["_date_str"].map(tf[feat].to_dict()).fillna(0)
                    else:
                        df[feat] = 0
                df.drop(columns=["_date_str"], inplace=True)
                fund_loaded = True
        except Exception as e:
            log.debug(f"Fundamentals load error: {e}") if 'log' in dir() else None

    if not fund_loaded:
        for feat in _fund_feats:
            df[feat] = 0

    df.dropna(subset=["rsi_14", "close_pct_1"], inplace=True)
    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df if not df.empty else None

def build_hourly_features(hourly_candles):
    if len(hourly_candles) < 55:
        return None
    df = pd.DataFrame(hourly_candles)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["time"] = pd.to_datetime(df["time"])
    df["close_pct_1h"] = df["close"].pct_change(1) * 100
    df["close_pct_3h"] = df["close"].pct_change(3) * 100
    df["vol_ratio_h"] = df["volume"] / df["volume"].rolling(20).mean()
    hour = df["time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["is_open_hour"] = (hour == 10).astype(int)
    df["is_close_hour"] = (hour >= 18).astype(int)
    df["date"] = df["time"].dt.date
    day_open = df.groupby("date")["close"].transform("first")
    df["day_pct"] = ((df["close"] - day_open) / (day_open + 1e-9)) * 100
    df.dropna(subset=["close_pct_1h"], inplace=True)
    for col in df.columns:
        if col not in ("time", "date"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df if not df.empty else None


# ================================================================
# LSTM INFERENCE HELPER
# ================================================================

def _load_lstm_arch(prefix, ticker, state_dict):
    params_path = MODELS_DIR / f"{prefix}_params_{ticker}.json"
    if params_path.exists():
        try:
            with open(params_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            return {
                "hidden_size": int(params.get("hidden", 128)),
                "num_layers": int(params.get("layers", 3)),
                "dropout": float(params.get("dropout", 0.3)),
            }
        except Exception as exc:
            print(f"  {prefix} params warning: {exc}")

    # Fallback for models trained before params json was copied.
    hidden_size = int(state_dict["lstm.weight_hh_l0"].shape[1])
    num_layers = sum(1 for key in state_dict if key.startswith("lstm.weight_ih_l"))
    return {"hidden_size": hidden_size, "num_layers": num_layers, "dropout": 0.3}


def run_lstm(ticker, df, features_default, lookback, prefix):
    """Run LSTM model and return raw output."""
    feat_path = MODELS_DIR / f"{prefix}_features_{ticker}.pkl"
    features = joblib.load(feat_path) if feat_path.exists() else \
        [f for f in features_default if f in df.columns]

    model_path = MODELS_DIR / f"{prefix}_{ticker}.pt"
    scaler_path = MODELS_DIR / f"{prefix}_scaler_{ticker}.pkl"

    if not model_path.exists() or not scaler_path.exists():
        return None
    if len(df) < lookback:
        return None

    try:
        scaler = joblib.load(scaler_path)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        arch = _load_lstm_arch(prefix, ticker, state)
        model = StockLSTM(input_size=len(features), **arch)
        model.load_state_dict(state)
        model.eval()

        seq_data = df[features].fillna(0).values[-lookback:].astype(np.float32)
        dummy = np.zeros((len(seq_data), 1), dtype=np.float32)
        scaled = scaler.transform(np.hstack([seq_data, dummy]))
        seq_tensor = torch.tensor(scaled[:, :-1].reshape(1, lookback, len(features)))

        with torch.no_grad():
            return float(model(seq_tensor).item())
    except Exception as e:
        print(f"  {prefix} error: {e}")
        return None


# ================================================================
# ENSEMBLE
# ================================================================

def predict_full(ticker, df_daily, df_hourly, news_sent, market_sent):
    row = df_daily.iloc[-1]
    current_close = float(row["close"])

    # --- 1. LightGBM ---
    lgbm_prob = 0.5
    top_features = {}

    feat_path = MODELS_DIR / f"lgbm_{ticker}_features.pkl"
    lgbm_features = joblib.load(feat_path) if feat_path.exists() else \
        [f for f in LGBM_FEATURES_DEFAULT if f in df_daily.columns]

    lgbm_path = MODELS_DIR / f"lgbm_{ticker}.pkl"
    if lgbm_path.exists():
        model = joblib.load(lgbm_path)
        X = pd.DataFrame([row[lgbm_features].fillna(0).to_dict()])
        lgbm_prob = float(model.predict_proba(X)[0, 1])

        exp_path = MODELS_DIR / f"lgbm_{ticker}_explainer.pkl"
        if exp_path.exists():
            try:
                exp = joblib.load(exp_path)
                sv = exp.shap_values(X)
                if isinstance(sv, list): sv = sv[1]
                imp = dict(zip(lgbm_features, np.abs(sv[0])))
                top = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
                top_features = {k: round(float(v), 4) for k, v in top}
            except: pass

    # --- 2. LSTM daily ---
    lstm_d = run_lstm(ticker, df_daily, LSTM_D_FEATURES_DEFAULT, LOOKBACK_D, "lstm")
    lstm_d = lstm_d if lstm_d is not None else 0.0

    # --- 3. LSTM hourly ---
    lstm_h = 0.0
    if df_hourly is not None:
        lstm_h_raw = run_lstm(ticker, df_hourly, LSTM_H_FEATURES_DEFAULT, LOOKBACK_H, "lstm_hourly")
        lstm_h = lstm_h_raw if lstm_h_raw is not None else 0.0

    # --- Ensemble: 50% LightGBM + 30% LSTM daily + 20% LSTM hourly ---
   # --- XGBoost ---
    xgb_prob = 0.5
    xgb_path = MODELS_DIR / f"xgb_{ticker}.pkl"
    if xgb_path.exists():
        xgb_model = joblib.load(xgb_path)
        xgb_prob = float(xgb_model.predict_proba(X)[0, 1])

    # --- Ensemble: 50% LightGBM + 10% XGBoost + 40% LSTM daily ---
    lstm_d_prob = 1 / (1 + np.exp(-lstm_d * 3))  # amplify LSTM signal
    lstm_h_prob = 1 / (1 + np.exp(-lstm_h * 3))

    ensemble_prob = 0.5 * lgbm_prob + 0.1 * xgb_prob + 0.4 * lstm_d_prob + 0.0 * lstm_h_prob

    # News and macro adjustment
    ensemble_prob += news_sent * 0.05 + market_sent * 0.03
    ensemble_prob = max(0.01, min(0.99, ensemble_prob))
    ensemble_delta = (ensemble_prob - 0.5) * 6

    # Signal
    if ensemble_prob > CONF_THRESHOLD and ensemble_delta > DELTA_MIN:
        signal = "BUY"
    elif ensemble_prob < (1 - CONF_THRESHOLD) and ensemble_delta < -DELTA_MIN:
        signal = "SELL"
    else:
        signal = "HOLD"

    td = date.today() + timedelta(days=1)
    while td.weekday() >= 5: td += timedelta(days=1)

    if abs(news_sent) > 0.01: top_features["news_sentiment"] = round(news_sent, 3)
    if abs(market_sent) > 0.01: top_features["market_sentiment"] = round(market_sent, 3)

    return {
        "target_date": td, "ticker": ticker, "model_version": "v1.0.0",
        "predicted_close": round(current_close * (1 + ensemble_delta / 100), 2),
        "predicted_delta": round(ensemble_delta, 2),
        "confidence": round(ensemble_prob, 4),
        "signal": signal,
        "top_features": top_features,
        "lgbm_prob": round(lgbm_prob, 4),
        "xgb_prob": round(xgb_prob, 4),
        "lstm_daily": round(lstm_d, 4),
        "lstm_hourly": round(lstm_h, 4),
    }


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("="*60)
    print("MOEX Predictor V3 - Full 3-Model Ensemble")
    print("  LightGBM (50%) + LSTM daily (30%) + LSTM hourly (20%)")
    print("  + News sentiment + Macro adjustment")
    print("="*60)

    macro = load_macro()
    print(f"\nMacro: IMOEX={macro['imoex']:.0f}, USD/RUB={macro['usd_rub']:.2f}")

    ticker_sentiment, market_sentiment = load_news()
    print(f"Market sentiment: {market_sentiment:+.3f}")

    results = []
    for i, ticker in enumerate(TICKERS):
        try:
            print(f"\n[{i+1}/{len(TICKERS)}] {ticker}")

            # Daily data
            candles, indicators = get_daily_data(ticker)
            news_sent = ticker_sentiment.get(ticker, 0.0)
            df_daily = build_daily_features(candles, indicators, macro, news_sent, market_sentiment, ticker=ticker)
            if df_daily is None:
                print("  Skip: not enough daily data")
                continue

            # Hourly data
            hourly = get_hourly_data(ticker)
            df_hourly = build_hourly_features(hourly)

            # Full ensemble
            pred = predict_full(ticker, df_daily, df_hourly, news_sent, market_sentiment)
            results.append(pred)

            h_str = f" h={pred['lstm_hourly']:+.2f}" if abs(pred['lstm_hourly']) > 0.001 else " h=N/A"
            n_str = f" news={news_sent:+.2f}" if abs(news_sent) > 0.01 else ""
            print(f"  => {pred['signal']:4s} delta={pred['predicted_delta']:+.1f}% "
                  f"conf={pred['confidence']:.0%} "
                  f"(lgbm={pred['lgbm_prob']:.0%} d={pred['lstm_daily']:+.2f}{h_str})"
                  f"{n_str}")

        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(0.2)

    # Summary
    buys  = sum(1 for r in results if r["signal"] == "BUY")
    sells = sum(1 for r in results if r["signal"] == "SELL")
    holds = sum(1 for r in results if r["signal"] == "HOLD")
    avg_conf = sum(r["confidence"] for r in results) / max(len(results), 1)
    print(f"\n{'='*60}")
    print(f"Summary: {buys} BUY | {sells} SELL | {holds} HOLD | avg conf: {avg_conf:.0%}")
    print(f"{'='*60}")

    # Save
    print(f"\nSaving {len(results)} predictions...")
    saved = 0
    for p in results:
        try:
            supabase.table("predictions").upsert({
                "target_date":     str(p["target_date"]),
                "ticker":          p["ticker"],
                "model_version":   p["model_version"],
                "predicted_close": p.get("predicted_close"),
                "predicted_delta": p["predicted_delta"],
                "confidence":      p["confidence"],
                "signal":          p["signal"],
                "top_features":    p["top_features"],
            }, on_conflict="ticker,target_date,model_version").execute()
            saved += 1
        except Exception as e:
            print(f"  [{p['ticker']}] save error: {e}")

    print(f"\nSaved {saved}/{len(predictions)} predictions to Supabase")
    print("=" * 60)
