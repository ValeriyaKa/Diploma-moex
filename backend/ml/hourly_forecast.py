from __future__ import annotations

import os
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from backend.db.database import get_candles

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "./models"))
LOOKBACK_H = 48
LSTM_H_FEATURES_DEFAULT = [
    "close_pct_1h", "close_pct_3h", "vol_ratio_h",
    "hour_sin", "hour_cos", "is_open_hour", "is_close_hour", "day_pct",
]
TRADING_HOURS = list(range(10, 19))


class HourlyLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.att = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        ctx = (out * self.att(out)).sum(1)
        return self.head(ctx).squeeze(-1)


def next_trading_day(start: date | None = None) -> date:
    target = (start or date.today()) + timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


def build_hourly_features(hourly_candles: list[dict]) -> pd.DataFrame | None:
    if len(hourly_candles) < LOOKBACK_H:
        return None

    df = pd.DataFrame(hourly_candles).copy()
    df["time"] = pd.to_datetime(df["time"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    df.sort_values("time", inplace=True)
    return add_hourly_features(df)


def add_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["close_pct_1h"] = df["close"].pct_change(1) * 100
    df["close_pct_3h"] = df["close"].pct_change(3) * 100
    df["vol_ratio_h"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)

    hour = df["time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["is_open_hour"] = (hour == 10).astype(int)
    df["is_close_hour"] = (hour >= 18).astype(int)
    df["date"] = df["time"].dt.date
    day_open = df.groupby("date")["close"].transform("first")
    df["day_pct"] = ((df["close"] - day_open) / (day_open + 1e-9)) * 100

    for col in LSTM_H_FEATURES_DEFAULT:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def inverse_scaled_target(scaler, scaled_value: float, n_features: int) -> float:
    row = np.zeros((1, n_features + 1), dtype=np.float32)
    row[0, -1] = scaled_value
    return float(scaler.inverse_transform(row)[0, -1])


async def forecast_next_day_hourly(ticker: str) -> dict | None:
    ticker = ticker.upper()
    candles = await get_candles(ticker, "1h", 500)
    df = build_hourly_features(candles)
    if df is None:
        return None

    feat_path = MODELS_DIR / f"lstm_hourly_features_{ticker}.pkl"
    model_path = MODELS_DIR / f"lstm_hourly_{ticker}.pt"
    scaler_path = MODELS_DIR / f"lstm_hourly_scaler_{ticker}.pkl"
    if not model_path.exists() or not scaler_path.exists():
        return None

    features = joblib.load(feat_path) if feat_path.exists() else LSTM_H_FEATURES_DEFAULT
    features = [f for f in features if f in df.columns]
    if len(features) < 3:
        return None

    scaler = joblib.load(scaler_path)
    model = HourlyLSTM(input_size=len(features))
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    target_day = next_trading_day()
    last_close = float(df["close"].dropna().iloc[-1])
    last_volume = float(df["volume"].dropna().iloc[-1]) if df["volume"].notna().any() else 0.0
    rows = []
    work = df.copy()
    prev_close = last_close

    for hour in TRADING_HOURS:
        ts = datetime.combine(target_day, dt_time(hour=hour))
        synthetic = pd.DataFrame([{
            "time": pd.Timestamp(ts),
            "close": prev_close,
            "volume": last_volume,
        }])
        work = pd.concat([work[["time", "close", "volume"]], synthetic], ignore_index=True)
        feat_df = add_hourly_features(work).fillna(0)
        if len(feat_df) < LOOKBACK_H:
            return None

        seq = feat_df[features].tail(LOOKBACK_H).values.astype(np.float32)
        dummy_target = np.zeros((len(seq), 1), dtype=np.float32)
        scaled = scaler.transform(np.hstack([seq, dummy_target]))
        tensor = torch.tensor(scaled[:, :-1].reshape(1, LOOKBACK_H, len(features)))

        with torch.no_grad():
            scaled_delta = float(model(tensor).item())
        delta_pct = inverse_scaled_target(scaler, scaled_delta, len(features))
        delta_pct = float(np.clip(delta_pct, -3.0, 3.0))
        pred_close = prev_close * (1 + delta_pct / 100)

        rows.append({
            "time": ts.isoformat(),
            "ticker": ticker,
            "predicted_close": round(pred_close, 4),
            "predicted_delta": round(delta_pct, 4),
            "cumulative_delta": round((pred_close / last_close - 1) * 100, 4),
        })

        prev_close = pred_close
        work.loc[work.index[-1], "close"] = pred_close

    return {
        "ticker": ticker,
        "target_date": target_day.isoformat(),
        "start_close": round(last_close, 4),
        "points": rows,
    }
