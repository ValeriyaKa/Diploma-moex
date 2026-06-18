"""
Training V3 for DataSphere Jobs:
  1-day target + LightGBM + XGBoost + LSTM + Confidence filter
  Uploads models to S3 after training.
"""
import os, glob, json, warnings, argparse
from contextlib import nullcontext
try:
    import boto3
except ImportError:
    boto3 = None
import numpy as np
import pandas as pd
try:
    import exchange_calendars as xcals
    _moex_cal = xcals.get_calendar("XMOS")
    def _get_moex_trading_days(date_from="2022-01-01", date_to="2026-12-31"):
        return set(_moex_cal.sessions_in_range(date_from, date_to).strftime("%Y-%m-%d"))
except ImportError:
    def _get_moex_trading_days(date_from="2022-01-01", date_to="2026-12-31"):
        return None  # если библиотека не установлена — фильтр не применяется
import joblib
import lightgbm as lgb
import xgboost as xgb
import optuna
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

S3_KEY    = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET = os.environ.get("S3_SECRET_KEY", "")
BUCKET    = os.environ.get("YC_BUCKET_NAME", "moex-models-diploma")

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
AMP_ENABLED = USE_CUDA and os.environ.get("DISABLE_AMP", "0") != "1"
PIN_MEMORY = USE_CUDA

if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def make_grad_scaler():
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    return torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)


def amp_autocast():
    if not AMP_ENABLED:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast()

LGBM_FEATURES = [
    "rsi_14", "macd_hist", "bb_position", "atr_14", "vol_ratio",
    "rsi_divergence", "volume_spike",
    "close_pct_1", "close_pct_3", "close_pct_5", "close_pct_10", "close_pct_20",
    "gap_open", "high_low_range",
    "rsi_lag_1", "rsi_lag_2", "close_lag_1", "close_lag_3",
    "day_of_week", "month", "is_monday", "is_friday",
    "imoex", "usd_rub", "imoex_pct", "usd_rub_pct",
    "news_sentiment", "news_sentiment_3d", "news_sentiment_7d",
    "news_count", "news_count_3d", "news_count_7d",
    "market_sentiment", "market_sentiment_3d", "market_sentiment_7d",
    "market_news_count", "market_news_count_3d", "market_news_count_7d",
    # Fundamental features
    "days_to_div", "days_from_div", "div_value", "is_div_week",
    "pe_ratio", "pb_ratio", "ev_ebitda", "div_yield", "roe",
    "is_report_season", "is_quarter_end",
]

LSTM_D_FEATURES = [
    "close_pct_1", "close_pct_3", "close_pct_5",
    "rsi_14", "macd_hist", "bb_position", "vol_ratio", "atr_14",
    "imoex_pct", "usd_rub_pct",
    "gap_open", "volume_spike",
    "news_sentiment", "news_sentiment_3d", "news_sentiment_7d",
    "market_sentiment", "market_sentiment_3d", "market_sentiment_7d",
    # Fundamental features (subset for LSTM)
    "days_to_div", "is_div_week", "pe_ratio", "ev_ebitda",
    "is_report_season",
]

LSTM_H_FEATURES = [
    "close_pct_1h", "close_pct_3h", "vol_ratio_h",
    "hour_sin", "hour_cos", "is_open_hour", "is_close_hour", "day_pct",
]

LOOKBACK_D = 60
LOOKBACK_H = 48

def progress_iter(iterable, **kwargs):
    return tqdm(iterable, **kwargs) if tqdm is not None else iterable


# S3 client (may fail in DataSphere if keys wrong, but training still works)
try:
    s3 = boto3.client("s3",
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET,
        region_name="ru-central1")
except:
    s3 = None


def winsorize(s, lo=0.01, hi=0.99):
    return s.clip(s.quantile(lo), s.quantile(hi))


def build_daily_features(df, ticker_sentiment, market_sentiment, df_sent=None, df_fund=None):
    out = []
    for ticker, g in df.groupby("ticker"):
        g = g.copy().reset_index(drop=True)
        g["close"] = g["close"].astype(float)
        g["volume"] = g["volume"].astype(float)
        g["open"] = pd.to_numeric(g.get("open", g["close"]), errors="coerce")
        g["high"] = pd.to_numeric(g.get("high", g["close"]), errors="coerce")
        g["low"] = pd.to_numeric(g.get("low", g["close"]), errors="coerce")

        for d in [1, 3, 5, 10, 20]:
            g[f"close_pct_{d}"] = g["close"].pct_change(d) * 100
        g["gap_open"] = ((g["open"] - g["close"].shift(1)) / (g["close"].shift(1) + 1e-9)) * 100
        g["high_low_range"] = ((g["high"] - g["low"]) / (g["close"] + 1e-9)) * 100

        vol_avg = g["volume"].rolling(20).mean()
        g["volume_spike"] = (g["volume"] > vol_avg * 2).astype(float)

        rsi = pd.to_numeric(g["rsi_14"], errors="coerce")
        price_5 = g["close"].rolling(5).max()
        rsi_5 = rsi.rolling(5).max()
        g["rsi_divergence"] = (
            (g["close"] >= price_5 * 0.99).astype(float) -
            (rsi_5 < rsi_5.shift(5)).astype(float)
        )

        g["rsi_lag_1"] = rsi.shift(1)
        g["rsi_lag_2"] = rsi.shift(2)
        g["close_lag_1"] = g["close"].shift(1)
        g["close_lag_3"] = g["close"].shift(3)

        bb_u = pd.to_numeric(g.get("bb_upper", 0), errors="coerce")
        bb_l = pd.to_numeric(g.get("bb_lower", 0), errors="coerce")
        g["bb_position"] = (g["close"] - bb_l) / (bb_u - bb_l + 1e-9)

        dt = pd.to_datetime(g["time"])
        g["day_of_week"] = dt.dt.dayofweek
        g["month"] = dt.dt.month
        g["is_monday"] = (g["day_of_week"] == 0).astype(int)
        g["is_friday"] = (g["day_of_week"] == 4).astype(int)

        g["imoex"] = pd.to_numeric(g.get("imoex", 3000), errors="coerce").fillna(3000)
        g["usd_rub"] = pd.to_numeric(g.get("usd_rub", 88), errors="coerce").fillna(88)
        g["imoex_pct"] = g["imoex"].pct_change(1) * 100
        g["usd_rub_pct"] = g["usd_rub"].pct_change(1) * 100

        # News sentiment - use historical data per day if available
        g["_date_str"] = pd.to_datetime(g["time"]).dt.strftime("%Y-%m-%d")
        sent_col = f"sent_{ticker}"
        if df_sent is not None and sent_col in df_sent.columns:
            sent_index = df_sent.set_index("date")
            sent_map = sent_index[sent_col].to_dict()
            g["news_sentiment"] = g["_date_str"].map(sent_map).fillna(0)
            for window in ["3d", "7d"]:
                col = f"{sent_col}_{window}"
                if col in sent_index.columns:
                    g[f"news_sentiment_{window}"] = g["_date_str"].map(sent_index[col].to_dict()).fillna(0)
                else:
                    g[f"news_sentiment_{window}"] = g["news_sentiment"]
            count_col = f"news_count_{ticker}"
            if count_col in sent_index.columns:
                g["news_count"] = g["_date_str"].map(sent_index[count_col].to_dict()).fillna(0)
            else:
                g["news_count"] = 0
            for window in ["3d", "7d"]:
                col = f"{count_col}_{window}"
                if col in sent_index.columns:
                    g[f"news_count_{window}"] = g["_date_str"].map(sent_index[col].to_dict()).fillna(0)
                else:
                    g[f"news_count_{window}"] = g["news_count"]
        else:
            g["news_sentiment"] = ticker_sentiment.get(ticker, 0.0)
            g["news_sentiment_3d"] = g["news_sentiment"]
            g["news_sentiment_7d"] = g["news_sentiment"]
            g["news_count"] = 0
            g["news_count_3d"] = 0
            g["news_count_7d"] = 0
        
        if df_sent is not None and "market_sentiment" in df_sent.columns:
            sent_index = df_sent.set_index("date")
            mkt_map = sent_index["market_sentiment"].to_dict()
            g["market_sentiment"] = g["_date_str"].map(mkt_map).fillna(0)
            for col in ["market_sentiment_3d", "market_sentiment_7d",
                        "market_news_count", "market_news_count_3d", "market_news_count_7d"]:
                if col in sent_index.columns:
                    g[col] = g["_date_str"].map(sent_index[col].to_dict()).fillna(0)
                else:
                    g[col] = 0
        else:
            g["market_sentiment"] = market_sentiment
            g["market_sentiment_3d"] = market_sentiment
            g["market_sentiment_7d"] = market_sentiment
            g["market_news_count"] = 0
            g["market_news_count_3d"] = 0
            g["market_news_count_7d"] = 0
        
        # Fundamental features (long format: date, ticker, features)
        if df_fund is not None and "ticker" in df_fund.columns:
            tf = df_fund[df_fund["ticker"] == ticker].copy()
            if not tf.empty:
                tf = tf.set_index("date")
                for feat in ["days_to_div", "days_from_div", "div_value", "is_div_week",
                             "pe_ratio", "pb_ratio", "ev_ebitda", "div_yield", "roe",
                             "is_report_season", "is_quarter_end"]:
                    if feat in tf.columns:
                        g[feat] = g["_date_str"].map(tf[feat].to_dict()).fillna(0)
                    else:
                        g[feat] = 0
            else:
                for feat in ["days_to_div", "days_from_div", "div_value", "is_div_week",
                             "pe_ratio", "pb_ratio", "ev_ebitda", "div_yield", "roe",
                             "is_report_season", "is_quarter_end"]:
                    g[feat] = 0
        elif df_fund is not None:
            # Legacy wide format support (backwards compat)
            fund_index = df_fund.set_index("date")
            for base_col, feat_name in [
                (f"days_to_div_{ticker}", "days_to_div"),
                (f"days_from_div_{ticker}", "days_from_div"),
                (f"div_value_{ticker}", "div_value"),
                (f"is_div_week_{ticker}", "is_div_week"),
                (f"pe_{ticker}", "pe_ratio"),
                (f"pb_{ticker}", "pb_ratio"),
                (f"ev_ebitda_{ticker}", "ev_ebitda"),
                (f"div_yield_{ticker}", "div_yield"),
                (f"roe_{ticker}", "roe"),
            ]:
                if base_col in fund_index.columns:
                    g[feat_name] = g["_date_str"].map(fund_index[base_col].to_dict()).fillna(0)
                else:
                    g[feat_name] = 0
            for col in ["is_report_season", "is_quarter_end"]:
                if col in fund_index.columns:
                    g[col] = g["_date_str"].map(fund_index[col].to_dict()).fillna(0)
                else:
                    g[col] = 0
        else:
            for feat in ["days_to_div", "days_from_div", "div_value", "is_div_week",
                         "pe_ratio", "pb_ratio", "ev_ebitda", "div_yield", "roe",
                         "is_report_season", "is_quarter_end"]:
                g[feat] = 0

        # Убираем нерабочие дни MOEX (праздники + выходные) — не заменять нулями, а исключать
        _trading_days = _get_moex_trading_days()
        if _trading_days is not None:
            g = g[g["_date_str"].isin(_trading_days)].copy()

        g.drop(columns=["_date_str"], inplace=True)

        for col in ["close_pct_1","close_pct_3","close_pct_5","close_pct_10",
                     "close_pct_20","gap_open","high_low_range"]:
            if g[col].notna().sum() > 50:
                g[col] = winsorize(g[col])

        g["target"] = g["close"].pct_change(1).shift(-1) * 100
        g["target_dir"] = (g["target"] > 0).astype(int)
        out.append(g)

    res = pd.concat(out)
    res.dropna(subset=["target", "rsi_14", "close_pct_1"], inplace=True)
    return res


def build_hourly_features(df):
    out = []
    for ticker, g in df.groupby("ticker"):
        g = g.copy().reset_index(drop=True)
        g["close"] = g["close"].astype(float)
        g["volume"] = g["volume"].astype(float)
        g["time"] = pd.to_datetime(g["time"])
        g["close_pct_1h"] = g["close"].pct_change(1) * 100
        g["close_pct_3h"] = g["close"].pct_change(3) * 100
        g["vol_ratio_h"] = g["volume"] / g["volume"].rolling(20).mean()
        hour = g["time"].dt.hour
        g["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        g["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        g["is_open_hour"] = (hour == 10).astype(int)
        g["is_close_hour"] = (hour >= 18).astype(int)
        g["date"] = g["time"].dt.date
        day_open = g.groupby("date")["close"].transform("first")
        g["day_pct"] = ((g["close"] - day_open) / (day_open + 1e-9)) * 100
        g["target_h"] = g["close"].pct_change(1).shift(-1) * 100
        g.dropna(subset=["close_pct_1h", "target_h"], inplace=True)
        out.append(g)
    if not out:
        return pd.DataFrame()
    return pd.concat(out)


def train_lgbm_xgb(df_ticker, ticker, cutoff_idx=None, sample_weight=None):
    features = [f for f in LGBM_FEATURES if f in df_ticker.columns]
    X_all = df_ticker[features].fillna(0).values
    y_all = df_ticker["target_dir"].values
    n = len(X_all)
    # Нормализуем веса если переданы
    w_all = sample_weight if sample_weight is not None else np.ones(n)

    if cutoff_idx is not None:
        # Fixed cutoff mode: train on [:cutoff], validate on [cutoff:]
        min_train = int(cutoff_idx * 0.85)
        val_size = cutoff_idx - min_train
    else:
        min_train = int(n * 0.6)
        val_size = int(n * 0.1)
    X_t, y_t = X_all[:min_train], y_all[:min_train]
    X_v, y_v = X_all[min_train:min_train+val_size], y_all[min_train:min_train+val_size]
    w_t = w_all[:min_train]

    def lgbm_obj(trial):
        p = {
            "objective":"binary","metric":"binary_logloss","verbosity":-1,
            "num_leaves": trial.suggest_int("nl", 15, 200),
            "learning_rate": trial.suggest_float("lr", 0.005, 0.3, log=True),
            "feature_fraction": trial.suggest_float("ff", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bf", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bfr", 1, 7),
            "min_child_samples": trial.suggest_int("mcs", 5, 100),
            "reg_alpha": trial.suggest_float("a", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("l", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("spw", 0.5, 2.0),
            "n_estimators": 500,
        }
        m = lgb.LGBMClassifier(**p)
        m.fit(X_t, y_t, sample_weight=w_t, eval_set=[(X_v, y_v)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        return f1_score(y_v, m.predict(X_v), zero_division=0)

    study_l = optuna.create_study(direction="maximize")
    study_l.optimize(lgbm_obj, n_trials=100, show_progress_bar=True)
    bp = study_l.best_params
    best_lgbm = {
        "objective":"binary","metric":"binary_logloss","verbosity":-1,"n_estimators":600,
        "num_leaves":bp["nl"],"learning_rate":bp["lr"],
        "feature_fraction":bp["ff"],"bagging_fraction":bp["bf"],
        "bagging_freq":bp["bfr"],"min_child_samples":bp["mcs"],
        "reg_alpha":bp["a"],"reg_lambda":bp["l"],"scale_pos_weight":bp["spw"],
    }

    def xgb_obj(trial):
        p = {
            "objective":"binary:logistic","eval_metric":"logloss",
            "verbosity":0,"use_label_encoder":False,
            "max_depth": trial.suggest_int("md", 3, 10),
            "learning_rate": trial.suggest_float("lr", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("ss", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("cs", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("a", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("l", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("spw", 0.5, 2.0),
            "n_estimators": 500,
        }
        m = xgb.XGBClassifier(**p)
        m.fit(X_t, y_t, sample_weight=w_t, eval_set=[(X_v, y_v)], verbose=False)
        return f1_score(y_v, m.predict(X_v), zero_division=0)

    study_x = optuna.create_study(direction="maximize")
    study_x.optimize(xgb_obj, n_trials=80, show_progress_bar=True)
    bx = study_x.best_params
    best_xgb = {
        "objective":"binary:logistic","eval_metric":"logloss",
        "verbosity":0,"use_label_encoder":False,"n_estimators":600,
        "max_depth":bx["md"],"learning_rate":bx["lr"],
        "subsample":bx["ss"],"colsample_bytree":bx["cs"],
        "reg_alpha":bx["a"],"reg_lambda":bx["l"],"scale_pos_weight":bx["spw"],
    }

    # Walk-Forward
    oos_lgbm, oos_xgb, oos_true = [], [], []
    for start in range(min_train, n - val_size, val_size):
        end = start + val_size
        Xt, yt = X_all[:start], y_all[:start]
        Xv, yv = X_all[start:end], y_all[start:end]
        ml = lgb.LGBMClassifier(**best_lgbm)
        ml.fit(Xt, yt, eval_set=[(Xv, yv)],
               callbacks=[lgb.early_stopping(30, verbose=False)])
        mx = xgb.XGBClassifier(**best_xgb)
        mx.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
        oos_lgbm.extend(ml.predict_proba(Xv)[:, 1])
        oos_xgb.extend(mx.predict_proba(Xv)[:, 1])
        oos_true.extend(yv)

    oos_ens = [(l+x)/2 for l, x in zip(oos_lgbm, oos_xgb)]
    oos_pred = [1 if p > 0.5 else 0 for p in oos_ens]

    conf_pred, conf_true = [], []
    for p, t in zip(oos_ens, oos_true):
        if p > 0.60 or p < 0.40:
            conf_pred.append(1 if p > 0.5 else 0)
            conf_true.append(t)

    acc_all = accuracy_score(oos_true, oos_pred)
    f1_all = f1_score(oos_true, oos_pred, zero_division=0)
    acc_conf = accuracy_score(conf_true, conf_pred) if conf_true else acc_all
    f1_conf = f1_score(conf_true, conf_pred, zero_division=0) if conf_true else f1_all
    coverage = len(conf_true) / max(len(oos_true), 1) * 100

    if cutoff_idx is not None:
        # Cutoff mode: train on data before cutoff, test on data after
        split = cutoff_idx
    else:
        split = int(n * 0.85)
    fl = lgb.LGBMClassifier(**best_lgbm)
    fl.fit(X_all[:split], y_all[:split],
           sample_weight=w_all[:split],
           eval_set=[(X_all[split:], y_all[split:])],
           callbacks=[lgb.early_stopping(30, verbose=False)])
    fx = xgb.XGBClassifier(**best_xgb)
    fx.fit(X_all[:split], y_all[:split],
           sample_weight=w_all[:split],
           eval_set=[(X_all[split:], y_all[split:])], verbose=False)

    explainer = shap.TreeExplainer(fl)
    joblib.dump(fl, f"models/lgbm_{ticker}.pkl")
    joblib.dump(fx, f"models/xgb_{ticker}.pkl")
    joblib.dump(explainer, f"models/lgbm_{ticker}_explainer.pkl")
    joblib.dump(features, f"models/lgbm_{ticker}_features.pkl")

    result = {
        "ticker":ticker, "accuracy_all":acc_all, "f1_all":f1_all,
        "accuracy_conf":acc_conf, "f1_conf":f1_conf, "coverage":coverage,
    }

    # Backtest on holdout if cutoff specified
    if cutoff_idx is not None and cutoff_idx < n:
        X_test = X_all[cutoff_idx:]
        y_test = y_all[cutoff_idx:]
        p_lgbm = fl.predict_proba(X_test)[:, 1]
        p_xgb = fx.predict_proba(X_test)[:, 1]
        p_ens = (p_lgbm + p_xgb) / 2
        pred = (p_ens > 0.5).astype(int)

        bt_acc = accuracy_score(y_test, pred)
        bt_f1 = f1_score(y_test, pred, zero_division=0)
        bt_prec = precision_score(y_test, pred, zero_division=0)
        bt_rec = recall_score(y_test, pred, zero_division=0)

        # Confident predictions only
        mask_conf = (p_ens > 0.60) | (p_ens < 0.40)
        if mask_conf.sum() > 0:
            bt_acc_c = accuracy_score(y_test[mask_conf], pred[mask_conf])
            bt_f1_c = f1_score(y_test[mask_conf], pred[mask_conf], zero_division=0)
            bt_cov = mask_conf.sum() / len(y_test) * 100
        else:
            bt_acc_c, bt_f1_c, bt_cov = bt_acc, bt_f1, 100.0

        result.update({
            "bt_days": len(y_test),
            "bt_accuracy": bt_acc, "bt_f1": bt_f1,
            "bt_precision": bt_prec, "bt_recall": bt_rec,
            "bt_accuracy_conf": bt_acc_c, "bt_f1_conf": bt_f1_c,
            "bt_coverage": bt_cov,
        })
        print(f"[{ticker}] BACKTEST ({len(y_test)} days): "
              f"acc={bt_acc:.3f} f1={bt_f1:.3f} prec={bt_prec:.3f} rec={bt_rec:.3f}")
        print(f"[{ticker}] BACKTEST CONF: acc={bt_acc_c:.3f} f1={bt_f1_c:.3f} cov={bt_cov:.0f}%")

    print(f"[{ticker}] ALL: acc={acc_all:.3f} f1={f1_all:.3f} | "
          f"CONF: acc={acc_conf:.3f} f1={f1_conf:.3f} cov={coverage:.0f}%")
    return result


class StockLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.att = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh(),
                                  nn.Linear(64, 1), nn.Softmax(dim=1))
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        ctx = (out * self.att(out)).sum(1)
        return self.head(ctx).squeeze(-1)


def train_lstm(df_t, ticker, features, target_col, lookback, prefix, epochs=80, cutoff_idx=None):
    avail = [f for f in features if f in df_t.columns]
    if len(avail) < 3: return None
    data = df_t[avail + [target_col]].fillna(0).values.astype(np.float32)
    scaler = RobustScaler()
    data = scaler.fit_transform(data)
    X_list, y_list = [], []
    for i in range(lookback, len(data)):
        X_list.append(data[i-lookback:i, :-1])
        y_list.append(data[i, -1])
    if len(X_list) < 100: return None
    X_arr, y_arr = np.array(X_list), np.array(y_list)
    if cutoff_idx is not None:
        # Adjust cutoff for lookback offset: sequences start at index lookback
        split = max(cutoff_idx - lookback, int(len(X_arr) * 0.7))
    else:
        split = int(len(X_arr) * 0.8)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_arr[:split], dtype=torch.float32),
                      torch.tensor(y_arr[:split], dtype=torch.float32)),
        batch_size=64, shuffle=True, drop_last=True,
        pin_memory=PIN_MEMORY)
    model = StockLSTM(input_size=len(avail), hidden_size=128, num_layers=3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, total_steps=epochs * max(len(loader), 1))
    crit = nn.HuberLoss()
    best_loss, bad = float("inf"), 0
    scaler = make_grad_scaler()
    X_val_t = torch.tensor(X_arr[split:], dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_arr[split:], dtype=torch.float32, device=DEVICE)
    for ep in progress_iter(range(epochs), desc=f"{ticker} {prefix}", unit="epoch", leave=False):
        model.train()
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=USE_CUDA)
            yb = yb.to(DEVICE, non_blocking=USE_CUDA)
            opt.zero_grad(set_to_none=True)
            with amp_autocast():
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt)
            scaler.update()
            sched.step()
        model.eval()
        with torch.no_grad(), amp_autocast():
            vl = crit(model(X_val_t), y_val_t).item()
        if vl < best_loss:
            best_loss = vl
            torch.save(model.state_dict(), f"models/{prefix}_{ticker}.pt")
            bad = 0
        else:
            bad += 1
        if bad >= 20: break
    joblib.dump(scaler, f"models/{prefix}_scaler_{ticker}.pkl")
    joblib.dump(avail, f"models/{prefix}_features_{ticker}.pkl")
    print(f"[{ticker}] {prefix} val_loss={best_loss:.4f}")
    return best_loss


def load_error_patterns(path: str = "data/error_patterns.json") -> dict:
    """Загружает паттерны ошибок из analyze_errors.py."""
    for p in [path, "/job/data/error_patterns.json", "error_patterns.json"]:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                patterns = json.load(f)
            print(f"  Error patterns loaded from {p}")
            return patterns
    return {}


def expected_model_files(ticker: str, train_daily: bool, train_hourly: bool) -> list[str]:
    """Files that must exist before --resume can skip a ticker."""
    files = []
    if train_daily:
        files += [
            f"models/lgbm_{ticker}.pkl",
            f"models/xgb_{ticker}.pkl",
            f"models/lgbm_{ticker}_explainer.pkl",
            f"models/lgbm_{ticker}_features.pkl",
            f"models/lstm_{ticker}.pt",
            f"models/lstm_scaler_{ticker}.pkl",
            f"models/lstm_features_{ticker}.pkl",
        ]
    if train_hourly:
        files += [
            f"models/lstm_hourly_{ticker}.pt",
            f"models/lstm_hourly_scaler_{ticker}.pkl",
            f"models/lstm_hourly_features_{ticker}.pkl",
        ]
    return files


def missing_model_files(ticker: str, train_daily: bool, train_hourly: bool) -> list[str]:
    return [
        p for p in expected_model_files(ticker, train_daily, train_hourly)
        if not os.path.exists(p)
    ]


def build_sample_weights(df_ticker: pd.DataFrame, ticker: str,
                         patterns: dict) -> np.ndarray:
    """
    Строит вектор весов для каждой строки обучающих данных.
    Строки соответствующие паттернам ошибок получают повышенный вес.

    Логика:
      weight = w_month * w_dow * w_ticker
      где каждый компонент = boost_weight из error_patterns.json
      (1.0 = норма, >1.0 = проблемный паттерн, давать больший вес)
    """
    if not patterns:
        return np.ones(len(df_ticker))

    boost = patterns.get("boost_weights", {})
    month_boost  = boost.get("months",  {})
    dow_boost    = boost.get("dow",     {})
    ticker_boost = boost.get("tickers", {})

    DOW_NAMES = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}

    weights = np.ones(len(df_ticker))

    if "time" in df_ticker.columns:
        times = pd.to_datetime(df_ticker["time"])
        months = times.dt.month.astype(str)
        dows   = times.dt.dayofweek.map(DOW_NAMES)

        for i, (m, d) in enumerate(zip(months, dows)):
            w = 1.0
            w *= float(month_boost.get(m, 1.0))
            w *= float(dow_boost.get(d, 1.0))
            w *= float(ticker_boost.get(ticker, 1.0))
            # Cap at 2.5 чтобы не перекосить обучение
            weights[i] = min(w, 2.5)

    n_boosted = (weights > 1.05).sum()
    if n_boosted > 0:
        print(f"    ↑ sample_weight: {n_boosted}/{len(weights)} строк boosted "
              f"(max={weights.max():.2f}, mean={weights.mean():.2f})")

    return weights


def main():
    parser = argparse.ArgumentParser(description="Train V3 models")
    parser.add_argument("--cutoff", type=str, default=None,
                        help="Date cutoff for backtesting, e.g. 2026-04-01. "
                             "Trains on data before this date, tests on data after.")
    parser.add_argument("--boost-errors", action="store_true",
                        help="Загрузить data/error_patterns.json и усилить веса "
                             "строк соответствующих паттернам ошибок.")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Train only one ticker without changing features.csv.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip ticker only when all expected model files exist.")
    args = parser.parse_args()

    # Загружаем паттерны ошибок если флаг указан
    _error_patterns = {}
    if args.boost_errors:
        _error_patterns = load_error_patterns()
        if _error_patterns:
            boost = _error_patterns.get("boost_weights", {})
            print(f"  Boost: {len(boost.get('tickers',{}))} тикеров, "
                  f"{len(boost.get('months',{}))} месяцев, "
                  f"{len(boost.get('dow',{}))} дней недели")

    print(f"Device: {DEVICE} | AMP: {AMP_ENABLED}")
    print("Training V3: 1-day + XGBoost + Confidence filter (DataSphere)")
    if args.cutoff:
        print(f"BACKTEST MODE: cutoff = {args.cutoff}")
        print(f"  Train: data before {args.cutoff}")
        print(f"  Test:  data from  {args.cutoff} onward")

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Get features.csv from job input
    if not os.path.exists("data/features.csv"):
        for src in ["features.csv", "/job/features.csv"]:
            if os.path.exists(src):
                import shutil
                shutil.copy(src, "data/features.csv")
                print(f"Copied {src}")
                break

    # Get news sentiment from job input
    for src in ["news_sentiment.json", "/job/news_sentiment.json",
                "data/news_sentiment.json"]:
        if os.path.exists(src) and os.path.getsize(src) > 10:
            if src != "data/news_sentiment.json":
                import shutil
                shutil.copy(src, "data/news_sentiment.json")
            break

    # Get macro from job input
    for src in ["macro.csv", "/job/macro.csv", "data/macro.csv"]:
        if os.path.exists(src):
            if src != "data/macro.csv":
                import shutil
                shutil.copy(src, "data/macro.csv")
            break

    # Get fundamentals from job input
    for src in ["fundamentals.csv", "/job/fundamentals.csv", "data/fundamentals.csv"]:
        if os.path.exists(src):
            if src != "data/fundamentals.csv":
                import shutil
                shutil.copy(src, "data/fundamentals.csv")
            break

    # Load fundamentals
    df_fund = None
    for src in ["data/fundamentals.csv", "fundamentals.csv"]:
        if os.path.exists(src):
            df_fund = pd.read_csv(src)
            df_fund["date"] = pd.to_datetime(df_fund["date"]).dt.strftime("%Y-%m-%d")
            print(f"Fundamentals: {len(df_fund)} rows, {len(df_fund.columns)} columns")
            break
    if df_fund is None:
        print("No fundamentals.csv found — fundamental features will be zeros")

   # Load historical sentiment for training
    ticker_sentiment, market_sentiment = {}, 0.0
    
    # Try historical CSV — prefer merged (FinBERT + GDELT), then FinBERT, then GDELT
    df_sent = None
    for src in ["news_sentiment_historical_merged.csv",
                "data/news_sentiment_historical_merged.csv",
                "news_finbert_historical.csv",
                "data/news_finbert_historical.csv",
                "news_sentiment_historical.csv",
                "data/news_sentiment_historical.csv"]:
        if os.path.exists(src):
            df_sent = pd.read_csv(src)
            df_sent["date"] = pd.to_datetime(df_sent["date"]).dt.strftime("%Y-%m-%d")
            print(f"Historical sentiment ({os.path.basename(src)}): {len(df_sent)} days, {len(df_sent.columns)} columns")
            break
    
    # Also load JSON for market sentiment
    for p in ["news_sentiment.json", "data/news_sentiment.json"]:
        if os.path.exists(p) and os.path.getsize(p) > 100:
            with open(p) as f:
                d = json.load(f)
            ticker_sentiment = d.get("ticker_sentiment", {})
            market_sentiment = d.get("market_sentiment", 0.0)
            break
    print(f"Market sentiment: {market_sentiment:+.3f}")

    df_raw = pd.read_csv("data/features.csv", parse_dates=["time"])
    if args.ticker:
        ticker_filter = args.ticker.upper()
        df_raw = df_raw[df_raw["ticker"].astype(str).str.upper() == ticker_filter].copy()
        if df_raw.empty:
            raise SystemExit(f"No rows found for ticker {ticker_filter} in data/features.csv")
        print(f"Ticker filter: {ticker_filter} ({len(df_raw):,} rows)")
    df_raw.sort_values(["ticker", "time"], inplace=True)

    df_d = df_raw[df_raw["timeframe"] == "1d"].copy()
    df_h = df_raw[df_raw["timeframe"] == "1h"].copy()
    print(f"Daily: {len(df_d):,} | Hourly: {len(df_h):,}")

    df_daily = build_daily_features(df_d, ticker_sentiment, market_sentiment, df_sent, df_fund)
    df_hourly = build_hourly_features(df_h)
    print(f"After FE: daily={len(df_daily):,} hourly={len(df_hourly):,}")

    all_metrics = []
    tickers = df_daily["ticker"].unique()

    for i, ticker in enumerate(progress_iter(tickers, desc="Tickers", unit="ticker")):
        print(f"\n{'='*55}")
        print(f"[{i+1}/{len(tickers)}] {ticker}")
        print(f"{'='*55}")

        dt = df_daily[df_daily["ticker"] == ticker].copy()
        ht = df_hourly[df_hourly["ticker"] == ticker].copy()
        train_daily = len(dt) >= 250
        train_hourly = len(ht) >= 500

        if args.resume:
            missing = missing_model_files(ticker, train_daily, train_hourly)
            if not missing and (train_daily or train_hourly):
                print("  SKIP: all expected model files already exist")
                continue
            if missing:
                print("  RESUME: ticker is incomplete, retraining")
                print("    Missing: " + ", ".join(os.path.basename(p) for p in missing[:8]))
                if len(missing) > 8:
                    print(f"    ... and {len(missing) - 8} more")

        # Compute cutoff index if specified
        cutoff_idx = None
        if args.cutoff:
            dt["time"] = pd.to_datetime(dt["time"], utc=True)
            cutoff_mask = dt["time"] < pd.Timestamp(args.cutoff, tz="UTC")
            cutoff_idx = cutoff_mask.sum()
            total = len(dt)
            print(f"  Cutoff: {cutoff_idx} train / {total - cutoff_idx} test days")
            if cutoff_idx < 200:
                print(f"  WARNING: Only {cutoff_idx} train days, minimum 200 recommended")

        if len(dt) >= 250:
            try:
                sw = build_sample_weights(dt, ticker, _error_patterns) if _error_patterns else None
                m = train_lgbm_xgb(dt, ticker, cutoff_idx=cutoff_idx, sample_weight=sw)
                all_metrics.append(m)
            except Exception as e:
                print(f"LGBM+XGB error: {e}")
            try:
                train_lstm(dt, ticker, LSTM_D_FEATURES, "target",
                           LOOKBACK_D, "lstm", epochs=80, cutoff_idx=cutoff_idx)
            except Exception as e:
                print(f"LSTM daily error: {e}")

        if len(ht) >= 500:
            # Compute hourly cutoff if date cutoff specified
            h_cutoff = None
            if args.cutoff:
                ht["time"] = pd.to_datetime(ht["time"], utc=True)
                h_cutoff = (ht["time"] < pd.Timestamp(args.cutoff, tz="UTC")).sum()
            try:
                train_lstm(ht, ticker, LSTM_H_FEATURES, "target_h",
                           LOOKBACK_H, "lstm_hourly", epochs=50, cutoff_idx=h_cutoff)
            except Exception as e:
                print(f"LSTM hourly error: {e}")

    # Summary
    if all_metrics:
        df_m = pd.DataFrame(all_metrics)
        print(f"\n{'='*60}")
        print("RESULTS V3 — Walk-Forward Validation")
        print(f"{'='*60}")
        print(f"ALL: acc={df_m['accuracy_all'].mean():.3f} f1={df_m['f1_all'].mean():.3f}")
        print(f"CONF: acc={df_m['accuracy_conf'].mean():.3f} f1={df_m['f1_conf'].mean():.3f} "
              f"cov={df_m['coverage'].mean():.0f}%")
        print(f"Best: {df_m.loc[df_m['f1_conf'].idxmax(),'ticker']} "
              f"(F1={df_m['f1_conf'].max():.3f})")

        if args.cutoff and "bt_accuracy" in df_m.columns:
            print(f"\n{'='*60}")
            print(f"BACKTEST RESULTS (cutoff: {args.cutoff})")
            print(f"{'='*60}")
            print(f"Avg backtest days: {df_m['bt_days'].mean():.0f}")
            print(f"ALL:  acc={df_m['bt_accuracy'].mean():.3f} "
                  f"f1={df_m['bt_f1'].mean():.3f} "
                  f"prec={df_m['bt_precision'].mean():.3f} "
                  f"rec={df_m['bt_recall'].mean():.3f}")
            print(f"CONF: acc={df_m['bt_accuracy_conf'].mean():.3f} "
                  f"f1={df_m['bt_f1_conf'].mean():.3f} "
                  f"cov={df_m['bt_coverage'].mean():.0f}%")
            best_bt = df_m.loc[df_m['bt_f1'].idxmax()]
            print(f"Best backtest: {best_bt['ticker']} (F1={best_bt['bt_f1']:.3f})")
            worst_bt = df_m.loc[df_m['bt_f1'].idxmin()]
            print(f"Worst backtest: {worst_bt['ticker']} (F1={worst_bt['bt_f1']:.3f})")
            print(f"\nPer-ticker backtest:")
            for _, row in df_m.sort_values("bt_f1", ascending=False).iterrows():
                print(f"  {row['ticker']:6s}  acc={row['bt_accuracy']:.3f}  "
                      f"f1={row['bt_f1']:.3f}  prec={row['bt_precision']:.3f}  "
                      f"rec={row['bt_recall']:.3f}  days={row['bt_days']:.0f}")

        df_m.to_csv("models/training_metrics_v3.csv", index=False)

    # Upload models to S3
    if s3 and S3_KEY:
        print("\nUploading models to S3...")
        uploaded = 0
        for f in glob.glob("models/*.pkl") + glob.glob("models/*.pt") + glob.glob("models/*.csv"):
            try:
                s3.upload_file(f, BUCKET, f"models/{os.path.basename(f)}")
                uploaded += 1
            except Exception as e:
                print(f"  Upload error: {e}")
        print(f"Uploaded {uploaded} files")
    else:
        print("\nNo S3 credentials, models saved locally only")

    count = len(glob.glob("models/*.pkl") + glob.glob("models/*.pt"))
    print(f"Total files: {count}")
    print("V3 complete!")


if __name__ == "__main__":
    main()
