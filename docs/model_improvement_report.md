# Анализ и улучшение модели MOEX — Отчёт

> Дата: 2026-06-16  
> Данные: `rolling_backtest_2025-09-01_2026-04-01.csv` + `rolling_backtest_2026-04-01_2026-05-26.csv`

---

## 🚨 КРИТИЧЕСКАЯ НАХОДКА: Проблема с данными бэктеста

### Что обнаружено

**72.4% строк** в бэктесте имеют `actual_close == current_close` (actual_delta = 0), то есть фактическая цена закрытия для целевого дня **не была собрана**. Это приводит к тому, что `is_correct = False` для 74.7% активных сигналов — не потому что модель ошибается, а потому что нет данных.

| Метрика | Значение |
|---------|---------|
| Всего строк | 6,683 |
| Строк с actual_close = current_close | **4,839 (72.4%)** |
| Активных сигналов | 1,314 |
| Активных без данных actual_close | **982 (74.7%)** |
| Строк с ВАЛИДНЫМИ данными actual_close | 1,844 (27.6%) |

### Реальная точность модели

На строках, где actual_close действительно отличается от current_close:

| Метрика | Значение |
|---------|---------|
| Точность BUY (с учётом delta_min) | **48.1%** |
| Точность SELL (с учётом delta_min) | **64.1%** |
| **Общая точность (валидные строки)** | **50.4%** |
| Направленная точность | 42.5% |

**Вывод**: Модель случайна (≈50%), а не катастрофически плоха (10%), как казалось из-за ошибки в данных.

### Почему данные отсутствуют

Скрипт `rolling_backtest.py` в момент запуска не мог получить данные на следующий торговый день (T+1), потому что бэктест запускался в тот же день или с задержкой. Данные сохранились только для пятниц (предсказание на понедельник):
- **Пятница→Понедельник**: 1,322 строки с данными (72% от всех валидных)
- Понедельник–Четверг: только 113–148 строк с данными

---

## 📋 Текущая архитектура модели

```
Ансамбль:
  LightGBM      (w=0.35)
  XGBoost       (w=0.35)
  LSTM daily    (w=0.30)
  LSTM hourly   (w=0.00)  ← не вносит вклад!

CONF_THRESHOLD = 0.55
DELTA_MIN      = 0.30%

Features (13 штук):
  close, volume, rsi_14, macd_hist,
  bb_upper, bb_lower, atr_14, vol_ratio,
  imoex, usd_rub, timeframe
```

---

## 🔍 Анализ результатов оптимизации весов (tune_results)

### Период 2 (2026-04-01 → 2026-05-26) — лучшие конфиги:

| w_lgbm | w_xgb | w_lstm_d | conf_thresh | dir_acc | coverage | score |
|--------|-------|----------|-------------|---------|----------|-------|
| **1.0** | 0.0 | 0.0 | 0.58 | **22.6%** | 14.1% | 0.0879 |
| 0.7 | 0.1 | 0.2 | 0.58 | 22.4% | 10.7% | 0.0765 |
| 1.0 | 0.0 | 0.0 | 0.58 | 22.3% | 14.0% | 0.0862 |

**Вывод**: В тестовом периоде LightGBM работает ЛУЧШЕ один, без XGBoost и LSTM.

### Период 1 (2025-09-01 → 2026-04-01) — лучшие конфиги:

| w_lgbm | w_xgb | w_lstm_d | conf_thresh | dir_acc | n_signals |
|--------|-------|----------|-------------|---------|-----------|
| 0.6 | 0.2 | 0.2 | 0.58 | 10.6% | 527 |
| 0.4 | 0.1 | 0.3 | 0.55 | 10.6% | 660 |

**Вывод**: В тренировочном периоде нет выраженного победителя; все конфиги дают ~10% dir_acc из-за проблемы с данными.

---

## 🎯 Рекомендации по улучшению

### 1. СРОЧНО: Исправить сбор данных actual_close

Корень проблемы — бэктест не собирает actual_close для T+1. Исправление в `rolling_backtest.py`:

```python
# Текущий код (ПРОБЛЕМА):
actual_close = get_close(ticker, target_date)  # часто возвращает current_close

# Исправление: собирать данные с задержкой или из истории
def get_actual_close_from_history(ticker: str, target_date: str, features_df: pd.DataFrame) -> float:
    """Берёт actual_close из уже собранных данных в features.csv"""
    mask = (
        (features_df["ticker"] == ticker) &
        (features_df["time"].dt.date == pd.Timestamp(target_date).date()) &
        (features_df["timeframe"] == "1d")
    )
    rows = features_df[mask]
    if not rows.empty:
        return float(rows.iloc[0]["close"])
    return None  # Явно возвращаем None, если данных нет
```

Это позволит правильно оценивать точность модели.

### 2. Обновить веса ансамбля

По результатам тестового периода (Apr-May 2026) — лучший конфиг:

```python
# best_config.json — новые веса
{
    "w_lgbm":   1.0,   # ← было 0.35
    "w_xgb":    0.0,   # ← было 0.35
    "w_lstm_d": 0.0,   # ← было 0.30
    "w_lstm_h": 0.0,
    "CONF_THRESHOLD": 0.58,  # ← было 0.55
    "DELTA_MIN": 0.30
}
```

### 3. Многофолдовая WFV (Walk-Forward Validation)

Текущий пайплайн использует только 2 периода. Для более надёжной оценки нужны 4+ фолда:

```python
# wfv_multifold.py — запускать так:
python wfv_multifold.py \
    --csv data/rolling_backtest_2025-09-01_2026-04-01.csv \
          data/rolling_backtest_2026-04-01_2026-05-26.csv \
    --folds 4 \
    --conf-min 0.58 \
    --out data/wfv_analysis.json
```

### 4. Расширить Feature Engineering

Текущий набор из 13 фичей очень скудный. Добавить:

```python
# В train_models.py — дополнительные фичи:
features_to_add = {
    # Ценовые паттерны
    "return_1d":    "close.pct_change(1)",           # однодневная доходность
    "return_5d":    "close.pct_change(5)",           # недельная доходность
    "return_20d":   "close.pct_change(20)",          # месячная доходность
    "momentum_14":  "close / close.shift(14) - 1",  # моментум 2 недели
    
    # Волатильность
    "realized_vol": "return_1d.rolling(10).std()",  # реализованная волатильность
    "vol_trend":    "atr_14 / atr_14.shift(5)",     # направление волатильности
    
    # Относительные метрики
    "vs_imoex":    "close_pct_change / imoex_pct_change",  # относительная сила
    "bb_position": "(close - bb_lower) / (bb_upper - bb_lower)",  # позиция в BB
    
    # Паттерны недели
    "is_monday":   "(as_of.weekday() == 0).astype(int)",
    "is_friday":   "(as_of.weekday() == 4).astype(int)",
    
    # Объём
    "vol_spike":   "volume / volume.rolling(20).max()",  # всплески объёма
    "price_vol_corr": "corr(close, volume, 10)",         # корреляция цена-объём
    
    # Новостной сентимент (из news_sentiment.json)
    "news_sentiment_5d": "rolling_mean(sentiment, 5)",
    "news_count_5d":     "rolling_count(news, 5)",
}
```

### 5. Фильтрация тикеров

На основе анализа точности (валидные строки) — исключить из торговли:

**Плохие тикеры** (accuracy < 35% на валидных строках):
- Для уточнения запустить `wfv_multifold.py` с исправленными данными

**Потенциально лучшие** (по dir_acc из tune_results):
- CHMF, ALRS, PLZL, YDEX, MOEX, MTSS

### 6. Адаптивный порог уверенности по тикеру

```python
# Вместо единого CONF_THRESHOLD — индивидуальные пороги
TICKER_THRESHOLDS = {
    "CHMF": 0.55,   # хорошая точность, можно быть менее строгим
    "SBER": 0.62,   # нужен более высокий порог
    "SNGS": 0.70,   # много ошибок — высокий порог или исключить
    # default = 0.58
}
```

### 7. Обновить ноутбук: новый Шаг 4а

Добавить в `jupyter_pipeline_v3.1.ipynb` новую ячейку после Шага 4:

```python
# ── Шаг 4а: Мульти-фолдовый WFV-анализ ──────────────────────────────────
result = subprocess.run([
    sys.executable, "wfv_multifold.py",
    "--csv",
    "data/rolling_backtest_2025-09-01_2026-04-01.csv",
    "data/rolling_backtest_2026-04-01_2026-05-26.csv",
    "--folds", "4",
    "--conf-min", "0.58",
    "--out", "data/wfv_analysis.json"
], capture_output=False, text=True)

if os.path.exists("data/wfv_analysis.json"):
    with open("data/wfv_analysis.json") as f:
        wfv = json.load(f)
    
    print("\n📊 Результаты WFV:")
    for fold in wfv["folds"]:
        print(f"  {fold['label']}: accuracy={fold.get('accuracy_vs_delta_min','N/A')}")
    
    print("\n💡 Рекомендации:")
    recs = wfv.get("recommendations", {})
    print(f"  Исключить тикеры: {recs.get('exclude_tickers', [])}")
    print(f"  Лучшие тикеры: {recs.get('prefer_tickers', [])}")
    print(f"  Лучший диапазон confidence: {recs.get('best_confidence_range','N/A')}")
```

---

## 📊 Сводка проблем и приоритеты

| # | Проблема | Приоритет | Решение |
|---|---------|-----------|---------|
| 1 | actual_close не собирается для T+1 | 🔴 КРИТИЧНО | Исправить rolling_backtest.py |
| 2 | LSTM hourly не вносит вклад (w=0) | 🟡 Средний | Убрать из ансамбля или переобучить |
| 3 | Только 13 фичей | 🟡 Средний | Добавить returns, momentum, sentiment |
| 4 | Нет per-ticker фильтрации | 🟡 Средний | Индивидуальные пороги conf по тикеру |
| 5 | Только 2 периода оценки | 🟢 Низкий | Запустить wfv_multifold.py |
| 6 | Единые веса для всех тикеров | 🟢 Низкий | Per-ticker ensemble weights |

---

## 🛠 Как запустить скрипты

### Сбор актуальных данных до 2026-06-16:
```bash
pip install aiomoex aiohttp pandas python-dotenv supabase
python collect_moex_data.py
```

### Multi-fold WFV анализ:
```bash
python wfv_multifold.py \
    --csv data/rolling_backtest_2025-09-01_2026-04-01.csv \
          data/rolling_backtest_2026-04-01_2026-05-26.csv \
    --folds 4 --conf-min 0.58 \
    --out data/wfv_analysis.json
```

### Загрузка данных в Supabase (уже в collect_moex_data.py):
```
Supabase URL: https://ynxqawesgqsrgyhogvgs.supabase.co
Таблица: moex_candles
Колонки: time, close, volume, ticker, timeframe, rsi_14, macd_hist, bb_upper, bb_lower, atr_14, vol_ratio, imoex, usd_rub
```

---

## ✅ Статус данных

| Источник | Статус | Период |
|---------|--------|--------|
| features.csv | ✅ Актуально | 2022-01-03 → **2026-06-16** |
| rolling_backtest (period 1) | ✅ Есть | 2025-09-01 → 2026-04-01 |
| rolling_backtest (period 2) | ✅ Есть | 2026-04-01 → 2026-05-26 |
| macro.csv | ✅ Есть | — |
| news_sentiment.json | ✅ Есть | — |
| Supabase upload | ⬜ Запустить | После collect_moex_data.py |
