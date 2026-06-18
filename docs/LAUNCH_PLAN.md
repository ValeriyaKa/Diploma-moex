# План запуска MOEX Predictor

## Предусловия

Обучение моделей завершено, файлы `.pkl` и `.pt` загружены в Yandex Object Storage (бакет `moex-models-diploma`). Бэктесты пройдены.

---

## Шаг 1. Подготовка окружения

Открой терминал в папке проекта `C:\Diploma\project`.

```bash
# Создай виртуальное окружение (если ещё нет)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Установи зависимости
pip install -r requirements.txt
```

Проверь что `.env` на месте и содержит:
- `SUPABASE_URL` и `SUPABASE_KEY` — доступ к базе данных
- `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET_NAME` — доступ к моделям в S3
- `API_BASE_URL=http://127.0.0.1:8000`

---

## Шаг 2. Скачай модели из S3

```bash
python -m backend.ml.download_models
```

Что происходит: скрипт подключается к Yandex Object Storage и скачивает все `.pkl` и `.pt` файлы в папку `./models/`. Если модели уже скачаны и совпадают по размеру — пропускает.

Проверка: в папке `models/` должны быть файлы вида:
- `lgbm_SBER.pkl`, `lgbm_SBER_explainer.pkl`, `lgbm_SBER_features.pkl` — для каждого тикера
- `lstm_daily.pt`, `lstm_hourly.pt` — LSTM модели (если обучены)
- `scaler.pkl` — нормализатор признаков

---

## Шаг 3. Обнови данные в базе

```bash
# Загрузи свежие свечи, индикаторы, новости
python -m backend.collector.run --daily-update
```

Что происходит: скрипт тянет с MOEX ISS API свежие свечи по всем 35 тикерам, считает индикаторы (RSI, MACD, BB и т.д.) и сохраняет в Supabase.

Если база пустая (первый запуск):
```bash
# Загрузи историю с 2021 года (~20 минут)
python -m backend.collector.run --load-history --from 2021-01-01
```

---

## Шаг 4. Сгенерируй прогнозы

```bash
python generate_predictions.py
```

Что происходит: скрипт для каждого из 35 тикеров:
1. Загружает последние данные из Supabase
2. Прогоняет через LightGBM (вероятность роста) + LSTM daily (дельта) + LSTM hourly (внутридневной импульс)
3. Собирает ансамбль: 50% LightGBM + 30% LSTM daily + 20% LSTM hourly
4. Считает SHAP-объяснения (top-10 факторов)
5. Применяет фильтр уверенности (порог 0.50) → BUY / SELL / HOLD
6. Сохраняет прогноз в Supabase

Занимает ~3–5 минут.

---

## Шаг 5. Запусти FastAPI сервер (терминал 1)

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Проверка — открой в браузере:
- http://127.0.0.1:8000/health — должен вернуть `{"status": "ok", "database": "ok"}`
- http://127.0.0.1:8000/api/predictions — JSON с прогнозами
- http://127.0.0.1:8000/api/candles/SBER — свечи Сбербанка

---

## Шаг 6. Запусти Streamlit (терминал 2)

Открой **второй** терминал (первый занят FastAPI):

```bash
streamlit run frontend/app.py
```

Streamlit откроется на http://localhost:8501.

Что должно работать:
- Дашборд — карточки с прогнозами (BUY/SELL/HOLD) по всем акциям
- Анализ акции — выпадающий список «SBER — Сбербанк», график цены, SHAP-факторы с понятными подписями
- Режим исследования — в сайдбаре переключи «Выбрать дату анализа» → выбери дату → увидишь прогноз на тот день

---

## Шаг 7. Запусти опрос (для эксперимента)

Открой файл `survey.html` в браузере (двойной клик). Убедись:
- Streamlit работает на http://localhost:8501 (для блока «С системой»)
- В `survey.html` переменная `APPS_SCRIPT_URL` указывает на твой Apps Script
- Даты `ANALYSIS_DATE` и `NEXT_TRADING_DATE` актуальны

---

## Ежедневный цикл (после первого запуска)

Каждый будний день после 19:30 МСК:

```bash
# 1. Обнови данные
python -m backend.collector.run --daily-update

# 2. Сгенерируй прогнозы
python generate_predictions.py

# 3. Перезапусти Streamlit (или он подхватит сам через кэш ~2 мин)
```

Шаги 1–2 можно автоматизировать через планировщик задач Windows:
- Создай `.bat` файл с командами
- Добавь задание в Планировщик задач на 19:45 каждый будний день

---

## Структура портов

| Сервис     | Порт  | URL                        |
|------------|-------|----------------------------|
| FastAPI    | 8000  | http://127.0.0.1:8000      |
| Streamlit  | 8501  | http://localhost:8501       |
| Supabase   | —     | облако (ynxqawesgqsrgyhogvgs.supabase.co) |
| S3 модели  | —     | Yandex Object Storage      |

---

## Типичные проблемы

**«Прогнозы пока не сформированы»** на дашборде
→ Запусти `python generate_predictions.py`

**«API недоступен»** на странице «О системе»
→ Запусти `uvicorn backend.api.main:app --port 8000` в отдельном терминале

**Ошибка `NoneType object is not subscriptable` при daily-update**
→ MOEX ISS API не вернул данные (выходной день или API недоступен). Попробуй позже.

**Модели не скачиваются**
→ Проверь `S3_ACCESS_KEY` и `S3_SECRET_KEY` в `.env`. Убедись что бакет `moex-models-diploma` существует.

**SHAP показывает переменные вместо названий**
→ Обнови `frontend/app.py` (словарь `feat_names` уже расширен на ~70 признаков).

---

## Краткая шпаргалка (копируй и выполняй)

```bash
# Активируй окружение
venv\Scripts\activate

# Скачай модели (первый раз)
python -m backend.ml.download_models

# Обнови данные + прогнозы
python -m backend.collector.run --daily-update
python generate_predictions.py

# Запусти API (терминал 1)
uvicorn backend.api.main:app --port 8000

# Запусти фронтенд (терминал 2)
streamlit run frontend/app.py
```
