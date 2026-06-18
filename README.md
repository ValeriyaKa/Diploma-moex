# MOEX Predictor

MOEX Predictor - приложение для прогнозирования сигналов по российским акциям. Проект объединяет FastAPI backend, Streamlit frontend, сбор данных MOEX/новостей/макроиндикаторов и ML-инференс на ансамбле LightGBM + LSTM.

## Возможности

- просмотр списка отслеживаемых бумаг MOEX;
- загрузка свечей, технических индикаторов и макроданных;
- генерация сигналов `BUY`, `SELL`, `HOLD`;
- история прогнозов и статистика точности;
- почасовой прогноз на следующий торговый день;
- локальный запуск backend + frontend или запуск в Docker-контейнере.

## Структура проекта

```text
backend/
  api/          FastAPI endpoints
  collector/    сбор рыночных, новостных и фундаментальных данных
  db/           подключение к Supabase/PostgreSQL
  ml/           загрузка моделей и инференс
frontend/       Streamlit-интерфейс
diagrams/       SVG-схемы архитектуры
models/         локальные ML-модели, не публикуются в Git
data/           локальные датасеты и выгрузки, не публикуются в Git
Dockerfile      контейнер для Streamlit + FastAPI
docker_start.py запуск двух процессов внутри контейнера
```

Исследовательские черновики, тестовые файлы, презентации, офисные документы, временные файлы и лишние рендеры перенесены в:

```text
C:\Diploma\Документы\project_archive_2026-06-18
```

## Требования

- Python 3.11 или 3.12;
- Supabase/PostgreSQL с таблицами проекта;
- локальная папка `models/` с обученными моделями или настроенное скачивание моделей из Object Storage;
- зависимости из `requirements.txt`.

## Настройка окружения

1. Создайте виртуальное окружение:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Установите зависимости:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Создайте `.env` на основе `.env.example`:

```powershell
Copy-Item .env.example .env
```

4. Заполните переменные:

```text
SUPABASE_URL=...
SUPABASE_KEY=...
MODELS_DIR=./models
MODEL_VERSION=v1.0.0
API_BASE_URL=http://127.0.0.1:8000
```

## Локальный запуск

В первом терминале запустите API:

```powershell
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Во втором терминале запустите интерфейс:

```powershell
streamlit run frontend/app.py
```

Откройте:

```text
http://localhost:8501
```

Проверка API:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

## Docker

Сборка:

```powershell
docker build -t moex-predictor:latest .
```

Запуск:

```powershell
docker run --rm -p 8501:8501 `
  -e SUPABASE_URL="..." `
  -e SUPABASE_KEY="..." `
  -e MODELS_DIR="./models" `
  moex-predictor:latest
```

Контейнер публикует Streamlit на порту `8501`, а FastAPI запускает внутри контейнера на `127.0.0.1:8000`.

## Основные API endpoints

- `GET /health` - проверка сервиса;
- `GET /api/securities` - список бумаг;
- `GET /api/candles/{ticker}` - свечи;
- `GET /api/indicators/{ticker}` - технические индикаторы;
- `GET /api/macro` - макроданные;
- `GET /api/predictions` - последние прогнозы;
- `GET /api/predictions/{ticker}` - история прогнозов;
- `POST /api/predictions/{ticker}/generate` - сгенерировать прогноз по тикеру;
- `POST /api/predictions/generate-all` - сгенерировать прогнозы по всем тикерам;
- `GET /api/hourly-forecast/{ticker}` - почасовой прогноз.

## Подготовка к публикации на GitHub

Перед первым push проверьте, что в Git не попали секреты и большие локальные артефакты:

```powershell
git status --short
git add README.md .gitignore .dockerignore .env.example Dockerfile docker_start.py deploy_cloudru.ps1 deploy_container_app.md requirements*.txt backend frontend diagrams config*.yaml *.py *.md
git status --short
git commit -m "Prepare MOEX Predictor project for GitHub"
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

Если ветка называется `master`, замените `main` на `master`.
