# MOEX Predictor

MOEX Predictor is a FastAPI + Streamlit application for MOEX stock signal forecasting. It combines market data collection, Supabase/PostgreSQL storage, and ML inference with a LightGBM + LSTM ensemble.

## Features

- MOEX ticker dashboard;
- candles, indicators, macro data, and prediction history;
- `BUY`, `SELL`, and `HOLD` signals;
- model accuracy statistics;
- hourly forecast for the next trading day;
- local and Docker-based launch.

## Project Layout

```text
backend/        FastAPI API, collectors, database access, ML inference
frontend/       Streamlit UI
scripts/        maintenance, training, evaluation, and data scripts
config/         YAML configs for experiments and training
datasphere/     DataSphere training/export helpers
deploy/         deployment notes and container startup helper
diagrams/       architecture diagrams in SVG format
docs/           project notes and reports
data/           local datasets, ignored by Git
models/         local trained models, ignored by Git
```

Root-level files are intentionally limited to repository/runtime basics such as `README.md`, `.gitignore`, `.env.example`, `Dockerfile`, `.dockerignore`, and `requirements*.txt`.

Files that are not part of the clean GitHub project were moved to:

```text
C:\Diploma\Документы\project_archive_2026-06-18
```

Local root exports were moved to:

```text
data/root_exports_2026-06-18/
```

## Requirements

- Python 3.11 or 3.12;
- Supabase/PostgreSQL with the project tables;
- local `models/` directory with trained models, or configured model download from Object Storage;
- dependencies from `requirements.txt`.

## Environment

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create `.env`:

```powershell
Copy-Item .env.example .env
```

Fill in:

```text
SUPABASE_URL=...
SUPABASE_KEY=...
MODELS_DIR=./models
MODEL_VERSION=v1.0.0
API_BASE_URL=http://127.0.0.1:8000
```

## Local Run

Start the API:

```powershell
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Start the UI:

```powershell
streamlit run frontend/app.py
```

Open:

```text
http://localhost:8501
```

API docs and health check:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

## Docker

Build:

```powershell
docker build -t moex-predictor:latest .
```

Run:

```powershell
docker run --rm -p 8501:8501 `
  -e SUPABASE_URL="..." `
  -e SUPABASE_KEY="..." `
  -e MODELS_DIR="./models" `
  moex-predictor:latest
```

The container exposes Streamlit on port `8501`; FastAPI runs inside the container on `127.0.0.1:8000`.

## API Endpoints

- `GET /health` - service health check;
- `GET /api/securities` - tracked securities;
- `GET /api/candles/{ticker}` - OHLCV candles;
- `GET /api/indicators/{ticker}` - technical indicators;
- `GET /api/macro` - macro data;
- `GET /api/predictions` - latest predictions;
- `GET /api/predictions/{ticker}` - prediction history;
- `POST /api/predictions/{ticker}/generate` - generate one prediction;
- `POST /api/predictions/generate-all` - generate predictions for all tickers;
- `GET /api/hourly-forecast/{ticker}` - hourly forecast.

## GitHub Push

Check what will be committed:

```powershell
git status --short
```

Commit changes:

```powershell
git add .
git commit -m "Organize project structure"
```

Push the current branch:

```powershell
git push
```
