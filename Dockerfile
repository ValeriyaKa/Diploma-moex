FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODELS_DIR=./models \
    API_BASE_URL=http://127.0.0.1:8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-container.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-container.txt

COPY backend ./backend
COPY frontend ./frontend
COPY models ./models
COPY deploy/docker_start.py ./docker_start.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["python", "docker_start.py"]
