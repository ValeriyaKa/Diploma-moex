# Deploy MOEX Predictor To Container Apps

This image runs:

- Streamlit public site on port `8501`
- FastAPI backend inside the same container on `127.0.0.1:8000`

Set these Container Apps environment variables:

```text
SUPABASE_URL=...
SUPABASE_KEY=...
MODELS_DIR=./models
MODEL_VERSION=v1.0.0
```

Build locally:

```bash
docker build -t moex-predictor:latest .
```

Run locally:

```bash
docker run --rm -p 8501:8501 \
  -e SUPABASE_URL="$SUPABASE_URL" \
  -e SUPABASE_KEY="$SUPABASE_KEY" \
  moex-predictor:latest
```

Open:

```text
http://localhost:8501
```

For Container Apps, use public port:

```text
8501
```

Artifact Registry push template:

```bash
docker tag moex-predictor:latest diploma-moex.cr.cloud.ru/moex-predictor:latest
docker push diploma-moex.cr.cloud.ru/moex-predictor:latest
```

PowerShell helper:

```powershell
cd C:\Diploma\project
powershell -ExecutionPolicy Bypass -File .\deploy_cloudru.ps1 -Image "diploma-moex.cr.cloud.ru/moex-predictor"
```

Use this image in Cloud.ru Container Apps:

```text
diploma-moex.cr.cloud.ru/moex-predictor:latest
```
