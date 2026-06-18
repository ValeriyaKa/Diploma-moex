"""
backend/ml/download_models.py
    python -m backend.ml.download_models
"""
import boto3
import os
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "./models"))
BUCKET     = os.environ.get("S3_BUCKET_NAME") or os.environ.get("YC_BUCKET_NAME", "moex-models-diploma")
ENDPOINT   = "https://storage.yandexcloud.net"
KEY_ID     = os.environ.get("S3_ACCESS_KEY", "") or os.environ.get("YC_ACCESS_KEY", "")
SECRET     = os.environ.get("S3_SECRET_KEY", "") or os.environ.get("YC_SECRET_KEY", "")
MAX_RETRIES = 3



def download_all():
    """Скачивает все файлы из папки models/ в Object Storage."""
    log.info(f"Models directory: {MODELS_DIR.absolute()}")
    log.info(f"Bucket: {BUCKET}")
    log.info(f"Endpoint: {ENDPOINT}")
    
    if not KEY_ID or not SECRET:
        log.error("YC_ACCESS_KEY / YC_SECRET_KEY не заданы в .env или переменных окружения")
        print("ERROR: S3 credentials not set. Set YC_ACCESS_KEY, YC_SECRET_KEY, YC_BUCKET_NAME")
        return False

    log.info(f"Using credentials: KEY_ID={KEY_ID[:8]}..., SECRET={'*'*8}...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url          = ENDPOINT,
            aws_access_key_id     = KEY_ID,
            aws_secret_access_key = SECRET,
            region_name           = "ru-central1",
        )
        log.info("Connected to S3")
    except Exception as e:
        log.error(f"Failed to create S3 client: {e}")
        return False

    count = 0
    try:
        paginator = s3.get_paginator("list_objects_v2")
        
        # Try different prefixes
        prefixes = ["models/", "models", ""]
        found_objects = []
        
        for prefix in prefixes:
            log.info(f"Checking prefix: '{prefix}'...")
            pages = list(paginator.paginate(Bucket=BUCKET, Prefix=prefix))
            
            for page in pages:
                for obj in page.get("Contents", []):
                    found_objects.append(obj)
            
            if found_objects:
                log.info(f"Found {len(found_objects)} objects with prefix '{prefix}'")
                break
        
        if not found_objects:
            log.warning("No objects found in bucket!")
            log.info("Available objects in bucket (all):")
            all_pages = list(paginator.paginate(Bucket=BUCKET))
            if all_pages and all_pages[0].get("Contents"):
                for obj in all_pages[0].get("Contents", [])[:50]:
                    log.info(f"  - {obj['Key']} ({obj['Size']//1024} KB)")
            else:
                log.warning("Bucket is empty!")
            print("ERROR: No models found in S3 bucket")
            return False
        
        for obj in found_objects:
            name = obj["Key"].split("/")[-1]
            if not name or not (name.endswith(".pkl") or name.endswith(".pt")):
                continue
            local = MODELS_DIR / name

            try:
                if local.exists() and local.stat().st_size == obj["Size"]:
                    log.info(f"  = {name} (up-to-date, {obj['Size']//1024} KB)")
                    continue

                log.info(f"  ↓ Downloading {name} ({obj['Size']//1024} KB)...")
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        s3.download_file(BUCKET, obj["Key"], str(local))
                        log.info(f"  ✓ {name} downloaded")
                        count += 1
                        break
                    except Exception as e:
                        if attempt < MAX_RETRIES:
                            log.warning(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
                            time.sleep(2)
                        else:
                            log.error(f"  Failed to download {name}: {e}")
            except Exception as e:
                log.error(f"Error processing {name}: {e}")
        
        log.info(f"Downloaded {count} new/updated files")
        log.info(f"Models saved to: {MODELS_DIR.absolute()}")
        print(f"SUCCESS: {count} models downloaded")
        return True
        
    except Exception as e:
        log.error(f"S3 operation failed: {e}")
        print(f"ERROR: {e}")
        return False


if __name__ == "__main__":
    success = download_all()
    exit(0 if success else 1)
