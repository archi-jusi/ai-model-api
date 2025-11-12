import os
import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
import tarfile
from transformers import pipeline, Pipeline, PipelineException

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Environment variables
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio.minio.svc.cluster.local:9000")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "models")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MODEL_FILE = os.getenv("MODEL_FILE", "model.tar.gz")
MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/model")
DOWNLOAD_RETRIES = 3


def download_model() -> str:
    """Download and extract the model from MinIO"""
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
        )
    except (NoCredentialsError, EndpointConnectionError) as e:
        logger.error(f"[ERROR] Failed to create S3 client: {e}")
        raise

    os.makedirs(MODEL_DIR, exist_ok=True)
    local_path = os.path.join(MODEL_DIR, MODEL_FILE)

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            logger.info(f"[INFO] Downloading model from MinIO: {MODEL_FILE} (attempt {attempt})")
            s3.download_file(MINIO_BUCKET, MODEL_FILE, local_path)
            logger.info("[INFO] Model downloaded successfully")
            break
        except ClientError as e:
            logger.error(f"[ERROR] Failed to download model: {e}")
            if attempt == DOWNLOAD_RETRIES:
                raise
            logger.info("[INFO] Retrying download...")

    # Extract if tar.gz
    if local_path.endswith(".tar.gz"):
        try:
            with tarfile.open(local_path, "r:gz") as tar:
                tar.extractall(MODEL_DIR)
            logger.info("[INFO] Model extracted successfully")
        except (tarfile.TarError, OSError) as e:
            logger.error(f"[ERROR] Failed to extract model: {e}")
            raise

    return MODEL_DIR


def load_model() -> Pipeline:
    """Load the AI model using Hugging Face transformers"""
    try:
        model_path = download_model()
        logger.info("[INFO] Loading model pipeline...")
        model = pipeline("text-generation", model=model_path)
        logger.info("[INFO] Model loaded successfully")
        return model
    except (PipelineException, Exception) as e:
        logger.error(f"[ERROR] Failed to load model: {e}")
        raise


if __name__ == "__main__":
    # Simple test
    try:
        model_pipeline = load_model()
        test_output = model_pipeline("Model exists", max_length=50)
        logger.info(f"[INFO] Test output: {test_output}")
    except Exception as e:
        logger.error(f"[FATAL] Unable to initialize model: {e}")
