import os
import boto3
import tarfile
import logging
from fastapi import FastAPI, HTTPException
from transformers import pipeline, Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Environment variables
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio.minio.svc.cluster.local:9000")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "models")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MODEL_FILE = os.getenv("MODEL_FILE", "model.tar.gz")
MODEL_DIR = "/tmp/model"

app = FastAPI(title="AI Model API", version="1.0")

# Global variable to store model pipeline
model_pipeline: Pipeline | None = None


def download_model() -> str:
    """Download the model from MinIO and extract it if needed."""
    try:
        logger.info(f"Connecting to MinIO at {MINIO_ENDPOINT}")
        s3 = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
        )

        os.makedirs(MODEL_DIR, exist_ok=True)
        local_path = f"{MODEL_DIR}/{MODEL_FILE}"

        logger.info(f"Attempting to download model '{MODEL_FILE}' from bucket '{MINIO_BUCKET}'...")
        s3.download_file(MINIO_BUCKET, MODEL_FILE, local_path)

        if local_path.endswith(".tar.gz"):
            with tarfile.open(local_path, "r:gz") as tar:
                tar.extractall(MODEL_DIR)
            logger.info("Model extracted successfully")

        return MODEL_DIR

    except s3.exceptions.NoSuchKey:
        logger.warning(f"Model '{MODEL_FILE}' not found in bucket '{MINIO_BUCKET}'")
        return None
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return None


def load_model() -> Pipeline | None:
    """Load the model pipeline if available."""
    model_path = download_model()
    if not model_path:
        logger.warning("No model available to load.")
        return None

    try:
        logger.info("Loading model pipeline...")
        model = pipeline("text-generation", model=model_path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


@app.on_event("startup")
def startup_event():
    global model_pipeline
    logger.info("Starting AI Model API...")
    model_pipeline = load_model()
    if model_pipeline is None:
        logger.warning("⚠️ No model loaded — API will still run, but generation is disabled.")


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model_pipeline is not None}


@app.post("/generate")
def generate_text(title: str, author: str, words: int, story: str):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="No model available yet. Please train and upload one first.")

    try:
        prompt = f"Title: {title}\nAuthor: {author}\nStory: {story}\n\nContinue ({words} words):"
        result = model_pipeline(prompt, max_length=words + len(prompt.split()))
        return {"generated_text": result[0]["generated_text"]}
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {e}")
