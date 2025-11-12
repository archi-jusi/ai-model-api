import os
import logging
import boto3
import tarfile
from botocore.exceptions import ClientError, EndpointConnectionError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, Pipeline, PipelineException

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ai-model-api")

# ----------------------------
# Environment variables
# ----------------------------
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio.minio.svc.cluster.local:9000")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "models")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/model")

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="AI Model API")

# Cache for loaded model
current_model_name: str | None = None
current_model: Pipeline | None = None

# ----------------------------
# Request model
# ----------------------------
class GenerateRequest(BaseModel):
    model_name: str
    title: str
    story: str
    author: str
    max_words: int = 200

# ----------------------------
# Helper functions
# ----------------------------
def get_s3_client():
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
        )
        logger.info("Connected to MinIO successfully")
        return s3
    except EndpointConnectionError as e:
        logger.error(f"Failed to connect to MinIO endpoint: {e}")
        raise HTTPException(status_code=500, detail="Cannot connect to MinIO")


def list_models():
    s3 = get_s3_client()
    try:
        response = s3.list_objects_v2(Bucket=MINIO_BUCKET)
        models = [obj["Key"] for obj in response.get("Contents", [])]
        logger.info(f"Found {len(models)} models in bucket '{MINIO_BUCKET}'")
        return models
    except ClientError as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


def download_and_load_model(model_name: str) -> Pipeline:
    global current_model_name, current_model

    if current_model_name == model_name and current_model is not None:
        logger.info(f"Using cached model: {model_name}")
        return current_model

    s3 = get_s3_client()
    os.makedirs(MODEL_DIR, exist_ok=True)
    local_path = os.path.join(MODEL_DIR, model_name)

    # Download model
    try:
        logger.info(f"Downloading model '{model_name}' from bucket '{MINIO_BUCKET}'...")
        s3.download_file(MINIO_BUCKET, model_name, local_path)
        logger.info("Download completed")
    except ClientError as e:
        logger.error(f"Failed to download model '{model_name}': {e}")
        raise HTTPException(status_code=500, detail="Failed to download model")

    # Extract if .tar.gz
    if local_path.endswith(".tar.gz"):
        try:
            with tarfile.open(local_path, "r:gz") as tar:
                tar.extractall(MODEL_DIR)
            logger.info("Model extracted successfully")
        except tarfile.TarError as e:
            logger.error(f"Failed to extract '{model_name}': {e}")
            raise HTTPException(status_code=500, detail="Failed to extract model")

    # Load model
    try:
        model_path = os.path.join(MODEL_DIR, model_name.replace(".tar.gz", ""))
        model = pipeline("text-generation", model=model_path)
        current_model_name = model_name
        current_model = model
        logger.info(f"Model '{model_name}' loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")

# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/models")
def get_models():
    """Return list of available models in MinIO"""
    return {"models": list_models()}


@app.post("/generate")
def generate_text(request: GenerateRequest):
    """Generate text using selected model"""
    model = download_and_load_model(request.model_name)

    prompt = (
        f"Title: {request.title}\n"
        f"Author: {request.author}\n"
        f"Story: {request.story}\n"
        f"Continue the story with up to {request.max_words} words."
    )

    try:
        logger.info(f"Generating text with model '{request.model_name}'")
        result = model(prompt, max_length=request.max_words)
        generated_text = result[0]["generated_text"]
        logger.info("Text generation completed successfully")
        return {"generated_text": generated_text}
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise HTTPException(status_code=500, detail="Text generation failed")
