import os
import boto3
import tarfile
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline, Pipeline
from typing import Optional, List

# Logging setup
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Environment variables
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio.minio.svc.cluster.local:9000")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "models")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MODEL_DIR = "/tmp/models"

app = FastAPI(title="AI Model API", version="2.0")

# Global state
available_models: List[str] = []
current_model_name: Optional[str] = None
model_pipeline: Optional[Pipeline] = None


# -----------------------------
# Helper Functions
# -----------------------------
def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )


def list_models_in_minio() -> List[str]:
    """List all available .tar.gz models from MinIO."""
    try:
        s3 = get_s3_client()
        objs = s3.list_objects_v2(Bucket=MINIO_BUCKET).get("Contents", [])
        models = sorted([obj["Key"] for obj in objs if obj["Key"].endswith(".tar.gz")])
        logger.info(f"Found {len(models)} models in bucket.")
        return models
    except Exception as e:
        logger.error(f"Failed to list models from MinIO: {e}")
        return []


def download_and_extract_model(model_file: str) -> str:
    """Download and extract the specified model."""
    try:
        s3 = get_s3_client()
        os.makedirs(MODEL_DIR, exist_ok=True)
        local_path = os.path.join(MODEL_DIR, os.path.basename(model_file))

        logger.info(f"Downloading model '{model_file}' from MinIO...")
        s3.download_file(MINIO_BUCKET, model_file, local_path)

        extract_path = os.path.join(MODEL_DIR, os.path.splitext(os.path.basename(model_file))[0])
        os.makedirs(extract_path, exist_ok=True)

        with tarfile.open(local_path, "r:gz") as tar:
            tar.extractall(extract_path)

        logger.info(f"Model '{model_file}' extracted to {extract_path}")
        return extract_path
    except Exception as e:
        logger.error(f"Error extracting model '{model_file}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract model: {e}")


def load_model(model_file: str) -> Pipeline:
    """Load a specific model from MinIO."""
    try:
        model_path = download_and_extract_model(model_file)
        logger.info(f"Loading model from {model_path}...")
        model = pipeline("text-generation", model=model_path)
        logger.info(f"✅ Model '{model_file}' loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model '{model_file}': {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")


# -----------------------------
# Request Models
# -----------------------------
class GenerateRequest(BaseModel):
    title: str
    author: str
    num_words: int = Field(..., gt=0)
    story: str


class LoadModelRequest(BaseModel):
    model_name: str


# -----------------------------
# API Endpoints
# -----------------------------
@app.on_event("startup")
def startup_event():
    global available_models
    logger.info("🚀 Starting AI Model API...")
    available_models = list_models_in_minio()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": current_model_name is not None,
        "current_model": current_model_name,
        "available_models": available_models,
    }


@app.get("/models")
def get_models():
    """Return all available models."""
    global available_models
    available_models = list_models_in_minio()
    return {"models": available_models}


@app.post("/load-model")
def load_model_endpoint(request: LoadModelRequest):
    """Load a specific model from MinIO."""
    global model_pipeline, current_model_name

    if request.model_name not in available_models:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found in bucket.")

    model_pipeline = load_model(request.model_name)
    current_model_name = request.model_name

    return {"message": f"Model '{request.model_name}' loaded successfully."}


@app.post("/generate")
def generate_text(request: GenerateRequest):
    """Generate text using the currently loaded model."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="No model loaded. Please load one first using /load-model.")

    try:
        prompt = f"Title: {request.title}\nAuthor: {request.author}\nStory: {request.story}\n\nContinue ({request.num_words} words):"
        result = model_pipeline(prompt, max_length=request.num_words + len(prompt.story.split()))
        return {
            "generated_text": result[0]["generated_text"],
            "model_used": current_model_name,
        }
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {e}")
