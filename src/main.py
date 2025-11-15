import os
import boto3
import tarfile
import logging
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from transformers import pipeline, Pipeline, AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("AIModelAPI")

logger.debug("Debug mode enabled. Full troubleshooting logs active.")

# -----------------------------
# Environment Variables
# -----------------------------
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "models")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
MODEL_DIR = "/tmp/models"

logger.debug(f"ENV -> MINIO_ENDPOINT = {MINIO_ENDPOINT}")
logger.debug(f"ENV -> MINIO_BUCKET   = {MINIO_BUCKET}")
logger.debug(f"ENV -> ACCESS_KEY     = {MINIO_ACCESS_KEY}")
logger.debug(f"ENV -> SECRET_KEY     = {'*' * len(MINIO_SECRET_KEY)}")
logger.debug(f"ENV -> MODEL_DIR      = {MODEL_DIR}")

# -----------------------------
# App Initialization
# -----------------------------
app = FastAPI(title="AI Model API (Debug Version)", version="3.0")

available_models: List[str] = []
current_model_name: Optional[str] = None
model_pipeline: Optional[Pipeline] = None


# -----------------------------
# Helper Functions
# -----------------------------
def get_s3_client():
    logger.debug("Creating boto3 S3 client...")

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
        )
        logger.debug("S3 client created successfully.")
        return s3

    except Exception as e:
        logger.exception(f"Failed to create S3 client: {e}")
        raise


def debug_s3_connectivity():
    """Run a series of checks to confirm MinIO connectivity."""
    logger.debug("Running MinIO connectivity test...")

    try:
        s3 = get_s3_client()

        # List buckets
        logger.debug("Testing list_buckets()")
        buckets = s3.list_buckets()
        logger.debug(f"Buckets visible: {buckets}")

        if MINIO_BUCKET not in [b["Name"] for b in buckets.get("Buckets", [])]:
            logger.warning(f"Bucket '{MINIO_BUCKET}' does not exist.")

        logger.debug(f"Testing list_objects_v2({MINIO_BUCKET})")
        objects = s3.list_objects_v2(Bucket=MINIO_BUCKET)
        logger.debug(f"Objects in bucket: {objects}")

    except Exception as e:
        logger.exception(f"MinIO connectivity test failed: {e}")


def list_models_in_minio() -> List[str]:
    logger.debug(f"Listing models from MinIO bucket '{MINIO_BUCKET}'...")
    try:
        s3 = get_s3_client()

        objs = s3.list_objects_v2(Bucket=MINIO_BUCKET).get("Contents", [])
        logger.debug(f"Raw response from MinIO list_objects: {objs}")

        models = sorted([obj["Key"] for obj in objs if obj["Key"].endswith(".tar.gz")])

        logger.info(f"Found {len(models)} model(s): {models}")
        return models

    except Exception:
        logger.exception("Failed to list models from MinIO")
        return []


def download_and_extract_model(model_file: str) -> str:
    logger.info(f"Downloading model '{model_file}'...")
    try:
        s3 = get_s3_client()

        os.makedirs(MODEL_DIR, exist_ok=True)
        local_path = os.path.join(MODEL_DIR, os.path.basename(model_file))

        logger.debug(f"Downloading MinIO -> local path: {local_path}")
        s3.download_file(MINIO_BUCKET, model_file, local_path)

        extract_dir = local_path.replace(".tar.gz", "")
        os.makedirs(extract_dir, exist_ok=True)

        logger.debug(f"Extracting TAR to: {extract_dir}")

        with tarfile.open(local_path, "r:gz") as tar:
            tar.extractall(extract_dir)

        logger.debug(f"Extracted contents: {os.listdir(extract_dir)}")

        return extract_dir

    except Exception:
        logger.exception(f"Error extracting model: {model_file}")
        raise HTTPException(status_code=500, detail=f"Failed to extract model: {model_file}")


def load_model(model_file: str) -> Pipeline:
    logger.info(f"Loading model: {model_file}")

    try:
        model_path = download_and_extract_model(model_file)

        logger.debug(f"Loading HuggingFace tokenizer/model from: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        logger.debug("Building HuggingFace pipeline...")
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    except Exception:
        logger.exception("Failed to load the model pipeline")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {model_file}")


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
# Startup
# -----------------------------
@app.on_event("startup")
def startup_event():
    logger.info("Starting AI Model API in debug mode.")
    logger.debug("Running MinIO connectivity diagnostics...")
    debug_s3_connectivity()

    global available_models
    available_models = list_models_in_minio()

    logger.info("Registered routes:")
    for route in app.routes:
        if isinstance(route, APIRoute):
            logger.info(f" -> {route.path}  [{','.join(route.methods)}]")


# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/health")
def health_check():
    logger.debug("Health check called.")
    return {
        "status": "ok",
        "model_loaded": current_model_name is not None,
        "current_model": current_model_name,
        "available_models": available_models,
        "minio_endpoint": MINIO_ENDPOINT,
        "minio_bucket": MINIO_BUCKET,
    }


@app.get("/models")
def get_models():
    logger.debug("GET /models called. Refreshing model list.")
    global available_models
    available_models = list_models_in_minio()
    return {"models": available_models}


@app.post("/load-model")
def load_model_endpoint(request: LoadModelRequest):
    logger.debug(f"POST /load-model -> {request.model_name}")

    global model_pipeline, current_model_name

    if request.model_name not in available_models:
        logger.error(f"Model '{request.model_name}' not found in available list: {available_models}")
        raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found.")

    model_pipeline = load_model(request.model_name)
    current_model_name = request.model_name

    return {"message": f"Model '{request.model_name}' loaded."}


@app.post("/generate")
def generate_text(request: GenerateRequest):
    logger.debug("POST /generate called.")

    if model_pipeline is None:
        logger.error("No model is loaded. Cannot generate.")
        raise HTTPException(status_code=503, detail="No model loaded.")

    try:
        prompt = (
            f"Title: {request.title}\n"
            f"Author: {request.author}\n"
            f"Story: {request.story}\n\n"
            f"Continue ({request.num_words} words):"
        )

        result = model_pipeline(prompt, max_length=request.num_words + len(prompt.split()))

        return {
            "generated_text": result[0]["generated_text"],
            "model_used": current_model_name,
        }

    except Exception:
        logger.exception("Text generation failed")
        raise HTTPException(status_code=500, detail="Generation error")
