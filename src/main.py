from fastapi import FastAPI
from pydantic import BaseModel
from model_loader import load_model

app = FastAPI(title="AI Model API")

model = load_model()

class Request(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "AI Model API is running!"}

@app.post("/predict")
def predict(request: Request):
    output = model(request.text, max_length=40, num_return_sequences=1)
    return {"result": output[0]["generated_text"]}
