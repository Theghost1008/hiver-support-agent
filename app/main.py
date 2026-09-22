from fastapi import FastAPI, Depends
from pydantic import BaseModel
from app.auth import verify_api_key
from src.pipeline import process_message

app = FastAPI(title="AppleSupport AI Agent")

class MessageRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/process")
def process(request: MessageRequest, _=Depends(verify_api_key)):
    result = process_message(request.message)
    return result
