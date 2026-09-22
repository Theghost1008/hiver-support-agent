from fastapi import FastAPI, Depends, Request
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.auth import verify_api_key, get_api_key_for_rate_limit
from src.pipeline import process_message

limiter = Limiter(key_func=get_api_key_for_rate_limit)

app = FastAPI(title="AppleSupport AI Agent")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)#type: ignore[arg-type]


class MessageRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/process")
@limiter.limit("8/minutes")
def process(request: Request, body: MessageRequest, _=Depends(verify_api_key)):
    result = process_message(body.message)
    return result
