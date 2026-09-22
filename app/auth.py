import os
from fastapi import Header, HTTPException, Request

def verify_api_key(x_api_key:str = Header(...)):
    expected_key = os.environ.get("GATEWAY_API_KEY")
    if expected_key is None:
        raise HTTPException(status_code=500, detail="Server misconfigured: GATEWAY_API_KEY not set")
    if x_api_key!=expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing api key")

def get_api_key_for_rate_limit(request: Request)->str:
    """Used by the rate limiter to group requests by caller, not by IP."""
    return request.headers.get("x-api-key","unknown")
