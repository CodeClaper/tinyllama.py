import argparse
import fastapi
import uvicorn

from http import HTTPStatus
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.defs import ErrorResponse

app = fastapi.FastAPI()

def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, type="invalid_request_error").model_dump(),
        status_code=status_code
    )

@app.get("/v1/models")
async def get_available_models():
    """Get all available models."""
    models_list = ["qwen2.5", "qwen3.5"]
    return models_list

@app.post("/v1/completions")
async def create_completion(raw_request: Request):
    """Completeion API similar to OpenAI's API. """
    return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, "Not support now.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tinyllama.py OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8080, help="port number")
    args = parser.parse_args()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    uvicorn.run(app=app, host=args.host, port=args.port, log_level="info", timeout_keep_alive=10)
