import argparse
import fastapi
import json
import uvicorn

from http import HTTPStatus
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from src.config import CacheConfig, ModelConfig
from src.defs import CompletionRequest, ErrorResponse
from src.engine.engine import Engine

app = fastapi.FastAPI()

def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, type="invalid_request_error").model_dump(),
        status_code=status_code
    )

def create_model_config(args: argparse.Namespace) -> ModelConfig:
    """Create the ModelConfig from the parsed command-line arguments. """
    return ModelConfig(
        model=args.model, 
        dtype=args.dtype, 
        download_dir=args.download_dir, 
        local_dir=args.local_dir, 
        use_np_weights=args.use_np_weights
    )

def create_cache_config(args: argparse.Namespace) -> CacheConfig:
    """Create the CacheConfig from the parsed command-line arguments. """
    return CacheConfig(block_size=args.block_size, swap_space=args.swap_space)

@app.get("/v1/models")
async def get_available_models():
    """Get all available models."""
    models_list = ["qwen2.5", "qwen3.5"]
    return models_list

@app.post("/v1/completions")
async def create_completion(raw_request: Request):
    """Completeion API similar to OpenAI's API. """
    try:
        request = CompletionRequest(**await raw_request.json())
    except (json.JSONDecodeError, ValidationError) as exc:
        return create_error_response(HTTPStatus.BAD_REQUEST, f"Invalid request: {exc}")
    return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, "Not support now.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m src.api", description="Tinyllama.py OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host",       type=str, default="localhost", help="host name")
    parser.add_argument("--port",       type=int, default=8080, help="port number")
    parser.add_argument("--model",      type=str, required=True, help="name or path the huggingface model to use")
    parser.add_argument("--block-size", type=int, default=16, choices=[8, 16, 32], help="token block size")
    parser.add_argument("--swap_space", type=int, default=8, help="CPU swap space size (GB) per GPU.")
    parser.add_argument("--dtype",      type=str, default="auto", choices=['auto', 'half', 'bfloat16', 'float'], help='data type for model weights and activations. '
                                                                                                                      'The "auto" option will use FP16 precision '
                                                                                                                      'for FP32 and FP16 models, and BF precision for BF16 models')
    parser.add_argument('--download-dir', type=str, help='directory to download and load the weights.') 
    parser.add_argument('--local-dir', type=str, help='local directory to load the weights.') 
    parser.add_argument('--use-np-weights', action='store_true', help='save a numpy copy of model weights for faster loading. This can increase the disk usage by up to 2x.')
    args = parser.parse_args()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    model_config = create_model_config(args)
    cache_config = create_cache_config(args)
    engine       = Engine(model_config=model_config, cache_config=cache_config)
    uvicorn.run(app=app, host=args.host, port=args.port, log_level="info", timeout_keep_alive=10)

