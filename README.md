# tinyllama.py

A lightweight and fast LLM inference engine **inspired by** [vLLM](https://github.com/vllm-project/vllm), exposing an OpenAI-compatible RESTful API. It is an independent implementation and does not depend on or build upon the vLLM codebase.

## Features

- Lightweight, easy-to-run inference server
- OpenAI-compatible `/v1/models` and `/v1/completions` endpoints
- CORS enabled for cross-origin clients
- Pluggable model backends (Qwen2.5, Qwen3.5)
- Load models from Hugging Face or a local directory
- Configurable dtype with automatic selection from the model config
- Unified logging via [Loguru](https://github.com/Delgan/loguru)
- Minimal server dependencies: FastAPI + Uvicorn

## Requirements

- Python 3.12+
- A CUDA-capable GPU (recommended for inference)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

`--model` is required. Load a model from Hugging Face:

```bash
python -m src.api --model Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8080
```

Or load an already-downloaded model from a local directory:

```bash
python -m src.api --model Qwen/Qwen2.5-3B-Instruct --local-dir /path/to/model --host 0.0.0.0 --port 8080
```

Options:

| Flag                | Default     | Description                                                              |
| ------------------- | ----------- | ------------------------------------------------------------------------ |
| `--model`           | required    | Hugging Face model name or path                                          |
| `--host`            | `localhost` | Host to bind                                                             |
| `--port`            | `8080`      | Port to listen on                                                        |
| `--dtype`           | `auto`      | Weight/activation dtype: `auto`, `half`, `bfloat16`, `float`             |
| `--download-dir`    | –           | Hugging Face cache directory to download weights into                    |
| `--local-dir`       | –           | Local model directory; enables offline loading                           |
| `--use-np-weights`  | `false`     | Save a numpy copy of weights for faster loading (up to 2x disk usage)    |
| `--block-size`      | `16`        | KV-cache token block size: `8`, `16`, `32`                               |
| `--swap_space`      | `8`         | CPU swap space (GB) per GPU                                              |

`--local-dir` and `--download-dir` are mutually exclusive. When `--local-dir` is set, weights are loaded from that directory and no download is performed.

## API

### List models

```bash
curl http://localhost:8080/v1/models
```

### Create a completion

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "prompt": "Hello, my name is",
    "max_tokens": 32
  }'
```

> **Note:** The completion endpoint is not yet implemented and currently returns an error response.

## Project Structure

```
.
├── src/
│   ├── api.py          # FastAPI application and CLI entrypoint
│   ├── config.py       # Model and cache configuration
│   ├── defs.py         # OpenAI-compatible request/response schemas
│   ├── logger.py       # Loguru logging setup and uvicorn interception
│   ├── sample_params.py # Sampling parameters
│   ├── utils.py        # Shared utilities
│   ├── engine/
│   │   └── engine.py   # Request handling and worker orchestration
│   ├── model/
│   │   ├── model_loader.py  # Model architecture resolution and loading
│   │   ├── qwen25.py   # Qwen2.5 backend
│   │   └── qwen35.py   # Qwen3.5 backend
│   └── worker/
│       └── worker.py   # Model execution worker
├── requirements.txt
└── setup.py
```

## Roadmap

- [ ] Implement `/v1/completions` inference
- [ ] Implement `/v1/chat/completions`
- [ ] Streaming responses
- [ ] Complete weight loading for all supported model backends
- [ ] OpenAI-compatible error handling

## License

No license specified.
