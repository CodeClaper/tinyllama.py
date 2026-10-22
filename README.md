# tinyllama.py

A lightweight and fast LLM inference engine built on top of [vLLM](https://github.com/vllm-project/vllm), exposing an OpenAI-compatible RESTful API.

## Features

- Lightweight, easy-to-run inference server
- OpenAI-compatible `/v1/models` and `/v1/completions` endpoints
- CORS enabled for cross-origin clients
- Pluggable model backends (Qwen2.5, Qwen3.5)
- Minimal dependencies: FastAPI + Uvicorn

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

Start the API server:

```bash
python -m src.api --host 0.0.0.0 --port 8080
```

Options:

| Flag     | Default     | Description          |
| -------- | ----------- | -------------------- |
| `--host` | `localhost` | Host to bind         |
| `--port` | `8080`      | Port to listen on    |

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
│   ├── defs.py         # OpenAI-compatible request/response schemas
│   ├── utils.py        # Shared utilities
│   └── model/
│       ├── qwen25.py   # Qwen2.5 backend
│       └── qwen35.py   # Qwen3.5 backend
├── requirements.txt
└── setup.py
```

## Roadmap

- [ ] Implement `/v1/completions` inference
- [ ] Implement `/v1/chat/completions`
- [ ] Streaming responses
- [ ] Model loading and scheduling via vLLM
- [ ] OpenAI-compatible error handling

## License

No license specified.
