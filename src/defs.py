import time

import torch
from pydantic import BaseModel, Field

from src.utils import random_uuid

KVCache = tuple[torch.Tensor, torch.Tensor]

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    suffix: str | None = None
    max_tokens: int | None = 16
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    logprobs: int | None = None
    echo: bool | None = False
    stop: str | list[str] | None = Field(default_factory=list)
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    best_of: int | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    # Additional parameters supported by vLLM
    top_k: int | None = -1
    ignore_eos: bool | None = False
    use_beam_search: bool | None = False

class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: str | None = None
    code: str | None = None

class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{random_uuid()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: str | None = None
    is_blocking: bool = False

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm"
    root: str | None = None
    parent: str | None = None
    permission: list[ModelPermission] = Field(default_factory=list)

class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)

