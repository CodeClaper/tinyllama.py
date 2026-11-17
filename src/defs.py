import time
from typing import Dict, List, Optional

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

class CompletionOuput:
    """The output data of one completion ouput of a request."""
    
    def __init__(
        self,
        index: int,
        text: str,
        token_ids: List[int],
        cumulative_logprob: float,
        logprobs: Optional[List[Dict[int, float]]],
        finish_reason: Optional[str] = None,
    ) -> None:
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = logprobs
        self.finish_reason = finish_reason

    def __repr__(self) -> str:
        return (f"CompletionOutput(index={self.index}, "
                f"text={self.text!r}, "
                f"token_ids={self.token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob}, "
                f"logprobs={self.logprobs}, "
                f"finish_reason={self.finish_reason})")


class RequestOutput:
    """The output data of a request to the LLM."""

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        outputs: List[CompletionOuput],
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs

    def __repr__(self) -> str:
        return (f"RequestOutput(request_id={self.request_id}, "
                f"prompt={self.prompt}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"outputs={self.outputs})")

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

