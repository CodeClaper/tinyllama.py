import torch
import torch.nn as nn

from typing import Any, Dict, List, Optional
from transformers import Qwen2Config
from defs import KVCache

class Qwen2DecoderLaer(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__(config)
        self.hidden_size = config.hidden_size

class Qwen2Model(nn.Module):

    def __init__(self, config:Qwen2Config) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

    def forward(
        self, 
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> Dict[int, Any]:
        return {}

    def load_weights(
        self, 
        model_name_or_path: str,
    ):
        pass


class Qwen2ForCausalLM(nn.Module):
    
    def __init__(self, config) -> None:
        super().__init__(config)
        self.model = Qwen2Model(config)

    def forward(
        self, 
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
    ) -> Dict[int, Any]:
        return {}

    def load_weights(
        self, 
        model_name_or_path: str,
    ):
        pass

