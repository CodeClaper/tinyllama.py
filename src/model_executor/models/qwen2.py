import torch
import torch.nn as nn

from typing import Any, Dict, List
from transformers import Qwen2Config
from src.defs import KVCache
from src.model_executor.layers.attention import PagedAttentionWithRoPE
from src.model_executor.layers.sampler import Sampler
from src.model_executor.layers.norm import RMSNorm

class Qwen2Attention(nn.Module):

    def __init__(
        self, 
        hidden_size:int, 
        num_heads: int
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.attn = PagedAttentionWithRoPE(num_heads==self.num_heads, head_size=self.head_dim, scale=self.scaling, rotary_dim=self.head_dim)
    
    def forwad(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache
    ):
        pass


class Qwen2DecoderLayer(nn.Module):

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(hidden_size=self.hidden_size, num_heads=config.num_attention_heads) 
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache
    ):
        pass

class Qwen2Model(nn.Module):

    def __init__(self, config:Qwen2Config) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)])

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


class Qwen2ForCausalLM(nn.Module):
    
    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen2Model(config)
        self.samper = Sampler(config.vocab_size)

    def forward(
        self, 
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
    ) -> Dict[int, Any]:
        hidden_states = self.model(input_ids, positions, kv_caches)
        next_tokens = self.samper()
        return {}

    def load_weights(
        self, 
        model_name_or_path: str,
        local_dir: str,
        use_np_weights: bool = False,
    ):
        from src.model_executor.model_loader import load_hf_weights

        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in load_hf_weights(
            model_name_or_path,
            local_dir,
            use_np_weights,
        ):
            param = params.get(name)
            if param is None:
                continue
            param.data.copy_(weight.to(device=param.device, dtype=param.dtype))
            loaded.add(name)

        missing = set(params) - loaded
        if missing:
            raise ValueError(
                f"Missing weights for {len(missing)} parameters: "
                f"{sorted(missing)[:5]}"
            )

