import torch
import torch.nn as nn

from typing import Any, Dict, List, Optional, Tuple
from transformers import Qwen2Config
from src.attention.backends.abstract import AttentionMetadata
from src.attention.layer import Attention
from src.defs import KVCache
from src.model_executor.layers.sampler import Sampler
from src.model_executor.layers.norm import RMSNorm

class Qwen2Attention(nn.Module):

    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000.0,
        use_sliding_window: bool = False,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_position = max_position
        self.sliding_window = sliding_window if use_sliding_window else None
        self.attn = Attention(
            num_heads=num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            sliding_window=self.sliding_window,
        )
    
    def forward(
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
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            rope_theta=getattr(config, "rope_theta", 10000.0)
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        return hidden_states, residual

class Qwen2Model(nn.Module):

    def __init__(self, config:Qwen2Config) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, 
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_caches[i],
                attn_metadata=attn_metadata,
                residual=residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    
    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen2Model(config)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self, 
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        attn_metadata: AttentionMetadata,
    ) -> Dict[int, Any]:
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
        # Sampling is not part of the model forward pass. The engine/worker
        # will sample from the hidden states once generation is implemented.
        return hidden_states

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

