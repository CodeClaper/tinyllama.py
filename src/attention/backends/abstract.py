
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Dict, List, Tuple, Any, Optional, Type

import torch

from src.defs import KVCache


@dataclass
class AttentionMetadata:

    def asdict_zero_copy(self) -> Dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deep copy."""
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }

class AttentionImpl(ABC):
    
    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: KVCache,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError

class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> Type[AttentionImpl]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def make_metadata(*args, **kwargs) -> AttentionMetadata:
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        raise NotImplementedError

