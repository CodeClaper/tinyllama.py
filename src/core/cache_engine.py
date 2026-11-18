from typing import List, Tuple

import torch
from torch._C import dtype
from src.config import CacheConfig, ModelConfig
from src.defs import KVCache
from src.utils import Device


class CacheEgnine:
    """
    Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU 
    KV cache. It also provides method for performing KV cache operations,
    such as swapping and copying.
    """
    
    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers()
        self.num_kv_heads = model_config.get_num_kv_heads()
        self.dtype = model_config.dtype

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        # The KV cache is allocated lazily in `initialize()` once the number
        # of available blocks has been profiled.
        self.gpu_cache: List[KVCache] = []
        self.cpu_cache: List[KVCache] = []

    def initialize(self) -> None:
        """Allocate the GPU and CPU KV caches.

        The number of blocks must be set on the cache config (usually by the
        engine after profiling the available memory) before calling this.
        """
        self.num_gpu_blocks = self.cache_config.num_gpu_blocks
        self.num_cpu_blocks = self.cache_config.num_cpu_blocks
        self.gpu_cache = self.allocate_cache(Device.GPU)
        self.cpu_cache = self.allocate_cache(Device.CPU)

    def allocate_cache(self, device: Device) -> List[KVCache]:
        num_blocks = (
            self.num_gpu_blocks if device == Device.GPU else self.num_cpu_blocks
        )
        if num_blocks <= 0:
            raise ValueError(
                f"No KV cache blocks allocated for {device.name}. "
                "Profile the available memory before creating the cache."
            )

        torch_device = "cuda" if device == Device.GPU else "cpu"
        key_block_shape = self._get_key_block_shape()
        value_block_shape = self._get_value_block_shape()

        caches: List[KVCache] = []
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(num_blocks, *key_block_shape),
                dtype=self.dtype,
                device=torch_device,
            )
            value_blocks = torch.empty(
                size=(num_blocks, *value_block_shape),
                dtype=self.dtype,
                device=torch_device,
            )
            caches.append((key_blocks, value_blocks))
        return caches
    
    def get_kv_cache(self, device: Device) -> List[KVCache]:
        match device:
            case Device.GPU:
                return self.gpu_cache
            case Device.CPU:
                return self.cpu_cache
            case _:
                assert False, "Not support device type."

    def _get_key_block_shape(self) -> Tuple[int, int, int, int]:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        x = 16 // element_size
        return (
            self.num_kv_heads,
            self.head_size // x,
            self.block_size,
            x
        )

    def _get_value_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_kv_heads,
            self.head_size,
            self.block_size,
        )

    def calculate_cache_block_size(self, block_size: int) -> int:
        key_cache_block = block_size * self.num_kv_heads * self.head_size
        value_cache_block = key_cache_block
        total = self.num_layers * (key_cache_block + value_cache_block)
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return total * element_size
        
