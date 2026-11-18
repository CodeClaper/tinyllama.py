

from src.config import CacheConfig, ModelConfig


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

