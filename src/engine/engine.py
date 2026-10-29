
from src.config import CacheConfig, ModelConfig


class Engine:
    """An engine that receives requests and generates texts."""

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.workers = []
        self._init_workers()
        self._init_cache()
    
    def _init_workers(self) -> None:
        """Initialize the workers and their configuration. """

    def _init_cache(self) -> None:
        """Profiles the memory usage and intializes the KV cache."""


