
from typing import List
from src.config import CacheConfig, ModelConfig, SchedulerConfig
from src.worker.worker import Worker


class Engine:
    """An engine that receives requests and generates texts."""

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self.workers: List[Worker] = []
        self._init_workers()
        self._init_cache()
    
    def _init_workers(self) -> None:
        """Initialize the workers and their configuration. """
        worker = Worker(self.model_config, self.scheduler_config)
        self.workers.append(worker)

    def _init_cache(self) -> None:
        """Profiles the memory usage and intializes the KV cache."""


