import time

from typing import List, Optional
from src.config import CacheConfig, ModelConfig, SchedulerConfig
from src.model_executor.model_loader import load_hf_tokenizer
from src.sampler_params import SamplerParams
from src.worker.worker import Worker
from src.utils import Counter
from src.sequence import Sequence, SequenceGroup

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
        self.tokenizer = load_hf_tokenizer(self.model_config.model)
        self.counter = Counter()
        self._init_workers()
        self._init_cache()

    
    def _init_workers(self) -> None:
        """Initialize the workers and their configuration. """
        worker = Worker(self.model_config, self.scheduler_config)
        self.workers.append(worker)

    def _init_cache(self) -> None:
        """Profiles the memory usage and intializes the KV cache."""

    
    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplerParams,
        prompt_token_ids: Optional[List[int]],
        arrival_time: Optional[float],
    ) -> None:
        """
        Add a request to the engine's request pool.

        The request is added to the request pool and will be prcessed by the scheduler as `engine.step()` is called.
        The exact scheduler policy is determined by the scheduler.
        """

        if arrival_time is None:
            arrival_time = time.time()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)
        
        block_size = self.cache_config.block_size
        seqs: List[Sequence] = []
        for _ in range(sampling_params.best_of):
            seq_id = next(self.counter)
            seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)
            seqs.append(seq)

        seq_group = SequenceGroup(request_id, seqs, sampling_params, arrival_time)
