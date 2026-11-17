import time

from typing import Any, List, Optional
from src.config import CacheConfig, ModelConfig, SchedulerConfig
from src.core.scheduler import Scheduler
from src.defs import RequestOutput
from src.model_executor.model_loader import load_hf_tokenizer
from src.sampler_params import SamplerParams
from src.worker.worker import Worker
from src.utils import Counter
from src.sequence import Sequence, SequenceGroup, SequenceStatus

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
        self.schduler = Scheduler(scheduler_config, cache_config)
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
        self.schduler.add_seq_group(seq_group)
    
    def step(self) -> List[RequestOutput]:
        """
        Performs one decoding iteration and return newly generted result.

        This function performs one decoding iteration of the engine. It first schedules
        the sequence to be executed in the next iteration and the token blocks to be swapped
        in/out/copy. Then, it executes the model and updates the scheduler with the model output.
        Finally, it decodes the sequence and return the newly generated results.
        """
        
        # Do schedules.
        seq_group_metadata_list, schdule_results = self.schduler.schedule()
        if (seq_group_metadata_list) and schdule_results.is_empty():
            ## Nothing to do.
            return []
        
        # Execute the model.
        output = self._run_worker(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=schdule_results.blocks_to_swap_in,
            blocks_to_swap_out=schdule_results.blocks_to_swap_out,
            blocks_to_copy=schdule_results.blocks_to_copy,
        )
        
        # Update the schduler with the mode outputs.
        seq_groups = self.schduler.update(output)

        return []


    def _run_worker(
        self, 
        method: str, 
        if_all_ouputs: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        """Run the given method on all workers."""
        all_ouputs = []
        for worker in self.workers:
            executor = getattr(worker, method)
            assert executor is not None, "Not found method: " + method
            output = executor(*args, **kwargs)
            all_ouputs.append(output)

        if if_all_ouputs:
            return all_ouputs
        
        # Make sure all works have the same results.
        first = all_ouputs[0]
        for other in all_ouputs[1:]:
            assert first == other
        return first


    def _decode_sequences(self, seq_groups: List[SequenceGroup]) -> None:
        """Decode the sequence outputs."""
        for seq_group in seq_groups:
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                pass

