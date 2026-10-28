import torch

from typing import Tuple
from src.config import ModelConfig, SchedulerConfig
from src.model.model_loader import load_model
from src.sampler_params import SamplerParams


class Worker:
    """A worker class that executes the model on a device."""

    def __init__(
        self,
        model_config: ModelConfig,
        schduler_config: SchedulerConfig,
    ) -> None:
        self.model_config = model_config
        self.schduler_config = schduler_config
        self.model = load_model(self.model_config)

    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int
    ) -> Tuple[int, int]:
        """
        Profile the memory usage of the model and get the maxium number of cache
        blocks than can be allocated with the remaining free memory.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_host_memory_stats()

        sample_params = SamplerParams(top_p=0.99, top_k=-1)
        max_num_batched_tokens = self.schduler_config.max_num_batched_tokens
        max_num_seqs = self.schduler_config.max_num_seqs
        seqs = []
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs + (group_id < max_num_batched_tokens % max_num_seqs))
        return 1, 2
