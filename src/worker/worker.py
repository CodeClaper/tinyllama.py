import torch

from typing import Tuple
from src.config import ModelConfig
from src.sample_params import SampleParams


class Worker:
    """A worker class that executes the model on a device."""

    def __init__(
        self,
        model_config: ModelConfig
    ) -> None:
        self.model_config = model_config

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
        return 1, 2
