
class ModelConfig:
    """Configuration for the model."""

    def __init__(
        self,
        model: str,
        dtype: str,
    ) -> None:
        self.model = model
        self.dtype = dtype


class CacheConfig:
    """Configuration for cache."""

    def __init__(
        self,
        block_size: int,
        swap_space: int
    ) -> None:
        self.block_size     = block_size
        self.swap_space     = swap_space
        self.num_gpu_blocks = 0
        self.num_cpu_blocks = 0
