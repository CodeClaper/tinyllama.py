from typing import Optional

import torch
from loguru import logger
from transformers import AutoConfig, PretrainedConfig


class ModelConfig:
    """Configuration for the model."""

    def __init__(
        self,
        model: str,
        dtype: str,
        download_dir: Optional[str],
        local_dir: Optional[str],
        use_np_weights: bool,
    ) -> None:
        self.model = model
        self.hf_config: PretrainedConfig = AutoConfig.from_pretrained(model)
        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)
        self.download_dir = download_dir
        self.local_dir = local_dir
        self.use_np_weights = use_np_weights


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

_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "bfloat16": torch.bfloat16
}

def _get_and_verify_dtype(
    hf_config: PretrainedConfig,
    dtype: str
):
    config_dtype = getattr(hf_config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.float32
    
    dtype = dtype.lower()
    if dtype == 'auto':
        if config_dtype == torch.float32:
            torch_dtype = torch.float16
        else:
            torch_dtype = config_dtype
    else:
        if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
            raise ValueError(f"Unknown dtype: {dtype}")
        torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]

    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            pass
        elif config_dtype == torch.float32:
            pass
        else:
            logger.warning(f"Casting {config_dtype} to {torch_dtype}")

    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}."
            )
    return torch_dtype

