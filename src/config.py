import torch

from loguru import logger
from transformers import AutoConfig, PretrainedConfig

_Gib = 1 << 30

def get_head_size(hf_config: PretrainedConfig) -> int:
    """Return the attention head size for a HuggingFace config."""
    text_config = getattr(hf_config, "text_config", hf_config)

    # Some models (e.g. Qwen3) set head_dim explicitly, and it is not
    # necessarily equal to hidden_size // num_attention_heads.
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is not None:
        return head_dim

    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = text_config.d_model  # encoder-decoder configs

    num_heads = text_config.num_attention_heads
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) is not divisible by "
            f"num_attention_heads ({num_heads})"
        )
    return hidden_size // num_heads


class ModelConfig:
    """Configuration for the model."""

    def __init__(
        self,
        model: str,
        dtype: str,
        download_dir: str | None,
        local_dir: str | None,
        use_np_weights: bool,
    ) -> None:
        self.model = model
        self.download_dir = download_dir
        self.local_dir = local_dir
        self.use_np_weights = use_np_weights

        if local_dir and download_dir:
            raise ValueError("--local-dir and --download-dir are mutually exclusive.")
        if local_dir:
            self.model_path = local_dir
            self.local_files_only = True
            logger.info(f"Load {model} from local directory.")
        else:
            self.model_path = model
            self.local_files_only = False
            logger.info(f"Load {model} by downloading.")

        hf_kwargs = {}
        if download_dir:
            hf_kwargs["cache_dir"] = download_dir
        self.hf_config: PretrainedConfig = AutoConfig.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            **hf_kwargs,
        )
        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)

    def get_head_size(self) -> int:
        """Return the attention head size for a HuggingFace config."""
        return get_head_size(self.hf_config)
    
    def get_num_layers(self) -> int:
        """Return the number of layers for a HuggingFace config."""
        text_config = getattr(self.hf_config, "text_config", self.hf_config)

        for attr in (
            "num_hidden_layers",
            "num_decoder_layers",
            "num_layers",
            "n_layer",
        ):
            num_layers = getattr(text_config, attr, None)
            if num_layers is not None:
                return num_layers

        raise AttributeError(
            "Cannot determine the number of layers from the HuggingFace "
            f"config of type {type(text_config).__name__}."
        )

    def get_num_heads(self) -> int:
        """Return the number of attention heads for a HuggingFace config."""
        text_config = getattr(self.hf_config, "text_config", self.hf_config)

        for attr in ("num_attention_heads", "n_head"):
            num_heads = getattr(text_config, attr, None)
            if num_heads is not None:
                return num_heads

        raise AttributeError(
            "Cannot determine the number of attention heads from the "
            f"HuggingFace config of type {type(text_config).__name__}."
        )

    def get_num_kv_heads(self) -> int:
        """Return the number of key/value heads for a HuggingFace config."""
        text_config = getattr(self.hf_config, "text_config", self.hf_config)

        num_kv_heads = getattr(text_config, "num_key_value_heads", None)
        if num_kv_heads is not None:
            return num_kv_heads

        # Multi-head attention: one key/value head per query head.
        return self.get_num_heads()

class CacheConfig:
    """
    Configuration for cache.
    
    Args:
        block_size: Size of a cache block in mumber of tokens.
        swap_space: Size of the CPU swap space per GPU(in GiB).
        gpu_memory_utilication: Fraction of GPU memory to use for the Engine execution.
    """

    def __init__(
        self,
        block_size: int,
        swap_space: int,
        gpu_memory_utilization: float,
    ) -> None:
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * _Gib
        self.num_gpu_blocks = 0
        self.num_cpu_blocks = 0


class SchedulerConfig:
    """
    Schduler configuration.

    Args:
        max_num_batched_tokens: Maximum number of tokens to be process in a single iteration.
        max_num_seqs: Maximum number of sequences to be processed in a single iteration.
    """
    def __init__(
        self,
        max_num_batched_tokens: int,
        max_num_seqs: int
    ):
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs


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
    config_dtype = getattr(hf_config, "dtype", None)
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
        if torch_dtype == torch.float32 or config_dtype == torch.float32:
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

