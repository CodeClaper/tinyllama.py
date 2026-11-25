import torch

from functools import lru_cache
from typing import Type
from loguru import logger

from src.attention.backends.abstract import AttentionBackend
from src.attention.backends.flash_attn import FlashAttentionBacked
from src.utils import is_hip


@lru_cache(maxsize=None)
def get_attn_backend(dtype: torch.dtype) -> Type[AttentionBackend]:
    if _can_use_flash_attn(dtype=dtype):
        return FlashAttentionBacked
    return FlashAttentionBacked


def _can_use_flash_attn(dtype: torch.dtype) -> bool:
    if is_hip():
        logger.info("Cannot use FlashAttention backend for AMD GPUs.")
        return False

    if torch.cuda.get_device_capability()[0] < 8:
        # Volta and Turing NVIDIA GPUs.
        logger.info("Cannot use FlashAttention backend for Volta and Turing GPUs.")
        return False
    if dtype not in (torch.float16, torch.bfloat16):
        logger.info("Cannot use FlashAttention backend for dtype other than torch.float16 or torch.bfloat16.")
        return False

    try:
        import flash_attn
    except ImportError:
        logger.info(
            "Cannot use FlashAttention because the package is not found. "
            "Please install it for better performance.")
        return False
    return True
