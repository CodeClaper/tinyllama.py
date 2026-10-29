from typing import Type

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.model_executor.models.qwen2 import Qwen2ForCausalLM
from transformers import PretrainedConfig

_MODEL_REGISTRY_LIST = {
    "qwen25": Qwen2ForCausalLM,
}

def _get_model_arch(config: PretrainedConfig) -> Type[nn.Module]:
    archs = getattr(config, "architectures", [])
    for arch in archs:
        if arch in _MODEL_REGISTRY_LIST:
            return _MODEL_REGISTRY_LIST[arch]
    raise ValueError(
        f"Model architectures {archs} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY_LIST.keys())}"
    )


def load_model(model_config: ModelConfig) -> nn.Module:
    model_class = _get_model_arch(model_config.hf_config)
    torch.set_default_dtype(model_config.dtype)
    model = model_class(model_config.hf_config)
    model.load_weights(model_config.model_path, model_config.download_dir, model_config.use_np_weights)
    model = model.cuda();
    return model

