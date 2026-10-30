import glob
import json
import os
from collections.abc import Generator

import filelock
import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import PretrainedConfig

from src.config import ModelConfig
from src.model_executor.models.qwen2 import Qwen2ForCausalLM

_MODEL_REGISTRY_LIST = {
    "qwen25": Qwen2ForCausalLM,
}

_WEIGHT_FILE_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.json",
    "*.model",
    "*.txt",
]

def _get_model_arch(config: PretrainedConfig) -> type[nn.Module]:
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


def _find_weight_files(hf_folder: str) -> tuple[list[str], str]:
    safetensor_files = sorted(glob.glob(os.path.join(hf_folder, "*.safetensors")))
    if safetensor_files:
        return safetensor_files, "safetensors"

    bin_files = sorted(glob.glob(os.path.join(hf_folder, "*.bin")))
    if bin_files:
        return bin_files, "bin"

    raise FileNotFoundError(
        f"No .safetensors or .bin weight files found in {hf_folder}. "
        "Expected a Qwen2/Qwen3 checkpoint in safetensors or PyTorch bin format."
    )


def _iter_hf_state_dict(weight_files: list[str], weight_format: str):
    if weight_format == "safetensors":
        from safetensors.torch import load_file

        for weight_file in weight_files:
            yield from load_file(weight_file, device="cpu").items()
    else:
        for weight_file in weight_files:
            state_dict = torch.load(weight_file, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            yield from state_dict.items()


def load_hf_weights(
    model_name_path: str,
    local_dir: str | None,
    use_np_weights: bool = False,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    lock_dir = local_dir if local_dir is not None else "/tmp"
    os.makedirs(lock_dir, exist_ok=True)
    lock_file_name = model_name_path.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))

    is_local = os.path.isdir(model_name_path)
    if not is_local:
        with lock:
            hf_folder = snapshot_download(
                model_name_path,
                allow_patterns=_WEIGHT_FILE_PATTERNS,
                local_dir=local_dir,
            )
    else:
        hf_folder = model_name_path

    weight_files, weight_format = _find_weight_files(hf_folder)

    if use_np_weights:
        np_folder = os.path.join(hf_folder, "np")
        weight_names_file = os.path.join(np_folder, "weight_names.json")
        with lock:
            if not os.path.exists(weight_names_file):
                os.makedirs(np_folder, exist_ok=True)
                weight_names = []
                for name, param in _iter_hf_state_dict(weight_files, weight_format):
                    param_path = os.path.join(np_folder, name)
                    with open(param_path, "wb") as f:
                        np.save(f, param.detach().cpu().numpy())
                    weight_names.append(name)
                with open(weight_names_file, "w") as f:
                    json.dump(weight_names, f)

        with open(weight_names_file, "r") as f:
            weight_names = json.load(f)
        for name in weight_names:
            param_path = os.path.join(np_folder, name)
            with open(param_path, "rb") as f:
                param = np.load(f)
            yield name, torch.from_numpy(param)
    else:
        yield from _iter_hf_state_dict(weight_files, weight_format)

