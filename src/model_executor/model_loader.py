import glob
import json
import os
from collections.abc import Generator
from typing import Tuple, Union, List, Optional

import filelock
import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, PretrainedConfig, PreTrainedTokenizerBase

from src.config import ModelConfig
from src.model_executor.models.qwen2 import Qwen2ForCausalLM

_MODEL_REGISTRY_LIST = {
    "Qwen2ForCausalLM": Qwen2ForCausalLM,
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


def load_hf_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Gets a tokenizer for the given model name via Huggingface."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left",
            truncation_side="left",
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer for model {model_name!r}."
        ) from e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def detokenize_incrementally(
    tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast],
    prev_output_tokens: List[str],
    new_token_id: int,
    skip_special_tokens: bool,
) -> Tuple[Optional[str], str]:
    """
    Detokenizes the new token in conjunction with the previous output tokens.

    NOTE: this function does not update the prev_output_tokens.

    Returns:
        new_token: The new token as string, or None if it should be skipped.
        output_text: The new output text as a string.
    """
    if skip_special_tokens and (new_token_id in tokenizer.all_special_ids):
        return None, ""
    new_token = tokenizer.convert_ids_to_tokens(
        ids=new_token_id,
        skip_special_tokens=skip_special_tokens
    )
    # `convert_ids_to_tokens` is typed as `str | list[str]`; passing a single
    # int always returns a single token, so narrow the type here.
    assert isinstance(new_token, str)
    output_tokens = prev_output_tokens + [new_token]

    # `added_tokens_encoder` is only exposed by concrete tokenizer
    # implementations (e.g. PreTrainedTokenizerFast), not by the
    # PreTrainedTokenizerBase type, so look it up dynamically.
    added_tokens_encoder = getattr(tokenizer, "added_tokens_encoder", {})

    # Convert the tokens to a string.
    # Optimization: If the tokenizer does not have `added_tokens_encoder`,
    # then we can directly use `convert_tokens_to_string`.
    if not added_tokens_encoder:
        output_text = tokenizer.convert_tokens_to_string(output_tokens)
        return new_token, output_text

    sub_texts = []
    current_sub_text = []
    for token in output_tokens:
        if skip_special_tokens and token in tokenizer.all_special_tokens:
            continue
        if token in added_tokens_encoder:
            if current_sub_text:
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                sub_texts.append(sub_text)
                current_sub_text = []
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
        sub_texts.append(sub_text)
    output_text = " ".join(sub_texts)
    return (new_token, output_text)
