from typing import Any

import torch
from torch import nn


class Qwen35Model(nn.Module):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


    def forward(
        self, 
        input_ids: torch.Tensor
    ) -> dict[int, Any]:
        return {}

    def load_weights(
        self, 
        model_name_or_path: str,
    ):
        pass
