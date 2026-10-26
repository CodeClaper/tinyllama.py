import torch
import torch.nn as nn
from typing import Any, Dict

class Qwen25Model(nn.Module):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


    def forward(
        self, 
        input_ids: torch.Tensor
    ) -> Dict[int, Any]:
        return {}
