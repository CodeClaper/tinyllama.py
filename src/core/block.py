
from torch import Device


class PhysicalTokenBlock:

    def __init__(
        self,
        device: Device,
        block_id: int,
        block_size: int,
    ) -> None:
        self.device = device
        self.block_id= block_id
        self.block_size = block_size
        self.ref_count = 0


    def __repr__(self) -> str:
        return (f'PhysicalTokenBlock(device={self.device},  '
                f'block_id={self.block_id}, '
                f'ref_count={self.ref_count}')
