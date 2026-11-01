
from torch import Device


class LogicTokenBlock:

    """ 
    A block that stores a contiguous chunk of tokens from left to right.

    A logic block is used to represent the states of the corresponding 
    physical block in th kvcache.
    """

    def __init__(
        self,
        block_id,
        block_size,
    ) -> None:
        self.block_id = block_id
        self.block_size = block_size
        self.num_tokens = 0


    def is_empty(self) -> bool:
        return self.num_tokens == 0


class PhysicalTokenBlock:
    """ Represents the state of a block in kvcache. """
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
