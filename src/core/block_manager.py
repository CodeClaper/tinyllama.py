
from torch import Device
from typing import List
from src.core import block
from src.core.block import PhysicalTokenBlock


class BlockAllocator:
    """
    Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when requested.
    When a block is freed, its reference count is decremented. When the reference count
    becomes zeor, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_list: List[PhysicalTokenBlock] = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(device, i, block_size)
            self.free_list.append(block)

    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_list:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_list.pop()
        block.ref_count += 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double freed! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_list.append(block)


class BlockSpaceManager:
    
    def __init__(
        self,
        block_size: int,
        num_gpu_block: int,
        num_cpu_block: int,
        watermark: float = 0.01,
    ) -> None:
        self.block_size = block_size
        self.num_gpu_block = num_gpu_block
        self.num_cpu_block = num_cpu_block
        self.watermark = watermark
        assert self.watermark > 0
        self.watermark_blocks = int(watermark * num_gpu_block)
