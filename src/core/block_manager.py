
from torch import Device
from typing import List, Dict
from src.config import CacheConfig
from src.core.block import PhysicalTokenBlock
from src.sequence import SequenceGroup


# Mapping: logical block number -> physical block.
BlockTable = List[PhysicalTokenBlock]

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
    
    def get_num_free_blocks(self) -> int:
        return len(self.free_list)

class BlockSpaceManager:
    
    def __init__(
        self,
        cache_config: CacheConfig,
        watermark: float = 0.01,
    ) -> None:
        self.cache_config = cache_config
        self.watermark = watermark
        assert self.watermark > 0
        self.watermark_blocks = int(watermark * self.cache_config.num_gpu_blocks)
        self.block_tabls :Dict[int, BlockTable] = {}

    def can_allocate(
        self, 
        seq_group: SequenceGroup
    ) -> bool:
        return True
