from typing import List, Dict
from src.config import CacheConfig
from src.utils import Device
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
        self.gpu_allocator = BlockAllocator(Device.GPU, block_size=self.cache_config.block_size, num_blocks=self.cache_config.num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size=self.cache_config.block_size, num_blocks=self.cache_config.num_cpu_blocks)
        # Mapping: seq_id -> BlockTable.
        self.block_tabls :Dict[int, BlockTable] = {}

    def can_allocate(
        self, 
        seq_group: SequenceGroup
    ) -> bool:
        seqs = seq_group.get_seqs()
        # Here we assume that all sequences in the group are the same prompt.
        seq = seqs[0]
        num_required_blocks = len(seq.logic_token_blocks)
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        return num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks


    def allocate(
        self, 
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs()
        # Here we assume that all sequences in the group are the same prompt.
        seq = seqs[0]

        block_table: BlockTable = []
        for _ in range(len(seq.logic_token_blocks)):
            block = self.gpu_allocator.allocate()
            block.ref_count = seq_group.get_num_seqs()
            block_table.append(block)

        for seq in seq_group.get_seqs():
            self.block_tabls[seq.seq_id] = block_table.copy()


