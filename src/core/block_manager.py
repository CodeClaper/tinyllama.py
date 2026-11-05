from typing import List, Dict, Optional, Set, Tuple
from src.config import CacheConfig
from src.utils import Device
from src.core.block import PhysicalTokenBlock
from src.sequence import Sequence, SequenceGroup, SequenceStatus


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
        self.block_tables :Dict[int, BlockTable] = {}

    def get_num_free_blocks(self, device: Device) -> int:
        match device:
            case Device.GPU:
                return self.gpu_allocator.get_num_free_blocks()
            case Device.CPU:
                return self.cpu_allocator.get_num_free_blocks()
            case _:
                assert False, 'Invalid device type.'

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_id for block in block_table]

    def can_allocate(
        self, 
        seq_group: SequenceGroup,
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
            self.block_tables[seq.seq_id] = block_table.copy()

    def can_append_slot(
        self, 
        seq_group: SequenceGroup,
    ) -> bool:
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.get_num_seqs(SequenceStatus.RUNNING)
        return num_free_gpu_blocks - num_seqs >= 0

    def append_slot(
        self, 
        seq: Sequence,
    ) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        logical_blocks = seq.logic_token_blocks
        block_table = self.block_tables[seq.seq_id]

        if len(block_table) < len(logical_blocks):
            # The sequence has a new logical block.
            # Allocate a new physical block.
            block = self.gpu_allocator.allocate()
            block_table.append(block)
            return None

        assert len(block_table) == len(logical_blocks)
        # We want to append the token to the last physical block.
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.
            return None
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            new_block = self.gpu_allocator.allocate()
            block_table[-1] = new_block
            self.gpu_allocator.free(last_block)
            return last_block.block_id, new_block.block_id
    
    def _free_block_table(self, block_table: BlockTable) -> None:
        for block in block_table:
            match block.device:
                case Device.GPU:
                    self.gpu_allocator.free(block)
                case Device.CPU:
                    self.cpu_allocator.free(block)
                case _:
                    assert False, 'Invalid device type.'

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]

    def _get_physical_blocks(self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            block_table = self.block_tables[seq.seq_id]
            for block in block_table:
                blocks.add(block)
        return list(blocks)

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.get_num_seqs(status=SequenceStatus.SWAPPED)
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        num_required_blocks = len(blocks) + num_swapped_seqs
        return num_free_blocks - num_required_blocks >= self.watermark_blocks


    def swap_in(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # CPU block -> GPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]
            for cpu_block in block_table:
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    gpu_block.ref_count += 1
                else:
                    gpu_block = self.gpu_allocator.allocate()
                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                self.cpu_allocator.free(cpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            cpu_block.block_id: gpu_block.block_id
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping


    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        num_free_blocks = self.cpu_allocator.get_num_free_blocks()
        return num_free_blocks >= len(blocks)

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # GPU block -> CPU block
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]
            for gpu_block in block_table:
                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.cpu_allocator.allocate()
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                self.cpu_allocator.free(gpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            gpu_block.block_id: cpu_block.block_id
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping
