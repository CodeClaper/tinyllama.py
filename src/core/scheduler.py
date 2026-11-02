import time

from typing import List, Dict, Optional
from src.config import CacheConfig, SchedulerConfig
from src.core.block_manager import BlockSpaceManager
from src.core.policy import PolicyFactory
from src.sequence import SequenceGroup, SequenceStatus
from src.utils import PreemptionMode

class Scheduler:
    
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.policy = PolicyFactory.get_policy("fcfs")
        self.block_manager = BlockSpaceManager(cache_config=self.cache_config)
        self.waiting: List[SequenceGroup] = []
        self.running: List[SequenceGroup] = []
        self.swapped: List[SequenceGroup] = []

    def add_seq_group(
        self, 
        seq_group: SequenceGroup,
    ) -> None:
        self.waiting.append(seq_group)


    def _schedule(self) -> None:
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, int] = {}
        
        now = time.time()
        self.running = self.policy.sort_by_priority(now, self.running)

        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []

        while self.running:
            seq_group = self.running.pop(0)
            while not self.block_manager.can_append_slot(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop(-1)
                    preempted.append(victim_seq_group)
                else:
                    # There is no other sequence to be preempted.
                    # Preempt the current sequence group.
                    preempted.append(seq_group)

    
    def _preempt(
        self, 
        seq_group: SequenceGroup, 
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by defaul, since it ususally needs lower overhead than swapping.
        # However, when the sequence group has multiple sequences (e.g. beam search), recomputation
        # is not supported. In such a case, we use swapping instead.
        if preemption_mode is None:
            running_num = seq_group.get_num_seqs(status=SequenceStatus.RUNNING)
            if running_num == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP

        match preemption_mode:
            case PreemptionMode.RECOMPUTE:
                self._preempt_by_recompute(seq_group)
            case PreemptionMode.SWAP:
                self._preempt_by_swap(seq_group, blocks_to_swap_out)
            case _:
                assert False, 'Invalid preemption mode.'

    def _preempt_by_recompute(
        self, 
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        self.waiting.insert(0, seq_group)


    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        for seq in seqs:
            seq.status = SequenceStatus.SWAPPED
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)


    def _swap_out(
        self, 
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int]
    ) -> None:
        pass
