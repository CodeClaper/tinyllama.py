import time

from loguru import logger
from typing import List, Dict, Optional, Tuple
from src.config import CacheConfig, SchedulerConfig
from src.core.block_manager import BlockSpaceManager
from src.core.policy import PolicyFactory
from src.sequence import SequenceGroup, SequenceStatus
from src.utils import Device, PreemptionMode

_LOGGING_INTERVAL_SECOND = 5

class SchedulerResult:

    def __init__(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        # Swap in and out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.blocks_to_swap_in  = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy     = blocks_to_copy

    def is_empty(self) -> bool:
        return (not self.blocks_to_swap_in and 
                not self.blocks_to_swap_out and 
                not self.blocks_to_copy)

class Scheduler:
    
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        is_log_stats: bool = False,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.is_log_stats = is_log_stats
        self.policy = PolicyFactory.get_policy("fcfs")
        self.block_manager = BlockSpaceManager(cache_config=self.cache_config)
        self.waiting: List[SequenceGroup] = []
        self.running: List[SequenceGroup] = []
        self.swapped: List[SequenceGroup] = []
        self.last_logging_time: float = 0.0
        self.num_input_tokens: List[Tuple[float, int]] = []

    def add_seq_group(
        self, 
        seq_group: SequenceGroup,
    ) -> None:
        self.waiting.append(seq_group)


    def _schedule(self) -> Tuple[SchedulerResult, List[str]]:
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}
        
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
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # There is no other sequence to be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
        self.running = running

        # Swap in the sequence group in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        while self.swapped and not blocks_to_swap_out:
            seq_group = self.swapped[0]
            if seq_group in preempted:
                break
            if not self.block_manager.can_swap_in(seq_group):
                break

            num_new_reqs = seq_group.get_num_seqs(status=SequenceStatus.SWAPPED)
            num_cur_seqs = len(self.running)
            if num_cur_seqs + num_new_reqs > self.scheduler_config.max_num_seqs:
                break
            
            seq_group = self.swapped.pop(0)
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slot(seq_group, blocks_to_copy)
            self.running.append(seq_group)
            
        num_batched_tokens = sum(
            seq_group.get_num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running
        )

        prompt_group_ids: List[str] = []
        if not self.swapped:
            while self.waiting:
                seq_group = self.waiting[0]
                if seq_group in preempted:
                    break
                if not self.block_manager.can_allocate(seq_group):
                    break
                assert len(seq_group.get_seqs()) > 0
                first_seq = seq_group.get_seqs()[0]
                num_prompt_tokens = first_seq.get_len()
                if num_batched_tokens + num_prompt_tokens > self.scheduler_config.max_num_batched_tokens:
                    break
                num_new_reqs = seq_group.get_num_seqs(status=SequenceStatus.WAITING)
                num_cur_seqs = len(self.running)
                if num_cur_seqs + num_new_reqs > self.scheduler_config.max_num_batched_tokens:
                    break

                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_batched_tokens += num_prompt_tokens
                prompt_group_ids.append(seq_group.reqeust_id)
        
        scheduler_result = SchedulerResult(
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
        )

        if not self.is_log_stats:
            return scheduler_result, prompt_group_ids
        
        now = time.time()
        if num_batched_tokens > 0:
            self.num_input_tokens.append((now, num_batched_tokens))
        elapsed_time = now - self.last_logging_time
        if elapsed_time > _LOGGING_INTERVAL_SECOND:
            self.last_logging_time = now
            self.num_input_tokens = [
                (t, n) for t, n in self.num_input_tokens
                if now - t < _LOGGING_INTERVAL_SECOND
            ]
            if len(self.num_input_tokens) > 1:
                total_num_tokens = sum(n for _, n in self.num_input_tokens[:-1])
                window = now - self.num_input_tokens[0][0]
                avg_throughput = total_num_tokens / window
            else:
                avg_throughput = 0.0

            total_num_gpu_blocks = self.cache_config.num_gpu_blocks
            num_free_gpu_blocks  = self.block_manager.get_num_free_blocks(Device.GPU)
            num_used_gpu_blocks  = total_num_gpu_blocks - num_free_gpu_blocks
            gpu_cache_usage      = num_used_gpu_blocks / total_num_gpu_blocks

            total_num_cpu_blocks = self.cache_config.num_cpu_blocks
            if total_num_cpu_blocks > 0:
                num_free_cpu_blocks = self.block_manager.get_num_free_blocks(Device.CPU)
                num_used_cpu_blocks = total_num_cpu_blocks - num_free_gpu_blocks
                cpu_cache_usage     = num_used_cpu_blocks / total_num_cpu_blocks
            else: 
                cpu_cache_usage     = 0.0
            
            logger.info(
                f"Throughput: {avg_throughput: .1f} tokens/s, "
                f"Running: {len(self.running)} reqs, "
                f"Swapped: {len(self.swapped)} reqs, "
                f"Waiting: {len(self.waiting)} reqs, "
                f"GPU KV cache usage: { gpu_cache_usage * 100:.1f}%"
                f"CPU KV cache usage: { cpu_cache_usage * 100:.1f}%")
        
        return scheduler_result, prompt_group_ids


    def _allocate(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING

    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    
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


    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_in(seq_group):
            raise RuntimeError("Aborted due to the laock of GPU swap sapce. Please increase the swap spaces to avoid this error.")
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self, 
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            raise RuntimeError("Aborted due to the laock of CPU swap sapce. Please increase the swap spaces to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
