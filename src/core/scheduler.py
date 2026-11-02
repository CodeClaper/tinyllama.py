
from typing import List
from src.config import CacheConfig, SchedulerConfig
from src.core.block_manager import BlockSpaceManager
from src.core.policy import PolicyFactory
from src.sequence import SequenceGroup

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


    def schedule(self) -> None:

        for seq_group in self.running:
            is_prompt = seq_group.reqeust_id in 
