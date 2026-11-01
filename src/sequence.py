import enum
from typing import Dict, List, Optional

from src.core.block import LogicTokenBlock, PhysicalTokenBlock
from src.sampler_params import SamplerParams


class SequenceStatus(enum.Enum):
    WAITING  = enum.auto()
    RUNNING  = enum.auto()
    SWAPPED  = enum.auto()
    FINISHED = enum.auto()


class SequenceData:

    def __init__(
        self,
        prompt_token_ids: List[int],
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.cumulative_logprob = 0.0

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids


class Sequence:

    def __init__(
        self,
        seq_id: int,
        prompt: Optional[str],
        prompt_token_ids: List[int],
        block_size: int,
    ) -> None:
        self.seq_id = seq_id
        self.prompt = prompt
        self.data = SequenceData(prompt_token_ids)
        self.block_size = block_size
        self.status = SequenceStatus.WAITING
        self.logic_token_blocks: List[LogicTokenBlock] = []

    def _append_logical_block(self) -> None:
        block = LogicTokenBlock(
            block_id=len(self.logic_token_blocks),
            block_size=self.block_size
        )
        self.logic_token_blocks.append(block)

    def _append_token_to_blocks(
        self,
        token_ids: List[int],
    ) -> None:
        while token_ids: 
            if not self.logic_token_blocks:
                self._append_logical_block()

            last_block = self.logic_token_blocks[-1]
            if last_block.is_full():
                self._append_logical_block()
                last_block = self.logic_token_blocks[-1]
            num_empty_slots = last_block.get_num_empty_slots()
            last_block.append_tokens(token_ids[:num_empty_slots])
            token_ids = token_ids[num_empty_slots:]

class SequenceGroup:

    def __init__(
        self,
        reqeust_id: str,
        seqs: List[Sequence],
        sampling_params: SamplerParams,
        arrival_time: float,
    ) -> None:
        self.reqeust_id = reqeust_id
        self.seqs = seqs
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time


class SequenceGroupMetadata:

    def __init__(
        self,
        reqeust_id: str,
        is_prompt: bool,
        seq_data: Dict[int, SequenceData],
        sampling_params: SamplerParams,
        block_tables: Dict[int, List[int]] | None # Seq id -> list of physical block number.
    ) -> None:
        self.reqeust_id = reqeust_id
        self.is_prompt = is_prompt
        self.seq_data = seq_data
        self.sampling_params = sampling_params
        self.block_tables = block_tables
