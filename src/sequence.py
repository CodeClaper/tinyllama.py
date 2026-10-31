import enum
from typing import Dict, List

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
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
    ) -> None:
        self.seq_id = seq_id
        self.prompt = prompt
        self.data = SequenceData(prompt_token_ids)
        self.block_size = block_size
        self.status = SequenceStatus.WAITING

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
