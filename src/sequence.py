import enum
import copy

from typing import Dict, List, Optional
from src.core.block import LogicTokenBlock
from src.sampler_params import SamplerParams


class SequenceStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    SWAPPED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    
    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED,
        ]
    
    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> Optional[str]:
        match status:
            case SequenceStatus.FINISHED_STOPPED:
                finish_reason = "stop"
            case SequenceStatus.FINISHED_LENGTH_CAPPED:
                finish_reason = "length"
            case SequenceStatus.FINISHED_ABORTED:
                finish_reason = "abort"
            case _:
                finish_reason = None
        return finish_reason


class SequenceData:

    def __init__(
        self,
        prompt_token_ids: List[int],
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.cumulative_logprob = 0.0

    def append_token_id(
        self, 
        token_id: int, 
        logprob: float
    ) -> None:
        self.output_token_ids.append(token_id)
        self.cumulative_logprob += logprob

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids
    
    def get_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_last_token_id(self) -> int:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]


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
        self.logic_token_blocks: List[LogicTokenBlock] = []
        self.output_logprobs: List[Dict[int, float]] = []
        self.output_tokens: List[str] = []
        self.output_text = ""

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

    def append_token_id(
        self,
        token_id: int,
        logprob: Dict[int, float],
    ) -> None:
        assert token_id in logprob
        self._append_token_to_blocks([token_id])
        self.output_logprobs.append(logprob)
        self.data.append_token_id(token_id, logprob=logprob[token_id])

    def get_len(self) -> int:
        return self.data.get_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def get_output_token_ids(self) -> List[int]:
        return self.data.output_token_ids

    def get_cumulative_logprob(self) -> float:
        return self.data.cumulative_logprob

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)
    
    def fork(self, child_seq: "Sequence") -> None:
        child_seq.logic_token_blocks = copy.deepcopy(self.logic_token_blocks)
        child_seq.output_logprobs = copy.deepcopy(self.output_logprobs)
        child_seq.data = copy.deepcopy(self.data)

    def __repr__(self) -> str:
        return (f'Sequence(seq_id={self.seq_id}, '
                f'status={self.status.name}, '
                f'num_blocks={len(self.logic_token_blocks)})')

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

    def get_seqs(
        self, 
        status: Optional[SequenceStatus] = None,
    ) -> List[Sequence]:
        if status is None:
            return self.seqs
        else:
            return [seq for seq in self.seqs if seq.status == status]

    def get_num_seqs(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> int:
        return len(self.get_seqs(status))

    def find(self, seq_id: int) -> Sequence:
        for seq in self.seqs:
            if seq.seq_id == seq_id:
                return seq
        raise ValueError(f"Sequence {seq_id} not found.")

    def is_finished(self) -> bool:
        return all(seq.is_finished() for seq in self.seqs)

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


class SequenceOutputs:
    
    def __init__(
        self,
        seq_id: int,
        parent_seq_id: int,
        output_token: int,
        logprobs: Dict[int, float],
    ) -> None:
        self.seq_id = seq_id
        self.parent_seq_id = parent_seq_id
        self.output_token = output_token
        self.logprobs = logprobs

    def __repr__(self) -> str:
        return (f'SequenceOutputs(seq_id={self.seq_id}, '
                f'parent_seq_id={self.parent_seq_id}, '
                f'output_token={self.output_token}), '
                f'logprobs={self.logprobs}')
