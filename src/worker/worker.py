import torch

from typing import List, Tuple
from src.config import ModelConfig, SchedulerConfig
from src.model_executor.model_loader import load_model
from src.sampler_params import SamplerParams
from src.sequence import SequenceData, SequenceGroupMetadata

class Worker:
    """A worker class that executes the model on a device."""

    def __init__(
        self,
        model_config: ModelConfig,
        schduler_config: SchedulerConfig,
    ) -> None:
        self.model_config = model_config
        self.schduler_config = schduler_config
        self.model = load_model(self.model_config)
        self.gpu_cache = None

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int
    ) -> tuple[int, int]:
        """
        Profile the memory usage of the model and get the maxium number of cache
        blocks than can be allocated with the remaining free memory.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_host_memory_stats()

        sample_params = SamplerParams(top_p=0.99, top_k=-1)
        max_num_batched_tokens = self.schduler_config.max_num_batched_tokens
        max_num_seqs = self.schduler_config.max_num_seqs
        seqs: List[SequenceGroupMetadata] = []
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs + (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data = SequenceData(prompt_token_ids=[0] * seq_len)
            seq = SequenceGroupMetadata(
                reqeust_id=str(group_id),
                is_prompt=False,
                seq_data={group_id: seq_data},
                sampling_params=sample_params,
                block_tables=None,
            )
            seqs.append(seq)
        self._prepare_inputs(seqs)
        return 1, 2


    def _prepare_inputs(
        self, 
        seqs: List[SequenceGroupMetadata]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_groups: List[Tuple[List[int], SamplerParams]] = []
        prompt_lens: List[int] = []
        input_tokens: List[int] = []
        input_postions: List[int] = []
        for seq in seqs:
            if not seq.is_prompt:
                continue
            seq_ids = list(seq.seq_data.keys())
            seq_groups.append((seq_ids, seq.sampling_params))
            seq_id = seq_ids[0]
            seq_data = seq.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.extend(prompt_tokens)
            input_postions.extend(range(len(prompt_tokens)));
        
        input_tensor = torch.LongTensor(input_tokens)
        position_tensor = torch.LongTensor(input_postions)

        return input_tensor, position_tensor


    def execute_model(
        self, 
        seqs: List[SequenceGroupMetadata]
    ):
        input_tensor, position_tensor = self._prepare_inputs(seqs)
        return self.model(
            input_ids=input_tensor,
            positions=position_tensor,
            kv_caches=self.gpu_cache,
        )


