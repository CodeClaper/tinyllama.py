import torch
import random
import numpy as np

from typing import Dict, List, Tuple
from src.config import CacheConfig, ModelConfig, SchedulerConfig
from src.core.cache_engine import CacheEgnine
from src.model_executor.model_loader import load_model
from src.sampler_params import SamplerParams
from src.sequence import SequenceData, SequenceGroupMetadata, SequenceOutputs
from src.utils import Device

class Worker:
    """A worker class that executes the model on a device."""

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        schduler_config: SchedulerConfig,
    ) -> None:
        self.model_config = model_config
        self.schduler_config = schduler_config
        self.model = load_model(self.model_config)
        self.cache_engine = CacheEgnine(
            cache_config=cache_config,
            model_config=model_config,
        )
        self.block_size = cache_config.block_size

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        cpu_swap_space: int,
        gpu_memory_utilization: float = 0.9,
    ) -> Tuple[int, int]:
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
        input_tokens, input_postions = self._prepare_inputs(seqs)
        num_layers = self.model_config.get_num_layers()
        self.model(
            input_ids=input_tokens,
            positions=input_postions,
            kv_caches=[(None, None)] * num_layers,
        )

        # Calculate the number of blocks that can be allocated with 
        # the profiled peak memory.
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        cache_block_size = self.cache_engine.calculate_cache_block_size(block_size)
        assert total_gpu_memory * gpu_memory_utilization - peak_memory >= 0, "Not enough GPU memory to run the engine."
        num_gpu_blocks = int((total_gpu_memory * gpu_memory_utilization - peak_memory) // cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        
        # Reset the seed to ensure that the random state is not
        # affected by the model initialization and profiling.
        self.set_random_seed(0)
        return num_gpu_blocks, num_cpu_blocks


    def _prepare_inputs(
        self, 
        seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_groups: List[Tuple[List[int], SamplerParams]] = []
        input_tokens: List[int] = []
        input_postions: List[int] = []
        slot_mappings: List[int] = []

        # Add prompt tokens.
        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue
            seq_ids = list(seq_group_metadata.seq_data.keys())
            seq_groups.append((seq_ids, seq_group_metadata.sampling_params))
            
            # Use any sequence in the group.
            seq_id = seq_ids[0]
            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.extend(prompt_tokens)
            #NOTE: here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_postions.extend(range(len(prompt_tokens)));

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block table are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mappings.extend([0] * prompt_len)
                continue
            
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(prompt_len):
                block_idx = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_idx * self.block_size + block_offset
                slot_mappings.append(slot)

        # Add generation tokens.
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        generation_block_tables: List[List[int]] = []
        for seq_group_metadata in seq_group_metadata_list:
            if seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                context_len = seq_data.get_len()
                position = context_len - 1
                input_postions.append(position)

                block_table = seq_group_metadata.block_tables[seq_id]
                generation_block_tables.append(block_table)

                max_context_len = max(max_context_len, context_len)
                max_num_blocks_per_seq = max(max_num_blocks_per_seq, len(block_table))
                context_lens.append(context_len)

                block_idx = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_idx * self.block_size + block_offset
                slot_mappings.append(slot)

        # Optimization: Pad the input length to be a alignment of 8.
        # This is requried for utilizing the Tensor Cores in NVIDIA GPUs.
        input_tokens = self._pad_to_aligment(input_tokens, alignment=8)
        input_postions = self._pad_to_aligment(input_postions, alignment=8)

        input_tensor = torch.LongTensor(input_tokens)
        position_tensor = torch.LongTensor(input_postions)

        return input_tensor, position_tensor

    
    @torch.inference_mode()
    def execute_model(
        self, 
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> Dict[int, SequenceOutputs]:
        # Handle caceh operatrions.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        input_tensor, position_tensor = self._prepare_inputs(seq_group_metadata_list)
        kv_caches = self.cache_engine.get_kv_cache(Device.GPU)

        # Execute the model.
        output = self.model(
            input_ids=input_tensor,
            positions=position_tensor,
            kv_caches=kv_caches,
        )
        return output


    def set_random_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def _pad_to_aligment(self, x: List[int], alignment: int) -> List[int]:
        pad = ((-len(x)) % alignment)
        return x + [0] * pad

