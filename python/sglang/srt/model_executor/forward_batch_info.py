from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Store information about a forward batch.

The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
"""

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

if TYPE_CHECKING:
    from sglang.srt.layers.attention import AttentionBackend
    from sglang.srt.managers.schedule_batch import ImageInputs, ModelWorkerBatch
    from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


class ForwardMode(IntEnum):
    # Prefill a new sequence. This is deprecated now. "EXTEND" covers this case.
    PREFILL = auto()
    # Extend a sequence. The KV cache of the first part of the sequence is already computed (e.g., system prompt).
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE.
    MIXED = auto()

    def is_prefill(self):
        return self == ForwardMode.PREFILL

    def is_extend(self):
        return self == ForwardMode.EXTEND or self == ForwardMode.MIXED

    def is_decode(self):
        return self == ForwardMode.DECODE

    def is_mixed(self):
        return self == ForwardMode.MIXED


@dataclass
class ForwardBatch:
    """Store all inputs of a forward pass."""

    # The forward mode
    forward_mode: ForwardMode
    # The batch size
    batch_size: int
    # The input ids
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    # The indices of output tokens in the token_to_kv_pool
    out_cache_loc: torch.Tensor

    # For logprob
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None

    # Position information
    positions: torch.Tensor = None

    # For extend
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_start_loc: Optional[torch.Tensor] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    extend_logprob_start_lens_cpu: Optional[List[int]] = None

    # For multimodal
    image_inputs: Optional[List[ImageInputs]] = None

    # For LoRA
    lora_paths: Optional[List[str]] = None

    # Sampling info
    sampling_info: SamplingBatchInfo = None

    # Attention backend
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool: BaseTokenToKVPool = None
    attn_backend: AttentionBackend = None

    # For Qwen2-VL
    mrope_positions: torch.Tensor = None

    def compute_mrope_positions(
        self, model_runner: ModelRunner, batch: ModelWorkerBatch
    ):
        hf_config = model_runner.model_config.hf_config
        mrope_positions_list = [None] * self.seq_lens.shape[0]
        if self.forward_mode.is_decode():
            for i, _ in enumerate(mrope_positions_list):
                mrope_positions_list[i] = MRotaryEmbedding.get_next_input_positions(
                    batch.mrope_positions_delta[i][0],
                    int(self.seq_lens[i]) - 1,
                    int(self.seq_lens[i]),
                )
        elif self.forward_mode.is_extend():
            for i, image_inputs in enumerate(batch.image_inputs):
                if image_inputs is None:
                    # text only
                    mrope_positions = [[i for i in range(self.seq_lens[i])]] * 3
                    mrope_position_delta = 0
                else:
                    extend_start_loc, extend_seq_len = (
                        self.extend_start_loc[i],
                        self.extend_seq_lens[i],
                    )
                    mrope_positions, mrope_position_delta = (
                        MRotaryEmbedding.get_input_positions(
                            input_tokens=self.input_ids[
                                extend_start_loc : extend_start_loc + extend_seq_len
                            ],
                            image_grid_thw=image_inputs.image_grid_thws,
                            video_grid_thw=None,
                            image_token_id=hf_config.image_token_id,
                            video_token_id=hf_config.video_token_id,
                            vision_start_token_id=hf_config.vision_start_token_id,
                            vision_end_token_id=hf_config.vision_end_token_id,
                            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
                            context_len=0,
                        )
                    )
                mrope_positions_list[i] = mrope_positions
                batch.mrope_positions_delta[i].append(mrope_position_delta)

        self.mrope_positions = torch.tensor(
            np.concatenate(
                [np.array(pos) for pos in mrope_positions_list],
                axis=1,
            ),
            device="cuda",
        ).to(torch.int64)

    def compute_positions(self, batch: ModelWorkerBatch):
        if self.forward_mode.is_decode():
            self.positions = (self.seq_lens - 1).to(torch.int64)
        else:
            self.positions = torch.tensor(
                np.concatenate(
                    [
                        np.arange(prefix_len, prefix_len + extend_len)
                        for prefix_len, extend_len in zip(
                            batch.extend_prefix_lens, batch.extend_seq_lens
                        )
                    ],
                    axis=0,
                ),
                device="cuda",
            ).to(torch.int64)

    @classmethod
    def init_new(
        cls,
        batch: ModelWorkerBatch,
        model_runner: ModelRunner,
    ):
        ret = cls(
            forward_mode=batch.forward_mode,
            batch_size=len(batch.seq_lens),
            input_ids=batch.input_ids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            out_cache_loc=batch.out_cache_loc,
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
            lora_paths=batch.lora_paths,
            sampling_info=batch.sampling_info,
        )

        if not batch.forward_mode.is_decode():
            ret.image_inputs = batch.image_inputs
            ret.extend_seq_lens = torch.tensor(batch.extend_seq_lens, device="cuda")
            ret.extend_prefix_lens = torch.tensor(
                batch.extend_prefix_lens, device="cuda"
            )
            ret.extend_start_loc = torch.zeros_like(ret.extend_seq_lens)
            ret.extend_start_loc[1:] = torch.cumsum(ret.extend_seq_lens[:-1], dim=0)
            ret.extend_seq_lens_cpu = batch.extend_seq_lens
            ret.extend_logprob_start_lens_cpu = batch.extend_logprob_start_lens

        # Init position information
        is_mrope = model_runner.model_is_mrope
        if is_mrope:
            ret.compute_mrope_positions(model_runner, batch)
        else:
            ret.compute_positions(batch)

        # Init attention information
        ret.req_to_token_pool = model_runner.req_to_token_pool
        ret.token_to_kv_pool = model_runner.token_to_kv_pool
        ret.attn_backend = model_runner.attn_backend
        model_runner.attn_backend.init_forward_metadata(ret)

        # Init lora information
        if model_runner.server_args.lora_paths is not None:
            model_runner.lora_manager.prepare_lora_batch(ret)

        return ret
