from typing import Optional, Union

import torch
import triton
import triton.language as tl
from einops import rearrange

from sglang.jit_kernel.cutedsl_gdn import cutedsl_fused_sigmoid_gating_delta_rule_update
from sglang.srt.environ import Envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.srt.layers.attention.fla.kda import chunk_kda
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    PAD_SLOT_ID,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
    fused_mamba_state_scatter_with_mask,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import is_npu
from sglang.srt.utils.common import rank0_log

# if is_cuda():
#     from sglang.srt.layers.attention.mamba.causal_conv1d import (
#         causal_conv1d_fn as causal_conv1d_fn_cuda,
#     )

#     causal_conv1d_fn = causal_conv1d_fn_cuda
# elif is_npu():
#     from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
#     from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
#         fused_sigmoid_gating_delta_rule_update_npu,
#     )
#     from sgl_kernel_npu.mamba.causal_conv1d import (
#         causal_conv1d_fn_npu,
#         causal_conv1d_update_npu,
#     )

#     chunk_gated_delta_rule = chunk_gated_delta_rule_npu
#     fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu
#     causal_conv1d_fn = causal_conv1d_fn_npu
#     causal_conv1d_update = causal_conv1d_update_npu


# Kernel to track mamba states if needed based on track mask
@triton.jit
def track_mamba_state_if_needed_kernel(
    conv_states_ptr,
    ssm_states_ptr,
    cache_indices_ptr,
    mamba_track_mask_ptr,
    mamba_track_indices_ptr,
    conv_state_stride_0,  # stride for first dimension (batch/pool index)
    ssm_state_stride_0,  # stride for first dimension (batch/pool index)
    conv_state_numel_per_row: tl.constexpr,  # total elements per row
    ssm_state_numel_per_row: tl.constexpr,  # total elements per row
    BLOCK_SIZE: tl.constexpr,
):
    """
    Track conv_states and ssm_states rows based on track mask.

    This kernel replaces a Python loop that copies state tensors for mamba attention.
    For each batch element, if the track mask is True, it copies the entire row from
    the source index (cache_indices[i]) to the destination index (mamba_track_indices[i]).

    Grid: (batch_size,)
    Each block handles one batch element, using multiple threads to copy data in parallel.
    """
    batch_idx = tl.program_id(0)

    # Load the copy mask for this batch element
    track_mask = tl.load(mamba_track_mask_ptr + batch_idx)

    # Early exit if we don't need to track
    if not track_mask:
        return

    # Load source and destination indices
    src_idx = tl.load(cache_indices_ptr + batch_idx)
    dst_idx = tl.load(mamba_track_indices_ptr + batch_idx)

    # Copy conv_states
    # Each thread handles BLOCK_SIZE elements
    for offset in range(0, conv_state_numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < conv_state_numel_per_row

        src_ptr = conv_states_ptr + src_idx * conv_state_stride_0 + element_indices
        dst_ptr = conv_states_ptr + dst_idx * conv_state_stride_0 + element_indices

        data = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, data, mask=mask)

    # Copy ssm_states
    for offset in range(0, ssm_state_numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < ssm_state_numel_per_row

        src_ptr = ssm_states_ptr + src_idx * ssm_state_stride_0 + element_indices
        dst_ptr = ssm_states_ptr + dst_idx * ssm_state_stride_0 + element_indices

        data = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, data, mask=mask)


def track_mamba_states_if_needed(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    mamba_track_mask: torch.Tensor,
    mamba_track_indices: torch.Tensor,
    batch_size: int,
):
    """
    Track mamba states using Triton kernel for better performance.

    Args:
        conv_states: Convolution states tensor [pool_size, ...]
        ssm_states: SSM states tensor [pool_size, ...]
        cache_indices: Source indices for each batch element [batch_size]
        mamba_track_mask: Boolean mask indicating which elements to track [batch_size]
        mamba_track_indices: Indices to track for each batch element [batch_size]
        batch_size: Number of batch elements
    """
    conv_state_numel_per_row = conv_states[0].numel()
    ssm_state_numel_per_row = ssm_states[0].numel()

    # Choose BLOCK_SIZE based on the size of the data
    BLOCK_SIZE = 1024

    # Launch kernel with batch_size blocks
    grid = (batch_size,)
    track_mamba_state_if_needed_kernel[grid](
        conv_states,
        ssm_states,
        cache_indices,
        mamba_track_mask,
        mamba_track_indices,
        conv_states.stride(0),
        ssm_states.stride(0),
        conv_state_numel_per_row,
        ssm_state_numel_per_row,
        BLOCK_SIZE,
    )


class MambaAttnBackendBase(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.pad_slot_id = PAD_SLOT_ID
        self.device = model_runner.device
        self.req_to_token_pool: HybridReqToTokenPool = model_runner.req_to_token_pool
        self.forward_metadata: ForwardMetadata = None
        self.state_indices_list = []
        self.query_start_loc_list = []
        self.retrieve_next_token_list = []
        self.retrieve_next_sibling_list = []
        self.retrieve_parent_token_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None
        self.conv_states_shape: tuple[int, int] = None

    def _forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size

        retrieve_next_token = None
        retrieve_next_sibling = None
        retrieve_parent_token = None
        track_conv_indices = None
        track_ssm_h_src = None
        track_ssm_h_dst = None
        track_ssm_final_src = None
        track_ssm_final_dst = None
        chunk_indices_with_16 = None
        chunk_indices_with_64 = None
        chunk_indices_with_o = None
        chunk_offsets_with_64 = None

        mamba_cache_indices = self.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        )

        if forward_batch.forward_mode.is_decode_or_idle():
            query_start_loc = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
        elif forward_batch.forward_mode.is_extend():
            if forward_batch.forward_mode.is_target_verify():
                query_start_loc = torch.arange(
                    0,
                    bs * (forward_batch.spec_info.draft_token_num + FLA_CHUNK_SIZE) + 1,
                    # forward_batch.input_ids.shape[0] + 1,
                    step=forward_batch.spec_info.draft_token_num + FLA_CHUNK_SIZE,
                    dtype=torch.int32,
                    device=forward_batch.input_ids.device,
                )
                chunk_indices_with_16 = prepare_chunk_indices(query_start_loc, 16)
                chunk_indices_with_64 = prepare_chunk_indices(query_start_loc, 64)
                chunk_offsets_with_64 = prepare_chunk_offsets(query_start_loc, 64)
                BT = min(
                    64,
                    # max(16, triton.next_power_of_2(forward_batch.input_ids.shape[0])),
                    max(
                        16,
                        triton.next_power_of_2(
                            bs
                            * (forward_batch.spec_info.draft_token_num + FLA_CHUNK_SIZE)
                        ),
                    ),
                )
                chunk_indices_with_o = prepare_chunk_indices(query_start_loc, BT)

                if forward_batch.spec_info.topk > 1:
                    retrieve_next_token = forward_batch.spec_info.retrive_next_token
                    retrieve_next_sibling = forward_batch.spec_info.retrive_next_sibling
                    # retrieve_next_token is None during dummy run so skip tensor creation
                    if retrieve_next_token is not None:
                        retrieve_parent_token = torch.empty_like(retrieve_next_token)
            else:
                query_start_loc = torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                )
                query_start_loc[:bs] = forward_batch.extend_start_loc
                query_start_loc[bs] = (
                    forward_batch.extend_start_loc[-1]
                    + forward_batch.extend_seq_lens[-1]
                )
                if (
                    forward_batch.mamba_track_mask is not None
                    and forward_batch.mamba_track_mask.any()
                ):
                    track_conv_indices = self._init_track_conv_indices(
                        query_start_loc, forward_batch
                    )

                    (
                        track_ssm_h_src,
                        track_ssm_h_dst,
                        track_ssm_final_src,
                        track_ssm_final_dst,
                    ) = self._init_track_ssm_indices(mamba_cache_indices, forward_batch)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")

        return ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
            track_conv_indices=track_conv_indices,
            track_ssm_h_src=track_ssm_h_src,
            track_ssm_h_dst=track_ssm_h_dst,
            track_ssm_final_src=track_ssm_final_src,
            track_ssm_final_dst=track_ssm_final_dst,
            chunk_indices_with_16=chunk_indices_with_16,
            chunk_indices_with_64=chunk_indices_with_64,
            chunk_offsets_with_64=chunk_offsets_with_64,
            chunk_indices_with_o=chunk_indices_with_o,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.forward_metadata = self._forward_metadata(forward_batch)

    def _init_track_conv_indices(
        self, query_start_loc: torch.Tensor, forward_batch: ForwardBatch
    ):
        """
        Compute indices for extracting conv states from the input sequence during extend.

        In Mamba models, the conv layer maintains a sliding window of recent inputs.
        After processing a prefill chunk, we need to save the last `conv_state_len` tokens
        of the processed region for prefix caching.

        The key insight is that FLA (Flash Linear Attention) processes sequences in chunks
        of FLA_CHUNK_SIZE. We only track the conv state up to the last complete chunk boundary
        (aligned_len).

        start_indices is the starting token index of the conv state to track in this extend batch.
        indices include all pos to track in this extend batch, conv_state_len for each req that
        needs to be tracked (i.e. mamba_track_mask is True)

        Returns:
            indices: Tensor of shape [num_tracked_requests, conv_state_len] containing
                     flattened positions into the packed input tensor.
        """
        conv_state_len = self.conv_states_shape[-1]

        # Calculate the end position of the last aligned chunk
        lens_to_track = (
            forward_batch.mamba_track_seqlens - forward_batch.extend_prefix_lens
        )
        mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
        aligned_len = (lens_to_track // mamba_cache_chunk_size) * mamba_cache_chunk_size
        start_indices = query_start_loc[:-1] + aligned_len - conv_state_len
        start_indices = start_indices[forward_batch.mamba_track_mask]

        # Create indices: [batch_size, conv_state_len]
        indices = start_indices.unsqueeze(-1) + torch.arange(
            conv_state_len,
            device=self.device,
            dtype=start_indices.dtype,
        )

        return indices.clamp(0, query_start_loc[-1] - 1)

    def _init_track_ssm_indices(
        self, mamba_cache_indices: torch.Tensor, forward_batch: ForwardBatch
    ):
        """
        Compute source and destination indices for tracking SSM states for prefix caching.

        After processing a prefill, we need to save the SSM recurrent state for prefix caching.
        The FLA kernel outputs intermediate hidden states `h` at each chunk boundary,
        plus a `last_recurrent_state` at the end of the chunked prefill size.

        The challenge is that sequences may or may not end on a chunk boundary:
          - Aligned case (len % FLA_CHUNK_SIZE == 0): In this case, FLA will store the to-cache
            state in the last_recurrent_state.
          - Unaligned case (len % FLA_CHUNK_SIZE != 0): The last_recurrent_state includes the
            unaligned position, but we only want state up to the last chunk boundary.
            We must extract from the intermediate `h` tensor at the appropriate chunk index.

        We compute the src and dst indices for all requests that need to be cached
        (i.e. mamba_track_mask is True) based on the rule above.

        For example:
        1. If chunked prefill length is < 64, then only final state has value. In this case we
           cache `final` state.
        2. if chunked prefill length == 64, then only final state has value. In this case we
           cache pos 64, from `final` state
        3. if chunked prefill length >64 and < 128, then both h and final state have value.
           We cache pos 64 from `h` state
        4. if chunked prefill length ==128, then both h and final state have value. We cache
           pos 128 from `final` state. Note `h` doesn't include the pos 128.

        Returns:
            track_ssm_h_src: Source indices into the packed `h` tensor (for unaligned seqs)
            track_ssm_h_dst: Destination cache slot indices (for unaligned seqs)
            track_ssm_final_src: Source indices into last_recurrent_state buffer (for aligned seqs)
            track_ssm_final_dst: Destination cache slot indices (for aligned seqs)
        """
        # Move to CPU to avoid kernel launches for masking operations
        mamba_track_mask = forward_batch.mamba_track_mask.cpu()
        extend_seq_lens = forward_batch.extend_seq_lens.cpu()
        mamba_track_indices = forward_batch.mamba_track_indices.cpu()
        mamba_cache_indices = mamba_cache_indices.cpu()
        mamba_track_seqlens = forward_batch.mamba_track_seqlens.cpu()
        prefix_lens = forward_batch.extend_prefix_lens.cpu()

        # Calculate the number of hidden states per request
        num_h_states = (extend_seq_lens - 1) // FLA_CHUNK_SIZE + 1

        # Calculate the starting offset for each sequence in the packed batch
        track_ssm_src_offset = torch.zeros_like(num_h_states)
        track_ssm_src_offset[1:] = torch.cumsum(num_h_states[:-1], dim=0)

        # Filter variables by track mask
        lens_to_track = mamba_track_seqlens - prefix_lens
        lens_masked = lens_to_track[mamba_track_mask]
        offset_masked = track_ssm_src_offset[mamba_track_mask]
        dst_masked = mamba_track_indices[mamba_track_mask]

        # Determine if the sequence ends at a chunk boundary
        is_aligned = (lens_masked % FLA_CHUNK_SIZE) == 0

        # Case 1: Aligned. Use last_recurrent_state from ssm_states.
        track_ssm_final_src = mamba_cache_indices[mamba_track_mask][is_aligned]
        track_ssm_final_dst = dst_masked[is_aligned]

        # Case 2: Unaligned. Use intermediate state from h.
        # TODO: if support FLA_CHUNK_SIZE % page size != 0, then need to modify this
        not_aligned = ~is_aligned
        track_ssm_h_src = offset_masked[not_aligned] + (
            lens_masked[not_aligned] // FLA_CHUNK_SIZE
        )
        track_ssm_h_dst = dst_masked[not_aligned]

        # Move back to GPU
        return (
            track_ssm_h_src.to(self.device, non_blocking=True),
            track_ssm_h_dst.to(self.device, non_blocking=True),
            track_ssm_final_src.to(self.device, non_blocking=True),
            track_ssm_final_dst.to(self.device, non_blocking=True),
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        self.forward_metadata = self._capture_metadata(
            bs, req_pool_indices, forward_mode, spec_info
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        self.forward_metadata = self._replay_metadata(
            bs, req_pool_indices, forward_mode, spec_info, seq_lens_cpu
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        assert (
            max_num_tokens % max_bs == 0
        ), f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        draft_token_num = max_num_tokens // max_bs
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.query_start_loc_list.append(
                torch.zeros((i + 2,), dtype=torch.int32, device=self.device)
            )
            self.retrieve_next_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_next_sibling_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_parent_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.cached_cuda_graph_verify_query_start_loc = torch.arange(
            0,
            max_bs * (draft_token_num + FLA_CHUNK_SIZE) + 1,
            step=draft_token_num + FLA_CHUNK_SIZE,
            dtype=torch.int32,
            device=self.device,
        )

        self.cached_cuda_graph_chunk_indices_with_16 = []
        self.cached_cuda_graph_chunk_indices_with_64 = []
        self.cached_cuda_graph_chunk_offsets_with_64 = []
        self.cached_cuda_graph_chunk_indices_with_o = []
        for i in range(1, max_bs + 1):
            BT = min(
                64,
                max(16, triton.next_power_of_2(i * (draft_token_num + FLA_CHUNK_SIZE))),
            )
            self.cached_cuda_graph_chunk_indices_with_16.append(
                prepare_chunk_indices(
                    self.cached_cuda_graph_verify_query_start_loc[: i + 1], 16
                )
            )
            self.cached_cuda_graph_chunk_indices_with_64.append(
                prepare_chunk_indices(
                    self.cached_cuda_graph_verify_query_start_loc[: i + 1], 64
                )
            )
            self.cached_cuda_graph_chunk_offsets_with_64.append(
                prepare_chunk_offsets(
                    self.cached_cuda_graph_verify_query_start_loc[: i + 1], 64
                )
            )
            self.cached_cuda_graph_chunk_indices_with_o.append(
                prepare_chunk_indices(
                    self.cached_cuda_graph_verify_query_start_loc[: i + 1], BT
                )
            )

    def _capture_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        chunk_indices_with_16 = None
        chunk_indices_with_64 = None
        chunk_indices_with_o = None
        chunk_offsets_with_64 = None
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif forward_mode.is_target_verify():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
            chunk_indices_with_16 = self.cached_cuda_graph_chunk_indices_with_16[bs - 1]
            chunk_indices_with_64 = self.cached_cuda_graph_chunk_indices_with_64[bs - 1]
            chunk_offsets_with_64 = self.cached_cuda_graph_chunk_offsets_with_64[bs - 1]
            chunk_indices_with_o = self.cached_cuda_graph_chunk_indices_with_o[bs - 1]
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and spec_info.topk > 1:
            # They are None during cuda graph capture so skip the copy_...
            # self.retrieve_next_token_list[bs - 1].copy_(spec_info.retrive_next_token)
            # self.retrieve_next_sibling_list[bs - 1].copy_(spec_info.retrive_next_sibling)
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                chunk_indices_with_16=chunk_indices_with_16,
                chunk_indices_with_64=chunk_indices_with_64,
                chunk_indices_with_o=chunk_indices_with_o,
                chunk_offsets_with_64=chunk_offsets_with_64,
            )

    def _replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        num_padding = torch.count_nonzero(
            seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
        )
        chunk_indices_with_16 = None
        chunk_indices_with_64 = None
        chunk_offsets_with_64 = None
        chunk_indices_with_o = None
        # Make sure forward metadata is correctly handled for padding reqs
        req_pool_indices[bs - num_padding :] = 0
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        mamba_indices[bs - num_padding :] = -1
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        if forward_mode.is_decode_or_idle():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].copy_(
                    bs - num_padding
                )
        elif forward_mode.is_target_verify():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                )
                chunk_indices_with_16 = self.cached_cuda_graph_chunk_indices_with_16[
                    bs - 1
                ]
                chunk_indices_with_64 = self.cached_cuda_graph_chunk_indices_with_64[
                    bs - 1
                ]
                chunk_indices_with_o = self.cached_cuda_graph_chunk_indices_with_o[
                    bs - 1
                ]
                chunk_offsets_with_64 = self.cached_cuda_graph_chunk_offsets_with_64[
                    bs - 1
                ]

            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].copy_(
                    (bs - num_padding) * spec_info.draft_token_num
                )
                chunk_indices_with_16 = self.cached_cuda_graph_chunk_indices_with_16[
                    bs - 1
                ]
                chunk_indices_with_64 = self.cached_cuda_graph_chunk_indices_with_64[
                    bs - 1
                ]
                chunk_indices_with_o = self.cached_cuda_graph_chunk_indices_with_o[
                    bs - 1
                ]
                chunk_offsets_with_64 = self.cached_cuda_graph_chunk_offsets_with_64[
                    bs - 1
                ]
                chunk_indices_with_16[bs - num_padding :].fill_(-1)
                chunk_indices_with_64[bs - num_padding :].fill_(-1)
                chunk_indices_with_o[bs - num_padding :].fill_(-1)
                chunk_offsets_with_64[bs - num_padding :].fill_(-1)
                # print(f"{bs=} {num_padding=} {chunk_indices_with_16=} {chunk_indices_with_64=} {chunk_indices_with_o=}")
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and spec_info.topk > 1:
            bs_without_pad = spec_info.retrive_next_token.shape[0]
            self.retrieve_next_token_list[bs - 1][:bs_without_pad].copy_(
                spec_info.retrive_next_token
            )
            self.retrieve_next_sibling_list[bs - 1][:bs_without_pad].copy_(
                spec_info.retrive_next_sibling
            )
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                chunk_indices_with_16=chunk_indices_with_16,
                chunk_indices_with_64=chunk_indices_with_64,
                chunk_offsets_with_64=chunk_offsets_with_64,
                chunk_indices_with_o=chunk_indices_with_o,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1  # Mamba attn does not use seq lens to index kv cache

    def _track_mamba_state_decode(
        self,
        forward_batch: ForwardBatch,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ):
        """
        Track and copy Mamba conv/SSM states during decode for prefix caching.

        During decode, each token update modifies conv_states and ssm_states in-place
        at positions indexed by cache_indices (the working slots). For prefix caching,
        we need to copy these updated states to persistent cache slots (mamba_track_indices)
        so they can be prefix cached.

        This delegates to `track_mamba_states_if_needed`, which performs:
            conv_states[mamba_track_indices[i]] = conv_states[cache_indices[i]]
            ssm_states[mamba_track_indices[i]] = ssm_states[cache_indices[i]]
        for all requests where mamba_track_mask[i] is True.
        """
        if forward_batch.mamba_track_mask is not None:
            track_mamba_states_if_needed(
                conv_states,
                ssm_states,
                cache_indices,
                forward_batch.mamba_track_mask,
                forward_batch.mamba_track_indices,
                forward_batch.batch_size,
            )

    def _track_mamba_state_extend(
        self,
        forward_batch: ForwardBatch,
        h: torch.Tensor,
        ssm_states: torch.Tensor,
        forward_metadata: ForwardMetadata,
    ):
        """
        Track and copy SSM states during extend for prefix caching.

        After the FLA chunked prefill kernel runs, we need to save the SSM recurrent
        state at the last chunk boundary so it can be reused for prefix caching.
        The source of the state depends on whether the sequence length is aligned
        to FLA_CHUNK_SIZE. See `_init_track_ssm_indices` for more details on how
        the source and destination indices are computed.

        Note: Conv state tracking for extend is handled separately via gather operations
        using indices computed by `_init_track_conv_indices`.
        """
        if (
            forward_batch.mamba_track_mask is not None
            and forward_batch.mamba_track_mask.any()
        ):
            h = h.squeeze(0)

            if forward_metadata.track_ssm_h_src.numel() > 0:
                ssm_states[forward_metadata.track_ssm_h_dst] = h[
                    forward_metadata.track_ssm_h_src
                ].to(ssm_states.dtype, copy=False)
            if forward_metadata.track_ssm_final_src.numel() > 0:
                ssm_states[forward_metadata.track_ssm_final_dst] = ssm_states[
                    forward_metadata.track_ssm_final_src
                ]


class KimiLinearAttnBackend(MambaAttnBackendBase):
    """Attention backend using Mamba kernel."""

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        q_proj_states = kwargs["q_proj_states"]
        k_proj_states = kwargs["k_proj_states"]
        v_proj_states = kwargs["v_proj_states"]
        q_conv_weights = kwargs["q_conv_weights"]
        k_conv_weights = kwargs["k_conv_weights"]
        v_conv_weights = kwargs["v_conv_weights"]

        q_conv_bias = kwargs["q_conv_bias"]
        k_conv_bias = kwargs["k_conv_bias"]
        v_conv_bias = kwargs["v_conv_bias"]

        head_dim = kwargs["head_dim"]
        layer_id = kwargs["layer_id"]
        beta = kwargs["beta"]
        g = kwargs["gate"]

        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        q_conv_state, k_conv_state, v_conv_state = layer_cache.conv
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        q_conv_state = q_conv_state.transpose(-1, -2)
        k_conv_state = k_conv_state.transpose(-1, -2)
        v_conv_state = v_conv_state.transpose(-1, -2)

        q = causal_conv1d_update(
            q_proj_states,
            q_conv_state,
            q_conv_weights,
            q_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )
        k = causal_conv1d_update(
            k_proj_states,
            k_conv_state,
            k_conv_weights,
            k_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )
        v = causal_conv1d_update(
            v_proj_states,
            v_conv_state,
            v_conv_weights,
            v_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )

        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=head_dim), (q, k, v)
        )

        core_attn_out = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=g,
            b=beta,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
        )

        return core_attn_out

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
            causal_conv1d_fn,
        )

        q_proj_states = kwargs["q_proj_states"]
        k_proj_states = kwargs["k_proj_states"]
        v_proj_states = kwargs["v_proj_states"]
        q_conv_weights = kwargs["q_conv_weights"]
        k_conv_weights = kwargs["k_conv_weights"]
        v_conv_weights = kwargs["v_conv_weights"]

        q_conv_bias = kwargs["q_conv_bias"]
        k_conv_bias = kwargs["k_conv_bias"]
        v_conv_bias = kwargs["v_conv_bias"]

        head_dim = kwargs["head_dim"]
        layer_id = kwargs["layer_id"]
        beta = kwargs["beta"]
        g = kwargs["gate"]

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_state_q, conv_state_k, conv_state_v = mamba_cache_params.conv
        # deal with strides
        conv_state_q = conv_state_q.transpose(-1, -2)
        conv_state_k = conv_state_k.transpose(-1, -2)
        conv_state_v = conv_state_v.transpose(-1, -2)

        ssm_states = mamba_cache_params.temporal

        has_initial_state = forward_batch.extend_prefix_lens > 0

        q_proj_states = q_proj_states.transpose(0, 1)
        k_proj_states = k_proj_states.transpose(0, 1)
        v_proj_states = v_proj_states.transpose(0, 1)

        q = causal_conv1d_fn(
            q_proj_states,
            q_conv_weights,
            q_conv_bias,
            activation="silu",
            conv_states=conv_state_q,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        k = causal_conv1d_fn(
            k_proj_states,
            k_conv_weights,
            k_conv_bias,
            activation="silu",
            conv_states=conv_state_k,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        v = causal_conv1d_fn(
            v_proj_states,
            v_conv_weights,
            v_conv_bias,
            activation="silu",
            conv_states=conv_state_v,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=head_dim), (q, k, v)
        )

        core_attn_out = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
        )

        return core_attn_out


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend using Mamba kernel."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        assert (
            self.conv_states_shape[-1] < FLA_CHUNK_SIZE
        ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"

        use_cutedsl = Envs.SGLANG_USE_CUTEDSL_GDN_DECODE.get()
        rank0_log(f"CuTe DSL GDN decode enabled: {use_cutedsl}")
        self._kernel_func = (
            cutedsl_fused_sigmoid_gating_delta_rule_update
            if use_cutedsl
            else fused_sigmoid_gating_delta_rule_update
        )

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,  # Unused, for compatibility with HybridLinearAttnBackend
    ):
        conv_weights = layer.conv_weights
        bias = layer.bias
        activation = layer.activation
        key_dim = layer.key_dim
        value_dim = layer.value_dim
        attn_tp_size = layer.attention_tp_size
        head_k_dim = layer.head_k_dim
        head_v_dim = layer.head_v_dim

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            conv_weights,
            bias,
            activation,
            conv_state_indices=cache_indices,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [
                key_dim // attn_tp_size,
                key_dim // attn_tp_size,
                value_dim // attn_tp_size,
            ],
            dim=-1,
        )
        # Reshape from [l, h*d] to [1, l, h, d]
        seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        query = query.view(1, seq_len, num_heads, head_k_dim)
        key = key.view(1, seq_len, num_heads, head_k_dim)
        value = value.view(1, seq_len, value.shape[1] // head_v_dim, head_v_dim)

        core_attn_out = self._kernel_func(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

        # For DVR, tracking decode is unnecessary
        # self._track_mamba_state_decode(
        #     forward_batch, conv_states, ssm_states, cache_indices
        # )

        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,  # Unused, for compatibility with HybridLinearAttnBackend
    ):
        seq_len = mixed_qkv.shape[0]

        conv_weights = layer.conv_weights
        bias = layer.bias
        activation = layer.activation
        key_dim = layer.key_dim
        value_dim = layer.value_dim
        attn_tp_size = layer.attention_tp_size
        head_k_dim = layer.head_k_dim
        head_v_dim = layer.head_v_dim

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            has_initial_states = torch.ones(
                seq_len // forward_batch.spec_info.draft_token_num,
                dtype=torch.bool,
                device=forward_batch.input_ids.device,
            )
            intermediate_state_indices = torch.arange(
                cache_indices.shape[0], dtype=torch.int32, device=cache_indices.device
            )
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                conv_weights,
                bias,
                activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if (
                forward_batch.mamba_track_mask is not None
                and forward_batch.mamba_track_mask.any()
            ):
                conv_dst = forward_batch.mamba_track_indices
                # Gather all slices at once: [:, track_conv_indices] -> [d, num_masked, slice_len]
                # track_conv_indices is already filtered and clamped in _init_track_conv_indices
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                # Apply mask and assign to destinations
                mask_indices = forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
                conv_states[conv_dst[mask_indices]] = mixed_qkv_to_track

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                conv_weights,
                bias,
                activation=activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        key_split_dim = key_dim // attn_tp_size
        value_split_dim = value_dim // attn_tp_size

        query, key, value = torch.split(
            mixed_qkv,
            [key_split_dim, key_split_dim, value_split_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        num_value_heads = value.shape[1] // head_v_dim

        query = query.view(1, actual_seq_len, num_heads, head_k_dim)
        key = key.view(1, actual_seq_len, num_heads, head_k_dim)
        value = value.view(1, actual_seq_len, num_value_heads, head_v_dim)

        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)

        if get_global_server_args().speculative_num_draft_tokens is not None:
            intermediate_q_state_cache = mamba_cache_params.intermediate_q_state_cache
            intermediate_k_state_cache = mamba_cache_params.intermediate_k_state_cache
            intermediate_v_state_cache = mamba_cache_params.intermediate_v_state_cache
            intermediate_beta_state_cache = (
                mamba_cache_params.intermediate_beta_state_cache
            )
            intermediate_g_state_cache = mamba_cache_params.intermediate_g_state_cache
            intermediate_kvug_pos = mamba_cache_params.intermediate_kvug_pos
        if is_target_verify:
            prefix_len = intermediate_kvug_pos[cache_indices]
            extend_len = forward_batch.spec_info.draft_token_num

            row = cache_indices.unsqueeze(1).expand(-1, extend_len)
            col = torch.arange(
                extend_len, device=intermediate_q_state_cache.device
            ).unsqueeze(0) + prefix_len.unsqueeze(1)

            intermediate_q_state_cache[row, col] = query.view(
                -1, extend_len, num_heads, head_k_dim
            )
            intermediate_k_state_cache[row, col] = key.view(
                -1, extend_len, num_heads, head_k_dim
            )
            intermediate_v_state_cache[row, col] = value.view(
                -1, extend_len, num_value_heads, head_v_dim
            )
            intermediate_beta_state_cache[row, col] = beta.view(
                -1, extend_len, num_value_heads
            )
            intermediate_g_state_cache[row, col] = g.view(
                -1, extend_len, num_value_heads
            )

            core_attn_out = chunk_gated_delta_rule(
                q=intermediate_q_state_cache[cache_indices].view(
                    1, -1, num_heads, head_k_dim
                ),
                k=intermediate_k_state_cache[cache_indices].view(
                    1, -1, num_heads, head_k_dim
                ),
                v=intermediate_v_state_cache[cache_indices].view(
                    1, -1, num_value_heads, head_v_dim
                ),
                g=intermediate_g_state_cache[cache_indices].view(
                    1, -1, num_value_heads
                ),
                beta=intermediate_beta_state_cache[cache_indices].view(
                    1, -1, num_value_heads
                ),
                initial_state=ssm_states,
                initial_state_indices=forward_batch.mamba_track_indices[:batch_size],
                cu_seqlens=query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
                forward_metadata=forward_metadata,
                inplace_update=False,
            )[0]
            core_attn_out = core_attn_out.view(
                batch_size,
                extend_len + FLA_CHUNK_SIZE,
                num_value_heads,
                head_v_dim,
            )
            row = (
                torch.arange(batch_size, device=core_attn_out.device)
                .unsqueeze(1)
                .expand(-1, extend_len)
            )
            core_attn_out = (
                core_attn_out[row, col]
                .view(1, -1, num_value_heads, head_v_dim)
                .contiguous()
            )
        else:
            # Only cuda env uses fuse ssm_states update
            recurrent_state = ssm_states
            recurrent_state_indices_args = {"initial_state_indices": cache_indices}
            if is_npu():
                recurrent_state = ssm_states[cache_indices]
                recurrent_state_indices_args = {}
            core_attn_out, last_recurrent_state, h = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                cu_seqlens=query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
                **recurrent_state_indices_args,
            )
            if is_npu():
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

            if get_global_server_args().speculative_num_draft_tokens is not None:
                intermediate_kvug_pos = mamba_cache_params.intermediate_kvug_pos
                for i, extend_len in enumerate(forward_batch.extend_seq_lens):
                    extend_len = extend_len % FLA_CHUNK_SIZE
                    start = query_start_loc[i + 1] - extend_len
                    end = query_start_loc[i + 1]
                    if extend_len > 0:
                        intermediate_q_state_cache[cache_indices[i], :extend_len] = (
                            query[:, start:end]
                        )
                        intermediate_k_state_cache[cache_indices[i], :extend_len] = key[
                            :, start:end
                        ]
                        intermediate_v_state_cache[cache_indices[i], :extend_len] = (
                            value[:, start:end]
                        )
                        intermediate_beta_state_cache[cache_indices[i], :extend_len] = (
                            beta[:, start:end]
                        )
                        intermediate_g_state_cache[cache_indices[i], :extend_len] = g[
                            :, start:end
                        ]
                        intermediate_kvug_pos[cache_indices[i]] = extend_len

        return core_attn_out


class Mamba2AttnBackend(MambaAttnBackendBase):
    """Attention backend wrapper for Mamba2Mixer kernels."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        config = model_runner.mamba2_config
        assert config is not None
        self.mamba_chunk_size = config.mamba_chunk_size

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        metadata = self._forward_metadata(forward_batch)
        self.forward_metadata = Mamba2Metadata.prepare_mixed(
            metadata,
            self.mamba_chunk_size,
            forward_batch,
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        metadata = self._capture_metadata(bs, req_pool_indices, forward_mode, spec_info)
        draft_token_num = spec_info.draft_token_num if spec_info is not None else 1
        self.forward_metadata = Mamba2Metadata.prepare_decode(
            metadata,
            seq_lens,
            is_target_verify=forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        metadata = self._replay_metadata(
            bs, req_pool_indices, forward_mode, spec_info, seq_lens_cpu
        )
        draft_token_num = spec_info.draft_token_num if spec_info is not None else 1
        self.forward_metadata = Mamba2Metadata.prepare_decode(
            metadata,
            seq_lens,
            is_target_verify=forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
        )

    def forward(
        self,
        mixer: MambaMixer2,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layer_id: int,
        mup_vector: Optional[torch.Tensor] = None,
        use_triton_causal_conv: bool = False,
    ):
        assert isinstance(self.forward_metadata, Mamba2Metadata)
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        return mixer.forward(
            hidden_states=hidden_states,
            output=output,
            layer_cache=layer_cache,
            metadata=self.forward_metadata,
            mup_vector=mup_vector,
            use_triton_causal_conv=use_triton_causal_conv,
        )

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )


class HybridLinearAttnBackend(AttentionBackend):
    """Manages a full and linear attention backend"""

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: MambaAttnBackendBase,
        full_attn_layers: list[int],
    ):
        self.full_attn_layers = full_attn_layers
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend
        self.attn_backend_list = [full_attn_backend, linear_attn_backend]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.full_attn_backend.get_cuda_graph_seq_len_fill_value()

    def forward_decode(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        mixed_qkv: Optional[torch.Tensor] = None,  # For GDN linear attention
        a: Optional[torch.Tensor] = None,  # For GDN linear attention
        b: Optional[torch.Tensor] = None,  # For GDN linear attention
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.full_attn_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        # Linear attention backend
        return self.linear_attn_backend.forward_decode(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )

    def forward_extend(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        mixed_qkv: Optional[torch.Tensor] = None,  # For GDN linear attention
        a: Optional[torch.Tensor] = None,  # For GDN linear attention
        b: Optional[torch.Tensor] = None,  # For GDN linear attention
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.full_attn_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        # Linear attention backend
        return self.linear_attn_backend.forward_extend(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )

    def forward(
        self,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        layer: RadixAttention = None,
        forward_batch: ForwardBatch = None,
        save_kv_cache: bool = True,
        mixed_qkv: Optional[torch.Tensor] = None,  # For GDN linear attention
        a: Optional[torch.Tensor] = None,  # For GDN linear attention
        b: Optional[torch.Tensor] = None,  # For GDN linear attention
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        is_linear_attn = layer_id not in self.full_attn_layers

        if forward_batch.forward_mode.is_idle():
            if is_linear_attn:
                return mixed_qkv.new_empty(
                    mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim
                )
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                layer,
                forward_batch,
                save_kv_cache,
                q,
                k,
                v,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )
        else:
            return self.forward_extend(
                layer,
                forward_batch,
                save_kv_cache,
                q,
                k,
                v,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )

    def update_mamba_state_after_mtp_verify(
        self,
        accepted_steps: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        model,
    ):
        request_number = accepted_steps.shape[0]

        state_indices_tensor = (
            self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                :request_number
            ]
        )
        intermediate_state_indices = torch.arange(
            request_number, dtype=torch.int32, device=state_indices_tensor.device
        )

        mamba_caches = (
            self.linear_attn_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        )

        conv_states = mamba_caches.conv[0]
        ssm_states = mamba_caches.temporal

        valid_mask = accepted_steps >= 0
        dst_state_indices = state_indices_tensor.to(torch.int64)  # [N]
        # [bs, accept len, dim]
        intermediate_q_state_cache = mamba_caches.intermediate_q_state_cache
        intermediate_k_state_cache = mamba_caches.intermediate_k_state_cache
        intermediate_v_state_cache = mamba_caches.intermediate_v_state_cache
        intermediate_beta_state_cache = mamba_caches.intermediate_beta_state_cache
        intermediate_g_state_cache = mamba_caches.intermediate_g_state_cache
        intermediate_kvug_pos = mamba_caches.intermediate_kvug_pos
        intermediate_conv_window_cache = mamba_caches.intermediate_conv_window[0]

        draft_tokens_length = get_global_server_args().speculative_num_draft_tokens

        layer = intermediate_k_state_cache.shape[0]
        query_start_loc = torch.arange(
            0,
            (draft_tokens_length + FLA_CHUNK_SIZE) * request_number + 1,
            draft_tokens_length + FLA_CHUNK_SIZE,
            device=dst_state_indices.device,
        )
        # 1. update kvug positions and clean buffers larger than accepted steps
        intermediate_kvug_pos[:, dst_state_indices] += accepted_steps + 1
        gdn_kuwg_start_pos = intermediate_kvug_pos[0, dst_state_indices]

        buffer_list = [
            intermediate_q_state_cache,
            intermediate_k_state_cache,
            intermediate_v_state_cache,
            intermediate_beta_state_cache,
            intermediate_g_state_cache,
        ]

        clear_buffers(
            buffer_list,
            intermediate_kvug_pos,
            dst_state_indices,
        )

        Hg, H, K, V = (
            intermediate_k_state_cache.shape[3],
            intermediate_v_state_cache.shape[3],
            intermediate_k_state_cache.shape[4],
            intermediate_v_state_cache.shape[4],
        )

        # 2. update conv states for all requests using intermediate_conv_window_cache
        fused_mamba_state_scatter_with_mask(
            conv_states,
            intermediate_conv_window_cache,
            state_indices_tensor,
            accepted_steps,
        )

        # 3. Compact mamba states if needed (position >= FLA_CHUNK_SIZE)
        kuwg_mask = (gdn_kuwg_start_pos >= FLA_CHUNK_SIZE).unsqueeze(1)
        kuwg_mask = kuwg_mask & (
            (
                torch.arange(
                    FLA_CHUNK_SIZE + draft_tokens_length, device=accepted_steps.device
                )
                < FLA_CHUNK_SIZE
            ).unsqueeze(0)
        )

        kuwg_mask.unsqueeze_(-1)
        q = intermediate_q_state_cache[:, dst_state_indices, :]
        k = intermediate_k_state_cache[:, dst_state_indices, :] * kuwg_mask.unsqueeze(
            0
        ).unsqueeze(-1)
        v = intermediate_v_state_cache[:, dst_state_indices, :] * kuwg_mask.unsqueeze(
            0
        ).unsqueeze(-1)
        g = intermediate_g_state_cache[:, dst_state_indices, :] * kuwg_mask.unsqueeze(0)
        beta = intermediate_beta_state_cache[
            :, dst_state_indices, :
        ] * kuwg_mask.unsqueeze(0)

        q_fused = q.view(1, -1, Hg, K)
        k_fused = k.view(1, -1, Hg, K)
        v_fused = v.view(1, -1, H, V)
        g_fused = g.view(1, -1, H)
        beta_fused = beta.view(1, -1, H)
        ssm_states_fused = ssm_states.view(
            -1, ssm_states.shape[2], ssm_states.shape[3], ssm_states.shape[4]
        )
        query_len_per_layer = query_start_loc[-1]
        cu_seqlens_fused = torch.cat(
            [query_start_loc[1:] + l * query_len_per_layer for l in range(layer)]
        )
        cu_seqlens_fused = torch.cat(
            [torch.tensor([0], device=cu_seqlens_fused.device), cu_seqlens_fused]
        )
        mamba_track_indices_fused = torch.cat(
            [mamba_track_indices + l * ssm_states.shape[1] for l in range(layer)]
        )

        chunk_gated_delta_rule(
            q=q_fused,
            k=k_fused,
            v=v_fused,
            g=g_fused,
            beta=beta_fused,
            initial_state=ssm_states_fused,
            initial_state_indices=mamba_track_indices_fused,
            cu_seqlens=cu_seqlens_fused,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )

        # 4. update tracked conv states
        fused_mamba_state_scatter_with_mask(
            conv_states,
            intermediate_conv_window_cache,
            mamba_track_indices,
            mamba_steps_to_track,
        )

        # 5. update buffer by shifting if needed (position >= FLA_CHUNK_SIZE)
        shift_buffers(
            buffer_list,
            intermediate_kvug_pos,
            dst_state_indices,
            draft_tokens_length,
            FLA_CHUNK_SIZE,
        )

        # 6. Update ssm states for decode stage by recompute
        # build mask: accept steps >= 0 and position < gdn_kuwg_start_pos
        if get_global_server_args().speculative_algorithm != "DECODE_VERIFY_ROLLBACK":
            return

        kuwg_mask = torch.arange(
            draft_tokens_length + FLA_CHUNK_SIZE, device=dst_state_indices.device
        ).unsqueeze(0) < gdn_kuwg_start_pos.unsqueeze(1)
        kuwg_mask.unsqueeze_(-1)

        k = intermediate_k_state_cache[:, dst_state_indices, :] * kuwg_mask.unsqueeze(
            0
        ).unsqueeze(-1)
        v = intermediate_v_state_cache[:, dst_state_indices, :] * kuwg_mask.unsqueeze(
            0
        ).unsqueeze(-1)
        g = intermediate_g_state_cache[:, dst_state_indices, :] * kuwg_mask.unsqueeze(0)
        beta = intermediate_beta_state_cache[
            :, dst_state_indices, :
        ] * kuwg_mask.unsqueeze(0)

        ssm_states[:, dst_state_indices, :] = ssm_states[:, mamba_track_indices]

        q_fused = q.view(1, -1, Hg, K)
        k_fused = k.view(1, -1, Hg, K)
        v_fused = v.view(1, -1, H, V)
        g_fused = g.view(1, -1, H)
        beta_fused = beta.view(1, -1, H)
        dst_state_indices_fused = torch.cat(
            [dst_state_indices + l * ssm_states.shape[1] for l in range(layer)]
        )
        chunk_gated_delta_rule(
            q=q_fused,
            k=k_fused,
            v=v_fused,
            g=g_fused,
            beta=beta_fused,
            initial_state=ssm_states_fused,
            initial_state_indices=dst_state_indices_fused,
            cu_seqlens=cu_seqlens_fused,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )


###############################################################################################
# mixin for update_mamba_state_after_mtp_verify to clear and shift buffers using triton kernels
###############################################################################################
@triton.jit
def _compute_valid_and_update_pos_kernel(
    dst_state_indices_ptr,  # [num_requests]
    kvug_pos_ptr,  # [num_layers, num_states]
    valid_mask_ptr,  # [num_requests] output
    kvug_pos_stride0,  # stride for layer dim
    kvug_pos_stride1,  # stride for state dim
    num_requests,
    FLA_CHUNK_SIZE,
):
    """
    Grid: (num_requests, num_layers)
    - Each thread checks one (request, layer) pair.
    - valid_mask is written only by layer 0 to avoid race conditions.
    - kvug_pos is updated in-place for each layer independently.
    """
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1)

    if pid_req >= num_requests:
        return

    dst_state_idx = tl.load(dst_state_indices_ptr + pid_req)

    pos_ptr = (
        kvug_pos_ptr + pid_layer * kvug_pos_stride0 + dst_state_idx * kvug_pos_stride1
    )
    kvug_pos_val = tl.load(pos_ptr)
    need_shift = kvug_pos_val >= FLA_CHUNK_SIZE

    # Only layer 0 writes valid_mask to avoid race condition.
    # This assumes all layers share the same need_shift decision,
    # which holds when kvug_pos is updated uniformly across layers.
    if pid_layer == 0:
        tl.store(valid_mask_ptr + pid_req, need_shift.to(tl.int32))

    if need_shift:
        tl.store(pos_ptr, kvug_pos_val - FLA_CHUNK_SIZE)


@triton.jit
def _shift_5d_buffer_kernel(
    buffer_ptr,
    valid_mask_ptr,  # [num_requests]
    dst_state_indices_ptr,  # [num_requests]
    # strides for [num_layers, num_states, seq_len, heads, dim]
    stride_layer,
    stride_state,
    stride_seq,
    stride_head,
    stride_dim,
    # shape
    num_layers,
    seq_len,
    num_heads,
    head_dim,
    # shift params
    draft_tokens_length,
    FLA_CHUNK_SIZE,
    # block sizes
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """
    Grid: (num_requests, num_layers, cdiv(seq_len, BLOCK_SEQ))

    For each (request, layer, seq_block):
      - Iterate over heads and dim with tl.arange blocks.
      - Shift seq_len dimension: src[FLA_CHUNK_SIZE:FLA_CHUNK_SIZE+draft] -> dst[0:draft]
      - Zero out dst[draft:]
    """
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1)
    pid_seq = tl.program_id(2)

    # Check if this request needs shifting (GPU-side, no sync)
    valid = tl.load(valid_mask_ptr + pid_req) != 0
    if not valid:
        return

    dst_state_idx = tl.load(dst_state_indices_ptr + pid_req)

    # Base pointer for this (layer, state)
    base = pid_layer * stride_layer + dst_state_idx * stride_state

    # Seq offsets for this block
    seq_offsets = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)

    # Head and dim offsets
    head_offsets = tl.arange(0, BLOCK_HEAD)
    dim_offsets = tl.arange(0, BLOCK_DIM)

    # 3D offset grid: [BLOCK_SEQ, BLOCK_HEAD, BLOCK_DIM]
    # Shape broadcasting: seq[:, None, None], head[None, :, None], dim[None, None, :]
    seq_off = seq_offsets[:, None, None]  # [BLOCK_SEQ,  1,          1        ]
    head_off = head_offsets[None, :, None]  # [1,          BLOCK_HEAD, 1        ]
    dim_off = dim_offsets[None, None, :]  # [1,          1,          BLOCK_DIM]

    # ── Copy: src[FLA_CHUNK_SIZE : FLA_CHUNK_SIZE + draft] -> dst[0 : draft] ──
    src_seq = seq_off + FLA_CHUNK_SIZE
    copy_mask = (
        (seq_offsets[:, None, None] < draft_tokens_length)
        & (src_seq < seq_len)
        & (head_off < num_heads)
        & (dim_off < head_dim)
    )

    src_offsets = (
        base + src_seq * stride_seq + head_off * stride_head + dim_off * stride_dim
    )
    data = tl.load(buffer_ptr + src_offsets, mask=copy_mask, other=0.0)

    dst_offsets = (
        base + seq_off * stride_seq + head_off * stride_head + dim_off * stride_dim
    )
    tl.store(buffer_ptr + dst_offsets, data, mask=copy_mask)

    # ── Zero out: dst[draft_tokens_length : seq_len] ──
    zero_seq = seq_off + draft_tokens_length
    zero_mask = (zero_seq < seq_len) & (head_off < num_heads) & (dim_off < head_dim)
    zero_offsets = (
        base + zero_seq * stride_seq + head_off * stride_head + dim_off * stride_dim
    )
    tl.store(
        buffer_ptr + zero_offsets,
        tl.zeros([BLOCK_SEQ, BLOCK_HEAD, BLOCK_DIM], dtype=buffer_ptr.dtype.element_ty),
        mask=zero_mask,
    )


@triton.jit
def _shift_4d_buffer_kernel(
    buffer_ptr,
    valid_mask_ptr,  # [num_requests]
    dst_state_indices_ptr,  # [num_requests]
    # strides for [num_layers, num_states, seq_len, heads]
    stride_layer,
    stride_state,
    stride_seq,
    stride_head,
    # shape
    num_layers,
    seq_len,
    num_heads,
    # shift params
    draft_tokens_length,
    FLA_CHUNK_SIZE,
    # block sizes
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    """
    Grid: (num_requests, num_layers, cdiv(seq_len, BLOCK_SEQ))

    Same as _shift_5d_buffer_kernel but for 4D buffers
    (intermediate_g and intermediate_beta which have no head_dim).
    """
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1)
    pid_seq = tl.program_id(2)

    valid = tl.load(valid_mask_ptr + pid_req) != 0
    if not valid:
        return

    dst_state_idx = tl.load(dst_state_indices_ptr + pid_req)

    base = pid_layer * stride_layer + dst_state_idx * stride_state

    seq_offsets = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    head_offsets = tl.arange(0, BLOCK_HEAD)

    seq_off = seq_offsets[:, None]  # [BLOCK_SEQ,  1         ]
    head_off = head_offsets[None, :]  # [1,          BLOCK_HEAD]

    # ── Copy ──
    src_seq = seq_off + FLA_CHUNK_SIZE
    copy_mask = (
        (seq_offsets[:, None] < draft_tokens_length)
        & (src_seq < seq_len)
        & (head_off < num_heads)
    )

    src_offsets = base + src_seq * stride_seq + head_off * stride_head
    data = tl.load(buffer_ptr + src_offsets, mask=copy_mask, other=0.0)

    dst_offsets = base + seq_off * stride_seq + head_off * stride_head
    tl.store(buffer_ptr + dst_offsets, data, mask=copy_mask)

    # ── Zero out ──
    zero_seq = seq_off + draft_tokens_length
    zero_mask = (zero_seq < seq_len) & (head_off < num_heads)

    zero_offsets = base + zero_seq * stride_seq + head_off * stride_head
    tl.store(
        buffer_ptr + zero_offsets,
        tl.zeros([BLOCK_SEQ, BLOCK_HEAD], dtype=buffer_ptr.dtype.element_ty),
        mask=zero_mask,
    )


def shift_buffers(
    buffer_list: list[torch.Tensor],
    # position tracker: [num_layers, num_states]
    intermediate_kvug_pos: torch.Tensor,
    # request info
    dst_state_indices: torch.Tensor,  # [num_requests]
    draft_tokens_length: int,
    FLA_CHUNK_SIZE: int,
    # optional pre-allocated cache
    _valid_mask_cache: torch.Tensor = None,
):
    (
        intermediate_q_state_cache,
        intermediate_k_state_cache,
        intermediate_v_state_cache,
        intermediate_g_state_cache,
        intermediate_beta_state_cache,
    ) = buffer_list
    num_requests = dst_state_indices.shape[0]
    num_layers = intermediate_kvug_pos.shape[0]
    device = dst_state_indices.device

    if num_requests == 0:
        return

    # Reuse pre-allocated valid_mask if provided
    if _valid_mask_cache is not None and _valid_mask_cache.shape[0] >= num_requests:
        valid_mask = _valid_mask_cache[:num_requests]
    else:
        valid_mask = torch.empty(num_requests, dtype=torch.int32, device=device)

    # ── Step 1: Compute valid_mask and update kvug_pos in-place ──
    _compute_valid_and_update_pos_kernel[(num_requests, num_layers)](
        dst_state_indices,
        intermediate_kvug_pos,
        valid_mask,
        kvug_pos_stride0=intermediate_kvug_pos.stride(0),
        kvug_pos_stride1=intermediate_kvug_pos.stride(1),
        num_requests=num_requests,
        FLA_CHUNK_SIZE=FLA_CHUNK_SIZE,
    )

    # ── Step 2: Shift 5D buffers (q, k, v) ──
    for buf in (
        intermediate_q_state_cache,
        intermediate_k_state_cache,
        intermediate_v_state_cache,
    ):
        num_layers_, num_states_, seq_len, num_heads, head_dim = buf.shape

        BLOCK_SEQ = 1  # one seq position per thread block along seq axis
        BLOCK_HEAD = triton.next_power_of_2(num_heads)
        BLOCK_DIM = triton.next_power_of_2(head_dim)

        grid = (
            num_requests,
            num_layers_,
            triton.cdiv(draft_tokens_length + (seq_len - FLA_CHUNK_SIZE), BLOCK_SEQ),
        )

        _shift_5d_buffer_kernel[grid](
            buf,
            valid_mask,
            dst_state_indices,
            buf.stride(0),  # stride_layer
            buf.stride(1),  # stride_state
            buf.stride(2),  # stride_seq
            buf.stride(3),  # stride_head
            buf.stride(4),  # stride_dim
            num_layers_,
            seq_len,
            num_heads,
            head_dim,
            draft_tokens_length,
            FLA_CHUNK_SIZE,
            BLOCK_SEQ=BLOCK_SEQ,
            BLOCK_HEAD=BLOCK_HEAD,
            BLOCK_DIM=BLOCK_DIM,
        )

    # ── Step 3: Shift 4D buffers (g, beta) ──
    for buf in (
        intermediate_g_state_cache,
        intermediate_beta_state_cache,
    ):
        num_layers_, num_states_, seq_len, num_heads = buf.shape

        BLOCK_SEQ = 1
        BLOCK_HEAD = triton.next_power_of_2(num_heads)

        grid = (
            num_requests,
            num_layers_,
            triton.cdiv(seq_len, BLOCK_SEQ),
        )

        _shift_4d_buffer_kernel[grid](
            buf,
            valid_mask,
            dst_state_indices,
            buf.stride(0),  # stride_layer
            buf.stride(1),  # stride_state
            buf.stride(2),  # stride_seq
            buf.stride(3),  # stride_head
            num_layers_,
            seq_len,
            num_heads,
            draft_tokens_length,
            FLA_CHUNK_SIZE,
            BLOCK_SEQ=BLOCK_SEQ,
            BLOCK_HEAD=BLOCK_HEAD,
        )


@triton.jit
def _clear_5d_buffer_kernel(
    buffer_ptr,
    dst_state_indices_ptr,  # [num_requests]
    start_pos_ptr,  # [num_requests] gdn_kuwg_start_pos
    stride_layer,
    stride_state,
    stride_seq,
    stride_head,
    stride_dim,
    seq_len,
    num_heads,
    head_dim,
    num_requests,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """
    Grid: (num_requests, num_layers, cdiv(seq_len, BLOCK_SEQ))

    Zero out buffer[:, dst_state_indices[i], start_pos[i]:, :, :]
    """
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1)
    pid_seq = tl.program_id(2)

    dst_state_idx = tl.load(dst_state_indices_ptr + pid_req)
    start_pos = tl.load(start_pos_ptr + pid_req)

    seq_offsets = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    head_offsets = tl.arange(0, BLOCK_HEAD)
    dim_offsets = tl.arange(0, BLOCK_DIM)

    seq_off = seq_offsets[:, None, None]
    head_off = head_offsets[None, :, None]
    dim_off = dim_offsets[None, None, :]

    # Only zero positions >= start_pos
    zero_mask = (
        (seq_off >= start_pos)
        & (seq_off < seq_len)
        & (head_off < num_heads)
        & (dim_off < head_dim)
    )

    base = pid_layer * stride_layer + dst_state_idx * stride_state

    offsets = (
        base + seq_off * stride_seq + head_off * stride_head + dim_off * stride_dim
    )

    tl.store(
        buffer_ptr + offsets,
        tl.zeros([BLOCK_SEQ, BLOCK_HEAD, BLOCK_DIM], dtype=buffer_ptr.dtype.element_ty),
        mask=zero_mask,
    )


@triton.jit
def _clear_4d_buffer_kernel(
    buffer_ptr,
    dst_state_indices_ptr,  # [num_requests]
    start_pos_ptr,  # [num_requests]
    stride_layer,
    stride_state,
    stride_seq,
    stride_head,
    seq_len,
    num_heads,
    num_requests,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    """
    Grid: (num_requests, num_layers, cdiv(seq_len, BLOCK_SEQ))

    Zero out buffer[:, dst_state_indices[i], start_pos[i]:, :]
    """
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1)
    pid_seq = tl.program_id(2)

    dst_state_idx = tl.load(dst_state_indices_ptr + pid_req)
    start_pos = tl.load(start_pos_ptr + pid_req)

    seq_offsets = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    head_offsets = tl.arange(0, BLOCK_HEAD)

    seq_off = seq_offsets[:, None]
    head_off = head_offsets[None, :]

    zero_mask = (seq_off >= start_pos) & (seq_off < seq_len) & (head_off < num_heads)

    base = pid_layer * stride_layer + dst_state_idx * stride_state

    offsets = base + seq_off * stride_seq + head_off * stride_head

    tl.store(
        buffer_ptr + offsets,
        tl.zeros([BLOCK_SEQ, BLOCK_HEAD], dtype=buffer_ptr.dtype.element_ty),
        mask=zero_mask,
    )


def clear_buffers(
    buffer_list: list[torch.Tensor],
    intermediate_kvug_pos: torch.Tensor,  # [num_layers, num_states]
    dst_state_indices: torch.Tensor,  # [num_requests]
):
    """
    Zero out buffer[:, dst_state_indices[i], start_pos[i]:, ...] for all buffers.
    No CPU/GPU sync.

    Equivalent to:
        gdn_kuwg_start_pos = intermediate_kvug_pos[0, dst_state_indices]
        for i in range(request_number):
            buffer[:, dst_state_indices[i], gdn_kuwg_start_pos[i]:] = 0
    """
    (
        intermediate_q_state_cache,
        intermediate_k_state_cache,
        intermediate_v_state_cache,
        intermediate_g_state_cache,
        intermediate_beta_state_cache,
    ) = buffer_list
    num_requests = dst_state_indices.shape[0]
    num_layers = intermediate_kvug_pos.shape[0]

    if num_requests == 0:
        return

    # Gather start positions on GPU, no .item() / .cpu() call
    # shape: [num_requests]
    start_pos = intermediate_kvug_pos[0, dst_state_indices].to(torch.int32)

    # ── 5D buffers: [layers, states, seq, heads, dim] ──
    for buf in (
        intermediate_q_state_cache,
        intermediate_k_state_cache,
        intermediate_v_state_cache,
    ):
        _, _, seq_len, num_heads, head_dim = buf.shape

        BLOCK_SEQ = 1
        BLOCK_HEAD = triton.next_power_of_2(num_heads)
        BLOCK_DIM = triton.next_power_of_2(head_dim)

        grid = (
            num_requests,
            num_layers,
            triton.cdiv(seq_len, BLOCK_SEQ),
        )

        _clear_5d_buffer_kernel[grid](
            buf,
            dst_state_indices,
            start_pos,
            buf.stride(0),
            buf.stride(1),
            buf.stride(2),
            buf.stride(3),
            buf.stride(4),
            seq_len,
            num_heads,
            head_dim,
            num_requests,
            BLOCK_SEQ=BLOCK_SEQ,
            BLOCK_HEAD=BLOCK_HEAD,
            BLOCK_DIM=BLOCK_DIM,
        )

    # ── 4D buffers: [layers, states, seq, heads] ──
    for buf in (
        intermediate_g_state_cache,
        intermediate_beta_state_cache,
    ):
        _, _, seq_len, num_heads = buf.shape

        BLOCK_SEQ = 1
        BLOCK_HEAD = triton.next_power_of_2(num_heads)

        grid = (
            num_requests,
            num_layers,
            triton.cdiv(seq_len, BLOCK_SEQ),
        )

        _clear_4d_buffer_kernel[grid](
            buf,
            dst_state_indices,
            start_pos,
            buf.stride(0),
            buf.stride(1),
            buf.stride(2),
            buf.stride(3),
            seq_len,
            num_heads,
            num_requests,
            BLOCK_SEQ=BLOCK_SEQ,
            BLOCK_HEAD=BLOCK_HEAD,
        )
