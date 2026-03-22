import contextlib
import logging
from typing import List, Optional, Tuple

import torch
from torch.nn import functional as F
from triton import next_power_of_2

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.eagle_worker import add_output_logprobs_for_spec_v1
from sglang.srt.speculative.spec_utils import (
    assign_draft_cache_locs,
    detect_nan,
    select_top_k_tokens,
)
from sglang.srt.utils import is_cuda
from sglang.srt.utils.common import (
    fast_sample,
    fast_topk,
    get_bool_env_var,
    get_int_env_var,
)

if is_cuda():
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob
logger = logging.getLogger(__name__)

USE_CHAIN_SPECULATIVE_SAMPLING = get_bool_env_var("USE_CHAIN_SPECULATIVE_SAMPLING")
USE_DECODE_TOPK_RENORM = get_bool_env_var("USE_DECODE_TOPK_RENORM", "false")
USE_DECODE_TOPP_RENORM = get_bool_env_var("USE_DECODE_TOPP_RENORM", "false")
VERIFY_CUDA_GRAPH_BS = get_int_env_var("VERIFY_CUDA_GRAPH_BS", 64)


class DecodeVerifyRollbackWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        assert (
            server_args.page_size == 1
        ), "DecodeVerifyRollbackWorker only supports page_size == 1"
        self.page_size = 1
        self.server_args = server_args

        self.max_batch_size = target_worker.max_running_requests
        self.device = server_args.device
        self.gpu_id = gpu_id
        assert server_args.speculative_eagle_topk == 1, "DVR always uses top-1 draft"
        self.topk = 1
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)
        self.model_config = self.model_runner.model_config

        self.init_attention_backend()
        self.init_cuda_graphs()

    def init_attention_backend(self):
        self.decode_attention_backend = self.model_runner.attn_backend
        self.target_verify_attention_backend = (
            self.model_runner._get_attention_backend()
        )

    def init_cuda_graphs(self):
        self.model_runner.enable_dvr_target_verify_cuda_graph = True
        capture_bs = list(range(1, VERIFY_CUDA_GRAPH_BS + 1))
        self.model_runner.server_args.cuda_graph_bs = (
            capture_bs  # [bs for bs in capture_bs if bs % get_attention_tp_size() == 0]
        )
        self.model_runner.attn_backend = self.target_verify_attention_backend
        if not self.model_runner.server_args.disable_cuda_graph:
            self.target_verify_cuda_graph_runner = CudaGraphRunner(self.model_runner)
        else:
            self.target_verify_cuda_graph_runner = None
        self.model_runner.attn_backend = self.decode_attention_backend

    @contextlib.contextmanager
    def target_verify_cuda_graph_context(self):
        """A context manager for running CUDA graph for target verify.

        Args:
            forward_batch: The forward batch to run.
        """
        origin_cuda_graph_runner = self.model_runner.graph_runner
        try:
            self.model_runner.graph_runner = self.target_verify_cuda_graph_runner
            self.model_runner.attn_backend = self.target_verify_attention_backend
            yield
        finally:
            self.model_runner.graph_runner = origin_cuda_graph_runner
            self.model_runner.attn_backend = self.decode_attention_backend

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # log_info_on_rank0(logger, "DVR: running target extend")
            logits_output, next_token_ids, _ = self.forward_target_extend(batch)
            # log_info_on_rank0(logger, "DVR: finished target extend")
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )
        else:
            with self._mamba_backup_conv_state(batch):
                spec_info, can_run_cuda_graph = self.draft(batch)

            logits_output, verify_output, _ = self.verify(batch, spec_info)
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
            )

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, int, Optional[torch.Tensor]]:
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
            next_token_ids: Next token ids generated.
        """
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.
        model_worker_batch = batch.get_model_worker_batch()
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        logits_output, next_token_ids = (
            batch_result.logits_output,
            batch_result.next_token_ids,
        )
        logits_output.next_token_logits = logits_output.next_token_logits.to(
            torch.float32
        )

        batch.spec_info = EagleDraftInput(
            verified_id=next_token_ids,
            num_tokens_per_batch=1,
            num_tokens_for_logprob_per_batch=1,
        )
        self.capture_for_decode(batch, logits_output, batch.spec_info)
        return (
            logits_output,
            next_token_ids,
            model_worker_batch.seq_lens_cpu,
        )

    def get_renorm_probs(
        self,
        batch: ForwardBatch,
        logits_output: LogitsProcessorOutput,
        use_topk_renorm: bool = True,
        use_topp_renorm: bool = True,
    ):
        sampling_info = batch.sampling_info
        if not USE_CHAIN_SPECULATIVE_SAMPLING or sampling_info.is_all_greedy:
            probs = F.softmax(logits_output.next_token_logits, dim=-1)
        else:
            temperature = sampling_info.temperatures
            probs = F.softmax(logits_output.next_token_logits / temperature, dim=-1)
            if use_topk_renorm:
                probs = top_k_renorm_prob(probs, sampling_info.top_ks)
            if use_topp_renorm and not torch.all(sampling_info.top_ps == 1.0):
                probs = top_p_renorm_prob(probs, sampling_info.top_ps)
        return probs

    def capture_for_decode(
        self,
        batch: ForwardBatch,
        logits_output: LogitsProcessorOutput,
        draft_input: EagleDraftInput,
    ):
        probs = self.get_renorm_probs(batch, logits_output)
        if USE_CHAIN_SPECULATIVE_SAMPLING:
            draft_input.topk_p, draft_input.topk_index = fast_sample(
                probs, num_samples=1
            )
            draft_input.draft_probs = probs
        else:
            draft_input.topk_p, draft_input.topk_index = fast_topk(
                probs, self.topk, dim=-1
            )
        draft_input.hidden_states = logits_output.hidden_states

    def _draft_preprocess_decode(self, batch: ScheduleBatch):
        # Parse args
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        # Accumulate penalty
        if batch.sampling_info.penalizer_orchestrator.is_required:
            # This is a relaxed version of penalties for speculative decoding.
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                spec_info.verified_id.to(torch.int64)
            )

        # Allocate cache locations
        # Layout of the out_cache_loc
        # [       topk 0         ] [       topk 1         ]
        # [iter=0, iter=1, iter=2] [iter=0, iter=1, iter=2]
        # TODO: We only need self.speculative_num_steps - 1 * topk cache loc
        out_cache_loc, _ = alloc_token_slots(
            batch.tree_cache,
            num_seqs * (self.speculative_num_steps + 1) * self.topk,
            backup_state=True,
        )
        # When source_cache_loc is not needed, simply skip
        duplicate_cache_len = 0
        source_cache_loc, target_cache_loc, last_page_lens_cumsum = None, None, None

        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self.extend_lens,
            self.num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            (self.speculative_num_steps + 1),
            self.page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps + 1 + self.page_size),
        )

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)

    def _draft_preprocess_idle(self, batch: ScheduleBatch):
        batch.spec_info = EagleDraftInput.create_idle_input(
            device=self.device,
            hidden_size=self.model_config.hidden_size,
            dtype=self.model_config.dtype,
            topk=self.topk,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    def draft(self, batch: ScheduleBatch):
        """
        draft decode, for DVR worker, we treat target model as draft model as well
        """
        # Parse args
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
        else:
            self._draft_preprocess_decode(batch)

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        spec_info.num_tokens_per_batch = self.topk
        spec_info.num_tokens_for_logprob_per_batch = self.topk
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL
        batch.return_hidden_states = False

        # Get forward batch
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)

        # Run forward steps
        # forward_batch.can_run_dp_cuda_graph = False
        parent_list, top_scores_index, draft_tokens, draft_probs = self.draft_forward(
            forward_batch
        )

        if batch.forward_mode.is_idle():
            idle_return = EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )
            idle_return.capture_hidden_mode = CaptureHiddenMode.NULL
            return idle_return, True

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

        return (
            EagleVerifyInput(
                draft_token=draft_tokens,
                custom_mask=tree_mask,
                positions=position,
                retrive_index=retrive_index,
                retrive_next_token=retrive_next_token,
                retrive_next_sibling=retrive_next_sibling,
                retrive_cum_len=None,
                spec_steps=self.speculative_num_steps,
                topk=self.topk,
                draft_token_num=self.server_args.speculative_num_draft_tokens,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                seq_lens_sum=forward_batch.seq_lens_sum,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                draft_probs=draft_probs,
            ),
            True,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc
        topk_index = spec_info.verified_id
        # TODO: We only need self.speculative_num_steps - 1 cache loc
        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, (self.speculative_num_steps + 1)
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps + 1, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        if USE_CHAIN_SPECULATIVE_SAMPLING:
            draft_probs_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None

        # NOTE: use normal decode for attention backend
        origin_seq_lens = forward_batch.seq_lens.clone()
        origin_seq_lens_cpu = forward_batch.seq_lens_cpu.clone()
        origin_spec_info = forward_batch.spec_info
        forward_batch.spec_info = None
        topk_p = None
        for i in range(self.speculative_num_steps + 1):
            # step 0 is like draft_extend/draft_extend_after_decode
            if i == 0:
                input_ids = topk_index.flatten()
            else:
                input_ids, _, scores, tree_info = select_top_k_tokens(
                    i - 1, topk_p, topk_index, None, scores, self.topk
                )
                score_list.append(tree_info[0])
                token_list.append(tree_info[1])
                parents_list.append(tree_info[2])
                forward_batch.positions.add_(1)

            if i == self.speculative_num_steps:
                break

            # Run forward
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i].contiguous()
            forward_batch.seq_lens = origin_seq_lens + i + 1
            forward_batch.seq_lens_cpu = origin_seq_lens_cpu + i + 1
            logits_output = self.model_runner.forward(
                forward_batch,
                skip_attn_backend_init=forward_batch.forward_mode.is_idle(),
            ).logits_output
            logits_output.next_token_logits = logits_output.next_token_logits.to(
                torch.float32
            )
            if self.server_args.enable_nan_detection:
                detect_nan(logits_output)
            probs = self.get_renorm_probs(
                forward_batch,
                logits_output,
                use_topk_renorm=USE_DECODE_TOPK_RENORM,
                use_topp_renorm=USE_DECODE_TOPP_RENORM,
            )
            if USE_CHAIN_SPECULATIVE_SAMPLING:
                topk_p, topk_index = fast_sample(probs, num_samples=1)
                draft_probs_list.append(probs)
            else:
                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)

        # rollback to original spec info
        forward_batch.seq_lens = origin_seq_lens
        forward_batch.seq_lens_cpu = origin_seq_lens_cpu
        forward_batch.spec_info = origin_spec_info

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )
        draft_probs = (
            torch.stack(draft_probs_list, dim=1)
            if USE_CHAIN_SPECULATIVE_SAMPLING
            else None
        )
        return parent_list, top_scores_index, draft_tokens, draft_probs

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        # NOTE: do not need alloc cache loc for verify, share the same cache loc as draft/decoce
        seq_lens_pre_verify = batch.seq_lens.clone()
        if get_global_server_args().enable_mamba_extra_buffer():
            batch.mamba_track_indices = torch.cat(
                [
                    req.mamba_ping_pong_track_buffer[
                        req.mamba_next_track_idx
                    ].unsqueeze(0)
                    for req in batch.reqs
                ],
            ).to(torch.int64)
        if not batch.forward_mode.is_idle():
            batch.input_ids = spec_info.draft_token
        spec_info.num_tokens_per_batch = self.speculative_num_steps
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )

        # Forward
        with self.target_verify_cuda_graph_context():
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )
        logits_output.next_token_logits = logits_output.next_token_logits.to(
            torch.float32
        )
        if self.server_args.enable_nan_detection:
            detect_nan(logits_output)

        vocab_mask = None
        res: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask,
        )

        # Post process based on verified outputs.
        # Pick indices that we care (accepted)
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]

        if self.target_worker.model_runner.hybrid_gdn_config is not None:
            self._mamba_verify_update(
                batch, res, logits_output, spec_info, seq_lens_pre_verify
            )

        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, res, logits_output)

        # Prepare the batch for the next draft forwards.
        self.postprocess_for_verify(batch, res)
        return logits_output, res, can_run_cuda_graph

    def postprocess_for_verify(
        self, batch: ScheduleBatch, verify_output: EagleVerifyOutput
    ):
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = verify_output.draft_input
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.NULL
        if batch.forward_mode.is_idle():
            return
        res_accept_length_cumsum = (
            torch.cumsum(batch.spec_info.accept_length + 1, dim=0) - 1
        )
        batch.spec_info.verified_id = batch.spec_info.verified_id[
            res_accept_length_cumsum
        ]
        # dummy topk_p and topk_index, it is just used for compatibility
        batch.spec_info.topk_index = batch.spec_info.verified_id.unsqueeze(-1)
        batch.spec_info.topk_p = torch.zeros(
            (batch.spec_info.verified_id.shape[0], 1),
            device=batch.spec_info.verified_id.device,
            dtype=torch.float32,
        )

    def clear_cache_pool(self):
        pass

    def _mamba_verify_update(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
        spec_info: EagleVerifyInput,
        seq_lens_pre_verify: torch.Tensor,
    ):
        accepted_length = (
            torch.tensor(
                res.accept_length_per_req_cpu,
                device=logits_output.next_token_logits.device,
                dtype=torch.int64,
            )
            + 1
        )
        cumulative_accepted_lengths = torch.cumsum(accepted_length, dim=0)
        # prepend 0 to the cumulative_accepted_lengths
        accepted_indices_start = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=cumulative_accepted_lengths.dtype,
                    device=cumulative_accepted_lengths.device,
                ),
                cumulative_accepted_lengths[:-1],
            ]
        )
        accepted_indices_offset = torch.arange(
            0,
            len(batch.seq_lens) * batch.spec_info.draft_token_num,
            step=batch.spec_info.draft_token_num,
            dtype=accepted_indices_start.dtype,
            device=accepted_indices_start.device,
        )

        accepted_steps = accepted_length - 1

        if batch.mamba_track_indices is not None:
            # If after verify, the request's seq_lens has crossed a mamba track interval,
            # we need to update the mamba state for the request at the crossing point.
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            mamba_steps_to_track = torch.where(
                to_track_mask,
                res.accepted_indices[to_track_ith + accepted_indices_start]
                - accepted_indices_offset,
                -1,
            )
        else:
            mamba_steps_to_track = None

        self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    @contextlib.contextmanager
    def _mamba_backup_conv_state(self, batch: ScheduleBatch):
        if not get_global_server_args().enable_mamba_extra_buffer():
            try:
                yield
            finally:
                return
        hybrid_attn_backend = self.target_worker.model_runner.attn_backend
        request_number = batch.batch_size()
        mamba_caches = (
            hybrid_attn_backend.linear_attn_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        )
        conv_states = mamba_caches.conv[0]
        dst_state_indices = torch.tensor(
            [req.mamba_pool_idx for req in batch.reqs], device=conv_states.device
        )
        conv_states_backup = conv_states[:, dst_state_indices].clone()
        try:
            yield
        finally:
            conv_states[:, dst_state_indices] = conv_states_backup
