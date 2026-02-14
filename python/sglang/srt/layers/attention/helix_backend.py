"""
Helix Parallelism Attention Backend for SGLang.

This backend wraps local attention computation with the Helix KV-parallel
combining step. During decode, each GPU computes attention over its local
KV shard, then uses All-to-All + logsumexp combining to produce the exact
attention output.

For the MVP, this uses a torch-native attention implementation that explicitly
computes LSE values needed for the combining step. Production deployments
would use FlashAttention/FlashInfer kernels that return LSE natively.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.helix_attn_utils import (
    attention_with_lse,
    helix_attention_with_kvp,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class HelixAttnBackend(AttentionBackend):
    """
    Attention backend implementing Helix KV Parallelism.

    Splits KV cache across KVP GPUs along the sequence dimension.
    Each GPU computes local attention, then results are combined via
    All-to-All + logsumexp rescaling to produce exact attention output.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.device = model_runner.device
        self.forward_metadata = None

        from sglang.srt.distributed.parallel_state import (
            get_helix_kvp_group,
            get_helix_kvp_size,
            is_helix_enabled,
        )

        self.helix_enabled = is_helix_enabled()
        self.kvp_size = get_helix_kvp_size()
        if self.helix_enabled:
            self.kvp_group = get_helix_kvp_group().device_group
        else:
            self.kvp_group = None

        logger.info(
            f"HelixAttnBackend initialized: helix_enabled={self.helix_enabled}, "
            f"kvp_size={self.kvp_size}"
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        pass

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        """
        Decode forward with Helix KV parallelism.

        1. Save new KV to local cache
        2. Compute local attention over local KV shard (returns output + LSE)
        3. If helix enabled: All-to-All exchange + logsumexp combine
        4. Return flattened output
        """
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        cache_loc = forward_batch.out_cache_loc
        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        num_tokens = q.shape[0]
        q_3d = q.view(num_tokens, layer.tp_q_head_num, layer.qk_head_dim)

        # Get KV cache buffers
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens

        # Compute attention per request with LSE
        all_outputs = []
        all_lse = []

        for seq_idx in range(seq_lens.shape[0]):
            seq_len_kv = seq_lens[seq_idx]
            per_req_query = q_3d[seq_idx : seq_idx + 1]  # (1, heads, head_dim)

            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens]  # (seq_kv, kv_heads, head_dim)
            per_req_value = v_cache[per_req_tokens]  # (seq_kv, kv_heads, v_head_dim)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            # Reshape to (1, heads, seq, dim) for attention
            q_4d = per_req_query.unsqueeze(0).transpose(1, 2)  # (1, heads, 1, dim)
            k_4d = per_req_key.unsqueeze(0).transpose(1, 2)  # (1, kv_heads, seq, dim)
            v_4d = per_req_value.unsqueeze(0).transpose(1, 2)

            out, lse = attention_with_lse(
                q_4d, k_4d, v_4d, scale=layer.scaling, causal=False
            )
            # out: (1, heads, 1, v_dim), lse: (1, heads, 1)
            all_outputs.append(out.squeeze(0).squeeze(-2))  # (heads, v_dim)
            all_lse.append(lse.squeeze(0).squeeze(-1))  # (heads,)

        if len(all_outputs) == 0:
            return o

        # Stack: (batch, heads, v_dim) and (batch, heads)
        batch_output = torch.stack(all_outputs, dim=0)
        batch_lse = torch.stack(all_lse, dim=0)

        # Apply Helix combining if enabled
        if self.helix_enabled and self.kvp_size > 1:
            batch_output = helix_attention_with_kvp(
                batch_output, batch_lse, self.kvp_group, self.kvp_size
            )

        # Flatten back to (num_tokens, heads * v_dim)
        o = batch_output.reshape(num_tokens, -1)
        return o

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        """
        Extend (prefill) forward with Helix KV parallelism.

        During prefill, each GPU processes the full sequence but only stores
        its KVP shard of the KV cache. For the MVP, we compute full attention
        locally (since all tokens are available during prefill) and only
        distribute the KV cache storage.
        """
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        cache_loc = forward_batch.out_cache_loc
        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        num_tokens = q.shape[0]
        q_3d = q.view(num_tokens, layer.tp_q_head_num, layer.qk_head_dim)
        o_3d = o.view(num_tokens, layer.tp_q_head_num, layer.v_head_dim)

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        extend_prefix_lens = forward_batch.extend_prefix_lens
        extend_seq_lens = forward_batch.extend_seq_lens

        start_q = 0
        for seq_idx in range(seq_lens.shape[0]):
            extend_seq_len = extend_seq_lens[seq_idx]
            prefix_len = extend_prefix_lens[seq_idx]
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len

            per_req_query = q_3d[start_q:end_q]  # (ext_len, heads, dim)

            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens]
            per_req_value = v_cache[per_req_tokens]

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            # (1, heads, ext_len, dim) format
            q_4d = per_req_query.unsqueeze(0).transpose(1, 2)
            k_4d = per_req_key.unsqueeze(0).transpose(1, 2)
            v_4d = per_req_value.unsqueeze(0).transpose(1, 2)

            out, _ = attention_with_lse(
                q_4d, k_4d, v_4d, scale=layer.scaling, causal=True
            )
            # out: (1, heads, ext_len, v_dim)
            o_3d[start_q:end_q] = out.squeeze(0).transpose(0, 1)

            start_q = end_q

        return o

    def support_triton(self):
        return False
