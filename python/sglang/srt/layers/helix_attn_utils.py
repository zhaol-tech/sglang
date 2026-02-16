"""
Helix Parallelism: Core attention algorithms for KV-parallel decoding.

This module is intentionally free of sglang-internal imports so it can be
unit-tested on CPU without pulling in triton/flashinfer/CUDA dependencies.

It provides three layers of abstraction:

1. attention_with_lse()  — drop-in replacement for scaled_dot_product_attention
   that also returns the log-sum-exp (LSE) values needed for combining.
   (Production would use FlashAttention/FlashInfer kernels that return LSE natively.)

2. helix_combine_partial_attention()  — the mathematical core: given partial
   attention outputs and LSE values from K different KV shards, combine them
   into the exact full-attention result using numerically stable logsumexp rescaling.

3. helix_all_to_all_exchange() / helix_attention_with_kvp()  — distributed
   communication: redistributes partial results across GPUs in the KVP group,
   then calls the combining step.

Reference: "Helix Parallelism: Rethinking Sharding Strategies for
Interactive Multi-Million-Token LLM Decoding" (arXiv:2507.07120)
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist


def attention_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention and return both output and LSE.

    Unlike torch.nn.functional.scaled_dot_product_attention, this also
    returns the log-sum-exp of the attention scores needed for Helix combining.

    Args:
        query: (batch, num_q_heads, seq_len_q, head_dim)
        key: (batch, num_kv_heads, seq_len_kv, head_dim)
        value: (batch, num_kv_heads, seq_len_kv, v_head_dim)
        scale: attention scaling factor
        causal: whether to apply causal masking

    Returns:
        output: (batch, num_q_heads, seq_len_q, v_head_dim)
        lse: (batch, num_q_heads, seq_len_q)
    """
    # Handle GQA: repeat KV heads to match Q heads
    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    if num_kv_heads < num_q_heads:
        repeat_factor = num_q_heads // num_kv_heads
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)

    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if causal:
        seq_q = query.shape[2]
        seq_kv = key.shape[2]
        row_idx = torch.arange(seq_q, device=query.device).unsqueeze(1)
        col_idx = torch.arange(seq_kv, device=query.device).unsqueeze(0)
        offset = seq_kv - seq_q
        mask = col_idx > (row_idx + offset)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Compute LSE: logsumexp over the key dimension
    lse = torch.logsumexp(scores, dim=-1)  # (batch, heads, seq_q)

    # Compute attention weights and output
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    return output, lse


def helix_combine_partial_attention(
    partial_outputs: List[torch.Tensor],
    lse_values: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine partial attention outputs from multiple KV cache shards using
    numerically stable logsumexp rescaling.

    Each KV shard produces a partial softmax-weighted output O_k and a
    log-sum-exp scalar L_k per (batch, head) pair. The exact combined
    output is:

        L_combined = logsumexp(L_1, ..., L_K)
        O_combined = sum_k exp(L_k - L_combined) * O_k

    Args:
        partial_outputs: List of K tensors, each (batch, num_heads, head_dim).
            Partial attention output from each KV shard.
        lse_values: List of K tensors, each (batch, num_heads) or (batch, num_heads, 1).
            Log-sum-exp values from each KV shard's FlashAttention.

    Returns:
        combined_output: (batch, num_heads, head_dim) - exact attention output
        combined_lse: (batch, num_heads) - combined log-sum-exp
    """
    assert len(partial_outputs) == len(lse_values), (
        f"Mismatch: {len(partial_outputs)} outputs vs {len(lse_values)} lse values"
    )
    assert len(partial_outputs) > 0, "Need at least one partial output"

    if len(partial_outputs) == 1:
        lse = lse_values[0]
        if lse.dim() == 3:
            lse = lse.squeeze(-1)
        return partial_outputs[0], lse

    # Normalize lse shapes to (batch, num_heads)
    normalized_lse = []
    for lse in lse_values:
        if lse.dim() == 3:
            lse = lse.squeeze(-1)
        normalized_lse.append(lse)

    # Stack for vectorized operations: (K, batch, num_heads)
    lse_stack = torch.stack(normalized_lse, dim=0)

    # Numerically stable logsumexp across shards
    # L_max = max over K for stability
    lse_max, _ = lse_stack.max(dim=0)  # (batch, num_heads)

    # exp(L_k - L_max) for each shard
    # Shape: (K, batch, num_heads)
    exp_diff = torch.exp(lse_stack - lse_max.unsqueeze(0))

    # Combined logsumexp: L_combined = L_max + log(sum_k exp(L_k - L_max))
    combined_lse = lse_max + torch.log(exp_diff.sum(dim=0))  # (batch, num_heads)

    # Rescaling weights: w_k = exp(L_k - L_combined)
    # Shape: (K, batch, num_heads, 1) for broadcasting with head_dim
    weights = torch.exp(lse_stack - combined_lse.unsqueeze(0)).unsqueeze(-1)

    # Stack outputs: (K, batch, num_heads, head_dim)
    output_stack = torch.stack(partial_outputs, dim=0)

    # Weighted sum: O_combined = sum_k w_k * O_k
    combined_output = (weights * output_stack).sum(dim=0)  # (batch, num_heads, head_dim)

    return combined_output, combined_lse


def helix_all_to_all_exchange(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    kvp_group: dist.ProcessGroup,
    kvp_size: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Perform the Helix All-to-All exchange within a KVP group.

    Before exchange: each GPU has partial attention output for H_q/TPA heads
    from its local KV shard.

    After exchange: each GPU has partial attention outputs from ALL KVP shards,
    but for only H_q/N heads (its final head assignment).

    The exchange splits the head dimension into KVP chunks and redistributes
    so each GPU receives one chunk from every KVP rank.

    Args:
        local_output: (batch, num_heads_tpa, head_dim) - local partial attention output
        local_lse: (batch, num_heads_tpa) - local log-sum-exp values
        kvp_group: ProcessGroup for the KVP group (GPUs with same tpa_rank)
        kvp_size: number of KVP shards

    Returns:
        received_outputs: list of kvp_size tensors, each (batch, num_heads_per_gpu, head_dim)
        received_lse: list of kvp_size tensors, each (batch, num_heads_per_gpu)
    """
    batch_size, num_heads_tpa, head_dim = local_output.shape
    assert num_heads_tpa % kvp_size == 0, (
        f"num_heads_tpa ({num_heads_tpa}) must be divisible by kvp_size ({kvp_size})"
    )
    num_heads_per_gpu = num_heads_tpa // kvp_size

    # Split local output into KVP chunks along the head dimension.
    # This GPU has H_q/TPA heads; we split into KVP chunks of H_q/N heads each.
    # Chunk k will be sent to KVP rank k, which owns those heads in the final layout.
    output_chunks = local_output.chunk(kvp_size, dim=1)
    lse_chunks = local_lse.chunk(kvp_size, dim=1)

    # contiguous() is required because chunk() may return views with non-contiguous
    # strides, and NCCL/gloo all_to_all requires contiguous tensors.
    send_outputs = [chunk.contiguous() for chunk in output_chunks]
    send_lse = [chunk.contiguous() for chunk in lse_chunks]

    # Pre-allocate receive buffers — after the all_to_all, recv_outputs[k] will
    # contain the partial attention from KVP rank k for THIS GPU's head slice.
    recv_outputs = [
        torch.empty_like(send_outputs[0]) for _ in range(kvp_size)
    ]
    recv_lse = [
        torch.empty_like(send_lse[0]) for _ in range(kvp_size)
    ]

    # All-to-All: each GPU sends chunk[k] to KVP rank k and receives from rank k.
    # Communication volume per GPU = KVP * B * (H_q/N) * D — independent of seq len S.
    # This is the key property from the paper: scaling to millions of tokens doesn't
    # increase inter-GPU communication, only the local FlashAttention compute grows.
    # We do two all_to_all calls: one for the attention outputs, one for the LSE values.
    dist.all_to_all(recv_outputs, send_outputs, group=kvp_group)
    dist.all_to_all(recv_lse, send_lse, group=kvp_group)

    return recv_outputs, recv_lse


def helix_attention_with_kvp(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    kvp_group: dist.ProcessGroup,
    kvp_size: int,
) -> torch.Tensor:
    """
    Full Helix attention combining: All-to-All exchange followed by
    logsumexp-based combining of partial results.

    This is the main entry point for the Helix attention decode path.

    Args:
        local_output: (batch, num_heads_tpa, head_dim) - partial attention from local KV shard
        local_lse: (batch, num_heads_tpa) or (batch, num_heads_tpa, 1) - local lse
        kvp_group: ProcessGroup for KVP communication
        kvp_size: number of KVP shards

    Returns:
        combined_output: (batch, num_heads_per_gpu, head_dim) - exact attention for this GPU's heads
    """
    if local_lse.dim() == 3:
        local_lse = local_lse.squeeze(-1)

    if kvp_size == 1:
        return local_output

    # Step 1: All-to-All exchange
    recv_outputs, recv_lse = helix_all_to_all_exchange(
        local_output, local_lse, kvp_group, kvp_size
    )

    # Step 2: Combine using logsumexp
    combined_output, _ = helix_combine_partial_attention(recv_outputs, recv_lse)

    return combined_output
