"""
Distributed tests for Helix Parallelism.

Tests the All-to-All communication pattern and end-to-end Helix attention
using torch.distributed with the gloo backend (works on CPU without CUDA).

Run with: python -m torch.distributed.launch --nproc_per_node=4 test_helix_distributed.py
Or:       torchrun --nproc_per_node=4 test_helix_distributed.py
"""

import math
import os
import sys
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _init_process(rank, world_size, fn, *args):
    """Initialize process group and run test function."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _run_all_to_all_test(rank, world_size):
    """Test helix_all_to_all_exchange with actual distributed communication."""
    from sglang.srt.layers.helix_attn_utils import helix_all_to_all_exchange

    B = 2
    H_q = 8
    D = 32
    KVP = world_size  # All ranks form one KVP group
    heads_per_tpa = H_q  # TPA=1 for simplicity
    heads_per_gpu = H_q // KVP

    torch.manual_seed(42 + rank)

    # Each rank has partial attention for all H_q heads from its KV shard
    local_output = torch.randn(B, heads_per_tpa, D)
    local_lse = torch.randn(B, heads_per_tpa)

    # Perform All-to-All
    kvp_group = dist.group.WORLD
    recv_outputs, recv_lse = helix_all_to_all_exchange(
        local_output, local_lse, kvp_group, KVP
    )

    # Verify: we should receive KVP chunks
    assert len(recv_outputs) == KVP, f"Expected {KVP} chunks, got {len(recv_outputs)}"
    assert len(recv_lse) == KVP

    for i, (out, lse) in enumerate(zip(recv_outputs, recv_lse)):
        assert out.shape == (B, heads_per_gpu, D), (
            f"Rank {rank}: chunk {i} output shape {out.shape}, "
            f"expected ({B}, {heads_per_gpu}, {D})"
        )
        assert lse.shape == (B, heads_per_gpu), (
            f"Rank {rank}: chunk {i} lse shape {lse.shape}, "
            f"expected ({B}, {heads_per_gpu})"
        )

    # Verify: chunk from self should match local data
    # Rank k sends its chunk k to itself
    my_chunk = local_output[:, rank * heads_per_gpu : (rank + 1) * heads_per_gpu, :]
    torch.testing.assert_close(recv_outputs[rank], my_chunk, atol=1e-6, rtol=1e-6)

    if rank == 0:
        print("PASS: All-to-All exchange test")


def _run_end_to_end_helix_test(rank, world_size):
    """
    End-to-end test: each rank computes partial attention on its KV shard,
    then All-to-All + combine produces exact attention.
    """
    from sglang.srt.layers.helix_attn_utils import (
        helix_all_to_all_exchange,
        helix_combine_partial_attention,
    )

    B = 2
    H_q = 4
    D = 32
    S = 64
    KVP = world_size
    heads_per_gpu = H_q // KVP
    scale = 1.0 / math.sqrt(D)

    # Use same seed for shared Q, K, V
    torch.manual_seed(123)
    q = torch.randn(B, H_q, 1, D)
    k = torch.randn(B, H_q, S, D)  # Using H_q heads for simplicity (MHA)
    v = torch.randn(B, H_q, S, D)

    # Reference full attention (all ranks compute this identically)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    ref_output = torch.matmul(torch.softmax(scores, dim=-1), v).squeeze(2)

    # Each rank gets its KV shard
    shard_size = S // KVP
    k_shard = k[:, :, rank * shard_size : (rank + 1) * shard_size, :]
    v_shard = v[:, :, rank * shard_size : (rank + 1) * shard_size, :]

    # Compute local partial attention on shard
    local_scores = torch.matmul(q, k_shard.transpose(-2, -1)) * scale
    local_lse = torch.logsumexp(local_scores, dim=-1).squeeze(2)  # (B, H_q)
    local_weights = torch.softmax(local_scores, dim=-1)
    local_output = torch.matmul(local_weights, v_shard).squeeze(2)  # (B, H_q, D)

    # All-to-All exchange
    kvp_group = dist.group.WORLD
    recv_outputs, recv_lse = helix_all_to_all_exchange(
        local_output, local_lse, kvp_group, KVP
    )

    # Combine partial results for this rank's head slice
    combined, _ = helix_combine_partial_attention(recv_outputs, recv_lse)

    # Verify against reference for this rank's heads
    my_ref = ref_output[:, rank * heads_per_gpu : (rank + 1) * heads_per_gpu, :]

    torch.testing.assert_close(combined, my_ref, atol=1e-4, rtol=1e-4)

    if rank == 0:
        print("PASS: End-to-end Helix attention test")


def _run_kvp_group_creation_test(rank, world_size):
    """Test that KVP groups are created correctly with proper rank assignments."""
    # Simulate KVP group creation logic
    N = world_size  # Total GPUs
    KVP = 2  # KV parallel shards
    TPA = N // KVP  # Attention TP size

    # My coordinates
    kvp_rank = rank // TPA
    tpa_rank = rank % TPA

    # My KVP group: all ranks with same tpa_rank
    my_kvp_group_ranks = [kvp_idx * TPA + tpa_rank for kvp_idx in range(KVP)]

    # Verify I'm in my group
    assert rank in my_kvp_group_ranks, (
        f"Rank {rank} (kvp={kvp_rank}, tpa={tpa_rank}) not in its KVP group {my_kvp_group_ranks}"
    )

    # Verify group has correct size
    assert len(my_kvp_group_ranks) == KVP, (
        f"KVP group should have {KVP} members, got {len(my_kvp_group_ranks)}"
    )

    if rank == 0:
        print(f"PASS: KVP group creation test (N={N}, KVP={KVP}, TPA={TPA})")


def _can_spawn():
    """Check if mp.spawn with gloo backend works on this platform."""
    import platform
    # mp.spawn + gloo is unreliable on macOS; skip gracefully
    if platform.system() == "Darwin":
        return False
    return True


class TestHelixDistributed(unittest.TestCase):
    """Test Helix distributed operations using gloo backend.

    These tests require Linux with torch.distributed support.
    On macOS, they are skipped (mp.spawn + gloo is unreliable).
    """

    @staticmethod
    def _spawn_test(fn, world_size):
        mp.spawn(_init_process, args=(world_size, fn), nprocs=world_size, join=True)

    @unittest.skipUnless(_can_spawn(), "mp.spawn + gloo not reliable on this platform")
    def test_all_to_all_exchange_2_ranks(self):
        """Test All-to-All with 2 ranks."""
        self._spawn_test(_run_all_to_all_test, 2)

    @unittest.skipUnless(_can_spawn(), "mp.spawn + gloo not reliable on this platform")
    def test_all_to_all_exchange_4_ranks(self):
        """Test All-to-All with 4 ranks."""
        self._spawn_test(_run_all_to_all_test, 4)

    @unittest.skipUnless(_can_spawn(), "mp.spawn + gloo not reliable on this platform")
    def test_end_to_end_2_ranks(self):
        """End-to-end Helix attention with 2 KVP ranks."""
        self._spawn_test(_run_end_to_end_helix_test, 2)

    @unittest.skipUnless(_can_spawn(), "mp.spawn + gloo not reliable on this platform")
    def test_end_to_end_4_ranks(self):
        """End-to-end Helix attention with 4 KVP ranks."""
        self._spawn_test(_run_end_to_end_helix_test, 4)

    @unittest.skipUnless(_can_spawn(), "mp.spawn + gloo not reliable on this platform")
    def test_kvp_group_creation_4_ranks(self):
        """Test KVP group creation logic with 4 ranks."""
        self._spawn_test(_run_kvp_group_creation_test, 4)


if __name__ == "__main__":
    unittest.main()
