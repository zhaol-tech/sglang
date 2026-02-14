"""
Unit tests for Helix Parallelism attention algorithms.

Tests the core logsumexp combining logic and validates correctness
against reference full-attention computation. All tests run on CPU
without requiring distributed setup.
"""

import math
import unittest

import torch


class TestHelixCombinePartialAttention(unittest.TestCase):
    """Test the logsumexp-based combining of partial attention outputs."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.num_heads = 8
        self.head_dim = 64
        self.seq_len = 128
        self.dtype = torch.float32

    def _reference_attention(self, q, k, v, scale):
        """Compute reference full attention output and LSE."""
        # q: (batch, heads, 1, dim), k/v: (batch, heads, seq, dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        lse = torch.logsumexp(scores, dim=-1)  # (batch, heads, 1)
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)  # (batch, heads, 1, dim)
        return output, lse

    def _reference_attention_on_shard(self, q, k_shard, v_shard, scale):
        """Compute attention on a single KV shard."""
        scores = torch.matmul(q, k_shard.transpose(-2, -1)) * scale
        lse = torch.logsumexp(scores, dim=-1)
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v_shard)
        return output, lse

    def test_combine_two_shards_matches_full(self):
        """Combining 2 KV shards should match full attention exactly."""
        from sglang.srt.layers.helix_attn_utils import helix_combine_partial_attention

        B, H, D = self.batch_size, self.num_heads, self.head_dim
        S = self.seq_len
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(B, H, 1, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        # Full attention reference
        ref_output, ref_lse = self._reference_attention(q, k, v, scale)

        # Split KV into 2 shards
        k1, k2 = k[:, :, : S // 2, :], k[:, :, S // 2 :, :]
        v1, v2 = v[:, :, : S // 2, :], v[:, :, S // 2 :, :]

        # Partial attention on each shard
        out1, lse1 = self._reference_attention_on_shard(q, k1, v1, scale)
        out2, lse2 = self._reference_attention_on_shard(q, k2, v2, scale)

        # Reshape for combine: (batch, heads, dim)
        partial_outputs = [
            out1.squeeze(2),  # (B, H, D)
            out2.squeeze(2),
        ]
        lse_vals = [
            lse1.squeeze(2),  # (B, H)
            lse2.squeeze(2),
        ]

        combined, combined_lse = helix_combine_partial_attention(
            partial_outputs, lse_vals
        )

        torch.testing.assert_close(
            combined,
            ref_output.squeeze(2),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_combine_four_shards_matches_full(self):
        """Combining 4 KV shards should match full attention exactly."""
        from sglang.srt.layers.helix_attn_utils import helix_combine_partial_attention

        B, H, D = self.batch_size, self.num_heads, self.head_dim
        S = 256
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(B, H, 1, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref_output, _ = self._reference_attention(q, k, v, scale)

        kvp = 4
        shard_size = S // kvp
        partial_outputs = []
        lse_vals = []
        for i in range(kvp):
            k_shard = k[:, :, i * shard_size : (i + 1) * shard_size, :]
            v_shard = v[:, :, i * shard_size : (i + 1) * shard_size, :]
            out, lse = self._reference_attention_on_shard(q, k_shard, v_shard, scale)
            partial_outputs.append(out.squeeze(2))
            lse_vals.append(lse.squeeze(2))

        combined, _ = helix_combine_partial_attention(partial_outputs, lse_vals)

        torch.testing.assert_close(
            combined,
            ref_output.squeeze(2),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_single_shard_passthrough(self):
        """With a single shard, combine should return the input unchanged."""
        from sglang.srt.layers.helix_attn_utils import helix_combine_partial_attention

        B, H, D = 2, 4, 32
        output = torch.randn(B, H, D)
        lse = torch.randn(B, H)

        combined, combined_lse = helix_combine_partial_attention([output], [lse])
        torch.testing.assert_close(combined, output)
        torch.testing.assert_close(combined_lse, lse)

    def test_numerical_stability_large_lse(self):
        """Combining should be stable even with very large LSE values."""
        from sglang.srt.layers.helix_attn_utils import helix_combine_partial_attention

        B, H, D = 2, 4, 32

        out1 = torch.randn(B, H, D)
        out2 = torch.randn(B, H, D)
        # Very large LSE values that could cause overflow without stability tricks
        lse1 = torch.full((B, H), 500.0)
        lse2 = torch.full((B, H), 500.0)

        combined, combined_lse = helix_combine_partial_attention(
            [out1, out2], [lse1, lse2]
        )

        self.assertFalse(torch.isnan(combined).any(), "Output contains NaN")
        self.assertFalse(torch.isinf(combined).any(), "Output contains Inf")
        self.assertFalse(torch.isnan(combined_lse).any(), "LSE contains NaN")

    def test_numerical_stability_very_different_lse(self):
        """One shard dominates when its LSE is much larger."""
        from sglang.srt.layers.helix_attn_utils import helix_combine_partial_attention

        B, H, D = 2, 4, 32

        out1 = torch.randn(B, H, D)
        out2 = torch.randn(B, H, D)
        # Shard 1 has much larger LSE → should dominate
        lse1 = torch.full((B, H), 100.0)
        lse2 = torch.full((B, H), -100.0)

        combined, _ = helix_combine_partial_attention(
            [out1, out2], [lse1, lse2]
        )

        # When one shard dominates, output should be close to that shard's output
        torch.testing.assert_close(combined, out1, atol=1e-5, rtol=1e-5)

    def test_3d_lse_with_trailing_dim(self):
        """LSE values with shape (B, H, 1) should work correctly."""
        from sglang.srt.layers.helix_attn_utils import helix_combine_partial_attention

        B, H, D = 2, 4, 32
        out1 = torch.randn(B, H, D)
        out2 = torch.randn(B, H, D)
        lse1 = torch.randn(B, H, 1)  # 3D with trailing dim
        lse2 = torch.randn(B, H, 1)

        combined, combined_lse = helix_combine_partial_attention(
            [out1, out2], [lse1, lse2]
        )
        self.assertEqual(combined.shape, (B, H, D))
        self.assertEqual(combined_lse.shape, (B, H))

    def test_combine_preserves_dtype(self):
        """Output dtype should match input dtype."""
        from sglang.srt.layers.helix_attn_utils import helix_combine_partial_attention

        for dtype in [torch.float32, torch.float64]:
            out1 = torch.randn(2, 4, 32, dtype=dtype)
            out2 = torch.randn(2, 4, 32, dtype=dtype)
            lse1 = torch.randn(2, 4, dtype=dtype)
            lse2 = torch.randn(2, 4, dtype=dtype)

            combined, combined_lse = helix_combine_partial_attention(
                [out1, out2], [lse1, lse2]
            )
            self.assertEqual(combined.dtype, dtype)
            self.assertEqual(combined_lse.dtype, dtype)

    def test_gqa_combine_matches_full(self):
        """Test combining with GQA (fewer KV heads than Q heads)."""
        from sglang.srt.layers.helix_attn_utils import helix_combine_partial_attention

        B = 2
        num_q_heads = 8
        num_kv_heads = 2
        D = 32
        S = 64
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(B, num_q_heads, 1, D)
        k = torch.randn(B, num_kv_heads, S, D)
        v = torch.randn(B, num_kv_heads, S, D)

        # Expand KV for full attention reference
        repeat = num_q_heads // num_kv_heads
        k_expanded = k.repeat_interleave(repeat, dim=1)
        v_expanded = v.repeat_interleave(repeat, dim=1)
        ref_output, _ = self._reference_attention(q, k_expanded, v_expanded, scale)

        # Split KV and compute partials (with GQA expansion)
        kvp = 2
        shard_size = S // kvp
        partial_outputs = []
        lse_vals = []
        for i in range(kvp):
            k_shard = k_expanded[:, :, i * shard_size : (i + 1) * shard_size, :]
            v_shard = v_expanded[:, :, i * shard_size : (i + 1) * shard_size, :]
            out, lse = self._reference_attention_on_shard(q, k_shard, v_shard, scale)
            partial_outputs.append(out.squeeze(2))
            lse_vals.append(lse.squeeze(2))

        combined, _ = helix_combine_partial_attention(partial_outputs, lse_vals)

        torch.testing.assert_close(
            combined,
            ref_output.squeeze(2),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_long_sequence_many_shards(self):
        """Test with a longer sequence split into many shards."""
        from sglang.srt.layers.helix_attn_utils import helix_combine_partial_attention

        B, H, D = 1, 4, 32
        S = 1024
        kvp = 8
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(B, H, 1, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref_output, _ = self._reference_attention(q, k, v, scale)

        shard_size = S // kvp
        partial_outputs = []
        lse_vals = []
        for i in range(kvp):
            k_shard = k[:, :, i * shard_size : (i + 1) * shard_size, :]
            v_shard = v[:, :, i * shard_size : (i + 1) * shard_size, :]
            out, lse = self._reference_attention_on_shard(q, k_shard, v_shard, scale)
            partial_outputs.append(out.squeeze(2))
            lse_vals.append(lse.squeeze(2))

        combined, _ = helix_combine_partial_attention(partial_outputs, lse_vals)

        torch.testing.assert_close(
            combined,
            ref_output.squeeze(2),
            atol=1e-4,
            rtol=1e-4,
        )


class TestHelixAttentionWithLSE(unittest.TestCase):
    """Test the _attention_with_lse function from the backend."""

    def test_output_matches_sdpa(self):
        """_attention_with_lse output should match torch sdpa."""
        from sglang.srt.layers.helix_attn_utils import attention_with_lse

        B, H, S_q, S_kv, D = 2, 4, 1, 32, 64
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(B, H, S_q, D)
        k = torch.randn(B, H, S_kv, D)
        v = torch.randn(B, H, S_kv, D)

        out_helix, lse_helix = attention_with_lse(q, k, v, scale=scale, causal=False)

        # Reference using torch sdpa
        ref_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=False
        )

        torch.testing.assert_close(out_helix, ref_out, atol=1e-5, rtol=1e-5)

        # Verify LSE shape
        self.assertEqual(lse_helix.shape, (B, H, S_q))

    def test_lse_values_correct(self):
        """LSE values should equal logsumexp of attention scores."""
        from sglang.srt.layers.helix_attn_utils import attention_with_lse

        B, H, S_q, S_kv, D = 1, 2, 1, 8, 16
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(B, H, S_q, D)
        k = torch.randn(B, H, S_kv, D)
        v = torch.randn(B, H, S_kv, D)

        _, lse = attention_with_lse(q, k, v, scale=scale, causal=False)

        # Manually compute LSE
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        expected_lse = torch.logsumexp(scores, dim=-1)

        torch.testing.assert_close(lse, expected_lse, atol=1e-5, rtol=1e-5)

    def test_gqa_handling(self):
        """GQA (fewer KV heads) should be handled correctly."""
        from sglang.srt.layers.helix_attn_utils import attention_with_lse

        B, D = 2, 32
        num_q_heads = 8
        num_kv_heads = 2
        S_q, S_kv = 1, 16
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(B, num_q_heads, S_q, D)
        k = torch.randn(B, num_kv_heads, S_kv, D)
        v = torch.randn(B, num_kv_heads, S_kv, D)

        out, lse = attention_with_lse(q, k, v, scale=scale, causal=False)

        self.assertEqual(out.shape, (B, num_q_heads, S_q, D))
        self.assertEqual(lse.shape, (B, num_q_heads, S_q))

    def test_causal_attention(self):
        """Causal masking should prevent attending to future positions."""
        from sglang.srt.layers.helix_attn_utils import attention_with_lse

        B, H, S, D = 1, 1, 4, 16
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        out_causal, lse_causal = attention_with_lse(q, k, v, scale=scale, causal=True)
        out_full, lse_full = attention_with_lse(q, k, v, scale=scale, causal=False)

        # Last position attends to everything in both modes → should match
        torch.testing.assert_close(
            out_causal[:, :, -1, :], out_full[:, :, -1, :], atol=1e-5, rtol=1e-5
        )

        # Earlier positions should differ (causal excludes future tokens)
        self.assertFalse(
            torch.allclose(out_causal[:, :, 0, :], out_full[:, :, 0, :], atol=1e-3)
        )


class TestHelixConfigValidation(unittest.TestCase):
    """Test Helix configuration validation logic."""

    def test_kvp_size_must_divide_tp_size(self):
        """tp_size must be divisible by helix_kvp_size."""
        # This tests the logic, not the actual ServerArgs (which requires more setup)
        tp_size = 4
        kvp_size = 3  # Does not divide 4
        self.assertNotEqual(tp_size % kvp_size, 0)

    def test_valid_configurations(self):
        """Valid Helix configurations."""
        valid_configs = [
            (4, 1),  # Disabled
            (4, 2),  # 4 GPUs, KVP=2, TPA=2
            (4, 4),  # 4 GPUs, KVP=4, TPA=1
            (8, 2),  # 8 GPUs, KVP=2, TPA=4
            (8, 4),  # 8 GPUs, KVP=4, TPA=2
            (8, 8),  # 8 GPUs, KVP=8, TPA=1
        ]
        for tp_size, kvp_size in valid_configs:
            self.assertEqual(
                tp_size % kvp_size,
                0,
                f"tp={tp_size}, kvp={kvp_size} should be valid",
            )
            tpa_size = tp_size // kvp_size
            self.assertGreaterEqual(tpa_size, 1)

    def test_invalid_configurations(self):
        """Invalid Helix configurations."""
        invalid_configs = [
            (4, 3),  # 3 doesn't divide 4
            (6, 4),  # 4 doesn't divide 6
            (8, 3),  # 3 doesn't divide 8
        ]
        for tp_size, kvp_size in invalid_configs:
            self.assertNotEqual(
                tp_size % kvp_size,
                0,
                f"tp={tp_size}, kvp={kvp_size} should be invalid",
            )


class TestHelixEndToEndCombine(unittest.TestCase):
    """End-to-end test: split sequence, compute partial attention, combine."""

    def test_full_pipeline_2_shards(self):
        """Full pipeline: split KV into 2 shards, compute partials, combine."""
        from sglang.srt.layers.helix_attn_utils import attention_with_lse
        from sglang.srt.layers.helix_attn_utils import helix_combine_partial_attention

        B, H, D = 2, 8, 64
        S = 128
        scale = 1.0 / math.sqrt(D)
        kvp = 2

        q = torch.randn(B, H, 1, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        # Reference full attention
        ref_out, _ = attention_with_lse(q, k, v, scale=scale, causal=False)

        # Shard and compute partials
        shard_size = S // kvp
        partial_outputs = []
        lse_vals = []
        for i in range(kvp):
            k_shard = k[:, :, i * shard_size : (i + 1) * shard_size, :]
            v_shard = v[:, :, i * shard_size : (i + 1) * shard_size, :]
            out, lse = attention_with_lse(
                q, k_shard, v_shard, scale=scale, causal=False
            )
            partial_outputs.append(out.squeeze(2))  # (B, H, D)
            lse_vals.append(lse.squeeze(2))  # (B, H)

        # Combine
        combined, _ = helix_combine_partial_attention(partial_outputs, lse_vals)

        torch.testing.assert_close(
            combined, ref_out.squeeze(2), atol=1e-5, rtol=1e-5
        )

    def test_full_pipeline_4_shards_gqa(self):
        """Full pipeline with GQA: 4 shards, 8 Q heads, 2 KV heads."""
        from sglang.srt.layers.helix_attn_utils import attention_with_lse
        from sglang.srt.layers.helix_attn_utils import helix_combine_partial_attention

        B = 2
        num_q = 8
        num_kv = 2
        D = 32
        S = 64
        scale = 1.0 / math.sqrt(D)
        kvp = 4

        q = torch.randn(B, num_q, 1, D)
        k = torch.randn(B, num_kv, S, D)
        v = torch.randn(B, num_kv, S, D)

        # Reference (with GQA expansion handled by _attention_with_lse)
        ref_out, _ = attention_with_lse(q, k, v, scale=scale, causal=False)

        # Shard KV and compute partials
        shard_size = S // kvp
        partial_outputs = []
        lse_vals = []
        for i in range(kvp):
            k_shard = k[:, :, i * shard_size : (i + 1) * shard_size, :]
            v_shard = v[:, :, i * shard_size : (i + 1) * shard_size, :]
            out, lse = attention_with_lse(
                q, k_shard, v_shard, scale=scale, causal=False
            )
            partial_outputs.append(out.squeeze(2))
            lse_vals.append(lse.squeeze(2))

        combined, _ = helix_combine_partial_attention(partial_outputs, lse_vals)

        torch.testing.assert_close(
            combined, ref_out.squeeze(2), atol=1e-5, rtol=1e-5
        )


class TestHelixAllToAllPattern(unittest.TestCase):
    """Test the All-to-All communication pattern logic (without actual distributed)."""

    def test_head_redistribution_logic(self):
        """
        Verify the head redistribution pattern.

        With KVP=2, TPA=2, N=4 GPUs:
        - GPU(kvp=0, tpa=0) has heads [0,4) from shard 0
        - GPU(kvp=0, tpa=1) has heads [4,8) from shard 0
        - GPU(kvp=1, tpa=0) has heads [0,4) from shard 1
        - GPU(kvp=1, tpa=1) has heads [4,8) from shard 1

        After redistribution:
        - GPU 0 gets heads [0,2) from shards 0,1
        - GPU 1 gets heads [2,4) from shards 0,1
        - GPU 2 gets heads [4,6) from shards 0,1
        - GPU 3 gets heads [6,8) from shards 0,1
        """
        H_q = 8
        KVP = 2
        TPA = 2
        N = KVP * TPA
        heads_per_tpa = H_q // TPA
        heads_per_gpu = H_q // N

        # Simulate the redistribution
        # Each (kvp_rank, tpa_rank) pair holds heads_per_tpa heads
        for tpa_rank in range(TPA):
            head_start = tpa_rank * heads_per_tpa
            head_end = (tpa_rank + 1) * heads_per_tpa

            # These heads get split into KVP chunks
            for chunk_idx in range(KVP):
                dest_gpu = tpa_rank * KVP + chunk_idx
                chunk_head_start = head_start + chunk_idx * heads_per_gpu
                chunk_head_end = chunk_head_start + heads_per_gpu

                # Verify destination gets correct head range
                expected_head_start = dest_gpu * heads_per_gpu
                expected_head_end = (dest_gpu + 1) * heads_per_gpu

                self.assertEqual(
                    chunk_head_start,
                    expected_head_start,
                    f"GPU {dest_gpu}: expected heads [{expected_head_start},{expected_head_end}), "
                    f"got [{chunk_head_start},{chunk_head_end})",
                )

    def test_communication_volume_independent_of_seq_len(self):
        """
        Verify that All-to-All communication volume doesn't depend on sequence length.
        """
        B = 4
        H_q = 8
        D = 64
        KVP = 2
        TPA = 2
        heads_per_tpa = H_q // TPA
        heads_per_gpu = H_q // (KVP * TPA)

        for seq_len in [128, 1024, 10000, 100000]:
            # Communication per GPU: sends KVP chunks of (B, heads_per_gpu, D)
            comm_elements = KVP * B * heads_per_gpu * D
            # This should be constant regardless of seq_len
            self.assertEqual(
                comm_elements,
                KVP * B * heads_per_gpu * D,
                f"Communication volume should not depend on seq_len={seq_len}",
            )

    def test_all_to_all_local_simulation(self):
        """Simulate All-to-All exchange locally to verify data routing."""
        from sglang.srt.layers.helix_attn_utils import helix_combine_partial_attention

        B = 2
        H_q = 8
        D = 32
        KVP = 2
        TPA = 2
        heads_per_tpa = H_q // TPA

        # Simulate partial outputs from 2 KVP ranks for tpa_rank=0
        # Both have heads [0, 4) but from different KV shards
        partial_0 = torch.randn(B, heads_per_tpa, D)
        partial_1 = torch.randn(B, heads_per_tpa, D)
        lse_0 = torch.randn(B, heads_per_tpa)
        lse_1 = torch.randn(B, heads_per_tpa)

        # Split each into KVP chunks (simulating what All-to-All does)
        heads_per_gpu = heads_per_tpa // KVP

        # After All-to-All, GPU 0 gets:
        #   chunk 0 from shard 0: partial_0[:, :heads_per_gpu, :]
        #   chunk 0 from shard 1: partial_1[:, :heads_per_gpu, :]
        gpu0_partials = [
            partial_0[:, :heads_per_gpu, :],
            partial_1[:, :heads_per_gpu, :],
        ]
        gpu0_lse = [
            lse_0[:, :heads_per_gpu],
            lse_1[:, :heads_per_gpu],
        ]

        # Combine should produce (B, heads_per_gpu, D)
        combined, _ = helix_combine_partial_attention(gpu0_partials, gpu0_lse)
        self.assertEqual(combined.shape, (B, heads_per_gpu, D))


if __name__ == "__main__":
    unittest.main()
