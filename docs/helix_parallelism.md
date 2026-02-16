# Helix Parallelism MVP — Design & Implementation

Reference: [Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding](https://arxiv.org/abs/2507.07120)

## How SGLang Currently Works (Standard TP for Attention)

In standard tensor parallelism, sglang shards a model across N GPUs. For attention:

1. **QKV Projection**: Each GPU holds `Q_heads/N` query heads and `K_heads/N` KV heads (via `QKVParallelLinear`). Each GPU projects its slice.
2. **KV Cache**: Every GPU stores the **full** KV cache for its assigned KV heads across the **entire** sequence length S.
3. **Attention**: Each GPU runs FlashAttention over all S tokens for its head slice. Output shape: `(batch, Q_heads/N, head_dim)`.
4. **Output Projection**: Each GPU applies its slice of W_o, then **All-Reduce** across N GPUs to get the full hidden state.
5. **FFN**: Standard TP — each GPU has `weights/N`, computes partial output, All-Reduce again.

**The bottleneck for long contexts**: KV cache read time scales linearly with S. When S reaches millions of tokens, reading the full KV cache from HBM dominates latency. You can't increase TP beyond the number of KV heads (K) without duplicating KV cache, which wastes memory and doesn't help bandwidth.

## What Helix Parallelism Changes

Helix's key insight: **decouple attention parallelism from FFN parallelism**. Instead of each GPU reading S tokens of KV, shard the KV cache along the sequence dimension across KVP groups, so each GPU only reads `S/KVP` tokens.

With N GPUs:

- **Attention phase**: `N = KVP x TPA` — KVP shards along sequence, TPA shards along heads (like standard TP but only for attention)
- **FFN phase**: same N GPUs do standard TP (`TPF = N`) — unchanged from current sglang

The attention flow becomes:

1. Each GPU holds only `S/KVP` of the KV cache (1/KVP of the sequence)
2. Each GPU computes local FlashAttention on its shard -> gets **partial output + log-sum-exp (LSE)**
3. **All-to-All exchange** within KVP groups redistributes partial results
4. **LSE-based combining** produces the mathematically exact full attention output
5. Output projection + FFN proceed as normal TP

The communication volume of the All-to-All is `O(B x H)` — **independent of sequence length S**. This is the core win: you can scale to millions of tokens without communication growing.

## Implementation Details

### 1. Configuration (`server_args.py`)

Added `--helix-kvp-size` parameter. When set to 1 (default), sglang behaves exactly as before. When >1, it activates Helix mode.

```
N=8 GPUs, --helix-kvp-size 4  ->  KVP=4, TPA=2, TPF=8
```

Validation ensures `tp_size % kvp_size == 0`, no PP, no context parallelism conflict.

### 2. Process Groups (`parallel_state.py`)

SGLang uses `GroupCoordinator` objects for collective communication. We added a new `_HELIX_KVP` group:

- **KVP group**: GPUs with the **same TPA rank** but **different KVP ranks**. These are the GPUs that hold different sequence shards for the same head slice and need to exchange partial attention results.
- Layout: GPU `g` maps to `kvp_rank = g // TPA`, `tpa_rank = g % TPA`. The KVP group for a given `tpa_idx` contains ranks `[tpa_idx, tpa_idx + TPA, tpa_idx + 2*TPA, ...]`.
- The existing `_TP` group is **reused unchanged** for FFN All-Reduce.

### 3. Core Algorithms (`helix_attn_utils.py`)

**`attention_with_lse(q, k, v, scale, causal)`** — Computes attention explicitly (not via `F.scaled_dot_product_attention`) so we can extract the LSE values that the combining step needs. Standard SDPA doesn't return LSE. Production would use FlashAttention/FlashInfer kernels that return LSE natively.

**`helix_combine_partial_attention(partial_outputs, lse_values)`** — The mathematical core. Given K partial attention outputs {O_k} and their LSE values {L_k} from K different KV shards:

```
L_max = max(L_1, ..., L_K)                          # numerical stability
L_combined = L_max + log(sum_k exp(L_k - L_max))    # exact combined LSE
w_k = exp(L_k - L_combined)                          # rescaling weights
O_combined = sum_k w_k * O_k                          # exact output
```

This is the same online softmax trick used in FlashAttention internally. It guarantees **mathematically exact** results — no approximation.

**`helix_all_to_all_exchange(local_output, local_lse, kvp_group, kvp_size)`** — The redistribution step. Each GPU has partial attention for `H_q/TPA` heads from its KV shard. It:

1. Splits its output into KVP chunks along the head dimension (each chunk: `H_q/N` heads)
2. Calls `dist.all_to_all` within the KVP group
3. Each GPU ends up with KVP partial results for its final `H_q/N` heads — one from each KV shard

**`helix_attention_with_kvp(...)`** — Orchestrates: All-to-All -> combine -> return exact output.

### 4. Attention Backend (`helix_backend.py`)

Implements sglang's `AttentionBackend` interface (same as FlashInfer, Triton, etc.):

- **`forward_decode`**: For each request, gathers KV from the local cache, computes attention with LSE, then calls `helix_attention_with_kvp` to combine across KVP ranks. Returns the flat `(tokens, heads*dim)` tensor sglang expects.
- **`forward_extend`**: Prefill path — computes full local attention (all tokens available during prefill), stores KV to cache. Helix combining isn't needed during prefill since each GPU sees the full input.

Registered as `"helix"` in the attention registry so it's selectable via `--attention-backend helix`.

### 5. All-to-All Communication Pattern (Concrete Example)

```
4 GPUs, KVP=2, TPA=2, H_q=8 heads

GPU layout (kvp_rank, tpa_rank):
  GPU0 = (0, 0): KV shard 0, heads [0,4)
  GPU1 = (0, 1): KV shard 0, heads [4,8)
  GPU2 = (1, 0): KV shard 1, heads [0,4)
  GPU3 = (1, 1): KV shard 1, heads [4,8)

After local FlashAttention, each GPU has partial output for its heads from its shard.

All-to-All splits each GPU's 4 heads into 2 chunks (KVP=2) and exchanges:
  GPU0 sends heads [0,2) to GPU0, heads [2,4) to GPU1
  GPU2 sends heads [0,2) to GPU0, heads [2,4) to GPU1
  GPU1 sends heads [4,6) to GPU2, heads [6,8) to GPU3
  GPU3 sends heads [4,6) to GPU2, heads [6,8) to GPU3

After exchange + LSE combine:
  GPU0: exact output for heads [0,2)  (combined from shards 0,1)
  GPU1: exact output for heads [2,4)  (combined from shards 0,1)
  GPU2: exact output for heads [4,6)  (combined from shards 0,1)
  GPU3: exact output for heads [6,8)  (combined from shards 0,1)

Then W_o projection + All-Reduce -> FFN with standard TP across all 4 GPUs.
```

## Usage

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3-70B \
  --tp-size 8 \
  --helix-kvp-size 4 \
  --attention-backend helix
```

## MVP Limitations

- **KV cache memory savings**: Currently each GPU still allocates a full-size KV pool. True Helix would allocate `1/KVP` per GPU, cutting memory proportionally. This requires deeper changes to `RadixCache` and `memory_pool.py`.
- **Optimized kernels**: The MVP uses explicit matmul-based attention to extract LSE. Production would hook into FlashInfer/FlashAttention which return LSE natively.
- **CUDA graphs**: Not yet supported for the Helix backend.
- **Hop-B**: The batch-wise communication-computation overlap from the paper is excluded.

## Tests

21 unit tests (all passing) + 5 distributed tests (skip on macOS, ready for Linux multi-GPU):

- Combining correctness: 2, 4, 8 KV shards match full attention exactly
- GQA support with fewer KV heads than Q heads
- Numerical stability with extreme LSE values
- Causal attention correctness
- Config validation for valid/invalid KVP sizes
- All-to-All head redistribution pattern verification
- Communication volume independence from sequence length
- End-to-end pipeline: shard -> partial attention -> combine
