# Tensor Cache Design

This document describes the tensor-cache mechanism implemented in `tensor_cache/model.py` and how it interacts with the KV cache for long-context streaming inference.

## 1) Goal

Support five comparable inference methods:

1. Full KV cache: exact attention over all past tokens (memory grows with sequence length).
2. Sliding-window KV: exact attention over last `W` tokens only (constant KV memory).
3. Sliding-window + Tensor Cache (TC): keep exact local window `W`, and compress evicted history into fixed-size per-layer memory state.
4. StreamingLLM (`kv_mode=streaming_llm`): attention sinks + sliding window baseline.
5. Infini-attention (`kv_mode=infini`): sliding window + compressive memory baseline.

The design target is constant memory w.r.t. prompt length (for fixed model size and `W`) while recovering long-range information lost by pure windowing.

## 2) Notation and Shapes

- `B`: batch size
- `H`: number of attention heads
- `D`: head dimension (`n_embd / n_head`)
- `T`: sequence length in full forward
- `W`: KV window length (`kv_window`)
- `S`: number of TC slots per head (`tc_num_slots`)

Per layer, per head:

- Query/key/value tensors in head form:
  - `q, k, v in R^{B x H x T x D}` (full forward)
  - `q_t, k_t, v_t in R^{B x H x D}` (single streaming step)
- Tensor-cache state:
  - Single-slot single timescale (default): `A in R^{B x H x D x D}`
  - Multi-slot single timescale: `A in R^{B x H x S x D x D}`
  - Two timescales (optional): `(A_fast, A_slow)`, same shape each

## 3) Local Attention Path

Standard attention projections:

```math
[Q, K, V] = X W_{qkv}
```

with optional RoPE on `Q, K`.

### Full forward

- If `attn_window == 0`: causal full attention.
- If `attn_window == W > 0`: causal sliding mask allowing only `j in [i-W, i]`.

### Streaming step

- Maintain per-layer KV cache.
- `kv_window = 0`: append forever (unbounded).
- `kv_window = W > 0`: ring buffer with fixed capacity `W`.
- Compute attention for current token against chronologically ordered cache contents.

## 4) Ring Buffer and Eviction Semantics

For fixed window `W`:

- While cache not full: insert new `(k_t, v_t)` at next slot.
- Once full: overwrite slot `pos` and increment `pos = (pos + 1) mod W`.
- On overwrite, optionally capture evicted `(k_old, v_old)` for TC write.

This eviction event is the key compressor trigger in your method.

## 5) Tensor Cache (Fast Weights)

The implementation supports:

- Single-slot TC (`tc_num_slots=1`): exact backward-compatible behavior with the original design.
- Multi-slot TC (`tc_num_slots=S>1`): routed read/write over multiple fast-weight cells per head.

### 5.1 Read (single-slot)

Given query `q_t in R^{B x H x D}`:

```math
r_t = q_t A_t
```

implemented as batched matmul over heads.

For full forward chunk:

```math
R_{t0:t1} = Q_{t0:t1} A
```

### 5.2 Read (multi-slot routed)

When `S > 1`, each head has learned slot keys:

```math
K^{slot} \in R^{H \times S \times D}
```

Router logits from query:

```math
\ell_{t,s} = \frac{\langle \hat q_t, \hat K^{slot}_s \rangle}{\sqrt{D}\,\tau}
```

where `tau = tc_router_temp`, and hats denote optional L2 normalization used by the router.

Top-k sparse slot weights:

```math
\pi_{t,:} = \operatorname{TopKSoftmax}(\ell_{t,:}, k_r),\quad k_r = tc_read_topk
```

Memory read is a weighted mixture of slot reads:

```math
r_t = \sum_{s=1}^{S} \pi_{t,s}\,(q_t A_{t,s})
```

(`tc_read_topk >= S` reduces to dense softmax over all slots.)

### 5.3 Write rules

Let `k_w, v_w` be the write key/value (from eviction or insertion policy).

General form:

```math
A_{t+1} = \lambda A_t + \eta (k_w \otimes u_w)
```

where `lambda in (0,1)` is decay and `eta in (0,1)` is learning rate (per head, sigmoid-parameterized).

Two supported targets:

1. Outer rule:
```math
u_w = v_w
```

2. Delta rule (error-correcting, default):
```math
\hat v_w = k_w A_t,\quad
u_w = v_w - \hat v_w
```
so
```math
A_{t+1} = \lambda A_t + \eta (k_w \otimes (v_w - k_w A_t))
```

### 5.4 Write routing in multi-slot mode

When `S > 1`, only one slot is updated per write event. Route source:

- `tc_write_route="k"`: route with write key `k_w`
- `tc_write_route="q"`: route with current query `q_t` (or chunk-mean query in full forward)

Slot index:

```math
s^* = \arg\max_s \ell_{route,s} 
```

Then update only `A_{s^*}` using outer/delta rule; all other slots are unchanged.

### 5.5 Read/write normalization and scaling

Optional switches:

- Normalize keys before write (`tc_normalize_k`)
- Normalize queries before read (`tc_normalize_q`)
- Scale values by `tc_value_scale`

## 6) Eviction-Conditioned Compression Policy

In streaming mode, write source is selected by:

- `tc_write_on_evict=True` (main setting): write only evicted KV pair.
- Else if `tc_write_on_insert=True`: write current `(k_t, v_t)`.
- Else: no write this step.

The intended paper interpretation:

- Local window keeps recent tokens exact.
- TC stores a compressed summary of distant evicted context.
- In multi-slot mode, that summary is partitioned across routed slots.

## 7) Output Fusion

Let local attention output be `y_local`.
Let memory read mapped back to model dim be:

```math
m_t = W_{tc} \cdot \operatorname{MergeHeads}(r_t)
```

Final output:

```math
y_t = y_{local,t} + \sigma(g)\, m_t
```

where `g` is a learned scalar gate per layer (`tc_gate` parameter).

## 8) Full-Forward Training Approximation (`tc_use_in_full_forward`)

In non-streaming full forward, TC state is scanned in chunks of size `C = tc_chunk_size`:

1. Read each chunk with current `A`.
2. Write once per chunk using chunk means:
```math
\bar k = \frac{1}{C}\sum k,\quad \bar v = \frac{1}{C}\sum v
```
3. Update `A` with `(\bar k, \bar v)`.
   - If `tc_write_route="k"`, route with `\bar k`.
   - If `tc_write_route="q"`, route with chunk-mean query `\bar q`.

This is a practical approximation for batched training speed and memory, not identical to token-by-token streaming updates.

## 9) Optional Two-Timescale Memory

If `tc_two_timescales=True`:

- Maintain `(A_fast, A_slow)` with separate `(lambda, eta)` parameters.
- Read:
```math
r = r_{fast} + \alpha r_{slow},\quad \alpha = \sigma(\alpha_{logit})
```
- Write both states each write event.

## 10) Layer-Selective TC (`tc_layers`)

`tc_layers` controls which layers run TC:

- `-1`: all layers
- `0`: no layers
- `k > 0`: top-`k` layers only

This is a compute/quality tradeoff knob for ablations.

## 11) Complexity and Memory

### KV memory

- Full KV (`kv_window=0`): grows as `O(B H D L)` with context length `L`.
- Window/TC (`kv_window=W`): fixed `O(B H D W)` per layer.

### TC memory

- Single-slot: `O(B H D^2)` per layer (or `2x` with two timescales).
- Multi-slot: `O(B H S D^2)` per layer (or `2x` with two timescales).
- Independent of context length `L`.

### Per-token streaming compute (rough)

- Local attention over window: `O(B H W D)`
- TC read:
  - single-slot: `O(B H D^2)`
  - multi-slot: `O(B H S D^2)` plus routing overhead
- TC write (if write event):
  - single-slot: `O(B H D^2)`
  - multi-slot: selected-slot update `O(B H D^2)` plus routing overhead

So TC adds constant-with-context compute overhead, while preserving constant memory.

## 12) Config Map (Implementation)

Primary switches in `GPTConfig`:

- `kv_window`: `0` unbounded, `>0` fixed window
- `tc_enabled`: enable TC module
- `tc_write_on_evict`, `tc_write_on_insert`: write policy
- `tc_update_rule`: `"delta"` or `"outer"`
- `tc_chunk_size`: full-forward chunk scan size
- `tc_use_in_full_forward`: include TC path during full forward
- `tc_two_timescales`: enable fast+slow states
- `tc_layers`: activate TC in top-k layers
- `tc_num_slots`: number of TC slots per head (`1` = original behavior)
- `tc_read_topk`: top-k slots used for read mixture
- `tc_router_temp`: router temperature
- `tc_write_route`: route writes by `"k"` or `"q"`

Preset helper:

- `apply_kv_mode("full_kv" | "window_kv" | "streaming_llm" | "infini" | "tc")`

## 13) Practical Interpretation of Your Results

Expected signatures (and what your runs showed):

- Full KV: memory grows with context and eventually OOMs.
- Window KV: flat memory, stable speed, worse far-context quality.
- TC: near-flat memory like window KV, lower throughput than window KV, but better far-context NLL/PPL.

That is the core tradeoff this design targets.

## 14) Training Objective and Token-Level Loss

The implementation uses standard next-token cross-entropy with teacher forcing.

Given token sequence `x_{1:T}`, model logits are:

```math
z_t = f_\theta(x_{1:t}) \in \mathbb{R}^{V}
```

and targets are `x_{2:T+1}`. The loss (ignoring masked targets with value `-1`) is:

```math
\mathcal{L}(\theta) = - \frac{1}{|\mathcal{I}|} \sum_{t \in \mathcal{I}} \log p_\theta(x_{t+1}\mid x_{\le t}),
\quad p_\theta = \mathrm{softmax}(z_t)
```

where `\mathcal{I}` is the set of valid target positions.

Implementation mapping:

- `GPT.forward(..., targets)` computes logits and `F.cross_entropy(..., ignore_index=-1)`.
- Full-forward TC state is re-initialized per batch forward call (no cross-batch recurrence).

## 15) Algorithms (Implementation-Level Pseudocode)

### Algorithm 1: Streaming Step With Ring KV + Eviction-Conditioned TC

```text
Inputs:
  x_t (current token), layer states {KV_l, A_l}, window W
For each layer l:
  1) Compute q_t, k_t, v_t (with optional RoPE on q_t, k_t)
  2) Update KV ring:
       if W == 0: append (k_t, v_t)
       else:
         if ring not full: insert (k_t, v_t)
         else:
           evict (k_old, v_old) at write pointer and overwrite with (k_t, v_t)
  3) Local attention over chronological KV contents -> y_local
  4) Select TC write pair:
       if tc_write_on_evict and eviction happened: (k_w, v_w) = (k_old, v_old)
       else if tc_write_on_insert: (k_w, v_w) = (k_t, v_t)
       else: no write
  5) TC read:
       single-slot: r_t = q_t A_l
       multi-slot: r_t = sum_s pi_{t,s} (q_t A_{l,s})
  6) TC write (optional):
       single-slot:
         A_l <- lambda A_l + eta (k_w \otimes u_w)
       multi-slot:
         choose slot s* by router(route_vec), route_vec in {k_w, q_t}
         update only A_{l,s*}
       u_w = v_w                    (outer)
       u_w = v_w - k_w A_selected   (delta)
  7) Fuse:
       y = y_local + sigmoid(g_l) * W_tc_l(merge_heads(r_t))
Return updated states and logits
```

### Algorithm 2: Full-Forward TC Scan (Chunked Approximation)

```text
Inputs:
  Q, K, V for one layer, initial A, chunk size C
For chunks [t0:t1):
  1) Read memory for all queries in chunk:
       single-slot: R[:, :, t0:t1, :] = Q[:, :, t0:t1, :] @ A
       multi-slot: routed mixture across slots
  2) Compute chunk summaries:
       k_bar = mean_t K[:, :, t0:t1, :]
       v_bar = mean_t V[:, :, t0:t1, :]
       q_bar = mean_t Q[:, :, t0:t1, :]   (used if tc_write_route="q")
  3) Update A once with (k_bar, v_bar) using outer/delta rule
     and routed selected-slot update when S > 1
Fuse R through W_tc and gate with local attention output
```

## 16) Train/Inference Mismatch (Important for Reviewers)

There is a deliberate mismatch:

- Streaming inference updates TC at token granularity (or eviction events).
- Full-forward training uses chunk-level summary writes (`tc_chunk_size`).

Why this exists:

- Token-by-token recurrence in full batched training is expensive.
- Chunked scan offers a practical approximation with good throughput.

What to report:

1. Ablate `tc_chunk_size` (e.g., 8, 16, 32, 64, 128).
2. Show sensitivity of far-context metrics and speed.
3. State this approximation explicitly in the paper body (not only appendix).

## 17) Evaluation Protocol (Systems + Quality)

### 17.1 Systems metrics (KV/OOM story)

For each mode (`full_kv`, `window_kv`, `tc`) and prompt length `L`:

1. Construct random prompt of length `L`.
2. Run streaming prefill (`stream_prefill`) once.
3. Record:
   - status (`ok`/`oom`)
   - peak memory (`torch.cuda.max_memory_allocated`)
   - throughput (`L / elapsed_seconds`)

Report table:

- `mode, length, status, peak_gb, prefill_tok_s`

### 17.2 Quality metrics (long-context story)

Use teacher-forced streaming evaluation on validation sequences.

Define:

- all-context NLL: all positions
- far-context NLL: positions `t >= W`
- very-far NLL: positions `t >= 4W` (or another fixed multiplier)

Convert with `ppl = exp(nll)`.

Primary claim target:

- At fixed `W`, `tc` should improve `nll_far` and `nll_vfar` vs `window_kv`.
