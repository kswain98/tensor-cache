"""
TensorCache: constant-memory long-context GPT with eviction-conditioned tensor cache.

Core idea:
  - Keep only a fixed sliding-window KV cache of length W for LOCAL attention.
  - Compress anything older than W into a fixed-size "Tensor Cache" state per layer:
      A ∈ R^{B × H × D × D}  (or A ∈ R^{B × H × S × D × D} for multi-slot TC)
    Read:   r = q @ A
    Write:  A <- decay * A + lr * (k ⊗ update_target)
            where update_target is:
              - "outer": v
              - "delta": (v - k@A)  (error-correcting / delta-rule)

Key knob:
  - tc_write_on_evict=True: only write evicted KV entries into tensor cache.
    => explicit "KV-cache compressor": local window keeps exact recent tokens,
       tensor cache summarizes the distant past.

This file provides:
  - GPTConfig
  - GPT (forward for training + streaming APIs for ultra-long prompts)
  - Constant-memory generate() via streaming

Baselines:
  - Full KV-cache: use_kv_cache=True, kv_window=0,  tc_enabled=False
  - Window-only :  use_kv_cache=True, kv_window=W,  tc_enabled=False
  - Ours        :  use_kv_cache=True, kv_window=W,  tc_enabled=True, tc_write_on_evict=True
  - StreamingLLM:  apply_kv_mode("streaming_llm") -- attention sinks + sliding window, no TC
"""

from __future__ import annotations

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional fast sliding-window attention (PyTorch 2.5+). Numerically equivalent
# to the bool-mask SDPA path (verified max|Δ| ~2e-6 in fp32 across W=128/256,
# T=256/512/1024, sinks 0/4), ~2.5-3.5x faster with a cached block mask, and
# torch.compile-friendly (the bool-mask path is what forced TC training to run
# eager). Falls back to the bool-mask path if flex_attention is unavailable.
try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask as _create_block_mask,
    )
    _HAS_FLEX = True
except Exception:  # pragma: no cover - depends on torch version
    _flex_attention = None
    _create_block_mask = None
    _HAS_FLEX = False

_FLEX_MASK_CACHE: Dict[Any, Any] = {}


def _disable_dynamo(fn):
    """Run `fn` outside torch.compile so create_block_mask is built eagerly
    (recommended) and the resulting BlockMask is passed into the compiled
    flex_attention call. No-op if torch._dynamo is unavailable."""
    dynamo = getattr(torch, "_dynamo", None)
    disable = getattr(dynamo, "disable", None) if dynamo is not None else None
    return disable(fn) if callable(disable) else fn


@_disable_dynamo
def _sliding_window_block_mask(T: int, window: int, n_sinks: int, device):
    """Cached BlockMask with the EXACT semantics of the bool-mask path:
        disallow = ((k > q) | (k < q - window)) & (k >= n_sinks)
        allowed  = ~disallow
    """
    key = (int(T), int(window), int(n_sinks), str(device))
    bm = _FLEX_MASK_CACHE.get(key)
    if bm is None:
        ns, w = int(n_sinks), int(window)

        def mask_mod(b, h, q_idx, k_idx):
            causal_win = (k_idx <= q_idx) & (k_idx >= q_idx - w)
            if ns > 0:
                return causal_win | (k_idx < ns)
            return causal_win

        bm = _create_block_mask(mask_mod, B=None, H=None,
                                Q_LEN=int(T), KV_LEN=int(T), device=device)
        _FLEX_MASK_CACHE[key] = bm
    return bm


# ----------------------------
# Config
# ----------------------------

TCUpdateRule = Literal["outer", "delta", "wedge"]

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 1024

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    dropout: float = 0.0
    bias: bool = False

    # Position encoding
    use_rope: bool = True
    rope_base: int = 10000
    rope_scaling_factor: float = 1.0  # NTK-aware RoPE scaling for context > block_size

    # KV cache (streaming/generation)
    use_kv_cache: bool = True
    kv_window: int = 512  # 0 => unbounded (full KV-cache)

    # Attention sinks (StreamingLLM baseline)
    kv_num_sinks: int = 0   # number of initial tokens to keep permanently (0 = disabled)

    # Tensor Cache
    tc_enabled: bool = True
    tc_layers: int = -1                      # -1 => all layers, 0 => none, k>0 => top-k layers
    tc_update_rule: TCUpdateRule = "delta"   # "outer", "delta", or "wedge"
    tc_chunk_size: int = 64                  # used in full forward scan
    tc_decay_init: float = 0.995             # sigmoid-parameterized
    tc_lr_init: float = 0.05                 # sigmoid-parameterized
    tc_gate_init: float = -2.0               # sigmoid(gate) initial contribution (~0.12)
    tc_normalize_k: bool = True
    tc_normalize_q: bool = False
    tc_value_scale: float = 1.0

    # Key knob: write policy
    tc_write_on_evict: bool = True           # compress evicted KV into tensor cache
    tc_write_on_insert: bool = False         # also write new tokens (usually False if write_on_evict=True)

    # Optional: two-timescale memory (still constant size, but 2 states)
    tc_two_timescales: bool = False
    tc_decay_slow_init: float = 0.9995
    tc_lr_slow_init: float = 0.01
    tc_alpha_init: float = 0.5               # mixing of slow read: r = r_fast + alpha * r_slow

    # Optional multi-slot tensor cache (S slots per head)
    tc_num_slots: int = 1                    # 1 keeps current behavior
    tc_read_topk: int = 1                    # top-k slots mixed at read time
    tc_router_temp: float = 1.0              # router temperature for slot softmax
    tc_write_route: Literal["k", "q"] = "k"  # route writes by evicted key or current query

    # Freeze decay/lr parameters (for ablation: learned vs fixed)
    tc_freeze_decay_lr: bool = False

    # V2.2: per-token query-conditional fusion gate (instead of scalar per-layer gate).
    # When True, replaces sigmoid(g)*mem with sigmoid(W_g x_t)*mem so the model can
    # selectively turn TC off when local attention already covers the prediction.
    tc_per_token_gate: bool = False

    # V2.3: read-time normalization (mLSTM/Infini-style z vector).
    # When True, maintains a z vector alongside A and divides reads by max(|q.z|, eps),
    # bounding the L2 read magnitude as A accumulates writes.
    tc_normalize_read: bool = False
    tc_normalize_read_eps: float = 1.0  # like Infini's max(., 1) clamp

    # Infini-attention baseline (Munkhdalai et al., 2024)
    infini_enabled: bool = False
    infini_update_rule: str = "delta"    # "linear" or "delta"
    infini_segment_size: int = 256  # must be < block_size for memory to be used during training

    # Apply TC in full forward (training/eval)
    tc_use_in_full_forward: bool = True

    def apply_kv_mode(self, mode: Literal["full_kv", "window_kv", "tc", "streaming_llm", "infini"]) -> None:
        """
        Canonicalize config flags for the KV/TC experiment settings.
        Resets all mechanism flags first to prevent stale state when switching modes.
        """
        # Reset all mechanism flags to safe defaults
        self.use_kv_cache = True
        self.tc_enabled = False
        self.infini_enabled = False
        self.kv_num_sinks = 0

        if mode == "full_kv":
            self.kv_window = 0
            return

        if mode == "window_kv":
            if self.kv_window <= 0:
                raise ValueError("window_kv mode requires kv_window > 0.")
            return

        if mode == "tc":
            if self.kv_window <= 0:
                raise ValueError("tc mode requires kv_window > 0.")
            self.tc_enabled = True
            self.tc_write_on_evict = True
            return

        if mode == "streaming_llm":
            if self.kv_window <= 0:
                raise ValueError("streaming_llm mode requires kv_window > 0.")
            self.kv_num_sinks = 4    # StreamingLLM default: 4 sink tokens
            return

        if mode == "infini":
            if self.kv_window <= 0:
                raise ValueError("infini mode requires kv_window > 0.")
            self.infini_enabled = True
            return

        raise ValueError(f"Unknown KV mode: {mode}")


# ----------------------------
# Helpers
# ----------------------------

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RotaryEmbedding(nn.Module):
    """Minimal RoPE for q/k with optional NTK-aware scaling."""
    def __init__(self, dim: int, base: int = 10000, scaling_factor: float = 1.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be even."
        # NTK-aware scaling: increase base to spread out frequencies for longer contexts
        if scaling_factor > 1.0:
            base = base * scaling_factor ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_cos_sin(self, positions: torch.Tensor, device, dtype):
        freqs = torch.einsum("...t,d->...td", positions.to(device=device), self.inv_freq.to(device=device))
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        return cos, sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q,k: [B, H, T, D]
        positions: [T] or [B,T]
        """
        cos, sin = self._build_cos_sin(positions, device=q.device, dtype=q.dtype)

        if cos.dim() == 2:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        else:
            cos = cos[:, None, :, :]
            sin = sin[:, None, :, :]

        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)

        q_out = (q * cos) + (self._rotate_half(q) * sin)
        k_out = (k * cos) + (self._rotate_half(k) * sin)
        return q_out, k_out


# ----------------------------
# Fast-Weight Tensor Cache
# ----------------------------

TCState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]  # A or (A_fast, A_slow), with optional slot axis

class FastWeightTensorCache(nn.Module):
    """
    Per-layer fixed state:
      - single-timescale:
          A [B,H,D,D] (default) or A [B,H,S,D,D] if tc_num_slots>1
      - two-timescale:
          (A_fast, A_slow), each shaped as above

    Read:
      step: r = q @ A
      full: r = q @ A (batched across time)

    Write:
      Outer: A <- decay*A + lr*(k ⊗ v)
      Delta: A <- decay*A + lr*(k ⊗ (v - k@A))   (error-correcting)
    """
    def __init__(self, n_head: int, head_dim: int, cfg: GPTConfig):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.chunk_size = max(1, int(cfg.tc_chunk_size))

        self.update_rule: TCUpdateRule = cfg.tc_update_rule
        self.normalize_k = bool(cfg.tc_normalize_k)
        self.normalize_q = bool(cfg.tc_normalize_q)
        self.value_scale = float(cfg.tc_value_scale)

        self.num_slots = max(1, int(cfg.tc_num_slots))
        self.read_topk = max(1, min(self.num_slots, int(cfg.tc_read_topk)))
        self.router_temp = max(1e-4, float(cfg.tc_router_temp))
        self.write_route = str(cfg.tc_write_route)
        if self.write_route not in ("k", "q"):
            raise ValueError(f"tc_write_route must be 'k' or 'q', got: {self.write_route}")
        if self.num_slots > 1:
            # Per-head slot keys used for lightweight routing.
            self.slot_keys = nn.Parameter(torch.randn(n_head, self.num_slots, head_dim) * 0.02)
        else:
            self.register_parameter("slot_keys", None)

        self.two_timescales = bool(cfg.tc_two_timescales)

        # V2.3: optional read-time normalization (mLSTM/Infini-style z vector).
        # Maintains z alongside A; reads divide by max(|q.z|, normalize_read_eps).
        # Currently supported only for single-timescale, single-slot.
        self.normalize_read = bool(getattr(cfg, "tc_normalize_read", False))
        self.normalize_read_eps = float(getattr(cfg, "tc_normalize_read_eps", 1.0))
        if self.normalize_read and (self.two_timescales or self.num_slots > 1):
            raise ValueError(
                "tc_normalize_read=True is only supported for single-timescale, "
                "single-slot Tensor Cache (set tc_two_timescales=False, tc_num_slots=1)."
            )

        # Fast params (learned per head, or frozen for ablation)
        freeze = bool(cfg.tc_freeze_decay_lr)
        decay_fast = torch.full((n_head,), float(cfg.tc_decay_init))
        lr_fast = torch.full((n_head,), float(cfg.tc_lr_init))
        logit_decay_fast = torch.log(decay_fast / (1.0 - decay_fast)).clamp(-10, 10)
        logit_lr_fast = torch.log(lr_fast / (1.0 - lr_fast)).clamp(-10, 10)
        if freeze:
            self.register_buffer("logit_decay_fast", logit_decay_fast)
            self.register_buffer("logit_lr_fast", logit_lr_fast)
        else:
            self.logit_decay_fast = nn.Parameter(logit_decay_fast)
            self.logit_lr_fast = nn.Parameter(logit_lr_fast)

        if self.two_timescales:
            decay_slow = torch.full((n_head,), float(cfg.tc_decay_slow_init))
            lr_slow = torch.full((n_head,), float(cfg.tc_lr_slow_init))
            logit_decay_slow = torch.log(decay_slow / (1.0 - decay_slow)).clamp(-10, 10)
            logit_lr_slow = torch.log(lr_slow / (1.0 - lr_slow)).clamp(-10, 10)
            if freeze:
                self.register_buffer("logit_decay_slow", logit_decay_slow)
                self.register_buffer("logit_lr_slow", logit_lr_slow)
            else:
                self.logit_decay_slow = nn.Parameter(logit_decay_slow)
                self.logit_lr_slow = nn.Parameter(logit_lr_slow)

            alpha = torch.full((n_head,), float(cfg.tc_alpha_init))
            self.logit_alpha = nn.Parameter(torch.log(alpha / (1.0 - alpha)).clamp(-10, 10))

    def init_state(self, B: int, device, dtype) -> TCState:
        if self.num_slots > 1:
            A = torch.zeros(B, self.n_head, self.num_slots, self.head_dim, self.head_dim, device=device, dtype=dtype)
        else:
            A = torch.zeros(B, self.n_head, self.head_dim, self.head_dim, device=device, dtype=dtype)
        if not self.two_timescales:
            if self.normalize_read:
                # State is (A, z); z has shape [B, H, D].
                z = torch.zeros(B, self.n_head, self.head_dim, device=device, dtype=dtype)
                return (A, z)
            return A
        A2 = torch.zeros_like(A)
        return (A, A2)

    def _split_state(self, state):
        """Return (A, z) where z is None when normalize_read is off.

        Centralizes the (A, z) tuple unpacking so callers don't have to branch.
        """
        if self.normalize_read and isinstance(state, tuple) and len(state) == 2 \
                and not self.two_timescales:
            return state[0], state[1]
        return state, None

    def _sigmoid_param(self, logit: torch.Tensor, dtype, device) -> torch.Tensor:
        return torch.sigmoid(logit).to(device=device, dtype=dtype)

    def _maybe_norm(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def _router_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,H,D] or [B,H,T,D]
        returns logits over slots: [B,H,S] or [B,H,T,S]
        """
        if self.num_slots <= 1:
            if x.dim() == 3:
                return torch.zeros(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
            return torch.zeros(x.shape[0], x.shape[1], x.shape[2], 1, device=x.device, dtype=x.dtype)

        assert self.slot_keys is not None
        keys = self._maybe_norm(self.slot_keys.to(device=x.device, dtype=x.dtype))
        x_norm = self._maybe_norm(x)
        scale = 1.0 / math.sqrt(self.head_dim)
        if x.dim() == 3:
            logits = torch.einsum("bhd,hsd->bhs", x_norm, keys)
        else:
            logits = torch.einsum("bhtd,hsd->bhts", x_norm, keys)
        return logits * (scale / self.router_temp)

    def _router_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [..., S]
        returns sparse/soft weights with top-k support.
        """
        S = logits.shape[-1]
        if S == 1:
            return torch.ones_like(logits)
        if self.read_topk >= S:
            return F.softmax(logits, dim=-1)

        topv, topi = torch.topk(logits, k=self.read_topk, dim=-1)
        topw = F.softmax(topv, dim=-1)
        weights = torch.zeros_like(logits)
        weights.scatter_(-1, topi, topw)
        return weights

    def _route_slot_index(self, route_vec: torch.Tensor) -> torch.Tensor:
        """
        route_vec: [B,H,D]
        returns slot indices [B,H]
        """
        logits = self._router_logits(route_vec)
        return torch.argmax(logits, dim=-1)

    # ---- reads ----
    def _read_step(self, q: torch.Tensor, A: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        # q: [B,H,D], A: [B,H,D,D] or [B,H,S,D,D] -> [B,H,D]
        if A.dim() == 4:
            num = torch.matmul(q.unsqueeze(-2), A).squeeze(-2)         # [B,H,D]
            if z is not None:
                # mLSTM/Infini-style normalization: divide by max(|q.z|, eps)
                denom = torch.einsum("bhd,bhd->bh", q, z).abs()         # [B,H]
                denom = denom.clamp(min=self.normalize_read_eps).unsqueeze(-1)  # [B,H,1]
                return num / denom
            return num

        # Multi-slot read: weighted mixture of slot reads.
        weights = self._router_weights(self._router_logits(q))         # [B,H,S]
        slot_reads = torch.matmul(q.unsqueeze(2).unsqueeze(-2), A).squeeze(-2)  # [B,H,S,D]
        return (weights.unsqueeze(-1) * slot_reads).sum(dim=2)

    def _read_full(self, q: torch.Tensor, A: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        # q: [B,H,T,D], A: [B,H,D,D] or [B,H,S,D,D] -> [B,H,T,D]
        if A.dim() == 4:
            num = torch.matmul(q, A)                                   # [B,H,T,D]
            if z is not None:
                # z is [B,H,D], q is [B,H,T,D] -> denom is [B,H,T]
                denom = torch.einsum("bhtd,bhd->bht", q, z).abs()
                denom = denom.clamp(min=self.normalize_read_eps).unsqueeze(-1)  # [B,H,T,1]
                return num / denom
            return num

        weights = self._router_weights(self._router_logits(q))         # [B,H,T,S]
        slot_reads = torch.einsum("bhtd,bhsde->bhtse", q, A)           # [B,H,T,S,D]
        return torch.einsum("bhts,bhtsd->bhtd", weights, slot_reads)

    # ---- writes ----
    def _write_single(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        A: torch.Tensor,
        decay: torch.Tensor,
        lr: torch.Tensor,
        route_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        k,v: [B,H,D]
        A:   [B,H,D,D] or [B,H,S,D,D]
        decay, lr: [1,H,1,1]
        """
        if A.dim() == 4:
            if self.update_rule == "outer":
                outer = k.unsqueeze(-1) * v.unsqueeze(-2)
                return decay * A + lr * outer

            if self.update_rule == "wedge":
                # Antisymmetric (exterior-algebra) write: A <- decay*A + lr*(k v^T - v k^T).
                # Same per-step storage as the rank-1 outer product, but with the
                # antisymmetric structure of the wedge product k /\ v.
                outer = (
                    k.unsqueeze(-1) * v.unsqueeze(-2)
                    - v.unsqueeze(-1) * k.unsqueeze(-2)
                )
                return decay * A + lr * outer

            # delta-rule: write residual v - (k@A)
            v_hat = torch.matmul(k.unsqueeze(-2), A).squeeze(-2)  # [B,H,D]
            err = (v - v_hat)
            outer = k.unsqueeze(-1) * err.unsqueeze(-2)
            return decay * A + lr * outer

        # Multi-slot write: route and update only selected slot per (B,H).
        if route_vec is None:
            route_vec = k
        B, H, S, D, _ = A.shape

        # Straight-through Gumbel-softmax for differentiable write routing.
        # _router_logits bakes in scale/router_temp; undo the temp so we can
        # pass it explicitly to gumbel_softmax as tau.
        route_logits = self._router_logits(route_vec) * self.router_temp  # [B,H,S]
        soft_weights = F.gumbel_softmax(route_logits, tau=self.router_temp, hard=True, dim=-1)  # [B,H,S]

        # Weighted sum over slots for the selected A (hard one-hot in forward).
        A_sel = torch.einsum("bhs,bhsde->bhde", soft_weights, A)  # [B,H,D,D]

        if self.update_rule == "outer":
            outer = k.unsqueeze(-1) * v.unsqueeze(-2)  # [B,H,D,D]
        elif self.update_rule == "wedge":
            outer = (
                k.unsqueeze(-1) * v.unsqueeze(-2)
                - v.unsqueeze(-1) * k.unsqueeze(-2)
            )  # [B,H,D,D]
        else:
            v_hat = torch.matmul(k.unsqueeze(-2), A_sel).squeeze(-2)
            update_target = (v - v_hat)
            outer = k.unsqueeze(-1) * update_target.unsqueeze(-2)  # [B,H,D,D]

        mask = soft_weights.unsqueeze(-1).unsqueeze(-1)  # [B,H,S,1,1]
        decay5 = decay.unsqueeze(2)  # [1,H,1,1,1]
        lr5 = lr.unsqueeze(2)        # [1,H,1,1,1]

        # Update only routed slot: A_s <- decay*A_s + lr*outer ; other slots unchanged.
        selected_scale = 1.0 + mask * (decay5 - 1.0)
        return A * selected_scale + lr5 * outer.unsqueeze(2) * mask

    # ---- public APIs ----
    def forward_step(
        self,
        q: torch.Tensor,                  # [B,H,D] (read query = current token query)
        write_k: Optional[torch.Tensor],  # [B,H,D] or None
        write_v: Optional[torch.Tensor],  # [B,H,D] or None
        state: TCState
    ) -> Tuple[torch.Tensor, TCState]:
        if self.normalize_q:
            q = self._maybe_norm(q)

        if not self.two_timescales:
            A, z = self._split_state(state)
            mem = self._read_step(q, A, z)

            if write_k is None or write_v is None:
                return mem, ((A, z) if self.normalize_read else A)

            k = write_k
            if self.normalize_k:
                k = self._maybe_norm(k)
            v = write_v * self.value_scale

            decay_fast = self._sigmoid_param(self.logit_decay_fast, dtype=q.dtype, device=q.device).view(1, self.n_head, 1, 1)
            lr_fast = self._sigmoid_param(self.logit_lr_fast, dtype=q.dtype, device=q.device).view(1, self.n_head, 1, 1)

            route_vec = k if self.write_route == "k" else q
            A_new = self._write_single(k, v, A, decay_fast, lr_fast, route_vec=route_vec)
            if self.normalize_read:
                # z update: z <- decay*z + lr*k  (matches the per-token recurrence on A)
                decay_z = decay_fast.view(1, self.n_head, 1)
                lr_z    = lr_fast.view(1, self.n_head, 1)
                z_new = decay_z * z + lr_z * k
                return mem, (A_new, z_new)
            return mem, A_new

        # two-timescale
        A_fast, A_slow = state  # type: ignore
        mem_fast = self._read_step(q, A_fast)
        mem_slow = self._read_step(q, A_slow)
        alpha = self._sigmoid_param(self.logit_alpha, dtype=q.dtype, device=q.device).view(1, self.n_head, 1)
        mem = mem_fast + alpha * mem_slow

        if write_k is None or write_v is None:
            return mem, (A_fast, A_slow)

        k = write_k
        if self.normalize_k:
            k = self._maybe_norm(k)
        v = write_v * self.value_scale

        decay_fast = self._sigmoid_param(self.logit_decay_fast, dtype=q.dtype, device=q.device).view(1, self.n_head, 1, 1)
        lr_fast = self._sigmoid_param(self.logit_lr_fast, dtype=q.dtype, device=q.device).view(1, self.n_head, 1, 1)
        route_vec = k if self.write_route == "k" else q
        A_fast_new = self._write_single(k, v, A_fast, decay_fast, lr_fast, route_vec=route_vec)

        decay_slow = self._sigmoid_param(self.logit_decay_slow, dtype=q.dtype, device=q.device).view(1, self.n_head, 1, 1)
        lr_slow = self._sigmoid_param(self.logit_lr_slow, dtype=q.dtype, device=q.device).view(1, self.n_head, 1, 1)
        A_slow_new = self._write_single(k, v, A_slow, decay_slow, lr_slow, route_vec=route_vec)

        return mem, (A_fast_new, A_slow_new)

    def forward_full(
        self,
        q: torch.Tensor,     # [B,H,T,D]
        k: torch.Tensor,     # [B,H,T,D]
        v: torch.Tensor,     # [B,H,T,D]
        state: TCState
    ) -> Tuple[torch.Tensor, TCState]:
        """
        Chunked scan for full forward:
          - read for tokens in chunk using current state
          - write once per chunk using mean(k), mean(v)
        Returns:
          mem: [B,H,T,D]
          state_new
        """
        B, H, T, D = q.shape
        if self.normalize_q:
            q = self._maybe_norm(q)
        if self.normalize_k:
            k = self._maybe_norm(k)
        if self.value_scale != 1.0:
            v = v * self.value_scale

        mem = torch.empty_like(q)

        decay_fast = self._sigmoid_param(self.logit_decay_fast, dtype=q.dtype, device=q.device).view(1, H, 1, 1)
        lr_fast = self._sigmoid_param(self.logit_lr_fast, dtype=q.dtype, device=q.device).view(1, H, 1, 1)

        cs = self.chunk_size

        if not self.two_timescales:
            A, z = self._split_state(state)
            # Per-head decay/lr scalars in shapes that broadcast cleanly.
            decay_h = decay_fast.view(1, H, 1, 1)        # [1,H,1,1] for matrix update
            lr_h    = lr_fast.view(1, H, 1, 1)
            decay_h_for_weights = decay_fast.view(1, H, 1)  # [1,H,1] for [B,H,T_chunk] weights

            for t0 in range(0, T, cs):
                t1 = min(t0 + cs, T)
                T_chunk = t1 - t0
                mem[:, :, t0:t1, :] = self._read_full(q[:, :, t0:t1, :], A, z)

                k_chunk = k[:, :, t0:t1, :]  # [B,H,T_chunk,D]
                v_chunk = v[:, :, t0:t1, :]

                # Per-token decay weights inside this chunk:
                #   weights[t] = decay^(T_chunk - 1 - t)
                # Implements the parallel-scan equivalent of T_chunk sequential
                # per-token writes A_t = decay*A_{t-1} + lr*(k_t (x) v_t).
                idx = torch.arange(T_chunk - 1, -1, -1, device=k.device, dtype=k.dtype)  # [T_chunk]
                weights = decay_h_for_weights ** idx.view(1, 1, T_chunk)  # [1,H,T_chunk]
                weights = weights.unsqueeze(-1)                            # [1,H,T_chunk,1]

                if self.update_rule == "outer":
                    k_weighted = k_chunk * weights
                    contribution = torch.einsum("bhtd,bhte->bhde", k_weighted, v_chunk)
                elif self.update_rule == "wedge":
                    # Antisymmetric write: contribution = sum_t w_t (k_t v_t^T - v_t k_t^T).
                    # Each token's wedge product is weighted by the same per-token
                    # decay weight w_t = decay^(T_chunk - 1 - t), exactly matching
                    # the per-token recurrence A_t = decay*A_{t-1} + lr*(k_t v_t^T - v_t k_t^T).
                    k_weighted = k_chunk * weights
                    v_weighted = v_chunk * weights
                    kv_contrib = torch.einsum("bhtd,bhte->bhde", k_weighted, v_chunk)
                    vk_contrib = torch.einsum("bhtd,bhte->bhde", v_weighted, k_chunk)
                    contribution = kv_contrib - vk_contrib
                else:
                    # Delta-rule with chunk-start A: small approximation
                    # vs strict per-token recurrence (mLSTM-style; tractable in parallel).
                    v_hat = torch.einsum("bhtd,bhde->bhte", k_chunk, A)
                    err = v_chunk - v_hat
                    k_weighted = k_chunk * weights
                    contribution = torch.einsum("bhtd,bhte->bhde", k_weighted, err)

                # Chunk-level decay on the previous A: A_new = decay^T * A_old + lr * contribution
                decay_chunk = decay_h ** T_chunk
                A = decay_chunk * A + lr_h * contribution

                # If normalize_read is on, update z analogously: z <- decay^T * z + lr * sum_t weights[t] * k_t.
                # This is the parallel-scan equivalent of z_t = decay*z_{t-1} + lr*k_t.
                if z is not None:
                    decay_chunk_z = decay_chunk.view(1, H, 1, 1).squeeze(-1)  # [1,H,1]
                    z_contribution = (k_chunk * weights).sum(dim=2)  # [B,H,D]
                    z = decay_chunk_z * z + lr_h.view(1, H, 1) * z_contribution
            return mem, ((A, z) if self.normalize_read else A)

        # two-timescale
        A_fast, A_slow = state  # type: ignore
        decay_slow = self._sigmoid_param(self.logit_decay_slow, dtype=q.dtype, device=q.device).view(1, H, 1, 1)
        lr_slow = self._sigmoid_param(self.logit_lr_slow, dtype=q.dtype, device=q.device).view(1, H, 1, 1)
        alpha = self._sigmoid_param(self.logit_alpha, dtype=q.dtype, device=q.device).view(1, H, 1)

        for t0 in range(0, T, cs):
            t1 = min(t0 + cs, T)
            q_chunk = q[:, :, t0:t1, :]  # [B,H,cs,D]
            r_fast = self._read_full(q_chunk, A_fast)
            r_slow = self._read_full(q_chunk, A_slow)
            mem[:, :, t0:t1, :] = r_fast + alpha.unsqueeze(2) * r_slow

            k_bar = k[:, :, t0:t1, :].mean(dim=2)
            v_bar = v[:, :, t0:t1, :].mean(dim=2)
            route_bar = k_bar if self.write_route == "k" else q[:, :, t0:t1, :].mean(dim=2)
            A_fast = self._write_single(k_bar, v_bar, A_fast, decay_fast, lr_fast, route_vec=route_bar)
            A_slow = self._write_single(k_bar, v_bar, A_slow, decay_slow, lr_slow, route_vec=route_bar)

        return mem, (A_fast, A_slow)


# ----------------------------
# Attention with streaming KV window + Tensor Cache
# ----------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0

        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.attn_dropout = nn.Dropout(cfg.dropout)

        # RoPE
        self.use_rope = bool(cfg.use_rope)
        self.rope = RotaryEmbedding(self.head_dim, base=cfg.rope_base, scaling_factor=cfg.rope_scaling_factor) if self.use_rope else None

        # Tensor Cache (mutually exclusive with Infini-attention)
        self.tc_enabled = bool(cfg.tc_enabled) and not bool(cfg.infini_enabled)
        if self.tc_enabled:
            self.tc = FastWeightTensorCache(self.n_head, self.head_dim, cfg)
            self.tc_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
            self.tc_per_token_gate = bool(getattr(cfg, "tc_per_token_gate", False))
            if self.tc_per_token_gate:
                # V2.2: per-token query-conditional gate, sigmoid(W_g x_t).
                # Initialize so the gate starts ~ sigmoid(tc_gate_init), matching scalar-gate init.
                self.tc_gate_proj = nn.Linear(cfg.n_embd, 1, bias=True)
                nn.init.zeros_(self.tc_gate_proj.weight)
                nn.init.constant_(self.tc_gate_proj.bias, float(cfg.tc_gate_init))
                self.tc_gate = None
            else:
                self.tc_gate = nn.Parameter(torch.tensor([float(cfg.tc_gate_init)]))  # scalar per layer
                self.tc_gate_proj = None

        # Infini-attention baseline (Munkhdalai et al., 2024)
        self.infini_enabled = bool(cfg.infini_enabled)
        if self.infini_enabled:
            from baselines.infini_attention import InfiniMemory
            self.infini = InfiniMemory(
                self.n_head, self.head_dim,
                update_rule=cfg.infini_update_rule,
                segment_size=cfg.infini_segment_size,
            )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_rope:
            return q, k
        return self.rope(q, k, pos)  # calls RotaryEmbedding.forward


    # ---------- full forward (training / normal eval) ----------
    def forward(
        self,
        x: torch.Tensor,
        tc_state: Optional[TCState] = None,
        attn_window: Optional[int] = None,
        tc_active: bool = True,
    ) -> Tuple[torch.Tensor, Optional[TCState]]:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = self._split_heads(q)  # [B,H,T,D]
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.use_rope:
            pos = torch.arange(T, device=x.device)
            q, k = self._apply_rope(q, k, pos)

        if attn_window is None:
            attn_window = 0

        if attn_window and attn_window < T:
            n_sinks = self.cfg.kv_num_sinks
            if _HAS_FLEX:
                # Fast, torch.compile-friendly sliding-window(+sinks) attention,
                # numerically equivalent to the bool-mask SDPA fallback below.
                # Attention-weight dropout is intentionally not applied on this
                # path (standard for windowed-attention LMs; residual dropout
                # still regularizes). Verified to match the bool-mask path's
                # val loss within seed noise.
                block_mask = _sliding_window_block_mask(T, attn_window, n_sinks, x.device)
                attn_out = _flex_attention(q, k, v, block_mask=block_mask)
            else:
                idx = torch.arange(T, device=x.device)
                i = idx[:, None]
                j = idx[None, :]
                disallow = (j > i) | (j < (i - attn_window))
                # Attention sinks: always allow attending to the first kv_num_sinks positions
                if n_sinks > 0:
                    disallow = disallow & (j >= n_sinks)
                # PyTorch bool attn_mask: True = ALLOWED, so negate disallow
                attn_out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=~disallow,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False
                )
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )

        # Infini-attention: per-head gated interpolation between memory and local attention
        next_tc_state = tc_state
        if self.infini_enabled and self.cfg.infini_enabled:
            if next_tc_state is None:
                next_tc_state = self.infini.init_state(B, x.device, x.dtype)
            mem_heads, next_tc_state = self.infini.forward_full(q, k, v, next_tc_state)
            gate = self.infini.gate().to(device=attn_out.device, dtype=attn_out.dtype)
            gate = gate.view(1, self.n_head, 1, 1)
            attn_out = gate * mem_heads + (1 - gate) * attn_out

        y = self._merge_heads(attn_out)   # [B,T,C]
        y = self.resid_dropout(self.c_proj(y))

        if tc_active and self.tc_enabled and self.cfg.tc_enabled and self.cfg.tc_use_in_full_forward:
            if next_tc_state is None:
                next_tc_state = self.tc.init_state(B, x.device, x.dtype)
            mem_heads, next_tc_state = self.tc.forward_full(q, k, v, next_tc_state)  # [B,H,T,D]
            mem = self._merge_heads(mem_heads)  # [B,T,C]
            mem = self.tc_proj(mem)
            if self.tc_per_token_gate:
                gate = torch.sigmoid(self.tc_gate_proj(x))  # [B,T,1]
            else:
                gate = torch.sigmoid(self.tc_gate)
            y = y + gate * mem

        return y, next_tc_state

    # ---------- streaming single-token step ----------
    def stream_step(
        self,
        x_t: torch.Tensor,                      # [B,1,C]
        pos_t: int,
        kv_cache: Optional[Dict[str, Any]],      # ring buffer or unbounded
        tc_state: Optional[TCState],
        kv_window: int,
        tc_active: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Optional[TCState]]:
        B, T, C = x_t.shape
        assert T == 1

        qkv = self.c_attn(x_t)
        q, k, v = qkv.split(C, dim=2)

        q = self._split_heads(q)  # [B,H,1,D]
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.use_rope:
            pos = torch.tensor([pos_t], device=x_t.device)
            q, k = self._apply_rope(q, k, pos)

        # --- KV cache update (and optional eviction extraction) ---
        if kv_cache is None:
            kv_cache = {"k": None, "v": None, "len": 0, "pos": 0}

        k_evict: Optional[torch.Tensor] = None
        v_evict: Optional[torch.Tensor] = None

        if kv_window == 0:
            # Unbounded cache (full KV-cache baseline)
            if kv_cache["k"] is None:
                k_all = k
                v_all = v
            else:
                k_all = torch.cat([kv_cache["k"], k], dim=2)
                v_all = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k_all, v_all
            kv_cache["len"] = k_all.shape[2]
        else:
            W = kv_window
            n_sinks = self.cfg.kv_num_sinks
            if kv_cache["k"] is None:
                kv_cache["k"] = torch.zeros(B, self.n_head, W, self.head_dim, device=x_t.device, dtype=x_t.dtype)
                kv_cache["v"] = torch.zeros(B, self.n_head, W, self.head_dim, device=x_t.device, dtype=x_t.dtype)
                kv_cache["len"] = 0
                kv_cache["pos"] = 0   # ring position within the non-sink section

            cur_len = int(kv_cache["len"])
            cur_pos = int(kv_cache["pos"])

            if cur_len < W:
                # Still filling — insert normally (sinks fill first naturally)
                kv_cache["k"][:, :, cur_len:cur_len+1, :] = k
                kv_cache["v"][:, :, cur_len:cur_len+1, :] = v
                kv_cache["len"] = cur_len + 1
            else:
                # Buffer full — overwrite in ring section (after sinks)
                ring_start = n_sinks
                ring_size = W - n_sinks
                actual_pos = ring_start + cur_pos

                # eviction happens here
                if tc_active and self.tc_enabled and self.cfg.tc_enabled and self.cfg.tc_write_on_evict:
                    k_evict = kv_cache["k"][:, :, actual_pos:actual_pos+1, :].clone()
                    v_evict = kv_cache["v"][:, :, actual_pos:actual_pos+1, :].clone()

                kv_cache["k"][:, :, actual_pos:actual_pos+1, :] = k
                kv_cache["v"][:, :, actual_pos:actual_pos+1, :] = v
                kv_cache["pos"] = (cur_pos + 1) % ring_size

        # Gather K/V in chronological order for attention
        if kv_window == 0:
            k_cat, v_cat = kv_cache["k"], kv_cache["v"]  # [B,H,L,D]
        else:
            W = kv_window
            n_sinks = self.cfg.kv_num_sinks
            L = int(kv_cache["len"])
            if L < W:
                # Still filling — already in chronological order
                k_cat = kv_cache["k"][:, :, :L, :]
                v_cat = kv_cache["v"][:, :, :L, :]
            elif n_sinks == 0:
                # No sinks — original ring buffer gather
                p = int(kv_cache["pos"])
                k_cat = torch.cat([kv_cache["k"][:, :, p:, :], kv_cache["k"][:, :, :p, :]], dim=2)
                v_cat = torch.cat([kv_cache["v"][:, :, p:, :], kv_cache["v"][:, :, :p, :]], dim=2)
            else:
                # Sinks + ring: sinks first (slots 0..n_sinks-1),
                # then ring in chronological order
                ring_start = n_sinks
                ring_size = W - n_sinks
                p = int(kv_cache["pos"])  # next-write pos within ring section
                abs_p = ring_start + p
                k_sinks = kv_cache["k"][:, :, :n_sinks, :]
                v_sinks = kv_cache["v"][:, :, :n_sinks, :]
                k_ring = torch.cat([kv_cache["k"][:, :, abs_p:ring_start+ring_size, :],
                                    kv_cache["k"][:, :, ring_start:abs_p, :]], dim=2)
                v_ring = torch.cat([kv_cache["v"][:, :, abs_p:ring_start+ring_size, :],
                                    kv_cache["v"][:, :, ring_start:abs_p, :]], dim=2)
                k_cat = torch.cat([k_sinks, k_ring], dim=2)
                v_cat = torch.cat([v_sinks, v_ring], dim=2)

        # --- Local attention for this token over windowed KV ---
        att = torch.matmul(q, k_cat.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,1,L]
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        out = torch.matmul(att, v_cat)  # [B,H,1,D]

        # --- Infini-attention: read and write every token, per-head gating ---
        next_tc_state = tc_state
        if self.infini_enabled and self.cfg.infini_enabled:
            if next_tc_state is None:
                next_tc_state = self.infini.init_state(B, x_t.device, x_t.dtype)
            q1 = q[:, :, 0, :]  # [B, H, D]
            k1 = k[:, :, 0, :]  # Infini writes every token
            v1 = v[:, :, 0, :]
            mem_h, next_tc_state = self.infini.forward_step(q1, k1, v1, next_tc_state)
            gate = self.infini.gate().to(device=out.device, dtype=out.dtype)
            gate = gate.view(1, self.n_head, 1)
            out_h = out[:, :, 0, :]  # [B, H, D]
            combined = gate * mem_h + (1 - gate) * out_h
            out = combined.unsqueeze(2)  # [B, H, 1, D]

        y = self._merge_heads(out)       # [B,1,C]
        y = self.resid_dropout(self.c_proj(y))

        # --- Tensor Cache: read with current q, write with eviction or insert policy ---
        if tc_active and self.tc_enabled and self.cfg.tc_enabled:
            if next_tc_state is None:
                next_tc_state = self.tc.init_state(B, x_t.device, x_t.dtype)

            q1 = q[:, :, 0, :]  # [B,H,D]

            write_k: Optional[torch.Tensor] = None
            write_v: Optional[torch.Tensor] = None

            if self.cfg.tc_write_on_evict and (k_evict is not None) and (v_evict is not None):
                write_k = k_evict[:, :, 0, :]
                write_v = v_evict[:, :, 0, :]
            elif self.cfg.tc_write_on_insert:
                write_k = k[:, :, 0, :]
                write_v = v[:, :, 0, :]

            mem_h, next_tc_state = self.tc.forward_step(q1, write_k, write_v, next_tc_state)  # [B,H,D]
            mem = mem_h.reshape(B, 1, self.n_head * self.head_dim)  # [B,1,C]
            mem = self.tc_proj(mem)
            if self.tc_per_token_gate:
                gate = torch.sigmoid(self.tc_gate_proj(x_t))  # [B,1,1]
            else:
                gate = torch.sigmoid(self.tc_gate)
            y = y + gate * mem

        return y, kv_cache, next_tc_state


# ----------------------------
# MLP and Block
# ----------------------------

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, tc_state: Optional[TCState] = None, attn_window: Optional[int] = None, tc_active: bool = True):
        a, next_tc = self.attn(self.ln1(x), tc_state=tc_state, attn_window=attn_window, tc_active=tc_active)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, next_tc

    def stream_step(self, x_t: torch.Tensor, pos_t: int, kv_cache: Optional[Dict[str, Any]], tc_state: Optional[TCState], kv_window: int, tc_active: bool = True):
        a, kv_cache, tc_state = self.attn.stream_step(self.ln1(x_t), pos_t, kv_cache, tc_state, kv_window=kv_window, tc_active=tc_active)
        x_t = x_t + a
        x_t = x_t + self.mlp(self.ln2(x_t))
        return x_t, kv_cache, tc_state


# ----------------------------
# GPT Model
# ----------------------------

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.config = cfg

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=None if cfg.use_rope else nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("attn.c_proj.weight") or pn.endswith("mlp.proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _tc_active_for_layer(self, layer_idx: int) -> bool:
        if not self.cfg.tc_enabled:
            return False
        k = int(self.cfg.tc_layers)
        if k < 0:
            return True
        if k == 0:
            return False
        k = min(k, self.cfg.n_layer)
        return layer_idx >= (self.cfg.n_layer - k)

    def _infini_active_for_layer(self, layer_idx: int) -> bool:
        return bool(self.cfg.infini_enabled)

    # -------- standard full forward (training / normal eval) --------
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, attn_window: Optional[int] = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size or self.cfg.use_rope, (
            "If use_rope=False (learned abs positions), T must be <= block_size."
        )

        tok_emb = self.transformer.wte(idx)  # [B,T,C]
        if self.cfg.use_rope:
            x = tok_emb
        else:
            pos = torch.arange(0, T, device=idx.device)
            pos_emb = self.transformer.wpe(pos)[None, :, :]
            x = tok_emb + pos_emb

        x = self.transformer.drop(x)

        tc_states: List[Optional[TCState]] = [None] * self.cfg.n_layer
        for i, block in enumerate(self.transformer.h):
            x, tc_states[i] = block(
                x,
                tc_state=tc_states[i],
                attn_window=attn_window,
                tc_active=self._tc_active_for_layer(i),
            )

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # [B,T,V]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    # -------- streaming state for long prompts / generation --------
    def init_stream_state(self, batch_size: int, device, dtype) -> Dict[str, Any]:
        layers = []
        for _ in range(self.cfg.n_layer):
            layers.append({"kv": None, "tc": None})
        return {"pos": 0, "layers": layers, "dtype": dtype}

    @torch.no_grad()
    def stream_step(self, idx_t: torch.Tensor, state: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        idx_t: [B,1] single token
        returns logits [B,1,V] and updated state
        """
        B, T = idx_t.shape
        assert T == 1

        pos_t = int(state["pos"])
        kv_window = int(self.cfg.kv_window) if self.cfg.use_kv_cache else 0

        x = self.transformer.wte(idx_t)  # [B,1,C]
        if not self.cfg.use_rope:
            assert pos_t < self.cfg.block_size
            x = x + self.transformer.wpe(torch.tensor([pos_t], device=idx_t.device))[None, :, :]

        x = self.transformer.drop(x)

        for li, block in enumerate(self.transformer.h):
            layer_state = state["layers"][li]
            tc_active = self._tc_active_for_layer(li)
            infini_active = self._infini_active_for_layer(li)
            # Pass state through if either TC or Infini is active for this layer
            mem_state = layer_state["tc"] if (tc_active or infini_active) else None
            x, layer_state["kv"], layer_state["tc"] = block.stream_step(
                x_t=x,
                pos_t=pos_t,
                kv_cache=layer_state["kv"],
                tc_state=mem_state,
                kv_window=kv_window,
                tc_active=tc_active,
            )

        x = self.transformer.ln_f(x)     # [B,1,C]
        logits = self.lm_head(x)         # [B,1,V]
        state["pos"] = pos_t + 1
        return logits, state

    @torch.no_grad()
    def stream_prefill(self, idx: torch.Tensor, state: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Stream a (potentially huge) prompt through the model with constant memory (if kv_window>0).
        Returns logits for the last prompt position and final state.
        """
        B, T = idx.shape
        if state is None:
            state = self.init_stream_state(B, idx.device, self.transformer.wte.weight.dtype)

        logits_last = None
        for t in range(T):
            logits_last, state = self.stream_step(idx[:, t:t+1], state)
        assert logits_last is not None
        return logits_last, state

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Constant-memory generation when use_kv_cache=True and kv_window>0.
        If kv_window=0, this reproduces unbounded KV-cache (can OOM for huge prompts).
        """
        B, T = idx.shape
        state = self.init_stream_state(B, idx.device, self.transformer.wte.weight.dtype)

        logits, state = self.stream_prefill(idx, state)

        out = idx
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(1e-6, float(temperature))

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(int(top_k), next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_id], dim=1)

            logits, state = self.stream_step(next_id, state)

        return out

    @classmethod
    def from_pretrained(cls, model_type: str, override_args: Optional[Dict[str, Any]] = None):
        """
        Load GPT-2 weights from Hugging Face into this implementation.
        Notes:
          - Uses learned absolute positions (use_rope=False) to match GPT-2 checkpoints.
          - Disables Tensor Cache by default for checkpoint compatibility.
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args), "Only dropout override is supported"

        from transformers import GPT2LMHeadModel

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args.update(dict(
            vocab_size=50257,
            block_size=1024,
            bias=True,
            use_rope=False,
            kv_window=0,
            tc_enabled=False,
        ))
        if "dropout" in override_args:
            config_args["dropout"] = override_args["dropout"]

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        loaded_keys = set()
        transposed = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}
        for k, v in sd_hf.items():
            if k.endswith(".attn.masked_bias") or k.endswith(".attn.bias"):
                continue

            mapped_k = (
                k.replace(".ln_1.", ".ln1.")
                 .replace(".ln_2.", ".ln2.")
                 .replace(".mlp.c_fc.", ".mlp.fc.")
                 .replace(".mlp.c_proj.", ".mlp.proj.")
            )

            if mapped_k not in sd:
                raise KeyError(f"Unexpected pretrained key after remap: {mapped_k}")

            if any(k.endswith(w) for w in transposed):
                assert v.shape[::-1] == sd[mapped_k].shape, f"shape mismatch for {mapped_k}"
                with torch.no_grad():
                    sd[mapped_k].copy_(v.t())
            else:
                assert v.shape == sd[mapped_k].shape, f"shape mismatch for {mapped_k}"
                with torch.no_grad():
                    sd[mapped_k].copy_(v)
            loaded_keys.add(mapped_k)

        missing = [k for k in sd.keys() if k not in loaded_keys]
        if missing:
            raise KeyError(f"Missing keys when loading pretrained weights: {missing[:8]}")

        return model

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str):
        """
        AdamW with weight decay applied only to 2D+ params (not biases/norms).
        """
        # Use named_parameters() to avoid duplicates from tied weights (wte/lm_head).
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and (device_type == "cuda")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    def get_num_params(self, non_embedding: bool = True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and (self.transformer.wpe is not None):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def get_param_breakdown(self):
        """Return a dict with parameter counts broken down by component.

        Useful for reporting the overhead of TC/Infini memory modules
        relative to the base transformer.
        """
        tc_params = 0
        infini_params = 0
        tc_proj_params = 0
        for block in self.transformer.h:
            attn = block.attn
            if hasattr(attn, 'tc') and attn.tc is not None:
                tc_params += sum(p.numel() for p in attn.tc.parameters())
            if hasattr(attn, 'tc_proj') and attn.tc_proj is not None:
                tc_proj_params += sum(p.numel() for p in attn.tc_proj.parameters())
            if hasattr(attn, 'tc_gate') and attn.tc_gate is not None:
                tc_params += attn.tc_gate.numel()
            if hasattr(attn, 'infini') and attn.infini is not None:
                infini_params += sum(p.numel() for p in attn.infini.parameters())
        total = self.get_num_params(non_embedding=False)
        memory_overhead = tc_params + tc_proj_params + infini_params
        return {
            'total': total,
            'base': total - memory_overhead,
            'tc_memory': tc_params,
            'tc_proj': tc_proj_params,
            'infini': infini_params,
            'memory_overhead': memory_overhead,
            'overhead_pct': 100.0 * memory_overhead / total if total > 0 else 0.0,
        }

    def set_rope_scaling(self, scaling_factor: float):
        """Rebuild RoPE inv_freq buffers with NTK-aware scaling for long-context eval.

        Call AFTER loading weights, BEFORE inference. Only affects the non-learned
        inv_freq buffers — trained parameters are untouched.
        Pass scaling_factor=1.0 to reset to the original (unscaled) frequencies.
        """
        if not self.cfg.use_rope:
            return
        self.cfg.rope_scaling_factor = scaling_factor
        for block in self.transformer.h:
            attn = block.attn
            if attn.rope is not None:
                dim = attn.head_dim
                base = float(self.cfg.rope_base)
                if scaling_factor > 1.0:
                    base = base * scaling_factor ** (dim / (dim - 2))
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
                attn.rope.inv_freq = inv_freq.to(device=attn.rope.inv_freq.device, dtype=attn.rope.inv_freq.dtype)

    def crop_block_size(self, block_size):
        # Only relevant if using learned absolute positions (use_rope=False)
        # Safe no-op for RoPE beyond updating config.
        self.config.block_size = block_size
        self.cfg.block_size = block_size
        if self.transformer.wpe is not None:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])



__all__ = ["GPTConfig", "GPT"]
