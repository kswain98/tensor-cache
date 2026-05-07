"""
Reference implementation of eviction-conditioned Tensor Cache.

This is a standalone teaching file that isolates the core mechanism used in
`model.py`:

1) Local exact memory with a KV ring buffer (size W):
     - W = 0  -> unbounded/full KV cache
     - W > 0  -> fixed sliding-window KV cache

2) Fixed-size tensor cache per head:
     A in R[B, H, D, D]

3) Read + write equations:
     mem_t = q_t @ A
     outer rule:  A <- decay*A + lr * (k_w outer v_w)
     delta rule:  A <- decay*A + lr * (k_w outer (v_w - k_w @ A))
     wedge rule:  A <- decay*A + lr * (k_w v_w^T - v_w k_w^T)
                  (antisymmetric / exterior-algebra write)

4) Eviction-conditioned compression (main paper setting):
     When KV ring is full and a slot is overwritten, write the evicted
     (k_old, v_old) into A.

This file is intentionally minimal and does not depend on the full GPT model.
Use it as publishable pseudocode-in-code for reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class TensorCacheConfig:
    n_head: int = 12
    head_dim: int = 64
    kv_window: int = 512              # 0 => unbounded/full KV

    # Tensor-cache update
    update_rule: str = "delta"       # "delta", "outer", or "wedge"
    decay: float = 0.995
    lr: float = 0.05

    # Read/write behavior
    gate: float = 0.12                # fused output: local + gate * mem
    normalize_q: bool = False
    normalize_k: bool = True
    write_on_evict: bool = True
    write_on_insert: bool = False


@dataclass
class TensorCacheState:
    # Fixed tensor cache per layer/head.
    A: torch.Tensor                   # [B,H,D,D]

    # KV cache storage.
    k_buf: Optional[torch.Tensor] = None  # [B,H,L,D] (unbounded) or [B,H,W,D] (ring)
    v_buf: Optional[torch.Tensor] = None  # [B,H,L,D] (unbounded) or [B,H,W,D] (ring)
    length: int = 0
    pos: int = 0                      # ring write pointer when length == W
    total_seen: int = 0


class TensorCache:
    """Standalone tensor-cache module for one layer of attention."""

    def __init__(self, cfg: TensorCacheConfig):
        self.cfg = cfg
        if cfg.kv_window < 0:
            raise ValueError("kv_window must be >= 0")
        if cfg.update_rule not in ("delta", "outer", "wedge"):
            raise ValueError("update_rule must be 'delta', 'outer', or 'wedge'")

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> TensorCacheState:
        A = torch.zeros(
            batch_size,
            self.cfg.n_head,
            self.cfg.head_dim,
            self.cfg.head_dim,
            device=device,
            dtype=dtype,
        )
        return TensorCacheState(A=A)

    @staticmethod
    def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def _memory_read(self, q: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # q: [B,H,D], A: [B,H,D,D] -> mem: [B,H,D]
        return torch.matmul(q.unsqueeze(-2), A).squeeze(-2)

    def _memory_write(self, A: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # k, v: [B,H,D]
        if self.cfg.update_rule == "outer":
            update = k.unsqueeze(-1) * v.unsqueeze(-2)
        elif self.cfg.update_rule == "wedge":
            # Antisymmetric (exterior-algebra) write: k v^T - v k^T.
            update = (
                k.unsqueeze(-1) * v.unsqueeze(-2)
                - v.unsqueeze(-1) * k.unsqueeze(-2)
            )
        else:
            # Delta rule: write the residual v - (k @ A).
            v_hat = torch.matmul(k.unsqueeze(-2), A).squeeze(-2)
            target = v - v_hat
            update = k.unsqueeze(-1) * target.unsqueeze(-2)

        return self.cfg.decay * A + self.cfg.lr * update  # [B,H,D,D]

    def _kv_insert_and_get_context(
        self,
        state: TensorCacheState,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Insert current key/value into cache, then return context K/V in
        chronological order plus optional evicted token.

        Returns:
          k_ctx, v_ctx, k_evict, v_evict
        """
        B, H, D = k.shape
        W = self.cfg.kv_window

        k_evict = None
        v_evict = None

        if W == 0:
            # Unbounded full KV cache (will grow with sequence length).
            if state.k_buf is None:
                state.k_buf = k.unsqueeze(2)
                state.v_buf = v.unsqueeze(2)
            else:
                state.k_buf = torch.cat([state.k_buf, k.unsqueeze(2)], dim=2)
                state.v_buf = torch.cat([state.v_buf, v.unsqueeze(2)], dim=2)
            state.length = int(state.k_buf.shape[2])
            return state.k_buf, state.v_buf, k_evict, v_evict

        # Fixed window ring buffer.
        if state.k_buf is None:
            state.k_buf = torch.zeros(B, H, W, D, device=k.device, dtype=k.dtype)
            state.v_buf = torch.zeros(B, H, W, D, device=v.device, dtype=v.dtype)
            state.length = 0
            state.pos = 0

        if state.length < W:
            idx = state.length
            state.k_buf[:, :, idx:idx + 1, :] = k.unsqueeze(2)
            state.v_buf[:, :, idx:idx + 1, :] = v.unsqueeze(2)
            state.length += 1
        else:
            idx = state.pos
            k_evict = state.k_buf[:, :, idx, :].clone()
            v_evict = state.v_buf[:, :, idx, :].clone()
            state.k_buf[:, :, idx:idx + 1, :] = k.unsqueeze(2)
            state.v_buf[:, :, idx:idx + 1, :] = v.unsqueeze(2)
            state.pos = (state.pos + 1) % W

        # Gather ring contents in chronological order.
        if state.length < W:
            k_ctx = state.k_buf[:, :, : state.length, :]
            v_ctx = state.v_buf[:, :, : state.length, :]
        else:
            p = state.pos
            k_ctx = torch.cat([state.k_buf[:, :, p:, :], state.k_buf[:, :, :p, :]], dim=2)
            v_ctx = torch.cat([state.v_buf[:, :, p:, :], state.v_buf[:, :, :p, :]], dim=2)

        return k_ctx, v_ctx, k_evict, v_evict

    def step(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: TensorCacheState,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, TensorCacheState]:
        """
        Single-token streaming step.

        Inputs:
          q, k, v: [B,H,D] for current token

        Returns:
          fused: local_attn + gate * mem
          local: local sliding-window attention output
          mem: tensor-cache read output
          new_state
        """
        if self.cfg.normalize_q:
            q = self._l2norm(q)
        if self.cfg.normalize_k:
            k = self._l2norm(k)

        k_ctx, v_ctx, k_evict, v_evict = self._kv_insert_and_get_context(state, k, v)

        # Local exact attention over current KV context.
        att = torch.matmul(q.unsqueeze(2), k_ctx.transpose(-2, -1)) / math.sqrt(self.cfg.head_dim)
        att = F.softmax(att, dim=-1)
        local = torch.matmul(att, v_ctx).squeeze(2)  # [B,H,D]

        # Tensor-cache read.
        mem = self._memory_read(q, state.A)

        # Tensor-cache write.
        write_k = None
        write_v = None
        if self.cfg.write_on_evict and (k_evict is not None) and (v_evict is not None):
            write_k = k_evict
            write_v = v_evict
        elif self.cfg.write_on_insert:
            write_k = k
            write_v = v

        if write_k is not None and write_v is not None:
            state.A = self._memory_write(state.A, write_k, write_v)

        state.total_seen += 1
        fused = local + self.cfg.gate * mem
        return fused, local, mem, state


def estimate_state_memory_mb(state: TensorCacheState) -> float:
    """Rough bytes for A + KV buffers currently materialized."""
    total = state.A.numel() * state.A.element_size()
    if state.k_buf is not None:
        total += state.k_buf.numel() * state.k_buf.element_size()
    if state.v_buf is not None:
        total += state.v_buf.numel() * state.v_buf.element_size()
    return total / (1024.0 * 1024.0)


def run_demo(mode: str, steps: int = 4096, kv_window: int = 512,
             update_rule: str = "delta") -> None:
    """
    Tiny demo to show growth behavior across modes.

    Modes:
      - full_kv: W=0, no tensor-cache writes
      - window_kv: W>0, no tensor-cache writes
      - tensor_cache: W>0, write evicted KV into A (uses `update_rule`)

    update_rule applies to the tensor_cache mode only:
      - "delta" (default), "outer", "wedge".
    """
    if mode == "full_kv":
        cfg = TensorCacheConfig(kv_window=0, write_on_evict=False, write_on_insert=False)
    elif mode == "window_kv":
        cfg = TensorCacheConfig(kv_window=kv_window, write_on_evict=False, write_on_insert=False)
    elif mode == "tensor_cache":
        cfg = TensorCacheConfig(kv_window=kv_window, write_on_evict=True, write_on_insert=False,
                                update_rule=update_rule)
    else:
        raise ValueError("mode must be one of: full_kv, window_kv, tensor_cache")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    B = 1
    cache = TensorCache(cfg)
    state = cache.init_state(batch_size=B, device=device, dtype=dtype)

    with torch.no_grad():
        for t in range(steps):
            q = torch.randn(B, cfg.n_head, cfg.head_dim, device=device, dtype=dtype)
            k = torch.randn(B, cfg.n_head, cfg.head_dim, device=device, dtype=dtype)
            v = torch.randn(B, cfg.n_head, cfg.head_dim, device=device, dtype=dtype)
            _, _, _, state = cache.step(q, k, v, state)

            if t in (0, 1, 2, kv_window - 1, kv_window, steps - 1):
                kv_len = 0 if state.k_buf is None else int(state.k_buf.shape[2])
                print(
                    f"t={t:6d} | kv_len={kv_len:6d} | ring_len={state.length:6d} | "
                    f"mem_mb={estimate_state_memory_mb(state):8.2f}"
                )

    final_kv_len = 0 if state.k_buf is None else int(state.k_buf.shape[2])
    print("\nSummary")
    print(f"mode={mode}")
    if mode == "tensor_cache":
        print(f"update_rule={cfg.update_rule}")
    print(f"tokens_seen={state.total_seen}")
    print(f"kv_storage_len={final_kv_len}")
    print(f"ring_length={state.length}")
    print(f"state_memory_mb={estimate_state_memory_mb(state):.2f}")


if __name__ == "__main__":
    print("=== Tensor Cache Reference Demo ===")
    run_demo(mode="full_kv", steps=2048, kv_window=512)
    print("\n---")
    run_demo(mode="window_kv", steps=2048, kv_window=512)
    for rule in ("outer", "delta", "wedge"):
        print("\n---")
        run_demo(mode="tensor_cache", steps=2048, kv_window=512, update_rule=rule)
