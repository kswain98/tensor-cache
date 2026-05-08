"""
Infini-attention: Compressive Memory for Infinite-Length Sequences.

Faithful reproduction of the Infini-attention mechanism from:
    "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"
    Munkhdalai, Pham, Yu, Gu, Shazeer (2024)

Integration notes:
    - This module is used as a drop-in alternative to FastWeightTensorCache.
    - It will be selected via kv_mode="infini" in the model config.
    - The gate (beta) is per-head, unlike TensorCache's per-layer gate.
    - This is a SEPARATE module from TensorCache; the math is fundamentally different.
      Infini-attention uses a linear associative memory (outer-product accumulation)
      with ELU+1 feature maps, whereas TensorCache uses learned decay and learning rates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfiniMemory(nn.Module):
    """Faithful Infini-attention compressive memory (Munkhdalai et al., 2024).

    Maintains a per-head associative memory matrix M and normalization vector z.
    Supports both "linear" and "delta" update rules from the paper.

    The memory is read via:
        A_mem = sigma(Q) @ M / (sigma(Q) @ z + eps)
    and written via (linear rule):
        M_s = M_{s-1} + sigma(K)^T @ V
        z_s = z_{s-1} + sigma(K).sum(dim=time)
    or (delta rule):
        V_retrieved = sigma(K) @ M_{s-1} / (sigma(K) @ z_{s-1} + eps)
        M_s = M_{s-1} + sigma(K)^T @ (V - V_retrieved)
        z_s = z_{s-1} + sigma(K).sum(dim=time)

    where sigma(x) = ELU(x) + 1.

    Args:
        n_head: Number of attention heads.
        head_dim: Dimension per head (used for both key and value).
        update_rule: "linear" or "delta". Delta writes the residual (V - V_retrieved).
        segment_size: Chunk size for segment-based processing in forward_full.
    """

    def __init__(self, n_head: int, head_dim: int, update_rule: str = "delta", segment_size: int = 2048):
        super().__init__()
        assert update_rule in ("linear", "delta"), f"update_rule must be 'linear' or 'delta', got '{update_rule}'"
        self.n_head = n_head
        self.head_dim = head_dim
        self.update_rule = update_rule
        self.segment_size = segment_size
        self.eps = 1e-5

        # Per-head learned gating scalar beta (Eq. 6 in the paper).
        # sigmoid(beta) gates between memory output and local attention output.
        # Initialized to 0 so sigmoid(beta) = 0.5 at the start.
        self.beta = nn.Parameter(torch.zeros(n_head))

    def init_state(self, B: int, device: torch.device, dtype: torch.dtype):
        """Initialize compressive memory state.

        Returns:
            Tuple (M, z) where:
                M: [B, H, D_key, D_val] associative memory matrix (zeros).
                z: [B, H, D_key] normalization vector (zeros).
        """
        M = torch.zeros(B, self.n_head, self.head_dim, self.head_dim, device=device, dtype=dtype)
        z = torch.zeros(B, self.n_head, self.head_dim, device=device, dtype=dtype)
        return (M, z)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feature map sigma(x) = ELU(x) + 1.

        This ensures all outputs are non-negative (ELU lower-bounds at -1, so +1
        shifts to non-negative), which is required for the linear attention
        normalization to be well-defined.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor of same shape with sigma applied element-wise.
        """
        return F.elu(x) + 1.0

    def _read(self, q_mapped: torch.Tensor, M: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Retrieve from memory using mapped queries.

        Computes: A_mem = (sigma(Q) @ M) / (sigma(Q) @ z + eps)

        Args:
            q_mapped: sigma(Q), shape [B, H, ..., D_key] where ... is T or empty.
            M: Memory matrix [B, H, D_key, D_val].
            z: Normalization vector [B, H, D_key].

        Returns:
            Memory retrieval [B, H, ..., D_val].
        """
        # Numerator and denominator computation differs by rank to avoid
        # torch.matmul broadcasting issues with 3D x 4D tensors.
        if q_mapped.dim() == 3:
            # Step: q_mapped [B,H,D], M [B,H,D,D] -> [B,H,D]
            numerator = torch.einsum("bhd,bhde->bhe", q_mapped, M)
            denominator = torch.einsum("bhd,bhd->bh", q_mapped, z)
        else:
            # Full: q_mapped [B,H,T,D], M [B,H,D,D] -> [B,H,T,D]
            numerator = torch.matmul(q_mapped, M)
            denominator = torch.einsum("bhtd,bhd->bht", q_mapped, z)
        denominator = denominator.unsqueeze(-1) + self.eps  # [B, H, ..., 1]

        return numerator / denominator

    def _write(self, k_mapped: torch.Tensor, v: torch.Tensor, M: torch.Tensor, z: torch.Tensor):
        """Write to memory and update normalization.

        Args:
            k_mapped: sigma(K), shape [B, H, T, D_key] or [B, H, D_key].
            v: Values to write, shape [B, H, T, D_val] or [B, H, D_val].
            M: Current memory [B, H, D_key, D_val].
            z: Current normalization [B, H, D_key].

        Returns:
            Tuple (M_new, z_new).
        """
        is_step = k_mapped.dim() == 3  # [B, H, D] for single step

        if self.update_rule == "delta":
            # Retrieve what is already stored for these keys
            v_retrieved = self._read(k_mapped, M, z)
            # Write the residual
            write_v = v - v_retrieved
        else:
            # Linear rule: write V directly
            write_v = v

        if is_step:
            # Single token: k_mapped [B, H, D_key], write_v [B, H, D_val]
            # Outer product: [B, H, D_key, 1] @ [B, H, 1, D_val] -> [B, H, D_key, D_val]
            M_new = M + k_mapped.unsqueeze(-1) * write_v.unsqueeze(-2)
            z_new = z + k_mapped
        else:
            # Segment: k_mapped [B, H, T, D_key], write_v [B, H, T, D_val]
            # sigma(K)^T @ write_v: [B, H, D_key, T] @ [B, H, T, D_val] -> [B, H, D_key, D_val]
            M_new = M + torch.matmul(k_mapped.transpose(-2, -1), write_v)
            # Sum over time dim for z
            z_new = z + k_mapped.sum(dim=-2)

        return M_new, z_new

    def gate(self) -> torch.Tensor:
        """Return the per-head gating coefficient sigmoid(beta).

        Returns:
            Tensor of shape [H] with values in (0, 1).
        """
        return torch.sigmoid(self.beta)

    def forward_step(self, q: torch.Tensor, write_k: torch.Tensor, write_v: torch.Tensor, state):
        """Single-token read from memory, then write to memory.

        In Infini-attention, every token is written to memory (unlike TensorCache
        which only writes evicted tokens). This is called during autoregressive
        generation.

        Args:
            q: Query for this step, [B, H, D_key].
            write_k: Key to write, [B, H, D_key], or None to skip write.
            write_v: Value to write, [B, H, D_val], or None to skip write.
            state: Tuple (M, z) from init_state or previous step.

        Returns:
            Tuple (mem_out, new_state) where:
                mem_out: Memory retrieval [B, H, D_val].
                new_state: Updated (M, z).
        """
        M, z = state

        # Read: retrieve from memory using current query
        q_mapped = self._feature_map(q)  # [B, H, D_key]

        # Numerator: [B, H, D_key] @ [B, H, D_key, D_val] -> [B, H, D_val]
        # Using einsum for the batched matmul with 3D query
        numerator = torch.einsum("bhd,bhde->bhe", q_mapped, M)

        # Denominator: [B, H, D_key] dot [B, H, D_key] -> [B, H]
        denominator = torch.einsum("bhd,bhd->bh", q_mapped, z).unsqueeze(-1) + self.eps  # [B, H, 1]

        mem_out = numerator / denominator  # [B, H, D_val]

        # Write: update memory with current token
        if write_k is not None and write_v is not None:
            k_mapped = self._feature_map(write_k)  # [B, H, D_key]
            M, z = self._write(k_mapped, write_v, M, z)

        return mem_out, (M, z)

    def forward_full(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state):
        """Segment-based processing for training.

        Processes the sequence in non-overlapping segments of self.segment_size.
        Within each segment:
            1. Read: retrieve A_mem for all tokens in the segment using current state.
            2. Write: update state with all tokens in the segment.

        This matches the paper's formulation where the memory from segment s-1
        is used to retrieve for segment s, and then segment s's tokens are written.

        Args:
            q: Queries [B, H, T, D_key].
            k: Keys [B, H, T, D_key].
            v: Values [B, H, T, D_val].
            state: Tuple (M, z) from init_state or previous call.

        Returns:
            Tuple (mem_out, new_state) where:
                mem_out: Memory retrievals [B, H, T, D_val].
                new_state: Updated (M, z) after processing all segments.
        """
        B, H, T, D = q.shape
        M, z = state

        # Collect outputs for all segments
        outputs = []

        for start in range(0, T, self.segment_size):
            end = min(start + self.segment_size, T)

            q_seg = q[:, :, start:end, :]  # [B, H, S, D]
            k_seg = k[:, :, start:end, :]
            v_seg = v[:, :, start:end, :]

            # Map queries and keys through the feature map
            q_mapped = self._feature_map(q_seg)  # [B, H, S, D_key]
            k_mapped = self._feature_map(k_seg)  # [B, H, S, D_key]

            # --- Read: retrieve from memory for all tokens in this segment ---
            # Numerator: [B, H, S, D_key] @ [B, H, D_key, D_val] -> [B, H, S, D_val]
            numerator = torch.matmul(q_mapped, M)

            # Denominator: [B, H, S, D_key] @ [B, H, D_key] -> [B, H, S]
            denominator = torch.einsum("bhsd,bhd->bhs", q_mapped, z).unsqueeze(-1) + self.eps  # [B, H, S, 1]

            seg_out = numerator / denominator  # [B, H, S, D_val]
            outputs.append(seg_out)

            # --- Write: update memory with all tokens in this segment ---
            M, z = self._write(k_mapped, v_seg, M, z)

        # Concatenate segment outputs along time dimension
        mem_out = torch.cat(outputs, dim=2)  # [B, H, T, D_val]

        return mem_out, (M, z)
