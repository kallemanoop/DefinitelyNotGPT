from __future__ import annotations
import torch
from torch import Tensor
import math

def build_rope_cache(seq_len: int, head_dim: int, rope_theta: float = 10000.0, device=None, dtype=None):
    """
    Precompute cos/sin for RoPE. head_dim must be even.
    Returns cos, sin of shape (seq_len, head_dim)
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    device = device or "cpu"
    dtype = dtype or torch.get_default_dtype()
    pos = torch.arange(seq_len, device=device, dtype=dtype)[:, None]  # (T, 1)
    idx = torch.arange(0, head_dim, 2, device=device, dtype=dtype)[None, :]  # (1, D/2)
    inv_freq = 1.0 / (rope_theta ** (idx / head_dim))
    freqs = pos * inv_freq  # (T, D/2)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    # interleave to (T, D)
    cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, head_dim)
    sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, head_dim)
    return cos, sin

def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, t_offset: int = 0):
    """
    q,k: (B, H, T, D) ; cos,sin: (T_total, D)
    t_offset: starting position (for KV-cache decoding)
    """
    T = q.size(-2)
    cos_t = cos[t_offset:t_offset+T].unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
    sin_t = sin[t_offset:t_offset+T].unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos_t) + (rotate_half(q) * sin_t)
    k_rot = (k * cos_t) + (rotate_half(k) * sin_t)
    return q_rot, k_rot

def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)
