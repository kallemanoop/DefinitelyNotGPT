from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .rope import apply_rope

class KVCache:
    """
    Simple per-layer KV cache.
    Stores tensors of shape:
      k: (B, H, T_cached, D)
      v: (B, H, T_cached, Dv)
    """
    def __init__(self):
        self.k: Tensor | None = None
        self.v: Tensor | None = None

    def append(self, k_new: Tensor, v_new: Tensor):
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)

    @property
    def len(self) -> int:
        return 0 if self.k is None else self.k.size(2)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout_p = attn_dropout

    def forward(
        self,
        x: Tensor,                    # (B, T, C)
        cos: Tensor, sin: Tensor,     # rope caches
        is_causal: bool = True,
        kv_cache: KVCache | None = None,
        t_offset: int = 0,
        attn_mask: Tensor | None = None
    ) -> Tensor:
        B, T, C = x.shape
        qkv = self.Wqkv(x)  # (B,T,3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to heads
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,D)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # RoPE
        q, k = apply_rope(q, k, cos, sin, t_offset=t_offset)

        # KV cache
        if kv_cache is not None and kv_cache.len > 0:
            k = torch.cat([kv_cache.k, k], dim=2)
            v = torch.cat([kv_cache.v, v], dim=2)

        # save new keys/values
        if kv_cache is not None:
            # only append the current step tokens (last T positions before concat)
            k_new = k[:, :, -T:, :]
            v_new = v[:, :, -T:, :]
            kv_cache.append(k_new.detach(), v_new.detach())

        # Attention via PyTorch 2 SDPA (uses Flash/Math as available)
        # SDPA expects q,k,v -> (B,H,T,D). is_causal can be True for autoreg.
        with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=True, enable_mem_efficient=True):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,       # typically None for SDPA causal
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=is_causal and attn_mask is None
            )  # (B,H,T,D)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out(y)
        return y
