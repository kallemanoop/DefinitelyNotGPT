from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from .config import ModelConfig
from .layer_norm import RMSNorm
from .rope import build_rope_cache
from .attention import MultiHeadSelfAttention, KVCache
from .mlp import MLP

class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, bias=cfg.bias, attn_dropout=cfg.attn_dropout)
        self.attn_drop = nn.Dropout(cfg.dropout)

        self.mlp_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.mlp = MLP(cfg.d_model, cfg.d_mlp, bias=cfg.bias, dropout=cfg.mlp_dropout)
        self.mlp_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, kv: KVCache | None, t_offset: int) -> Tensor:
        # self-attn
        a = self.attn(self.attn_norm(x), cos, sin, is_causal=True, kv_cache=kv, t_offset=t_offset, attn_mask=None)
        x = x + self.attn_drop(a)
        # mlp
        m = self.mlp(self.mlp_norm(x))
        x = x + self.mlp_drop(m)
        return x

class TransformerLM(nn.Module):
    """
    Causal decoder-only LM with:
      - token embeddings + tied output head
      - RMSNorm, RoPE, SDPA attention, KV cache
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tie weights

        # rope caches built lazily per device/seq len
        self.register_buffer("_rope_cos", None, persistent=False)
        self.register_buffer("_rope_sin", None, persistent=False)
        self._rope_cached_len = 0

    @torch.no_grad()
    def build_rope_if_needed(self, device, dtype):
        L = self.cfg.max_seq_len
        if self._rope_cos is None or self._rope_cached_len != L:
            cos, sin = build_rope_cache(L, self.cfg.d_model // self.cfg.n_heads, self.cfg.rope_theta, device=device, dtype=dtype)
            self._rope_cos = cos
            self._rope_sin = sin
            self._rope_cached_len = L

    def forward(
        self,
        input_ids: Tensor,              # (B, T)
        kv_caches: list[KVCache] | None = None,
        start_pos: int = 0,
        labels: Tensor | None = None,
    ):
        device = input_ids.device
        dtype = self.tok_emb.weight.dtype
        self.build_rope_if_needed(device, dtype)

        x = self.tok_emb(input_ids)  # (B,T,C)
        cos, sin = self._rope_cos, self._rope_sin

        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)

        t_offset = start_pos
        for blk, kv in zip(self.blocks, kv_caches):
            x = blk(x, cos, sin, kv, t_offset)

        x = self.norm_f(x)
        logits = self.lm_head(x)  # (B,T,V)

        loss = None
        if labels is not None:
            # shift for next-token prediction
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = 50):
        device = input_ids.device
        B, T = input_ids.shape
        kv_caches = [KVCache() for _ in self.blocks]
        self.build_rope_if_needed(device, self.tok_emb.weight.dtype)

        cur_ids = input_ids
        t_offset = 0
        for _ in range(max_new_tokens):
            logits, _ = self.forward(cur_ids[:, -1:].contiguous() if t_offset>0 else cur_ids,
                                     kv_caches=kv_caches,
                                     start_pos=t_offset)
            t_offset += 1
            next_logits = logits[:, -1, :] / max(1e-6, temperature)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                next_logits = torch.where(next_logits < thresh, torch.full_like(next_logits, -1e10), next_logits)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B,1)
            cur_ids = torch.cat([cur_ids, next_id], dim=1)
        return cur_ids
