from __future__ import annotations
from pydantic import BaseModel
from typing import Optional

class ModelConfig(BaseModel):
    # vocab + sequence
    vocab_size: int
    max_seq_len: int = 2048

    # widths
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_mlp: int = 2048  # FFN inner size

    # regularization
    dropout: float = 0.0
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0

    # init / misc
    rope_theta: float = 10000.0
    qk_layer_norm: bool = False  # rarely used; keep false by default
    bias: bool = False           # use bias in linear layers?
    rms_eps: float = 1e-5
    # precision is set at runtime (AMP), not in config
