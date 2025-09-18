from __future__ import annotations
from pydantic import BaseModel
from typing import List, Tuple

class TrainConfig(BaseModel):
    run_name: str = "run"
    seed: int = 1337

    tokenizer_dir: str
    train_shards: List[str]
    val_shards: List[str]

    vocab_json: str
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_mlp: int = 1024
    dropout: float = 0.0
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    rope_theta: float = 10000.0
    bias: bool = False
    rms_eps: float = 1e-5

    device: str = "cuda"
    dtype: str = "bfloat16"       # "float32"/"float16"/"bfloat16"
    batch_size: int = 32
    grad_accum_steps: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 200
    max_steps: int = 5000
    grad_clip: float = 1.0

    ckpt_dir: str = "artifacts/checkpoints/run"
    eval_every_steps: int = 250
    save_every_steps: int = 500
    log_every_steps: int = 50

    # data prep
    corpus_train: List[str] = []
    corpus_val: List[str] = []
    shard_tokens: int = 2_000_000
