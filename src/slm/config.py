from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional

class TokenizerConfig(BaseModel):
    model_dir: str = "artifacts/tokenizer"
    vocab_size: int = 32000 
    min_freq: int = 2
    special_tokens: List[str] = ["<pad>", "<unk>", "<bos>", "<eos>"]
    training_corpus: List[str] = Field(default_factory=list)
    byte_fallback: bool = True
    lowercase: bool = False
    max_input_chars_per_token: int = 100
    save_merges: bool = True
    random_seed: int = 42
