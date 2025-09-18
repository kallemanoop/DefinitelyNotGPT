# SLM — Production-Style Small Language Model (from scratch)

This repo implements an inspectable Transformer-decoder SLM with:
- Byte-level BPE tokenizer (trainable from raw text)
- RoPE, RMSNorm, multi-head attention w/ KV cache
- Mixed precision (AMP), AdamW w/ decoupled weight decay
- Cosine LR with warmup, gradient clipping, EMA (optional)
- Streaming dataloader, checkpointing, eval (perplexity)
- Minimal FastAPI inference server + sampling utilities
- Clean config via YAML + Pydantic models
- Tests

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/train_tokenizer.py --config configs/tokenizer.yaml
python scripts/inspect_tokenizer.py --model_dir artifacts/tokenizer
pytest
