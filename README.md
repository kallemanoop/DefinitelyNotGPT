# DefinitelyNotGPT - Small Language Model from scratch 

This repository is an experimental implementation of a **Transformer-decoder small language model (SLM)**.  
The aim is to build a clean, extensible, and inspectable framework that resembles the engineering style of modern large-model codebases, while remaining lightweight and open for study.

At this stage, the project is **pre-alpha**: the core pieces exist and run, but stability and scalability are limited.  
It is intended as a **research and learning platform**, not yet a production-ready model.

---

## Current Status

- Tokenizer: Byte-level BPE training is implemented and functional, but large corpora (tens of GB) can still cause memory errors.  
- Training: Runs end-to-end with PyTorch (AMP, AdamW, cosine LR, gradient clipping), but speed is limited on consumer GPUs.  
- Configs: YAML + Pydantic-driven, but only partially validated across different model sizes.  
- Evaluation: Perplexity is available, though not yet benchmarked at scale.  
- Inference: Minimal FastAPI server and sampling utilities are present, but not production-grade.  
- Tests: Early-stage, currently incomplete.

This repository should be seen as **a foundation under active development**. It demonstrates design principles for training and serving small language models, while leaving room for future refinement.

---

## Features

- **Tokenizer**
  - Byte-level BPE tokenizer (HF-compatible, trainable from raw text)
  - Artifacts saved for reuse across training and inference

- **Model**
  - Transformer decoder-only architecture
  - Rotary Position Embeddings (RoPE)
  - RMSNorm normalization
  - Multi-head attention with KV cache
  - Configurable hidden size, depth, and attention heads

- **Training**
  - Mixed precision training (AMP) with automatic scaling
  - AdamW optimizer with decoupled weight decay
  - Cosine learning rate schedule with warmup
  - Gradient clipping
  - Exponential Moving Average (EMA, optional)
  - Streaming dataset loader (NumPy shards)

- **Evaluation**
  - Perplexity metric for validation

- **Serving**
  - Minimal FastAPI inference server
  - Greedy, top-k, and nucleus sampling

---

## Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install in editable mode
pip install -e .


# Train tokenizer (configurable in YAML)
python scripts/train_tokenizer.py --config configs/tokenizer.yaml

# Inspect tokenizer artifacts
python scripts/inspect_tokenizer.py --model_dir artifacts/tokenizer

# Train a model (example: small config)
python scripts/train.py --config configs/train_small.yaml

# Repository structure- data/ and artifacts/ not present in the original repo
SLM/
├── configs/       # YAML configs (tokenizer, training)
├── data/          # Raw training data (ignored in git)
├── artifacts/     # Tokenizer, checkpoints, logs (ignored in git)
├── scripts/       # CLI entrypoints
├── src/slm/       # Core library code
│   ├── mytokenizer/ # BPE tokenizer
│   ├── model/     # Transformer, attention, norms
│   ├── train/     # Optimizers, schedulers, EMA
│   ├── data/      # Streaming dataloader
│   ├── utils/     # Logging, config, helpers
│   └── server/    # FastAPI inference server
└── tests/         # Unit tests (incomplete)


---

## Future Work
The project is in early stages and not yet stable. Upcoming work includes improving tokenizer training for large corpora, expanding evaluation metrics, and strengthening inference and testing to move closer to a production-ready SLM.

