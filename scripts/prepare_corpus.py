from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from tokenizers import ByteLevelBPETokenizer

def encode_files(tok: ByteLevelBPETokenizer, files, out_dir: Path, shard_tokens: int, split: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = []
    total = 0
    shard_id = 0

    def flush():
        nonlocal buf, shard_id
        if not buf: return
        arr = np.array(buf, dtype=np.uint32)  # ids fit in uint32
        np.save(out_dir / f"{split}_{shard_id:05d}.npy", arr)
        buf = []
        shard_id += 1

    for p in files:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                ids = tok.encode(line.rstrip("\n")).ids
                if not ids: continue
                buf.extend(ids)
                total += len(ids)
                if len(buf) >= shard_tokens:
                    flush()
    flush()
    print(f"[{split}] wrote {shard_id} shards (~{total} tokens) to {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)  # configs/train_small.yaml
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config, "r"))
    tok_dir = Path(cfg["tokenizer_dir"])
    tok = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))

    out_dir = Path("artifacts/shards")
    encode_files(tok, cfg["corpus_train"], out_dir, int(cfg["shard_tokens"]), "train")
    encode_files(tok, cfg["corpus_val"],   out_dir, int(cfg["shard_tokens"]), "val")

if __name__ == "__main__":
    main()
