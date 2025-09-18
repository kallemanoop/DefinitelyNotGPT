from __future__ import annotations
import json, argparse, torch
from slm.model import ModelConfig, TransformerLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_json", default="artifacts/tokenizer/vocab.json")
    args = ap.parse_args()

    with open(args.vocab_json, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    cfg = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=128,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_mlp=1024,
        dropout=0.0
    )
    model = TransformerLM(cfg)
    x = torch.randint(0, vocab_size, (1, 16))
    logits, loss = model(x, labels=x)
    print("logits:", tuple(logits.shape), "loss:", float(loss))
    out = model.generate(x[:, :4], max_new_tokens=8)
    print("generated ids:", out[0].tolist())

if __name__ == "__main__":
    main()
