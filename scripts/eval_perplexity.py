from __future__ import annotations
import argparse, json, yaml, math
import torch
from slm.train.config import TrainConfig
from slm.model import ModelConfig, TransformerLM
from slm.data.dataset import iter_memmap_windows, batch_to_x_y
from slm.train.checkpoint import load_checkpoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = TrainConfig(**yaml.safe_load(open(args.config)))
    vocab = json.load(open(cfg.vocab_json, "r", encoding="utf-8"))
    model = TransformerLM(ModelConfig(
        vocab_size=len(vocab), max_seq_len=cfg.max_seq_len,
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
        d_mlp=cfg.d_mlp, dropout=cfg.dropout, attn_dropout=cfg.attn_dropout,
        mlp_dropout=cfg.mlp_dropout, rope_theta=cfg.rope_theta, bias=cfg.bias, rms_eps=cfg.rms_eps
    )).to(cfg.device if torch.cuda.is_available() else "cpu")
    step = load_checkpoint(args.ckpt, model)
    model.eval()

    loader = iter_memmap_windows(cfg.val_shards, cfg.max_seq_len, cfg.batch_size)
    losses = []
    with torch.no_grad():
        for _ in range(100):
            x, y = batch_to_x_y(next(loader).to(model.lm_head.weight.device))
            _, loss = model(x, labels=y)
            losses.append(loss.item())
    ppl = math.exp(sum(losses)/len(losses))
    print(f"step={step} perplexity={ppl:.2f}")

if __name__ == "__main__":
    main()
