from __future__ import annotations
import argparse, json, math, os, random
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from rich.console import Console
from rich.table import Table
from glob import glob

from slm.train.config import TrainConfig
from slm.model import ModelConfig, TransformerLM
from slm.data.dataset import iter_memmap_windows, batch_to_x_y
from slm.train.schedule import cosine_warmup
from slm.train.checkpoint import save_checkpoint, load_checkpoint

console = Console()

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def infer_dtype(s: str):
    if s == "float32": return torch.float32
    if s == "float16": return torch.float16
    if s == "bfloat16": return torch.bfloat16
    raise ValueError(f"unknown dtype {s}")

def build_model(cfg: TrainConfig, vocab_size: int):
    mcfg = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=cfg.max_seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_mlp=cfg.d_mlp,
        dropout=cfg.dropout,
        attn_dropout=cfg.attn_dropout,
        mlp_dropout=cfg.mlp_dropout,
        rope_theta=cfg.rope_theta,
        bias=cfg.bias,
        rms_eps=cfg.rms_eps,
    )
    return TransformerLM(mcfg)

def estimate_ppl(model, loader, device, steps=50):
    model.eval()
    losses = []
    with torch.no_grad():
        it = iter(loader)
        for _ in range(steps):
            batch = next(it)
            x, y = batch_to_x_y(batch.to(device))
            logits, loss = model(x, labels=torch.where(x>=0, y, torch.full_like(y, -100)))
            losses.append(loss.item())
    model.train()
    return math.exp(sum(losses)/len(losses)) if losses else float("inf")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    cfg = TrainConfig(**yaml.safe_load(open(args.config, "r")))
    set_seed(cfg.seed)

    vocab = json.load(open(cfg.vocab_json, "r", encoding="utf-8"))
    vocab_size = len(vocab)

    model = build_model(cfg, vocab_size)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device=="cpu" else "cpu")
    model.to(device)

    use_amp = (cfg.dtype in ["float16", "bfloat16"]) and device.type == "cuda"
    scaler = GradScaler(enabled=(cfg.dtype=="float16"))
    if cfg.dtype == "bfloat16": autocast_dtype = torch.bfloat16
    elif cfg.dtype == "float16": autocast_dtype = torch.float16
    else: autocast_dtype = torch.float32

    optim = AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)

    # data
    train_loader = iter_memmap_windows(cfg.train_shards, cfg.max_seq_len, cfg.batch_size)
    val_loader   = iter_memmap_windows(cfg.val_shards,   cfg.max_seq_len, cfg.batch_size)

    start_step = 0
    ckpt_latest = Path(cfg.ckpt_dir) / "latest.pt"
    if args.resume and os.path.exists(args.resume):
        start_step = load_checkpoint(args.resume, model, optim, scaler)
        console.print(f"[yellow]Resumed from {args.resume} at step {start_step}[/]")
    elif ckpt_latest.exists():
        start_step = load_checkpoint(str(ckpt_latest), model, optim, scaler)
        console.print(f"[yellow]Resumed from {ckpt_latest} at step {start_step}[/]")

    table = Table(title=f"Training {cfg.run_name}")
    for k in ["device","dtype","batch_size","grad_accum_steps","lr","max_steps","warmup_steps"]:
        table.add_row(k, str(getattr(cfg,k)))
    console.print(table)

    step = start_step
    model.train()
    grad_accum = cfg.grad_accum_steps

    while step < cfg.max_steps:
        optim.zero_grad(set_to_none=True)
        total_loss = 0.0

        for micro in range(grad_accum):
            batch = next(train_loader)
            x, y = batch_to_x_y(batch.to(device))

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                logits, loss = model(x, labels=y)
                loss = loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()

        if cfg.grad_clip and cfg.grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        lr_now = cosine_warmup(step, cfg.warmup_steps, cfg.max_steps, cfg.lr)
        for g in optim.param_groups: g["lr"] = lr_now

        if scaler.is_enabled():
            scaler.step(optim); scaler.update()
        else:
            optim.step()

        step += 1

        if step % cfg.log_every_steps == 0 or step == 1:
            console.print(f"[cyan]step {step}[/] loss={(total_loss*grad_accum):.4f} lr={lr_now:.6f}")

        if step % cfg.eval_every_steps == 0:
            ppl = estimate_ppl(model, val_loader, device, steps=20)
            console.print(f"[magenta]eval[/] step={step} ppl={ppl:.2f}")

        if step % cfg.save_every_steps == 0 or step == cfg.max_steps:
            Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
            path = Path(cfg.ckpt_dir) / f"step_{step:06d}.pt"
            save_checkpoint(str(path), step, model, optim, scaler)
            save_checkpoint(str(Path(cfg.ckpt_dir) / "latest.pt"), step, model, optim, scaler)
            console.print(f"[green]saved[/] {path}")

    console.print("[bold green]Training finished[/]")

if __name__ == "__main__":
    main()
