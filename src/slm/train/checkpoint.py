from __future__ import annotations
import torch, os
from pathlib import Path
from typing import Optional

def save_checkpoint(path: str, step: int, model, optim, scaler=None):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    obj = {
        "step": step,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(obj, path)

def load_checkpoint(path: str, model, optim=None, scaler=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optim is not None and "optim" in ckpt and ckpt["optim"] is not None:
        optim.load_state_dict(ckpt["optim"])
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0))
