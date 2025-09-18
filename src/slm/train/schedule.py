from __future__ import annotations
import math

def cosine_warmup(step: int, warmup: int, max_steps: int, base_lr: float):
    if step < warmup:
        return base_lr * (step / max(1, warmup))
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
