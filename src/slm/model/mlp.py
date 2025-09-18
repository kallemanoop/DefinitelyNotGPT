from __future__ import annotations
import torch.nn as nn
import torch

class MLP(nn.Module):
    """
    Standard GELU MLP: Linear -> GELU -> Dropout -> Linear
    """
    def __init__(self, d_model: int, d_hidden: int, bias: bool, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden, bias=bias)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))
