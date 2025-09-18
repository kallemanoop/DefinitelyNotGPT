from __future__ import annotations
import os, json
from pathlib import Path
from typing import Iterable, List

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_text_files(paths: Iterable[str | Path], encoding: str = "utf-8") -> Iterable[str]:
    for p in paths:
        with open(p, "r", encoding=encoding, errors="ignore") as f:
            for line in f:
                yield line.rstrip("\n")

def write_json(obj, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
