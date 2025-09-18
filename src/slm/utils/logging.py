from __future__ import annotations
from rich.console import Console
from rich.theme import Theme

_theme = Theme({
    "ok": "bold cyan",
    "warn": "bold yellow",
    "err": "bold red",
    "info": "bold magenta",
})

console = Console(theme=_theme)

def info(msg: str) -> None:
    console.print(f"[info]•[/] {msg}")

def ok(msg: str) -> None:
    console.print(f"[ok]✓[/] {msg}")

def warn(msg: str) -> None:
    console.print(f"[warn]![/] {msg}")

def err(msg: str) -> None:
    console.print(f"[err]×[/] {msg}")
