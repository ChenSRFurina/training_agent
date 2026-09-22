from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameworkSpec:
    name: str
    version: str
    tasks: tuple[str, ...]
    source: str


BUILTIN = {
    "local": FrameworkSpec("local", "user-supplied", ("sft", "rl"), "local command supplied with --train-command"),
}


def select(name: str | None, task: str, command: str | None) -> FrameworkSpec:
    selected = name or ("local" if command else None)
    if not selected:
        raise ValueError("no framework selected; provide --framework or --train-command")
    if selected not in BUILTIN:
        raise ValueError(f"framework '{selected}' is not registered")
    spec = BUILTIN[selected]
    if task not in spec.tasks:
        raise ValueError(f"framework '{selected}' does not support task '{task}'")
    return spec
