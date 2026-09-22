from __future__ import annotations

from collections.abc import Callable


def approve(action: str, high_risk: bool, prompt: Callable[[str], str] = input) -> bool:
    if high_risk:
        return True
    answer = prompt(f"风险操作：{action}\n继续？[y/N] ")
    return answer.strip().lower() in {"y", "yes"}
