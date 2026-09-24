from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Adjustment:
    applied: bool
    changes: dict[str, object]
    reason: str
    evidence: dict[str, Any]


def propose(analysis: dict[str, Any], parameters: dict[str, object], parameter_flags: dict[str, str], target_loss: float | None = None) -> Adjustment:
    """Return one bounded, explainable change; never invent a flag mapping."""
    if analysis.get("outcome") == "unverified":
        return Adjustment(False, {}, "metrics are insufficient for a safe adjustment", analysis)
    if target_loss is not None and analysis.get("last_loss") is not None and float(analysis["last_loss"]) > target_loss and "learning_rate" in parameter_flags:
        current = float(parameters.get("learning_rate", 2e-5))
        return Adjustment(True, {"learning_rate": current / 2}, f"loss {analysis['last_loss']} is above target {target_loss}: halve learning_rate", analysis)
    if analysis.get("oom") and "batch_size" in parameter_flags:
        current = int(parameters.get("batch_size", 8))
        if current > 1:
            return Adjustment(True, {"batch_size": max(1, current // 2)}, "out-of-memory: halve batch_size", analysis)
    if analysis.get("returncode", 0) != 0 and "learning_rate" in parameter_flags:
        current = float(parameters.get("learning_rate", 2e-5))
        return Adjustment(True, {"learning_rate": current / 2}, "failed attempt: halve learning_rate", analysis)
    return Adjustment(False, {}, "no safe parameter adjustment is available", analysis)
