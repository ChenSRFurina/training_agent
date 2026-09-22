from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any

_NUMBER = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf(?:inity)?)"
_PATTERNS = {
    "loss": re.compile(rf"\bloss\s*[=:]\s*({_NUMBER})", re.I),
    "eval_loss": re.compile(rf"\beval[_ -]?loss\s*[=:]\s*({_NUMBER})", re.I),
    "epoch": re.compile(rf"\bepoch\s*[=:]\s*({_NUMBER})", re.I),
    "step": re.compile(r"\bstep\s*[=:]\s*(\d+)", re.I),
    "learning_rate": re.compile(rf"\b(?:learning[_ -]?rate|lr)\s*[=:]\s*({_NUMBER})", re.I),
}


def _number(value: Any) -> float | int | str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else str(value)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
            return lowered
        try:
            result = float(value)
            return result if math.isfinite(result) else lowered
        except ValueError:
            return None
    return None


def _from_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    aliases = {"eval-loss": "eval_loss", "lr": "learning_rate"}
    result: dict[str, Any] = {}
    for key, raw in value.items():
        target = aliases.get(str(key).lower(), str(key).lower())
        if target in {"loss", "eval_loss", "epoch", "step", "learning_rate"}:
            converted = _number(raw)
            if converted is not None:
                result[target] = converted
    return result


def _parse_line(line: str) -> dict[str, Any]:
    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
    if clean.strip().startswith("{") and clean.strip().endswith("}"):
        for parser in (json.loads, ast.literal_eval):
            try:
                result = _from_mapping(parser(clean.strip()))
                if result:
                    return result
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass
    result: dict[str, Any] = {}
    for name, pattern in _PATTERNS.items():
        match = pattern.search(clean)
        if match:
            value = _number(match.group(1))
            if value is not None:
                result[name] = value
    return result


def extract_metrics(raw_log: Path, concise_log: Path) -> int:
    count = 0
    concise_log.parent.mkdir(parents=True, exist_ok=True)
    with raw_log.open(encoding="utf-8", errors="replace") as source, concise_log.open("w", encoding="utf-8") as destination:
        for line_number, line in enumerate(source, 1):
            event = _parse_line(line)
            if event:
                event["source_line"] = line_number
                destination.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
    return count


def analyze_metrics(raw_log: Path, concise_log: Path, returncode: int) -> dict[str, Any]:
    metric_count = extract_metrics(raw_log, concise_log)
    events: list[dict[str, Any]] = []
    if concise_log.exists():
        with concise_log.open(encoding="utf-8") as stream:
            events = [json.loads(line) for line in stream if line.strip()]
    losses = [float(item["loss"]) for item in events if isinstance(item.get("loss"), (int, float)) and math.isfinite(float(item["loss"]))]
    eval_losses = [float(item["eval_loss"]) for item in events if isinstance(item.get("eval_loss"), (int, float)) and math.isfinite(float(item["eval_loss"]))]
    epochs = [float(item["epoch"]) for item in events if isinstance(item.get("epoch"), (int, float))]
    nonfinite = any(any(isinstance(value, str) and value.lower() in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity"} for value in event.values()) for event in events)
    raw_tail = raw_log.read_text(encoding="utf-8", errors="replace")[-1000:] if raw_log.exists() else ""
    oom = bool(re.search(r"out of memory|cuda out of memory|oom", raw_tail, re.I))
    if returncode != 0:
        outcome = "failed"
    elif nonfinite or not losses:
        outcome = "unverified"
    else:
        outcome = "observed"
    return {"outcome": outcome, "returncode": returncode, "metric_count": metric_count,
            "has_loss": bool(losses), "nonfinite": nonfinite, "oom": oom,
            "last_loss": losses[-1] if losses else None,
            "best_eval_loss": min(eval_losses) if eval_losses else None,
            "last_epoch": max(epochs) if epochs else None, "evidence": raw_tail}
