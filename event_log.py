from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

_SECRET = re.compile(r"(?i)(api[_-]?key|token|password|secret)(\s*[=:]\s*)([^\s,]+)")
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")


def redact(value: Any) -> Any:
    if isinstance(value, str):
        return _BEARER.sub(r"\1[REDACTED]", _SECRET.sub(r"\1\2[REDACTED]", value))
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if re.search(r"(?i)(key|token|password|secret)", k) else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


class EventLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, event: str, summary: str, attempt: int | None = None, **payload: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "training_attempts_n": attempt,
            "summary": redact(summary),
            **redact(payload),
        }
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            stream.flush()
