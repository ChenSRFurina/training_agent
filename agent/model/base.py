from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelResponse:
    content: str
    request_id: str | None = None
    raw: dict[str, Any] | None = None


class ModelClient(Protocol):
    def complete(self, messages: list[dict[str, str]], response_schema: dict[str, Any] | None = None,
                 temperature: float = 0.0, timeout: int | None = None) -> ModelResponse: ...
