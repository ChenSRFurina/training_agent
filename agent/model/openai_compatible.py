from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .base import ModelResponse


class OpenAICompatibleClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, default_timeout: int = 120,
                 response_mode: str | None = None, max_retries: int = 2):
        self.api_key = api_key or os.getenv("MODEL_API_KEY")
        self.base_url = (base_url or os.getenv("MODEL_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("MODEL_NAME")
        self.default_timeout = default_timeout
        self.response_mode = response_mode or os.getenv("MODEL_RESPONSE_MODE", "json_object")
        if self.response_mode not in {"json_object", "json_schema", "none"}:
            raise ValueError("MODEL_RESPONSE_MODE must be json_object, json_schema, or none")
        self.max_retries = max(0, max_retries)
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("model base URL must use http or https")
        if not self.api_key or not self.model:
            raise ValueError("MODEL_API_KEY and MODEL_NAME are required for model calls")

    def complete(self, messages: list[dict[str, str]], response_schema: dict[str, Any] | None = None,
                 temperature: float = 0.0, timeout: int | None = None) -> ModelResponse:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if response_schema and self.response_mode == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "agent_result", "schema": response_schema}}
        elif response_schema and self.response_mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        payload: dict[str, Any] | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout or self.default_timeout) as response:
                    payload = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 408 or exc.code == 429 or exc.code >= 500
                if not retryable or attempt >= self.max_retries:
                    raise RuntimeError(f"model API request failed with HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError("model API request failed") from exc
            time.sleep(min(2 ** attempt, 8))
        if payload is None:
            raise RuntimeError("model API returned no payload")
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("model API returned no choices")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("model API returned an empty response")
        return ModelResponse(content, payload.get("id"), payload)
