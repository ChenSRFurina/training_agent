from __future__ import annotations

import os

from .openai_compatible import OpenAICompatibleClient


class DeepSeekClient(OpenAICompatibleClient):
    def __init__(self, **kwargs):
        if not kwargs.get("base_url"):
            kwargs["base_url"] = os.getenv("MODEL_BASE_URL") or "https://api.deepseek.com/v1"
        if not kwargs.get("model"):
            kwargs["model"] = os.getenv("MODEL_NAME") or "deepseek-chat"
        super().__init__(**kwargs)
