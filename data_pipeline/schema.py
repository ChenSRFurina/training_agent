from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DataContract:
    """Deterministic requirements for the currently supported SFT task.

    An instruction pair or a complete conversation is required. Conversion may
    change representation, but may never invent, coerce, or discard its content.
    """

    task: str = "sft"
    required_fields: tuple[str, ...] = ()
    max_chars: int | None = None
    validation_ratio: float = 0.05
    allow_transform: bool = True

    def __post_init__(self) -> None:
        if self.task != "sft":
            raise ValueError(f"unsupported data contract for task {self.task!r}; only sft is implemented")
        if self.max_chars is not None and self.max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if not 0 <= self.validation_ratio < 1:
            raise ValueError("validation_ratio must be in [0, 1)")

    def validate_record(self, record: Any) -> list["Issue"]:
        if not isinstance(record, dict):
            return [Issue("not_object", "error", None, "record must be an object")]
        issues: list[Issue] = []

        def error(code: str, detail: str) -> None:
            issues.append(Issue(code, "error", None, detail))

        for name in ("instruction", "output", "answer"):
            if name in record and (not isinstance(record[name], str) or not record[name].strip()):
                error("invalid_field", f"{name} must be a non-empty string")
        if "input" in record and not isinstance(record["input"], str):
            error("invalid_field", "input must be a string when present")
        if "output" in record and "answer" in record and record["output"] != record["answer"]:
            error("conflicting_answers", "output and answer disagree; choosing an answer would change training semantics")

        text_length = 0
        if "messages" in record:
            messages = record["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                error("invalid_messages", "messages must contain at least a user and an assistant message")
            else:
                expected = "user"
                for index, message in enumerate(messages):
                    if not isinstance(message, dict):
                        error("invalid_message", f"messages[{index}] must be an object")
                        continue
                    role, content = message.get("role"), message.get("content")
                    if not isinstance(content, str) or not content.strip():
                        error("invalid_message", f"messages[{index}].content must be a non-empty string")
                    elif isinstance(content, str):
                        text_length += len(content)
                    if role == "system" and index == 0:
                        continue
                    if role not in {"user", "assistant"} or role != expected:
                        error("message_sequence", f"messages[{index}].role must be {expected!r}; system is only allowed first")
                    elif role == "user":
                        expected = "assistant"
                    else:
                        expected = "user"
                if not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
                    error("missing_assistant", "conversation must end with an assistant answer")
                if not any(isinstance(message, dict) and message.get("role") == "user" for message in messages):
                    error("missing_user", "conversation must contain a user before an assistant")
        else:
            if "instruction" not in record or not ({"output", "answer"} & record.keys()):
                error("missing_training_pair", "expected messages or instruction plus output/answer")
            answer = record.get("output", record.get("answer", ""))
            text_length = sum(len(value) for value in (record.get("instruction", ""), record.get("input", ""), answer) if isinstance(value, str))
        for name in self.required_fields:
            if name not in record or record[name] is None or record[name] == "":
                error("missing_field", f"required field is absent or empty: {name}")
        if self.max_chars is not None and text_length > self.max_chars:
            error("too_long", f"training text has {text_length} characters; maximum is {self.max_chars}")
        return issues


@dataclass
class Issue:
    code: str
    severity: str
    row: int | None
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity, "row": self.row, "detail": self.detail}
