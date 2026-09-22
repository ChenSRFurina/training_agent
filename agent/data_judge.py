from __future__ import annotations

import json
from typing import Any

from .model.base import ModelClient

JUDGE_SCHEMA = {
    "type": "object",
    "required": ["status", "issues", "transformations", "full_scan_required", "reason"],
    "properties": {
        "status": {"type": "string", "enum": ["pass", "needs_transform", "reject", "needs_full_scan"]},
        "issues": {"type": "array"},
        "transformations": {"type": "array"},
        "full_scan_required": {"type": "boolean"},
        "reason": {"type": "string"},
    },
}


def _deterministic(preview: dict[str, Any]) -> dict[str, Any]:
    issues = preview.get("issues", [])
    status = "reject" if any(item.get("severity") == "error" for item in issues) else "pass"
    transformations = []
    if "instruction_output" == preview.get("detected_format"):
        transformations.append({"type": "messages_conversion", "reason": "normalize instruction/output to messages"})
    if transformations and status == "pass":
        status = "needs_transform"
    return {"status": status, "issues": issues, "transformations": transformations, "full_scan_required": bool(preview.get("full_scan_required")), "reason": "deterministic schema and format inspection"}


def _safe_preview(value: Any) -> Any:
    if isinstance(value, str):
        return value[:500] + ("…" if len(value) > 500 else "")
    if isinstance(value, list):
        return [_safe_preview(item) for item in value[:20]]
    if isinstance(value, dict):
        return {key: ("[REDACTED]" if any(secret in key.lower() for secret in ("api_key", "token", "password", "secret")) else _safe_preview(item)) for key, item in value.items() if key != "samples" or item}
    return value


def judge(preview: dict[str, Any], client: ModelClient | None = None) -> dict[str, Any]:
    if client is None:
        return _deterministic(preview)
    messages = [
        {"role": "system", "content": "You inspect a small, possibly sensitive training-data preview. Do not invent answers or execute actions. Return only the requested JSON."},
        {"role": "user", "content": json.dumps({"task": "sft", "preview": _safe_preview(preview), "output_schema": JUDGE_SCHEMA}, ensure_ascii=False)},
    ]
    response = client.complete(messages, response_schema=JUDGE_SCHEMA, temperature=0.0)
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError as exc:
        raise ValueError("data judge returned invalid JSON") from exc
    if not isinstance(result, dict) or result.get("status") not in {"pass", "needs_transform", "reject", "needs_full_scan"}:
        raise ValueError("data judge returned an invalid decision")
    required = {"issues", "transformations", "full_scan_required", "reason"}
    if not required.issubset(result) or not isinstance(result["issues"], list) or not isinstance(result["transformations"], list) or not isinstance(result["full_scan_required"], bool) or not isinstance(result["reason"], str):
        raise ValueError("data judge returned an incomplete decision")
    allowed = {"messages_conversion", "rename_field"}
    for operation in result["transformations"]:
        if not isinstance(operation, dict) or operation.get("type") not in allowed:
            raise ValueError("data judge proposed an unsupported data transformation")
    return result
