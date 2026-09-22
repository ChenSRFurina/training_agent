from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .inspector import _records
from .schema import DataContract


def normalize_record(record: dict[str, Any], contract: DataContract | None = None) -> dict[str, Any]:
    """Add the canonical messages representation without removing original fields."""
    contract = contract or DataContract()
    issues = contract.validate_record(record)
    if any(issue.severity == "error" for issue in issues):
        raise ValueError("invalid training record: " + "; ".join(issue.detail for issue in issues[:5]))
    result = dict(record)
    if "messages" not in result:
        if not contract.allow_transform:
            raise ValueError("data contract does not allow conversion to messages")
        user = result["instruction"]
        if result.get("input"):
            user += "\n" + result["input"]
        result["messages"] = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": result.get("output", result.get("answer"))},
        ]
        # Keep a compatibility summary for older consumers that looked for
        # metadata, while retaining every original extra key verbatim.
        if "metadata" not in result:
            extras = {key: value for key, value in record.items()
                      if key not in {"instruction", "input", "output", "answer", "messages"}}
            if extras:
                result["metadata"] = dict(extras)
    issues = contract.validate_record(result)
    if any(issue.severity == "error" for issue in issues):
        raise ValueError("converted training record failed validation: " + "; ".join(issue.detail for issue in issues[:5]))
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform_jsonl(source: Path, destination: Path, contract: DataContract | None = None) -> dict[str, Any]:
    if Path(source).suffix.lower() != ".jsonl":
        raise ValueError("transform_jsonl requires a .jsonl source")
    return transform_any(source, destination, contract)


def transform_any(source: Path, destination: Path, contract: DataContract | None = None) -> dict[str, Any]:
    """Publish a validated JSONL version atomically, never replacing an existing file.

    All original keys and values are retained; a structural conversion only adds
    messages. Failure removes the private temporary file and leaves source and
    destination untouched. The returned hashes identify the exact data version.
    """
    source, destination = Path(source), Path(destination)
    contract = contract or DataContract()
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must be different paths")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")
    source_hash = _sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    changed = total = 0
    output_hash = hashlib.sha256()
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
                                         dir=destination.parent, delete=False) as stream:
            temp_path = Path(stream.name)
            for row_number, record, error in _records(source):
                if error is not None or record is None:
                    raise ValueError(f"invalid input at row {row_number}: {error or 'record must be an object'}")
                try:
                    normalized = normalize_record(record, contract)
                except ValueError as exc:
                    raise ValueError(f"invalid input at row {row_number}: {exc}") from exc
                changed += normalized != record
                total += 1
                encoded = (json.dumps(normalized, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                stream.write(encoded)
                output_hash.update(encoded)
            if total == 0:
                raise ValueError("cannot transform an empty dataset")
            stream.flush()
            os.fsync(stream.fileno())
        if _sha256(source) != source_hash:
            raise ValueError("source changed during transformation; refusing to publish the result")
        # Atomic creation, unlike replace()/rename(): also protects against a
        # destination appearing between the initial check and publication.
        os.link(temp_path, destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return {"total": total, "changed": changed, "source": str(source.resolve()),
            "destination": str(destination.resolve()), "source_sha256": source_hash,
            "destination_sha256": output_hash.hexdigest(), "transformer_version": 2,
            "operations": ["add_messages_preserving_source_fields"] if changed else []}
