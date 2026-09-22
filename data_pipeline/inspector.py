from __future__ import annotations

import csv
import json
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

from .schema import DataContract, Issue

MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_JSONL_LINE_CHARS = 8 * 1024 * 1024
MAX_REPORTED_ISSUES = 100


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not supported: {value}")


def _records(path: Path, limit: int | None = None) -> Iterator[tuple[int, dict[str, Any] | None, str | None]]:
    """Stream records with source positions; limit includes malformed JSONL lines.

    Blank JSONL lines count toward a preview's physical-line budget, but are not
    data records. CSV uses a record budget and reports physical starting lines.
    JSON documents are explicitly size-limited before being decoded in memory.
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            number = 0
            while limit is None or number < limit:
                line = stream.readline(MAX_JSONL_LINE_CHARS + 1)
                if not line:
                    break
                number += 1
                if len(line) > MAX_JSONL_LINE_CHARS:
                    # Consume the remainder of this physical line once.  Do
                    # not let a huge malformed line turn into thousands of
                    # apparent records during a bounded preview.
                    if not line.endswith("\n"):
                        stream.readline()
                    yield number, None, f"line exceeds {MAX_JSONL_LINE_CHARS} characters"
                    continue
                if not line.strip():
                    continue
                try:
                    value = json.loads(line, parse_constant=_reject_constant)
                    yield number, value if isinstance(value, dict) else None, None
                except (json.JSONDecodeError, ValueError, RecursionError) as exc:
                    yield number, None, str(exc)
    elif suffix == ".json":
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError(f"JSON input exceeds {MAX_JSON_BYTES} bytes; convert it to streaming JSONL")
        with path.open("rb") as stream:
            content = stream.read(MAX_JSON_BYTES + 1)
        if len(content) > MAX_JSON_BYTES:
            raise ValueError(f"JSON input exceeds {MAX_JSON_BYTES} bytes; convert it to streaming JSONL")
        try:
            value = json.loads(content, parse_constant=_reject_constant)
        except (ValueError, UnicodeError, RecursionError) as exc:
            yield 1, None, str(exc)
            return
        values = value if isinstance(value, list) else [value]
        for i, row in enumerate(islice(values, limit), 1):
            yield i, row if isinstance(row, dict) else None, None
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream, strict=True)
            try:
                header = next(reader, None)
            except csv.Error as exc:
                yield 1, None, str(exc)
                return
            if header is None:
                return
            if not header or any(not field for field in header) or len(set(header)) != len(header):
                yield 1, None, "CSV requires unique, non-empty column names"
                return
            read = 0
            while limit is None or read < limit:
                start = reader.line_num + 1
                try:
                    values = next(reader)
                except StopIteration:
                    return
                except csv.Error as exc:
                    yield start, None, str(exc)
                    return
                read += 1
                if len(values) != len(header):
                    yield start, None, f"CSV row has {len(values)} values for {len(header)} columns"
                else:
                    yield start, dict(zip(header, values)), None
    else:
        raise ValueError(f"unsupported data format: {path.suffix}")


def _format(record: dict[str, Any]) -> str:
    if "messages" in record:
        return "jsonl_messages"
    if "instruction" in record and ({"answer", "output"} & record.keys()):
        return "instruction_output"
    return "unknown"


def _scan(path: Path, contract: DataContract, preview_rows: int | None) -> dict[str, Any]:
    total = valid = error_count = 0
    issues: list[Issue] = []
    samples: list[dict[str, Any]] = []
    sample_rows: list[int] = []
    fields: set[str] = set()
    formats: set[str] = set()
    issue_counts: Counter[str] = Counter()

    def add(issue: Issue) -> None:
        nonlocal error_count
        issue_counts[issue.code] += 1
        error_count += issue.severity == "error"
        if len(issues) < MAX_REPORTED_ISSUES:
            issues.append(issue)

    for row_number, row, parse_error in _records(Path(path), preview_rows):
        total += 1
        if parse_error is not None:
            add(Issue("parse_error", "error", row_number, parse_error))
            formats.add("unknown")
            continue
        if row is None:
            add(Issue("not_object", "error", row_number, "record is not an object"))
            formats.add("unknown")
            continue
        if preview_rows is not None:
            samples.append(row)
            sample_rows.append(row_number)
        fields.update(row)
        formats.add(_format(row))
        row_issues = contract.validate_record(row)
        for issue in row_issues:
            issue.row = row_number
            add(issue)
        if not any(issue.severity == "error" for issue in row_issues):
            valid += 1
    if total == 0:
        code = "empty_file" if preview_rows is None else "empty_preview"
        add(Issue(code, "error", None, "no data records were found in the inspected input"))
    detected = next(iter(formats)) if len(formats) == 1 else "mixed" if formats else "unknown"
    result: dict[str, Any] = {
        "status": "reject" if error_count else "pass",
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "error_count": error_count,
        "issues": [issue.as_dict() for issue in issues],
        "issue_counts": dict(issue_counts),
        "issues_truncated": sum(issue_counts.values()) > len(issues),
        "detected_format": detected,
        "fields": sorted(fields),
    }
    if preview_rows is not None:
        result.update(preview_rows=len(samples), preview_limit=preview_rows,
                      samples=samples, sample_rows=sample_rows, full_scan_required=True)
    return result


def inspect(path: Path, contract: DataContract, preview_rows: int = 20) -> dict[str, Any]:
    if preview_rows <= 0:
        raise ValueError("preview_rows must be positive")
    return _scan(path, contract, preview_rows)


def full_scan(path: Path, contract: DataContract) -> dict[str, Any]:
    return _scan(path, contract, None)
