from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from .registry import FrameworkSpec


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                target = (destination / member.filename).resolve()
                if not str(target).startswith(str(destination.resolve()) + os.sep):
                    raise ValueError("framework archive contains an unsafe path")
            handle.extractall(destination)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                target = (destination / member.name).resolve()
                if not str(target).startswith(str(destination.resolve()) + os.sep):
                    raise ValueError("framework archive contains an unsafe path")
            handle.extractall(destination, filter="data")
        return
    raise ValueError(f"unsupported framework archive: {archive.name}")


def ensure_installed(spec: FrameworkSpec, root: Path, high_risk: bool, approve: Callable[[str, bool], bool], max_bytes: int = 2 * 1024 * 1024 * 1024) -> Path:
    target = root / (spec.install_dir or spec.name)
    if target.exists():
        if spec.is_remote and spec.sha256:
            marker = target / ".training_agent_framework.json"
            if not marker.exists():
                raise ValueError(f"existing framework is not verified: {target}")
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            if metadata.get("sha256", "").lower() != spec.sha256.lower() or metadata.get("version") != spec.version:
                raise ValueError(f"installed framework does not match registry: {spec.name}")
        return target
    if not spec.is_remote:
        target.mkdir(parents=True, exist_ok=True)
        return target
    if not approve(f"下载框架 {spec.name}@{spec.version} from {spec.source} to {target}", high_risk):
        raise PermissionError(f"framework download rejected: {spec.name}")
    root.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        archive = Path(temporary) / "framework.archive"
        try:
            with urllib.request.urlopen(spec.source, timeout=60) as response, archive.open("wb") as output:
                headers = getattr(response, "headers", {})
                content_length = headers.get("Content-Length") if hasattr(headers, "get") else None
                if content_length and int(content_length) > max_bytes:
                    raise ValueError("framework archive exceeds the configured size limit")
                copied = 0
                while block := response.read(1024 * 1024):
                    copied += len(block)
                    if copied > max_bytes:
                        raise ValueError("framework archive exceeds the configured size limit")
                    output.write(block)
        except Exception as exc:
            raise RuntimeError(f"framework download failed: {spec.name}") from exc
        if spec.sha256 and _hash(archive).lower() != spec.sha256.lower():
            raise ValueError(f"framework checksum mismatch: {spec.name}")
        extracted = Path(temporary) / "extracted"
        _safe_extract(archive, extracted)
        staging = root / f".{spec.name}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.move(str(extracted), staging)
        try:
            os.replace(staging, target)
        except FileExistsError:
            shutil.rmtree(staging, ignore_errors=True)
        (target / ".training_agent_framework.json").write_text(json.dumps({"name": spec.name, "version": spec.version, "source": spec.source, "sha256": spec.sha256}, indent=2), encoding="utf-8")
    return target
