from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .config import Settings
from .event_log import EventLog


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
            stream.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class RunContext:
    def __init__(self, settings: Settings):
        stamp = datetime.now().strftime("%Y_%m_%d_%H-%M")
        base_output = settings.output_root / stamp
        if base_output.exists():
            stamp = f"{stamp}-{uuid.uuid4().hex[:4]}"
        self.settings = settings
        self.run_id = stamp
        self.output_dir = settings.output_root / stamp
        self.tmp_dir = settings.tmp_root / stamp
        self.progress_dir = settings.progress_root / stamp
        self.agent_log = EventLog(self.output_dir / "agent_logs" / "agent.log")
        self.state_path = self.output_dir / "state.json"
        self.state = "INIT"
        self.attempt = 0
        for path in [self.output_dir / "training_logs", self.output_dir / "report", self.tmp_dir, self.progress_dir]:
            path.mkdir(parents=True, exist_ok=True)
        self.persist()
        self.agent_log.write("action", "initialized run", run_id=self.run_id, settings=asdict(settings))

    @classmethod
    def resume(cls, settings: Settings) -> "RunContext":
        run_dir = settings.resume.resolve() if settings.resume else None
        if run_dir is None or not run_dir.is_dir():
            raise FileNotFoundError(f"resume run directory does not exist: {run_dir}")
        state_path = run_dir / "state.json"
        if not state_path.exists():
            raise ValueError(f"resume run has no state.json: {run_dir}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self = cls.__new__(cls)
        self.settings = settings
        self.run_id = str(state.get("run_id") or run_dir.name)
        self.output_dir = run_dir
        self.tmp_dir = settings.tmp_root / self.run_id
        self.progress_dir = settings.progress_root / self.run_id
        self.agent_log = EventLog(self.output_dir / "agent_logs" / "agent.log")
        self.state_path = state_path
        self.state = str(state.get("state", "INIT"))
        self.attempt = int(state.get("attempt", 0))
        for path in [self.output_dir / "training_logs", self.output_dir / "report", self.tmp_dir, self.progress_dir]:
            path.mkdir(parents=True, exist_ok=True)
        self.agent_log.write("action", "resumed run", attempt=self.attempt, previous_state=self.state)
        return self

    def persist(self) -> None:
        _atomic_json(self.state_path, {"run_id": self.run_id, "state": self.state, "attempt": self.attempt})

    def write_manifest(self, **values: object) -> None:
        path = self.output_dir / "manifest.json"
        current: dict[str, object] = {}
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {}
        _atomic_json(path, {**current, "run_id": self.run_id, **values})

    def transition(self, state: str) -> None:
        self.agent_log.write("action", f"state {self.state} -> {state}", attempt=self.attempt)
        self.state = state
        self.persist()

    def attempt_dirs(self, number: int) -> tuple[Path, Path]:
        base = self.output_dir / "training_logs" / f"training_attempts_{number}"
        raw, concise = base / "raw_log", base / "concise_logs"
        raw.mkdir(parents=True, exist_ok=True)
        concise.mkdir(parents=True, exist_ok=True)
        (self.progress_dir / f"training_attempts_{number}").mkdir(parents=True, exist_ok=True)
        return raw, concise
