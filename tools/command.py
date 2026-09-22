from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path


def _terminate_group(process: subprocess.Popen[str], force: bool = False) -> None:
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def run_command(args: list[str], cwd: Path, output_path: Path, timeout: int | None = None) -> int:
    """Run a command without a shell and stream combined output to a file.

    A reader thread prevents a chatty child from blocking on a full pipe while
    the parent enforces a monotonic wall-clock deadline. The child is started
    in its own process group so timeout cleanup includes descendants.
    """
    if not args or any(not isinstance(item, str) or not item for item in args):
        raise ValueError("command must be a non-empty list of non-empty strings")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        lines: queue.Queue[str | None] = queue.Queue()

        def collect() -> None:
            try:
                for line in process.stdout:  # type: ignore[union-attr]
                    lines.put(line)
            finally:
                lines.put(None)

        reader = threading.Thread(target=collect, name="training-log-reader", daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout if timeout is not None else None
        eof = False
        try:
            while not eof:
                if deadline is not None and time.monotonic() >= deadline:
                    _terminate_group(process)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        _terminate_group(process, force=True)
                        process.wait()
                    process.stdout.close()
                    log.write("[training-agent] command timed out\n")
                    log.flush()
                    return 124
                wait_for = 0.1 if deadline is None else min(0.1, max(0.01, deadline - time.monotonic()))
                try:
                    line = lines.get(timeout=wait_for)
                except queue.Empty:
                    continue
                if line is None:
                    eof = True
                else:
                    log.write(line)
                    log.flush()
        except KeyboardInterrupt:
            _terminate_group(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _terminate_group(process, force=True)
                process.wait()
            process.stdout.close()
            log.write("[training-agent] command interrupted\n")
            log.flush()
            raise
        result = process.wait()
        process.stdout.close()
        return result
