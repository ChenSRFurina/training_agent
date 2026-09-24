from __future__ import annotations

import tempfile
import unittest
import io
import zipfile
from pathlib import Path
from unittest.mock import patch

from training_agent.frame.adjustments import propose
from training_agent.frame.downloader import ensure_installed
from training_agent.frame.registry import FrameworkSpec, build_command, load_registry, select


class FrameworkTests(unittest.TestCase):
    def test_registry_template_and_adjustment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "registry.yaml"
            registry.write_text(
                """frameworks:\n  - name: toy\n    version: '1'\n    tasks: [sft]\n    source: local\n    command_template: [python, train.py, --data, '${data_path}']\n    parameter_flags:\n      learning_rate: --learning_rate\n""",
                encoding="utf-8",
            )
            spec = select("toy", "sft", None, registry)
            command = build_command(spec, None, {"data_path": "/tmp/data"}, {"learning_rate": 0.0001})
            self.assertEqual(command[-2:], ["--learning_rate", "0.0001"])
            adjustment = propose({"outcome": "failed", "returncode": 1, "oom": False}, {"learning_rate": 0.0001}, spec.parameter_flags)
            self.assertTrue(adjustment.applied)
            self.assertEqual(adjustment.changes["learning_rate"], 0.00005)

    def test_local_builtin_is_available(self) -> None:
        self.assertEqual(load_registry()["local"].name, "local")

    def test_remote_archive_is_verified_and_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("package/train.py", "print(1)")

            class Response(io.BytesIO):
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return None

            spec = FrameworkSpec("package", "1", ("sft",), "https://example.invalid/package.zip")
            with patch("training_agent.frame.downloader.urllib.request.urlopen", return_value=Response(archive.getvalue())):
                target = ensure_installed(spec, Path(directory) / "frame", True, lambda action, risk: True)
            self.assertTrue((target / "package" / "train.py").exists())


if __name__ == "__main__":
    unittest.main()
