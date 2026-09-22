from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from training_agent.tools.command import run_command
from training_agent.tools.metrics import analyze_metrics, extract_metrics


class ExecutionTests(unittest.TestCase):
    def test_command_captures_output_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.log"
            concise = Path(directory) / "metrics.jsonl"
            code = run_command(["python", "-c", "print('loss=0.25 epoch=2')"], Path(directory), raw)
            self.assertEqual(code, 0)
            summary = analyze_metrics(raw, concise, code)
            self.assertEqual(summary["outcome"], "observed")
            self.assertEqual(summary["last_loss"], 0.25)
            self.assertEqual(summary["last_epoch"], 2.0)

    def test_timeout_terminates_quiet_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.log"
            started = time.monotonic()
            code = run_command(["python", "-c", "import time; time.sleep(5)"], Path(directory), raw, timeout=1)
            self.assertEqual(code, 124)
            self.assertLess(time.monotonic() - started, 4)
            self.assertIn("timed out", raw.read_text(encoding="utf-8"))

    def test_metrics_supports_json_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.log"
            concise = Path(directory) / "metrics.jsonl"
            raw.write_text("{'loss': 1e-2, 'epoch': 3}\nloss=nan\n", encoding="utf-8")
            self.assertEqual(extract_metrics(raw, concise), 2)
            summary = analyze_metrics(raw, concise, 0)
            self.assertTrue(summary["nonfinite"])
            self.assertEqual(summary["last_loss"], 0.01)
            self.assertEqual(summary["outcome"], "unverified")


if __name__ == "__main__":
    unittest.main()
