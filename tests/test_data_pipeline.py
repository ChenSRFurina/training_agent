from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training_agent.data_pipeline.inspector import MAX_JSONL_LINE_CHARS, full_scan, inspect
from training_agent.data_pipeline.schema import DataContract
from training_agent.data_pipeline.transformer import transform_any, transform_jsonl


class DataPipelineTests(unittest.TestCase):
    def test_missing_answer_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.jsonl"
            source.write_text('{"instruction":"hello"}\n', encoding="utf-8")
            result = inspect(source, DataContract(), 20)
            self.assertEqual(result["status"], "reject")
            self.assertEqual(result["error_count"], 1)

    def test_late_corruption_is_seen_by_full_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.jsonl"
            source.write_text("".join('{"instruction":"ok","output":"yes"}\n' for _ in range(25)) + "bad\n", encoding="utf-8")
            preview = inspect(source, DataContract(), 2)
            self.assertEqual(preview["status"], "pass")
            scan = full_scan(source, DataContract())
            self.assertEqual(scan["total"], 26)
            self.assertEqual(scan["valid"], 25)
            self.assertEqual(scan["invalid"], 1)
            self.assertEqual(scan["error_count"], 1)

    def test_preview_is_bounded_and_keeps_physical_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.jsonl"
            source.write_text("\n\nnot-json\n" + '{"instruction":"x","output":"y"}\n', encoding="utf-8")
            result = inspect(source, DataContract(), 3)
            self.assertEqual(result["total"], 1)
            self.assertEqual(result["sample_rows"], [])
            self.assertEqual(result["preview_rows"], 0)
            self.assertEqual(result["issues"][0]["row"], 3)

    def test_types_and_conflicting_answers_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.jsonl"
            source.write_text('{"instruction":7,"output":"a","answer":"b"}\n', encoding="utf-8")
            result = full_scan(source, DataContract())
            codes = set(result["issue_counts"])
            self.assertIn("invalid_field", codes)
            self.assertIn("conflicting_answers", codes)

    def test_messages_require_user_then_assistant_and_end_assistant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.jsonl"
            source.write_text(json.dumps({"messages": [{"role": "assistant", "content": "answer"}, {"role": "user", "content": "question"}]}) + "\n", encoding="utf-8")
            result = full_scan(source, DataContract())
            self.assertEqual(result["status"], "reject")
            self.assertIn("message_sequence", result["issue_counts"])
            self.assertIn("missing_assistant", result["issue_counts"])

    def test_normalization_preserves_metadata_and_creates_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.jsonl"
            destination = Path(directory) / "normalized.jsonl"
            source.write_text('{"instruction":"hello","answer":"world","id":7}\n', encoding="utf-8")
            result = transform_jsonl(source, destination)
            self.assertEqual(result["changed"], 1)
            record = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(record["messages"][1]["content"], "world")
            self.assertEqual(record["id"], 7)
            self.assertEqual(record["metadata"]["id"], 7)

    def test_transform_rejects_empty_answer_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.jsonl"
            source.write_text('{"instruction":"hello","output":""}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                transform_jsonl(source, Path(directory) / "out.jsonl")
            destination = Path(directory) / "existing.jsonl"
            destination.write_text("keep\n", encoding="utf-8")
            valid = Path(directory) / "valid.jsonl"
            valid.write_text('{"instruction":"hello","output":"world"}\n', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                transform_jsonl(valid, destination)
            with self.assertRaises(ValueError):
                transform_jsonl(valid, valid)

    def test_json_and_csv_convert_to_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_source = root / "data.json"
            json_source.write_text('[{"instruction":"a","output":"b"}]', encoding="utf-8")
            csv_source = root / "data.csv"
            csv_source.write_text("instruction,output\na,b\n", encoding="utf-8")
            for index, source in enumerate((json_source, csv_source)):
                destination = root / f"converted-{index}.jsonl"
                result = transform_any(source, destination)
                self.assertEqual(result["total"], 1)
                self.assertEqual(json.loads(destination.read_text())["messages"][0]["content"], "a")


if __name__ == "__main__":
    unittest.main()
