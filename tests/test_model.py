from __future__ import annotations

import unittest

from training_agent.agent.data_judge import judge
from training_agent.agent.model.base import ModelResponse


class FakeClient:
    def __init__(self, content: str):
        self.content = content

    def complete(self, messages, response_schema=None, temperature=0.0, timeout=None):
        return ModelResponse(self.content)


class ModelTests(unittest.TestCase):
    def test_judge_accepts_structured_whitelisted_decision(self) -> None:
        result = judge({"detected_format": "instruction_output", "issues": []}, FakeClient('{"status":"needs_transform","issues":[],"transformations":[{"type":"messages_conversion"}],"full_scan_required":true,"reason":"format"}'))
        self.assertEqual(result["status"], "needs_transform")

    def test_judge_rejects_unsupported_operation(self) -> None:
        with self.assertRaises(ValueError):
            judge({"detected_format": "instruction_output", "issues": []}, FakeClient('{"status":"needs_transform","issues":[],"transformations":[{"type":"invent_answer"}],"full_scan_required":true,"reason":"bad"}'))


if __name__ == "__main__":
    unittest.main()
