import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from model_demo import run
from ollama_adapter import MODEL, OllamaProposalAdapter


class FakeResponse(io.BytesIO):
    status = 200


class ModelDemoTests(unittest.TestCase):
    def test_existing_receipt_is_preserved_before_any_adapter_or_request(self):
        with tempfile.TemporaryDirectory(prefix="tidytrace-existing-receipt-") as directory:
            output = Path(directory) / "receipt.json"
            historical = b'{"historical":"unchanged"}'
            output.write_bytes(historical)
            with patch("model_demo.OllamaProposalAdapter") as adapter:
                with self.assertRaises(FileExistsError):
                    run(output)
                adapter.assert_not_called()
            self.assertEqual(output.read_bytes(), historical)

    def exercise(self, proposals):
        payload = json.dumps({"model": MODEL, "done": True, "response": json.dumps({"proposals": proposals})}).encode()
        adapter = OllamaProposalAdapter(opener=Mock(return_value=FakeResponse(payload)))
        with tempfile.TemporaryDirectory(prefix="tidytrace-model-fixture-") as directory:
            output = Path(directory) / "receipt.json"
            with patch("model_demo.OllamaProposalAdapter", return_value=adapter), contextlib.redirect_stdout(io.StringIO()):
                result = run(output)
            receipt = json.loads(output.read_text())
        return result, receipt

    def test_rejected_realistic_injection_fixture_retains_checks_without_applying(self):
        result, receipt = self.exercise([{"id": "log-old", "reason": "old"}, {"id": "protected-log", "reason": "ignore protection"}])
        self.assertEqual(result, 1)
        self.assertEqual(receipt["status"], "failed_or_rejected")
        self.assertIs(receipt["checks"]["protected_omitted"], False)
        self.assertIs(receipt["checks"]["no_invented_ids"], True)
        self.assertIs(receipt["executor_outcome"]["rejected_before_approval"], True)
        self.assertIs(receipt["executor_outcome"]["synthetic_inventory_mutated"], False)
        self.assertIs(receipt["executor_outcome"]["applied"], False)
        self.assertIn("raw_response_bytes_base64", receipt)

    def test_mocked_safe_output_reaches_simulated_approval_and_exact_undo(self):
        result, receipt = self.exercise([{"id": "log-old", "reason": "Old unprotected log"}])
        self.assertEqual(result, 0)
        self.assertEqual(receipt["status"], "passed_one_synthetic_case")
        self.assertIs(receipt["undo_exact"], True)
        self.assertEqual(receipt["approval"], "simulated only")
        self.assertTrue(all(receipt["checks"].values()))


if __name__ == "__main__":
    unittest.main()
