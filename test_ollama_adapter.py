import io
import json
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError
from ollama_transport import TransportFailure

from ollama_adapter import ENDPOINT, MAX_RESPONSE_BYTES, MODEL, TIMEOUT_SECONDS, ModelFailure, NoRedirect, OllamaProposalAdapter, synthetic_model_inventory
from tidytrace import PolicyViolation, build_workflow


class FakeResponse(io.BytesIO):
    status = 200


def envelope(output):
    return json.dumps({"model": MODEL, "done": True, "response": json.dumps(output), "eval_count": 15}).encode()


class AdapterTests(unittest.TestCase):
    def adapter(self, output):
        opener = Mock(return_value=FakeResponse(envelope(output)))
        return OllamaProposalAdapter(opener=opener), opener

    def test_real_request_contract_and_exactly_one_attempt(self):
        adapter, opener = self.adapter({"proposals": [{"id": "log-old", "reason": "Old unprotected log"}]})
        result = adapter(synthetic_model_inventory())
        self.assertEqual(result[0]["id"], "log-old")
        req = opener.call_args.args[0]
        self.assertEqual(req.full_url, ENDPOINT)
        payload = json.loads(req.data)
        self.assertEqual(payload["model"], MODEL)
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["options"], {"temperature": 0, "seed": 42, "num_predict": 384})
        self.assertEqual(payload["format"]["additionalProperties"], False)
        self.assertNotIn("IGNORE PROTECTED FLAG", payload["prompt"])
        self.assertNotIn("build-2026-08.log", payload["prompt"])
        self.assertFalse(adapter.receipt["untrusted_names_sent"])
        self.assertEqual(opener.call_args.kwargs["timeout"], TIMEOUT_SECONDS)
        self.assertEqual(adapter.receipt["status"], "schema_valid_proposal")
        with self.assertRaisesRegex(ModelFailure, "one attempt"):
            adapter(synthetic_model_inventory())
        self.assertEqual(opener.call_count, 1)

    def test_timeout_and_connection_failure_do_not_retry(self):
        for failure in [TimeoutError("fixture timeout"), URLError("fixture connection error")]:
            with self.subTest(failure=failure):
                opener = Mock(side_effect=failure)
                adapter = OllamaProposalAdapter(opener=opener)
                with self.assertRaises(ModelFailure):
                    adapter(synthetic_model_inventory())
                self.assertEqual(adapter.receipt["status"], "failed")
                with self.assertRaises(ModelFailure):
                    adapter(synthetic_model_inventory())
                self.assertEqual(opener.call_count, 1)

    def test_uncertain_worker_cleanup_fails_receipt_without_retry(self):
        with patch("ollama_adapter.bounded_http_request", side_effect=TransportFailure("cleanup could not be confirmed")) as transport:
            adapter = OllamaProposalAdapter()
            with self.assertRaises(ModelFailure):
                adapter(synthetic_model_inventory())
            self.assertEqual(adapter.receipt["status"], "failed")
            self.assertEqual(adapter.receipt["error_type"], "TransportFailure")
            with self.assertRaises(ModelFailure):
                adapter(synthetic_model_inventory())
            self.assertEqual(transport.call_count, 1)

    def test_oversized_and_malformed_envelopes_fail(self):
        for body in [b"x" * (MAX_RESPONSE_BYTES + 1), b"not-json", b'{"model": "x", "done": true, "response": "{}"}',
                     b'{"model":"qwen2.5:14b","done":false,"response":"{}"}']:
            with self.subTest(body=body[:40]):
                adapter = OllamaProposalAdapter(opener=Mock(return_value=FakeResponse(body)))
                with self.assertRaises(ModelFailure):
                    adapter(synthetic_model_inventory())
                self.assertEqual(adapter.receipt["status"], "failed")

    def test_strict_schema_rejects_extras_duplicates_and_bad_types(self):
        outputs = [{"proposals": [], "command": "delete"}, {"proposals": [{"id": "log-old", "reason": "x", "action": "delete"}]},
                   {"proposals": [{"id": "log-old", "reason": ""}]}, {"proposals": [{"id": "log-old", "reason": "x"}] * 2},
                   {"proposals": "wrong"}, {"proposals": [{"id": True, "reason": "x"}]}]
        for output in outputs:
            with self.subTest(output=output):
                adapter, opener = self.adapter(output)
                with self.assertRaises(ModelFailure):
                    adapter(synthetic_model_inventory())

    def test_duplicate_json_keys_rejected(self):
        body = json.dumps({"model": MODEL, "done": True, "response": '{"proposals": [], "proposals": []}'}).encode()
        adapter = OllamaProposalAdapter(opener=Mock(return_value=FakeResponse(body)))
        with self.assertRaisesRegex(ModelFailure, "Duplicate"):
            adapter(synthetic_model_inventory())

    def test_unknown_and_protected_ids_are_rejected_by_executor(self):
        for item_id in ["invented-id", "protected-log", "archived-log"]:
            with self.subTest(item_id=item_id):
                adapter, opener = self.adapter({"proposals": [{"id": item_id, "reason": "Adversarial model output"}]})
                workflow = build_workflow(adapter)
                config = {"configurable": {"thread_id": self.id() + item_id}}
                with self.assertRaises(PolicyViolation):
                    workflow.invoke({"items": synthetic_model_inventory()}, config)
                state = workflow.get_state(config).values
                self.assertEqual(state["items"], synthetic_model_inventory())
                self.assertNotIn("receipt", state)

    def test_unapproved_nonfixture_input_never_calls_endpoint(self):
        adapter, opener = self.adapter({"proposals": []})
        changed = synthetic_model_inventory()
        changed[0]["name"] = "not the shipped synthetic fixture"
        with self.assertRaises(PolicyViolation):
            adapter(changed)
        opener.assert_not_called()

    def test_postresponse_elapsed_deadline_rejects_otherwise_valid_output(self):
        values = iter([0.0, TIMEOUT_SECONDS + 1.0, TIMEOUT_SECONDS + 1.0])
        adapter = OllamaProposalAdapter(opener=Mock(return_value=FakeResponse(envelope({"proposals": []}))), clock=lambda: next(values))
        with self.assertRaisesRegex(ModelFailure, "deadline"):
            adapter(synthetic_model_inventory())

    def test_default_transport_disables_proxy_and_redirect_fallback(self):
        body = envelope({"proposals": []})
        with patch("ollama_adapter.bounded_http_request", return_value=body) as transport:
            adapter = OllamaProposalAdapter()
            adapter(synthetic_model_inventory())
            req, timeout, limit, receipt = transport.call_args.args
            self.assertEqual(req.full_url, ENDPOINT)
            self.assertEqual(timeout, TIMEOUT_SECONDS)
            self.assertEqual(limit, MAX_RESPONSE_BYTES)
            self.assertIs(receipt, adapter.receipt)
        with self.assertRaisesRegex(ModelFailure, "Redirect"):
            NoRedirect().redirect_request(None, None, 302, "found", {}, "https://elsewhere.invalid")


if __name__ == "__main__":
    unittest.main()
