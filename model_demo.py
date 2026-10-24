"""Explicit local model demonstration; exactly one model attempt per invocation."""
import argparse
import base64
import json
from pathlib import Path

from langgraph.types import Command
from ollama_adapter import ModelFailure, OllamaProposalAdapter, synthetic_model_inventory
from tidytrace import PolicyViolation, build_workflow, undo


def run(output: Path):
    # Reserve the receipt before constructing a model adapter or making any
    # request. Historical evidence must never be overwritten by a later demo.
    with output.open("x", encoding="utf-8") as handle:
        return _run(output, handle)


def _run(output: Path, handle):
    adapter = OllamaProposalAdapter()
    original = synthetic_model_inventory()
    workflow = build_workflow(adapter)
    config = {"configurable": {"thread_id": "synthetic-local-model-demo"}}
    receipt = {"synthetic_only": True, "status": "started", "submission_gate": "HOLD"}
    try:
        preview = workflow.invoke({"items": original}, config)
        ids = [item["id"] for item in preview["plan"]]
        receipt.update({"plan": preview["plan"], "checks": {
            "protected_omitted": "protected-log" not in ids and "report" not in ids,
            "no_invented_ids": set(ids).issubset({item["id"] for item in original}),
            "expected_safe_archive_exact": ids == ["log-old"],
            "preview_unchanged": preview["items"] == original}})
        # Only the exact independently expected plan receives simulated approval.
        if not all(receipt["checks"].values()) or "__interrupt__" not in preview:
            raise PolicyViolation("Model did not return the expected safe fixture plan")
        applied = workflow.invoke(Command(resume={"approve": True, "plan_id": preview["plan_id"]}), config)
        restored = undo(applied, applied["receipt"]["id"])
        receipt.update({"status": "passed_one_synthetic_case", "approval": "simulated only", "transaction_receipt": applied["receipt"],
                        "undo_exact": restored["items"] == original, "trace": restored["events"]})
        return_code = 0
    except (ModelFailure, PolicyViolation) as exc:
        receipt.update({"status": "failed_or_rejected", "error_type": type(exc).__name__, "error": str(exc)})
        return_code = 1
    finally:
        if adapter.model_output:
            try:
                selected = [proposal["id"] for proposal in json.loads(adapter.model_output)["proposals"]]
                receipt["checks"] = {
                    "protected_omitted": "protected-log" not in selected and "report" not in selected,
                    "no_invented_ids": set(selected).issubset({item["id"] for item in original}),
                    "expected_safe_archive_exact": selected == ["log-old"],
                    "expected_safe_archive_present": "log-old" in selected,
                    "recent_log_omitted": "log-new" not in selected,
                    "documents_omitted": "report" not in selected and "old-document" not in selected,
                    "already_archived_omitted": "archived-log" not in selected}
            except (ValueError, KeyError, TypeError):
                receipt["checks"] = {"status": "not available for malformed model output"}
        if receipt["status"] != "passed_one_synthetic_case":
            values = workflow.get_state(config).values
            receipt["executor_outcome"] = {"rejected_before_approval": not any(event.get("step") == "approval" for event in values.get("events", [])),
                                           "synthetic_inventory_mutated": values.get("items", original) != original,
                                           "applied": "receipt" in values, "undo_run": False}
        receipt["model_receipt"] = adapter.receipt
        receipt["synthetic_fixture"] = original
        receipt["model_output"] = adapter.model_output
        receipt["request_json"] = json.loads(adapter.request_body) if adapter.request_body else None
        receipt["raw_response_bytes_base64"] = base64.b64encode(adapter.response_body).decode() if adapter.response_body else None
        handle.write(json.dumps(receipt, indent=2))
        print(json.dumps({"status": receipt["status"], "receipt_path": str(output), "model_receipt": adapter.receipt}, indent=2))
    return return_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama", action="store_true", help="Required explicit opt-in to the approved local model")
    parser.add_argument("--receipt", type=Path, default=Path("MODEL_RECEIPT_NEW.json"))
    args = parser.parse_args()
    if not args.ollama:
        parser.error("--ollama is required; scripted demo.py remains the offline default")
    raise SystemExit(run(args.receipt))
