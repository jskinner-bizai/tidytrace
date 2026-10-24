"""Offline demonstration. The approval below is simulated, not a human receipt."""
import json

from langgraph.types import Command
from tidytrace import build_workflow, sample_inventory, undo


def demo():
    workflow = build_workflow()
    config = {"configurable": {"thread_id": "synthetic-demo"}}
    preview = workflow.invoke({"items": sample_inventory()}, config)
    assert "__interrupt__" in preview
    print("SYNTHETIC ONLY | deterministic planner | real LangGraph runtime")
    print("Preview: " + json.dumps(preview["plan"]))
    applied = workflow.invoke(Command(resume={"approve": True, "plan_id": preview["plan_id"]}), config)
    print("SIMULATED APPROVAL: " + json.dumps(applied["receipt"]))
    restored = undo(applied, applied["receipt"]["id"])
    assert restored["items"] == sample_inventory()
    print("Undo restored exact original inventory.")
    print("Trace: " + json.dumps(restored["events"]))
    return restored


if __name__ == "__main__":
    demo()
