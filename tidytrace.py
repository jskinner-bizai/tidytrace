"""Synthetic archive workflow. Inventory entries are objects, never real files."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Callable, TypedDict

# Remote tracing stays disabled, including when explicit local Ollama mode is used.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class PolicyViolation(ValueError):
    pass


class State(TypedDict, total=False):
    items: list[dict]
    plan: list[dict]
    plan_id: str
    snapshot: str
    approved: bool
    receipt: dict
    events: list[dict]
    status: str


def fingerprint(items: list[dict]) -> str:
    data = json.dumps(sorted(items, key=lambda item: item["id"]), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def plan_fingerprint(snapshot: str, plan: list[dict]) -> str:
    return hashlib.sha256((snapshot + json.dumps(plan, sort_keys=True)).encode()).hexdigest()


def validate_inventory(items: list[dict]) -> None:
    if not isinstance(items, list):
        raise PolicyViolation("Inventory must be a list of synthetic objects")
    seen = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise PolicyViolation("Every synthetic object needs a string id")
        if item["id"] in seen:
            raise PolicyViolation("Duplicate synthetic object id")
        seen.add(item["id"])
        if item.get("location") not in {"inbox", "archive"}:
            raise PolicyViolation("Unknown synthetic location")
        if type(item.get("protected")) is not bool:
            raise PolicyViolation("Protected flag must be explicit")


def scripted_planner(items: list[dict]) -> list[dict]:
    """Deterministic demonstration adapter, not model inference."""
    return [
        {"id": item["id"], "reason": "Synthetic generated log older than 30 days"}
        for item in items
        if item["location"] == "inbox" and not item["protected"]
        and item.get("kind") == "generated-log" and item.get("age_days", 0) > 30
    ]


def build_workflow(planner: Callable[[list[dict]], list[dict]] = scripted_planner):
    """LangGraph owns branching, checkpointing and approval interruption."""
    def propose(state: State) -> dict:
        validate_inventory(state["items"])
        items = copy.deepcopy(state["items"])
        proposals = planner(copy.deepcopy(items))
        if not isinstance(proposals, list):
            raise PolicyViolation("Planner output must be a list")
        by_id = {item["id"]: item for item in items}
        seen = set()
        plan = []
        for proposal in proposals:
            if not isinstance(proposal, dict):
                raise PolicyViolation("Malformed planner action")
            item_id = proposal.get("id")
            if not isinstance(item_id, str) or item_id not in by_id or item_id in seen:
                raise PolicyViolation("Planner referenced an unknown or duplicate object")
            item = by_id[item_id]
            if item["protected"] or item["location"] != "inbox":
                raise PolicyViolation("Protected or already archived object cannot be changed")
            reason = proposal.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise PolicyViolation("Planner must explain each recommendation")
            # Action is constrained here, never accepted from model-provided text.
            plan.append({"id": item_id, "name": item.get("name", item_id), "action": "archive", "reason": reason})
            seen.add(item_id)
        snapshot = fingerprint(items)
        plan_id = plan_fingerprint(snapshot, plan)
        return {"items": items, "plan": plan, "plan_id": plan_id, "snapshot": snapshot,
                "status": "preview" if plan else "no_changes", "events": [{"step": "preview", "count": len(plan)}]}

    def approve(state: State) -> dict:
        response = interrupt({"kind": "approval_required", "plan_id": state["plan_id"], "plan": state["plan"], "synthetic_only": True})
        if not isinstance(response, dict) or response.get("plan_id") != state["plan_id"] or type(response.get("approve")) is not bool:
            raise PolicyViolation("Approval must explicitly match this exact plan")
        approved = response["approve"]
        return {"approved": approved, "status": "approved" if approved else "declined",
                "events": state["events"] + [{"step": "approval", "approved": approved}]}

    def apply(state: State) -> dict:
        if state.get("approved") is not True:
            raise PolicyViolation("Explicit approval required")
        if fingerprint(state["items"]) != state["snapshot"]:
            raise PolicyViolation("Inventory changed after preview; create a new plan")
        if plan_fingerprint(state["snapshot"], state["plan"]) != state["plan_id"]:
            raise PolicyViolation("Plan changed after preview; create a new plan")
        items = copy.deepcopy(state["items"])
        by_id = {item["id"]: item for item in items}
        changes = []
        for action in state["plan"]:
            item = by_id[action["id"]]
            if item["protected"] or item["location"] != "inbox" or action["action"] != "archive":
                raise PolicyViolation("Plan failed execution policy")
            changes.append({"id": item["id"], "before": "inbox", "after": "archive"})
            item["location"] = "archive"
        receipt = {"id": state["plan_id"], "status": "applied", "before_snapshot": state["snapshot"],
                   "after_snapshot": fingerprint(items), "changes": changes, "synthetic_only": True}
        return {"items": items, "receipt": receipt, "status": "applied",
                "events": state["events"] + [{"step": "apply", "receipt_id": receipt["id"], "count": len(changes)}]}

    graph = StateGraph(State)
    graph.add_node("propose", propose)
    graph.add_node("approve", approve)
    graph.add_node("apply", apply)
    graph.add_edge(START, "propose")
    graph.add_conditional_edges("propose", lambda state: "approve" if state["plan"] else END)
    graph.add_conditional_edges("approve", lambda state: "apply" if state["approved"] else END)
    graph.add_edge("apply", END)
    return graph.compile(checkpointer=InMemorySaver())


def undo(state: State, receipt_id: str) -> State:
    """Undo exactly the recorded synthetic transaction, refusing intervening edits."""
    receipt = state.get("receipt")
    if not receipt or receipt.get("id") != receipt_id:
        raise PolicyViolation("Matching synthetic receipt required")
    if receipt["status"] == "undone":
        return copy.deepcopy(state)
    if receipt["status"] != "applied" or fingerprint(state["items"]) != receipt["after_snapshot"]:
        raise PolicyViolation("Inventory changed after apply; cannot safely undo")
    result = copy.deepcopy(state)
    by_id = {item["id"]: item for item in result["items"]}
    for change in receipt["changes"]:
        by_id[change["id"]]["location"] = change["before"]
    if fingerprint(result["items"]) != receipt["before_snapshot"]:
        raise PolicyViolation("Receipt does not restore the exact original inventory")
    result["receipt"]["status"] = "undone"
    result["status"] = "undone"
    result["events"].append({"step": "undo", "receipt_id": receipt_id})
    return result


def sample_inventory() -> list[dict]:
    return [
        {"id": "log-old", "name": "build-2026-08.log", "kind": "generated-log", "age_days": 61, "protected": False, "location": "inbox"},
        {"id": "log-new", "name": "build-today.log", "kind": "generated-log", "age_days": 0, "protected": False, "location": "inbox"},
        {"id": "report", "name": "quarterly-report.txt", "kind": "document", "age_days": 72, "protected": True, "location": "inbox"},
    ]
