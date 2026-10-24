"""Interactive, offline walkthrough of the fixed synthetic TidyTrace inventory."""
from __future__ import annotations

import json
import sys
from typing import TextIO

from langgraph.types import Command
from tidytrace import build_workflow, sample_inventory, undo


def _show_inventory(items: list[dict], output: TextIO) -> None:
    for item in items:
        print(json.dumps({key: item[key] for key in
                          ("id", "name", "kind", "age_days", "protected", "location")},
                         ensure_ascii=True, sort_keys=True), file=output)


def _read_confirmation(prompt: str, source: TextIO, output: TextIO) -> str | None:
    print(prompt, file=output, flush=True)
    try:
        # Each confirmation is one short exact command. Longer input cannot match.
        value = source.readline(512)
    except (EOFError, KeyboardInterrupt):
        return None
    if not value:
        return None
    return value.removesuffix("\n").removesuffix("\r")


def run(source: TextIO | None = None, output: TextIO | None = None) -> dict:
    """Return the final in-memory state; no user inventory, network or file access."""
    source = sys.stdin if source is None else source
    output = sys.stdout if output is None else output
    print("TidyTrace | SYNTHETIC INVENTORY ONLY | offline scripted planner", file=output)
    print("This walkthrough changes in-memory objects only. All state disappears on exit.", file=output)
    print("Terminal confirmations are demonstration choices, not authenticated consent or contest receipts.", file=output)
    print("\nOriginal synthetic inventory:", file=output)
    original = sample_inventory()
    _show_inventory(original, output)

    workflow = build_workflow()
    config = {"configurable": {"thread_id": "terminal-synthetic-demo"}}
    preview = workflow.invoke({"items": original}, config)
    print("\nArchive preview:", file=output)
    for action in preview["plan"]:
        print(json.dumps(action, ensure_ascii=True, sort_keys=True), file=output)
    print("Exact plan ID: " + preview["plan_id"], file=output)
    if not preview["plan"]:
        print("No changes proposed.", file=output)
        return preview

    try:
        interactive = source.isatty()
    except (AttributeError, OSError):
        interactive = False
    if interactive:
        choice = _read_confirmation(
            "Type 'approve " + preview["plan_id"] + "' to apply this exact synthetic plan.\n"
            "Type 'decline', press Enter, or end input to leave it unchanged.", source, output)
        approved = choice == "approve " + preview["plan_id"]
    else:
        print("Noninteractive input: approval is disabled; the plan is declined.", file=output)
        approved = False

    result = workflow.invoke(Command(resume={"approve": approved, "plan_id": preview["plan_id"]}), config)
    if not approved:
        print("Declined. Synthetic inventory is unchanged; no transaction receipt was created.", file=output)
        return result

    print("\nApplied to synthetic inventory:", file=output)
    _show_inventory(result["items"], output)
    print("Synthetic transaction receipt (not a competition submission receipt):", file=output)
    print(json.dumps(result["receipt"], ensure_ascii=True, indent=2, sort_keys=True), file=output)
    receipt_id = result["receipt"]["id"]
    choice = _read_confirmation(
        "Type 'undo " + receipt_id + "' to restore the exact original synthetic inventory.\n"
        "Press Enter or end input to finish with the in-memory change.", source, output)
    if choice == "undo " + receipt_id:
        result = undo(result, receipt_id)
        print("Undo restored the exact original synthetic inventory:", file=output)
        _show_inventory(result["items"], output)
    else:
        print("No undo requested. The synthetic change lasts only until this process exits.", file=output)
    print("Trace: " + json.dumps(result["events"], ensure_ascii=True, sort_keys=True), file=output)
    return result


def main() -> int:
    if len(sys.argv) > 1:
        if sys.argv[1:] in (["--help"], ["-h"]):
            print("Usage: python terminal_demo.py\nInteractive offline walkthrough; no arguments or input files.")
            return 0
        print("No arguments or input files are accepted. Use --help.", file=sys.stderr)
        return 2
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
