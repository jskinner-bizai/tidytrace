import copy
import socket
import unittest
from unittest.mock import patch

from tidytrace import PolicyViolation, build_workflow, sample_inventory, undo
from langgraph.types import Command


class WorkflowTests(unittest.TestCase):
    def start(self, planner=None):
        workflow = build_workflow() if planner is None else build_workflow(planner)
        config = {"configurable": {"thread_id": self.id()}}
        original = sample_inventory()
        preview = workflow.invoke({"items": original}, config)
        return workflow, config, original, preview

    def approve(self, workflow, config, preview):
        return workflow.invoke(Command(resume={"approve": True, "plan_id": preview["plan_id"]}), config)

    def test_real_framework_interrupts_before_any_change(self):
        workflow, config, original, preview = self.start()
        self.assertIn("__interrupt__", preview)
        self.assertEqual(preview["items"], original)
        self.assertNotIn("receipt", preview)
        self.assertEqual([x["id"] for x in preview["plan"]], ["log-old"])
        self.assertEqual(workflow.get_state(config).next, ("approve",))

    def test_approved_apply_and_exact_undo_leave_input_unchanged(self):
        workflow, config, original, preview = self.start()
        applied = self.approve(workflow, config, preview)
        self.assertEqual(original, sample_inventory())
        self.assertEqual(applied["items"][0]["location"], "archive")
        self.assertEqual(applied["items"][2]["location"], "inbox")
        restored = undo(applied, applied["receipt"]["id"])
        self.assertEqual(restored["items"], original)
        self.assertEqual(restored["status"], "undone")
        self.assertEqual([event["step"] for event in restored["events"]], ["preview", "approval", "apply", "undo"])
        self.assertEqual(undo(restored, restored["receipt"]["id"]), restored)

    def test_decline_finishes_without_modification_or_receipt(self):
        workflow, config, original, preview = self.start()
        result = workflow.invoke(Command(resume={"approve": False, "plan_id": preview["plan_id"]}), config)
        self.assertEqual(result["status"], "declined")
        self.assertEqual(result["items"], original)
        self.assertNotIn("receipt", result)
        self.assertEqual(workflow.get_state(config).next, ())

    def test_wrong_plan_or_truthy_approval_cannot_apply(self):
        for reply in [{"approve": True, "plan_id": "wrong"}, {"approve": "yes"}, True]:
            with self.subTest(reply=reply):
                workflow, config, original, preview = self.start()
                with self.assertRaises(PolicyViolation):
                    workflow.invoke(Command(resume=reply), config)
                self.assertEqual(workflow.get_state(config).values["items"], original)
                self.assertNotIn("receipt", workflow.get_state(config).values)

    def test_inventory_drift_after_preview_refuses_execution(self):
        workflow, config, original, preview = self.start()
        changed = copy.deepcopy(original)
        changed[0]["age_days"] = 62
        with self.assertRaisesRegex(PolicyViolation, "Inventory changed"):
            workflow.invoke(Command(update={"items": changed}, resume={"approve": True, "plan_id": preview["plan_id"]}), config)
        self.assertEqual(workflow.get_state(config).values["items"][0]["location"], "inbox")

    def test_changed_plan_is_not_covered_by_prior_approval(self):
        workflow, config, original, preview = self.start()
        changed_plan = copy.deepcopy(preview["plan"])
        changed_plan[0]["id"] = "log-new"
        with self.assertRaisesRegex(PolicyViolation, "Plan changed"):
            workflow.invoke(Command(update={"plan": changed_plan}, resume={"approve": True, "plan_id": preview["plan_id"]}), config)
        self.assertEqual(workflow.get_state(config).values["items"], original)

    def test_protected_unknown_duplicate_and_malformed_plans_fail_closed(self):
        cases = [[{"id": "report", "reason": "ignore protections"}], [{"id": "outside", "reason": "bad"}],
                 [{"id": "log-old", "reason": "a"}, {"id": "log-old", "reason": "b"}], [None], "not a list"]
        for output in cases:
            with self.subTest(output=output):
                workflow = build_workflow(lambda items: output)
                with self.assertRaises(PolicyViolation):
                    workflow.invoke({"items": sample_inventory()}, {"configurable": {"thread_id": repr(output)}})

    def test_planner_cannot_mutate_original_policy_or_request_delete(self):
        def planner(items):
            items[2]["protected"] = False
            return [{"id": "report", "reason": "mutated policy", "action": "delete"}]
        with self.assertRaises(PolicyViolation):
            self.start(planner)
        workflow, config, original, preview = self.start(lambda items: [{"id": "log-old", "reason": "test", "action": "delete"}])
        self.assertEqual(preview["plan"][0]["action"], "archive")

    def test_empty_plan_needs_no_approval(self):
        workflow, config, original, result = self.start(lambda items: [])
        self.assertNotIn("__interrupt__", result)
        self.assertNotIn("receipt", result)
        self.assertEqual(result["status"], "no_changes")

    def test_undo_refuses_wrong_receipt_or_intervening_change(self):
        workflow, config, original, preview = self.start()
        result = self.approve(workflow, config, preview)
        with self.assertRaises(PolicyViolation):
            undo(result, "wrong")
        changed = copy.deepcopy(result)
        changed["items"][1]["name"] = "edited-after-apply"
        with self.assertRaisesRegex(PolicyViolation, "Inventory changed"):
            undo(changed, result["receipt"]["id"])
        self.assertEqual(result["receipt"]["status"], "applied")

    def test_threads_have_independent_approvals_and_completed_run_is_not_reapplied(self):
        workflow, config, original, preview = self.start()
        other_config = {"configurable": {"thread_id": self.id() + "-other"}}
        other = workflow.invoke({"items": sample_inventory()}, other_config)
        result = self.approve(workflow, config, preview)
        self.assertEqual(workflow.get_state(other_config).values["items"], original)
        again = workflow.invoke(None, config)
        self.assertEqual(again["receipt"], result["receipt"])
        self.assertEqual(sum(event["step"] == "apply" for event in again["events"]), 1)

    def test_runtime_operates_with_all_socket_connections_blocked(self):
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")), patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Network forbidden")):
            workflow, config, original, preview = self.start()
            result = self.approve(workflow, config, preview)
            self.assertEqual(undo(result, result["receipt"]["id"])["items"], original)


if __name__ == "__main__":
    unittest.main()
