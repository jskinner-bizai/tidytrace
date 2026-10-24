import io
import socket
import unittest
from unittest.mock import patch

from terminal_demo import run
from tidytrace import fingerprint, plan_fingerprint, sample_inventory, scripted_planner


class TerminalInput(io.StringIO):
    def isatty(self):
        return True


def expected_plan_id():
    items = sample_inventory()
    by_id = {item["id"]: item for item in items}
    plan = [{"id": proposal["id"], "name": by_id[proposal["id"]]["name"],
             "action": "archive", "reason": proposal["reason"]}
            for proposal in scripted_planner(items)]
    return plan_fingerprint(fingerprint(items), plan)


class TerminalDemoTests(unittest.TestCase):
    def invoke(self, text, interactive=True):
        output = io.StringIO()
        source = TerminalInput(text) if interactive else io.StringIO(text)
        state = run(source, output)
        return state, output.getvalue(), source

    def assert_declined(self, state):
        self.assertEqual(state["status"], "declined")
        self.assertEqual(state["items"], sample_inventory())
        self.assertNotIn("receipt", state)

    def test_decline_shows_inventory_and_preview_without_apply(self):
        state, text, _ = self.invoke("decline\n")
        self.assert_declined(state)
        self.assertIn('"id": "log-old"', text)
        self.assertIn('"id": "report"', text)
        self.assertIn("Exact plan ID: " + expected_plan_id(), text)
        self.assertIn("offline scripted planner", text)
        self.assertIn("no transaction receipt", text)

    def test_exact_plan_confirmation_applies_and_prints_synthetic_receipt(self):
        state, text, _ = self.invoke("approve " + expected_plan_id() + "\n\n")
        self.assertEqual(state["status"], "applied")
        self.assertEqual(state["items"][0]["location"], "archive")
        self.assertEqual(state["items"][2]["location"], "inbox")
        self.assertEqual(state["receipt"]["id"], expected_plan_id())
        self.assertTrue(state["receipt"]["synthetic_only"])
        self.assertIn("not a competition submission receipt", text)

    def test_wrong_plan_confirmation_cannot_apply(self):
        for choice in ("yes", "approve wrong", "approve " + expected_plan_id() + " "):
            with self.subTest(choice=choice):
                state, _, _ = self.invoke(choice + "\n")
                self.assert_declined(state)

    def test_exact_receipt_undo_restores_original(self):
        plan_id = expected_plan_id()
        state, text, _ = self.invoke("approve " + plan_id + "\nundo " + plan_id + "\n")
        self.assertEqual(state["status"], "undone")
        self.assertEqual(state["receipt"]["status"], "undone")
        self.assertEqual(state["items"], sample_inventory())
        self.assertEqual([event["step"] for event in state["events"]],
                         ["preview", "approval", "apply", "undo"])
        self.assertIn("Undo restored the exact original", text)

    def test_wrong_receipt_does_not_undo(self):
        state, _, _ = self.invoke("approve " + expected_plan_id() + "\nundo wrong\n")
        self.assertEqual(state["status"], "applied")
        self.assertEqual(state["receipt"]["status"], "applied")
        self.assertNotIn("undo", [event["step"] for event in state["events"]])

    def test_eof_before_approval_declines(self):
        state, _, _ = self.invoke("")
        self.assert_declined(state)

    def test_eof_after_approval_leaves_only_in_memory_apply(self):
        state, text, _ = self.invoke("approve " + expected_plan_id() + "\n")
        self.assertEqual(state["status"], "applied")
        self.assertIn("only until this process exits", text)

    def test_pipe_cannot_approve_even_with_the_exact_command(self):
        command = "approve " + expected_plan_id() + "\n"
        state, text, source = self.invoke(command, interactive=False)
        self.assert_declined(state)
        self.assertEqual(source.tell(), 0)
        self.assertIn("Noninteractive input: approval is disabled", text)

    def test_approval_and_undo_need_no_socket_connections(self):
        plan_id = expected_plan_id()
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")), \
             patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Network forbidden")):
            state, _, _ = self.invoke("approve " + plan_id + "\nundo " + plan_id + "\n")
        self.assertEqual(state["items"], sample_inventory())


if __name__ == "__main__":
    unittest.main()
