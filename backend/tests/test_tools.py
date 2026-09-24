import dataclasses

import pytest

from app.tools import TOOL_REGISTRY, execute_tool, list_tickets, requires_approval


# ---- lookup_order_status -------------------------------------------------
def test_lookup_existing_order():
    result = execute_tool("lookup_order_status", {"order_id": "ORD-1001"})
    assert result.success
    assert result.data["order_id"] == "ORD-1001"
    assert result.data["status"] == "active"


def test_lookup_normalizes_case_and_whitespace():
    result = execute_tool("lookup_order_status", {"order_id": " ord-1002 "})
    assert result.success and result.data["order_id"] == "ORD-1002"


def test_lookup_not_found():
    result = execute_tool("lookup_order_status", {"order_id": "ORD-9999"})
    assert not result.success
    assert "not found" in result.error


@pytest.mark.parametrize(
    "bad_id", ["1001", "ORD-12", "ORD-ABCD", "ORD-1001; DROP TABLE orders", "../etc/passwd", ""]
)
def test_lookup_rejects_malformed_ids(bad_id):
    result = execute_tool("lookup_order_status", {"order_id": bad_id})
    assert not result.success
    assert result.error.startswith("Invalid arguments")


def test_lookup_rejects_wrong_types_missing_and_extra_args():
    assert not execute_tool("lookup_order_status", {"order_id": 1001}).success
    assert not execute_tool("lookup_order_status", {}).success
    injected = execute_tool("lookup_order_status", {"order_id": "ORD-1001", "admin": True})
    assert not injected.success


# ---- allowlist -------------------------------------------------------------
def test_unknown_tool_is_rejected():
    result = execute_tool("delete_account", {"user": "someone"})
    assert not result.success
    assert "not an allowed tool" in result.error


def test_requires_approval_flags():
    assert requires_approval("create_support_ticket") is True
    assert requires_approval("lookup_order_status") is False
    assert requires_approval("something_unknown") is True  # fail safe


def test_tool_crash_becomes_failure_not_exception(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("secret internal detail")

    spec = dataclasses.replace(TOOL_REGISTRY["lookup_order_status"], func=boom)
    monkeypatch.setitem(TOOL_REGISTRY, "lookup_order_status", spec)

    result = execute_tool("lookup_order_status", {"order_id": "ORD-1001"})
    assert not result.success
    assert "secret internal detail" not in result.error


# ---- create_support_ticket -------------------------------------------------
def test_create_ticket_success_and_persisted(tools_env):
    result = execute_tool(
        "create_support_ticket",
        {"issue": "Customer reports a duplicate charge on ORD-1002", "priority": "high"},
        thread_id="t-1",
    )
    assert result.success
    assert result.data["ticket_id"] == "TCK-1001"
    assert result.data["priority"] == "high"

    tickets = list_tickets()
    assert [t.ticket_id for t in tickets] == ["TCK-1001"]


def test_ticket_ids_increment(tools_env):
    args = {"issue": "Cannot log in after password reset", "priority": "medium"}
    first = execute_tool("create_support_ticket", args, thread_id="t-1")
    second = execute_tool("create_support_ticket", args, thread_id="t-2")
    assert (first.data["ticket_id"], second.data["ticket_id"]) == ("TCK-1001", "TCK-1002")


def test_create_ticket_is_idempotent_per_thread(tools_env):
    args = {"issue": "Refund request for annual plan", "priority": "low"}
    first = execute_tool("create_support_ticket", args, thread_id="same-thread")
    again = execute_tool("create_support_ticket", args, thread_id="same-thread")
    assert again.success
    assert again.data["ticket_id"] == first.data["ticket_id"]
    assert again.data["already_existed"] is True
    assert len(list_tickets()) == 1


@pytest.mark.parametrize(
    "args",
    [
        {"issue": "too short", "priority": "high"},
        {"issue": "x" * 501, "priority": "high"},
        {"issue": "A perfectly fine issue description", "priority": "urgent"},
        {"issue": "A perfectly fine issue description"},
        {"issue": "A perfectly fine issue description", "priority": "low", "status": "closed"},
    ],
)
def test_create_ticket_rejects_invalid_args(tools_env, args):
    result = execute_tool("create_support_ticket", args)
    assert not result.success
    assert result.error.startswith("Invalid arguments")
    assert list_tickets() == []  # nothing was written