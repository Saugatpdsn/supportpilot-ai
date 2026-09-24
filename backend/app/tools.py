"""Mock business tools and the allowlisted tool registry.

`execute_tool()` is the ONLY way the agent runs a tool. It:
  1. rejects tools that are not in TOOL_REGISTRY,
  2. validates arguments with a strict Pydantic model (unknown args are rejected),
  3. never raises: every failure becomes ToolResult(success=False, ...), so the
     agent can never claim success for an action that did not happen.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.config import get_settings
from app.schemas import Priority, Ticket, ToolResult

logger = logging.getLogger(__name__)

ORDER_ID_PATTERN = re.compile(r"^ORD-\d{4}$")
_ticket_lock = threading.Lock()  # protects the JSON ticket file within one process


# --------------------------------------------------------------------------
# Argument models (strict)
# --------------------------------------------------------------------------
class LookupOrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    order_id: str

    @field_validator("order_id")
    @classmethod
    def _valid_order_id(cls, v: str) -> str:
        v = v.upper()
        if not ORDER_ID_PATTERN.fullmatch(v):
            raise ValueError("order_id must look like ORD-1234")
        return v


class CreateTicketArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    issue: str = Field(min_length=10, max_length=500)
    priority: Priority


# --------------------------------------------------------------------------
# Tool 1: lookup_order_status (read-only)
# --------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_orders(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["orders"]


def lookup_order_status(order_id: str) -> ToolResult:
    """Look up an order/subscription in the mock data. Expects an already-validated id."""
    tool = "lookup_order_status"
    try:
        orders = _load_orders(get_settings().mock_data_path)
    except (OSError, ValueError, KeyError):
        logger.exception("Could not load mock order data")
        return ToolResult(tool=tool, success=False, error="Order data is temporarily unavailable.")

    order = orders.get(order_id)
    if order is None:
        return ToolResult(tool=tool, success=False, error=f"Order {order_id} was not found.")
    return ToolResult(tool=tool, success=True, data={"order_id": order_id, **copy.deepcopy(order)})


# --------------------------------------------------------------------------
# Tool 2: create_support_ticket (write action -> requires human approval)
# --------------------------------------------------------------------------
def _read_tickets(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _write_tickets(path: Path, tickets: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(tickets, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic swap: never leaves a half-written file


def create_support_ticket(
    issue: str, priority: Priority | str, thread_id: Optional[str] = None
) -> ToolResult:
    """Create a mock ticket. Idempotent per thread_id (a double approval creates one ticket)."""
    tool = "create_support_ticket"
    path = get_settings().tickets_path
    try:
        with _ticket_lock:
            tickets = _read_tickets(path)
            if thread_id:
                for existing in tickets:
                    if existing.get("thread_id") == thread_id:
                        return ToolResult(
                            tool=tool, success=True, data={**existing, "already_existed": True}
                        )
            number = max((int(t["ticket_id"].split("-")[1]) for t in tickets), default=1000) + 1
            ticket = Ticket(
                ticket_id=f"TCK-{number}",
                issue=issue,
                priority=Priority(priority),
                thread_id=thread_id,
            )
            tickets.append(ticket.model_dump(mode="json"))
            _write_tickets(path, tickets)
    except (OSError, ValueError, KeyError):
        logger.exception("Ticket store failure")
        return ToolResult(
            tool=tool, success=False, error="Ticket could not be created. No ticket was saved."
        )

    # Log the id and priority only: the issue text may contain personal data.
    logger.info("Created ticket %s (priority=%s)", ticket.ticket_id, ticket.priority.value)
    return ToolResult(tool=tool, success=True, data=ticket.model_dump(mode="json"))


def list_tickets() -> list[Ticket]:
    with _ticket_lock:
        raw = _read_tickets(get_settings().tickets_path)
    return sorted((Ticket.model_validate(t) for t in raw), key=lambda t: t.created_at, reverse=True)


# --------------------------------------------------------------------------
# Registry (the allowlist) + dispatcher
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[..., ToolResult]
    requires_approval: bool
    accepts_thread_id: bool = False


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "lookup_order_status": ToolSpec(
        name="lookup_order_status",
        description="Look up the status, charges, and dates of an order or subscription by "
        "order ID (format ORD-1234). Read-only.",
        args_model=LookupOrderArgs,
        func=lookup_order_status,
        requires_approval=False,
    ),
    "create_support_ticket": ToolSpec(
        name="create_support_ticket",
        description="Create a support ticket for a human agent. Arguments: issue (10-500 chars), "
        "priority (low, medium, or high). Requires human approval.",
        args_model=CreateTicketArgs,
        func=create_support_ticket,
        requires_approval=True,
        accepts_thread_id=True,
    ),
}


def requires_approval(tool_name: str) -> bool:
    """Fail safe: unknown tools are treated as needing approval."""
    spec = TOOL_REGISTRY.get(tool_name)
    return True if spec is None else spec.requires_approval


def _summarize_validation_error(exc: ValidationError) -> str:
    parts = [
        f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
        for err in exc.errors()
    ]
    return "Invalid arguments: " + "; ".join(parts)


def execute_tool(name: str, args: dict[str, Any], thread_id: Optional[str] = None) -> ToolResult:
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return ToolResult(tool=str(name)[:60], success=False, error="This is not an allowed tool.")

    try:
        validated = spec.args_model.model_validate(args)
    except ValidationError as exc:
        return ToolResult(tool=name, success=False, error=_summarize_validation_error(exc))

    kwargs = validated.model_dump()
    if spec.accepts_thread_id:
        kwargs["thread_id"] = thread_id

    try:
        return spec.func(**kwargs)
    except Exception:  # noqa: BLE001 - a tool must never crash the graph
        logger.exception("Tool %s crashed", name)
        return ToolResult(
            tool=name,
            success=False,
            error="The tool failed unexpectedly. The action was not completed.",
        )