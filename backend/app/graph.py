"""Graph wiring, routing functions, and the two entry points used by the API."""
from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app import nodes
from app.rag import to_source_docs
from app.schemas import (
    ChatResponse,
    Intent,
    ProposedAction,
    ReviewStatus,
    ToolResult,
    TraceStep,
)
from app.state import AgentState
from app.tools import requires_approval

MAX_VALIDATION_RETRIES = 1
HOLDING_MESSAGE = (
    "Your request needs review by a support agent before we can proceed. "
    "It is pending approval, and nothing has been done yet."
)


class NoPendingReview(Exception):
    """No paused review exists for this thread (unknown, finished, or already handled)."""


# --------------------------------------------------------------------------
# Routing functions: read state, return a label; the label maps to the next node
# --------------------------------------------------------------------------
def route_on_failure(state: AgentState) -> str:
    return "error" if state.get("failed") else "ok"


def route_after_decide(state: AgentState) -> str:
    if state.get("failed"):
        return "error"
    proposed = state.get("proposed_action")
    if not proposed:
        return "answer"
    return "review" if requires_approval(proposed["tool"]) else "tool"


def route_after_review(state: AgentState) -> str:
    return "approved" if state.get("review_status") == "approved" else "rejected"


def route_after_validate(state: AgentState) -> str:
    if state.get("failed"):
        return "error"
    validation = state.get("validation", {})
    if validation.get("passes"):
        return "done"
    if validation.get("needs_human") and not state.get("escalated"):
        return "escalate"
    if not validation.get("needs_human") and state.get("retry_count", 0) <= MAX_VALIDATION_RETRIES:
        return "retry"
    injection = state.get("analysis", {}).get("injection_suspected", False)
    if not state.get("escalated") and not injection:
        return "escalate"
    return "error"


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------
def build_graph(checkpointer: Any = None):
    builder = StateGraph(AgentState)

    builder.add_node("understand_query", nodes.understand_query)
    builder.add_node("retrieve_knowledge", nodes.retrieve_knowledge)
    builder.add_node("decide_action", nodes.decide_action)
    builder.add_node("human_review", nodes.human_review)
    builder.add_node("prepare_escalation", nodes.prepare_escalation)
    builder.add_node("execute_tool", nodes.execute_tool_node)
    builder.add_node("draft_answer", nodes.draft_answer)
    builder.add_node("validate_response", nodes.validate_response)
    builder.add_node("safe_error", nodes.safe_error)

    builder.add_edge(START, "understand_query")
    builder.add_conditional_edges(
        "understand_query", route_on_failure, {"ok": "retrieve_knowledge", "error": "safe_error"}
    )
    builder.add_conditional_edges(
        "retrieve_knowledge", route_on_failure, {"ok": "decide_action", "error": "safe_error"}
    )
    builder.add_conditional_edges(
        "decide_action",
        route_after_decide,
        {
            "answer": "draft_answer",
            "tool": "execute_tool",
            "review": "human_review",
            "error": "safe_error",
        },
    )
    builder.add_conditional_edges(
        "human_review", route_after_review, {"approved": "execute_tool", "rejected": "draft_answer"}
    )
    builder.add_edge("execute_tool", "draft_answer")
    builder.add_conditional_edges(
        "draft_answer", route_on_failure, {"ok": "validate_response", "error": "safe_error"}
    )
    builder.add_conditional_edges(
        "validate_response",
        route_after_validate,
        {
            "done": END,
            "retry": "draft_answer",
            "escalate": "prepare_escalation",
            "error": "safe_error",
        },
    )
    builder.add_edge("prepare_escalation", "human_review")
    builder.add_edge("safe_error", END)

    # InMemorySaver: real interrupt/resume, but pending approvals are lost on restart
    # and it only works within one process (run a single uvicorn worker).
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


@lru_cache
def get_graph():
    """One graph (and therefore one checkpointer) per process."""
    return build_graph()


# --------------------------------------------------------------------------
# Entry points used by FastAPI
# --------------------------------------------------------------------------
def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _build_response(thread_id: str) -> ChatResponse:
    snapshot = get_graph().get_state(_config(thread_id))
    values = snapshot.values
    pending = "human_review" in snapshot.next  # a paused graph is waiting at human_review

    analysis = values.get("analysis")
    review_status = ReviewStatus(values.get("review_status", "not_required"))
    pending_action = None
    answer = values.get("final_answer")
    if pending:
        review_status = ReviewStatus.PENDING
        pending_action = ProposedAction.model_validate(values["proposed_action"])
        answer = HOLDING_MESSAGE

    return ChatResponse(
        thread_id=thread_id,
        answer=answer,
        intent=Intent(analysis["intent"]) if analysis else None,
        sources=to_source_docs(values.get("retrieved", [])),
        tool_actions=[ToolResult.model_validate(r) for r in values.get("tool_results", [])],
        review_status=review_status,
        pending_action=pending_action,
        trace=[TraceStep.model_validate(t) for t in values.get("trace", [])],
        errors=list(values.get("errors", [])),
    )


def run_chat(query: str) -> ChatResponse:
    """Start a new run. Returns either a final answer or a pending human review."""
    thread_id = str(uuid.uuid4())
    initial: AgentState = {
        "thread_id": thread_id,
        "query": query,
        "review_status": "not_required",
        "retry_count": 0,
        "escalated": False,
        "failed": False,
        "tool_results": [],
        "errors": [],
        "trace": [],
    }
    get_graph().invoke(initial, _config(thread_id))
    return _build_response(thread_id)


def resume_chat(thread_id: str, approve: bool, reviewer_note: Optional[str] = None) -> ChatResponse:
    """Resume a paused run with the human's decision."""
    snapshot = get_graph().get_state(_config(thread_id))
    if "human_review" not in snapshot.next:
        raise NoPendingReview(thread_id)
    get_graph().invoke(
        Command(resume={"approved": approve, "reviewer_note": reviewer_note}), _config(thread_id)
    )
    return _build_response(thread_id)