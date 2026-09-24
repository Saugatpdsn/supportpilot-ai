"""LangGraph nodes. Each takes the AgentState and returns a PARTIAL state update."""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

from langgraph.types import interrupt

from app import guardrails, llm, prompts
from app.rag import DEFAULT_TOP_K, retrieve
from app.schemas import (
    ActionDecision,
    ActionType,
    DraftedAnswer,
    Intent,
    Priority,
    ProposedAction,
    QueryAnalysis,
    ToolResult,
    ValidationResult,
)
from app.state import AgentState
from app.tools import execute_tool, requires_approval

logger = logging.getLogger(__name__)

SAFE_ERROR_MESSAGE = (
    "Sorry, I couldn't process your request automatically right now. "
    "Please try again in a few minutes or contact support directly."
)


def _trace(node: str, summary: str) -> list[dict[str, str]]:
    return [{"node": node, "summary": summary}]


def _failure(node: str, message: str) -> dict[str, Any]:
    return {"failed": True, "errors": [message], "trace": _trace(node, f"FAILED: {message}")}


# ---------------------------------------------------------------- 1. understand
def understand_query(state: AgentState) -> dict[str, Any]:
    query = state["query"]
    try:
        analysis = llm.structured_call(
            QueryAnalysis, prompts.UNDERSTAND_SYSTEM, prompts.understand_input(query)
        )
    except llm.LLMError:
        return _failure("understand_query", "Could not classify the request.")

    if guardrails.heuristic_injection_signal(query):  # backup signal, OR-ed with the LLM flag
        analysis.injection_suspected = True

    summary = (
        f"intent={analysis.intent.value}, sensitive={analysis.sensitive}, "
        f"injection_suspected={analysis.injection_suspected}"
    )
    return {"analysis": analysis.model_dump(mode="json"), "trace": _trace("understand_query", summary)}


# ---------------------------------------------------------------- 2. retrieve
def retrieve_knowledge(state: AgentState) -> dict[str, Any]:
    analysis = state["analysis"]
    # For suspected injections search with the neutral summary, not the raw text.
    search_text = analysis["summary"] if analysis.get("injection_suspected") else state["query"]
    try:
        chunks = llm.with_one_retry(lambda: retrieve(search_text, k=DEFAULT_TOP_K), what="retrieval")
    except llm.LLMError:
        return _failure("retrieve_knowledge", "Knowledge base lookup failed.")

    sources = sorted({c["source"] for c in chunks})
    return {
        "retrieved": chunks,
        "trace": _trace("retrieve_knowledge", f"{len(chunks)} chunks from {', '.join(sources) or 'nothing'}"),
    }


# ---------------------------------------------------------------- 3. decide
def _order_id_in_query(order_id: str | None, query: str) -> bool:
    """The order ID must come from the customer, not from the model's imagination."""
    digits = (order_id or "").strip().split("-")[-1]
    return digits.isdigit() and digits in query


def _enforce_policy(decision: ActionDecision, analysis: QueryAnalysis, query: str) -> ActionDecision:
    """Code overrides the LLM where policy is non-negotiable."""
    if analysis.sensitive and decision.action != ActionType.ESCALATE:
        return decision.model_copy(
            update={
                "action": ActionType.ESCALATE,
                "reasoning": f"{decision.reasoning} [policy: sensitive request goes to a human]",
            }
        )
    if decision.action == ActionType.LOOKUP_ORDER and not _order_id_in_query(decision.order_id, query):
        return decision.model_copy(
            update={
                "action": ActionType.ANSWER,
                "order_id": None,
                "reasoning": f"{decision.reasoning} [policy: order ID not found in the message]",
            }
        )
    return decision


def _to_proposed_action(
    decision: ActionDecision, analysis: QueryAnalysis, query: str
) -> ProposedAction | None:
    if decision.action == ActionType.LOOKUP_ORDER:
        return ProposedAction(
            tool="lookup_order_status", args={"order_id": decision.order_id}, reason=decision.reasoning
        )
    if decision.action in (ActionType.CREATE_TICKET, ActionType.ESCALATE):
        issue = (decision.ticket_issue or analysis.summary or "").strip()
        if len(issue) < 10:
            issue = f"Customer needs help: {query}".strip()
        priority = decision.ticket_priority or Priority.MEDIUM
        return ProposedAction(
            tool="create_support_ticket",
            args={"issue": issue[:500], "priority": priority.value},
            reason=decision.reasoning,
        )
    return None


def decide_action(state: AgentState) -> dict[str, Any]:
    analysis = QueryAnalysis.model_validate(state["analysis"])
    query = state["query"]

    if analysis.injection_suspected:
        # Tools are disabled for suspected injections. We do not even ask the LLM.
        decision = ActionDecision(
            action=ActionType.ANSWER,
            reasoning="Possible prompt injection: tools disabled for this request.",
        )
    else:
        try:
            decision = llm.structured_call(
                ActionDecision,
                prompts.DECIDE_SYSTEM,
                prompts.decide_input(query, analysis, state["retrieved"]),
            )
        except llm.LLMError:
            return _failure("decide_action", "Could not decide on an action.")
        decision = _enforce_policy(decision, analysis, query)

    proposed = _to_proposed_action(decision, analysis, query)
    return {
        "decision": decision.model_dump(mode="json"),
        "proposed_action": proposed.model_dump(mode="json") if proposed else None,
        "trace": _trace("decide_action", f"action={decision.action.value}: {decision.reasoning[:120]}"),
    }


# ---------------------------------------------------------------- 4a. human review
def human_review(state: AgentState) -> dict[str, Any]:
    """Pauses the graph. NOTE: on resume LangGraph re-runs this node from the top,
    so nothing with side effects may happen before interrupt()."""
    proposed = state["proposed_action"]
    reply = interrupt({"proposed_action": proposed})

    reply = reply if isinstance(reply, dict) else {}
    approved = reply.get("approved") is True  # strict: anything else means "rejected"
    status = "approved" if approved else "rejected"
    return {
        "review_status": status,
        "reviewer_note": reply.get("reviewer_note"),
        "escalated": True,
        "trace": _trace("human_review", f"human {status} {proposed['tool']}"),
    }


def prepare_escalation(state: AgentState) -> dict[str, Any]:
    """Used when validation cannot be satisfied: propose a ticket for a human."""
    summary = state["analysis"]["summary"]
    proposed = ProposedAction(
        tool="create_support_ticket",
        args={"issue": f"Escalated to a human agent: {summary}"[:500], "priority": "medium"},
        reason="The automated reply could not be validated or the request needs a human.",
    )
    return {
        "proposed_action": proposed.model_dump(mode="json"),
        "trace": _trace("prepare_escalation", "proposed a ticket for human review"),
    }


# ---------------------------------------------------------------- 4b. execute tool
def execute_tool_node(state: AgentState) -> dict[str, Any]:
    proposed = state.get("proposed_action")
    if not proposed:
        result = ToolResult(tool="none", success=False, error="No action was proposed.")
    elif requires_approval(proposed["tool"]) and state.get("review_status") != "approved":
        # Defense in depth: even if the graph were mis-wired, no approval means no action.
        result = ToolResult(
            tool=proposed["tool"],
            success=False,
            error="This action requires human approval and was not approved. Nothing was done.",
        )
    else:
        result = execute_tool(proposed["tool"], proposed["args"], thread_id=state.get("thread_id"))

    outcome = "success" if result.success else f"failed ({result.error})"
    return {
        "tool_results": [result.model_dump(mode="json")],
        "trace": _trace("execute_tool", f"{result.tool}: {outcome}"),
    }


# ---------------------------------------------------------------- 5. draft
def draft_answer(state: AgentState) -> dict[str, Any]:
    analysis = QueryAnalysis.model_validate(state["analysis"])
    retrieved = state.get("retrieved", [])

    validation = state.get("validation") or {}
    feedback = list(validation.get("code_issues", [])) + list(validation.get("unsupported_claims", []))

    try:
        draft = llm.structured_call(
            DraftedAnswer,
            prompts.DRAFT_SYSTEM,
            prompts.draft_input(
                query=state["query"],
                analysis=analysis,
                retrieved=retrieved,
                tool_results=state.get("tool_results", []),
                review_status=state.get("review_status", "not_required"),
                today=date.today().isoformat(),
                feedback=feedback,
            ),
        )
    except llm.LLMError:
        return _failure("draft_answer", "Could not draft a reply.")

    known_sources = {c["source"] for c in retrieved}
    draft.cited_sources = [s for s in draft.cited_sources if s in known_sources]
    return {
        "draft": draft.model_dump(mode="json"),
        "trace": _trace("draft_answer", f"drafted {len(draft.answer)} chars, information_missing={draft.information_missing}"),
    }


# ---------------------------------------------------------------- 6. validate
def validate_response(state: AgentState) -> dict[str, Any]:
    analysis = QueryAnalysis.model_validate(state["analysis"])
    draft = DraftedAnswer.model_validate(state["draft"])
    tool_results = state.get("tool_results", [])
    review_status = state.get("review_status", "not_required")

    code_issues = guardrails.deterministic_answer_issues(
        draft.answer, query=state["query"], tool_results=tool_results
    )
    try:
        result = llm.structured_call(
            ValidationResult,
            prompts.VALIDATE_SYSTEM,
            prompts.validate_input(
                query=state["query"],
                retrieved=state.get("retrieved", []),
                tool_results=tool_results,
                review_status=review_status,
                draft_answer=draft.answer,
                information_missing=draft.information_missing,
            ),
        )
    except llm.LLMError:
        return _failure("validate_response", "Could not validate the reply.")

    # Backstop for policy: sensitive or uncertain cases that no human has seen yet.
    needs_human = (
        review_status == "not_required"
        and not analysis.injection_suspected
        and (analysis.sensitive or (draft.information_missing and analysis.intent != Intent.OTHER))
    )
    passes = result.passes and not code_issues and not needs_human

    update: dict[str, Any] = {
        "validation": {
            **result.model_dump(),
            "code_issues": code_issues,
            "needs_human": needs_human,
            "passes": passes,
        },
        "trace": _trace(
            "validate_response",
            f"passes={passes}, needs_human={needs_human}, "
            f"issues={len(code_issues) + len(result.unsupported_claims)}",
        ),
    }
    if passes:
        update["final_answer"] = draft.answer
    else:
        update["retry_count"] = state.get("retry_count", 0) + 1
    return update


# ---------------------------------------------------------------- safe error
def safe_error(state: AgentState) -> dict[str, Any]:
    message = SAFE_ERROR_MESSAGE
    # Be honest about anything that DID happen (e.g. an approved ticket that was created).
    for result in reversed(state.get("tool_results", [])):
        if result.get("tool") == "create_support_ticket" and result.get("success") and result.get("data"):
            message += f" Your support ticket {result['data']['ticket_id']} was created."
            break
    return {"final_answer": message, "trace": _trace("safe_error", "returned a safe fallback message")}