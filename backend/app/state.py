"""The single state object that flows through the LangGraph workflow.

Values are stored as plain dicts (Pydantic `.model_dump()`), so the checkpointer
can serialize them safely. Fields annotated with `operator.add` are *reducers*:
when a node returns {"trace": [x]}, LangGraph appends x instead of overwriting.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # --- input ---
    thread_id: str
    query: str

    # --- understand_query ---
    analysis: dict[str, Any]           # QueryAnalysis

    # --- retrieve_knowledge ---
    retrieved: list[dict[str, Any]]    # [{source, content, score}]

    # --- decide_action ---
    decision: dict[str, Any]           # ActionDecision

    # --- human review / tools ---
    proposed_action: Optional[dict[str, Any]]  # ProposedAction awaiting approval
    review_status: str                          # ReviewStatus value
    reviewer_note: Optional[str]
    escalated: bool                             # guard: only escalate once
    tool_results: Annotated[list[dict[str, Any]], operator.add]  # ToolResult list

    # --- draft / validate ---
    draft: dict[str, Any]              # DraftedAnswer
    validation: dict[str, Any]         # ValidationResult
    retry_count: int                   # draft retries (max 1)

    # --- output / observability ---
    final_answer: Optional[str]
    errors: Annotated[list[str], operator.add]
    trace: Annotated[list[dict[str, str]], operator.add]  # [{node, summary}]