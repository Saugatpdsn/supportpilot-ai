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
    retrieved: list[dict[str, Any]]    # [{source, chunk_id, content, score}]

    # --- decide_action ---
    decision: dict[str, Any]           # ActionDecision

    # --- human review / tools ---
    proposed_action: Optional[dict[str, Any]]  # ProposedAction
    review_status: str                          # ReviewStatus value
    reviewer_note: Optional[str]
    escalated: bool                             # a human review has been requested once
    tool_results: Annotated[list[dict[str, Any]], operator.add]  # ToolResult list

    # --- draft / validate ---
    draft: dict[str, Any]              # DraftedAnswer
    validation: dict[str, Any]         # ValidationResult + code checks
    retry_count: int                   # number of failed validations so far

    # --- output / observability ---
    failed: bool                       # set by a node when it cannot continue
    final_answer: Optional[str]
    errors: Annotated[list[str], operator.add]
    trace: Annotated[list[dict[str, str]], operator.add]  # [{node, summary}]